"""
主实验脚本

用法:
    python run_experiment.py --task ci --model gpt-4o --project openfoam
    python run_experiment.py --task dp --model deepseek-v3 --project all
    python run_experiment.py --task all --model all --project all
    python run_experiment.py --baselines --project all
"""

import json
import os
import re
import sys
import random
import argparse
import time
from collections import defaultdict

random.seed(42)

from config import MODELS, PROJECTS, SAMPLE_SIZE_CI, SAMPLE_SIZE_DP, SAMPLE_SIZE_AR
from prompts import build_ci_prompt, build_dp_prompt, build_ar_prompt, build_ci_hard_prompt
from llm_client import call_llm
from baselines import (
    load_project_data,
    baseline_ci_random, baseline_ci_frequency, baseline_ci_path_heuristic,
    baseline_dp_random, baseline_dp_frequency, baseline_dp_graph_neighbor,
    baseline_ar_random, baseline_ar_frequency, baseline_ar_graph_neighbor,
)
from build_sample_manifest import MANIFEST_PATH as MANIFEST_V1_PATH, item_key

# 题目清单版本：环境变量 SAB_MANIFEST=v1（默认，原始 AKG）/ v2（akg_v2 严格解析重建）。
# 用环境变量而非命令行参数，是因为 run_level1_icse / run_level2 / 重打分脚本在 import 时直接取 MANIFEST。
# v2：CI/CI-Hard 不变；DP/AR 题目文件取自 manifest 的 source_file（../akg_v2/...），
#     AR 真解 = manifest 中 true_targets（新图出边中按固定种子隐藏的 ceil(n/2) 条，
#     见 akg_v2/apply_ar_partial_mask.py；all_targets 为完整出边），无出边的源模块已剔除。
HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_VERSION = os.environ.get("SAB_MANIFEST", "v1")
assert MANIFEST_VERSION in ("v1", "v2"), f"SAB_MANIFEST must be v1 or v2, got {MANIFEST_VERSION}"
MANIFEST_PATH = (MANIFEST_V1_PATH if MANIFEST_VERSION == "v1"
                 else os.path.join(HERE, "samples", "manifest_v2.json"))
AKG_V2_DIR = os.path.join(HERE, "..", "systems")

with open(MANIFEST_PATH, encoding="utf-8") as _f:
    MANIFEST = json.load(_f)


# =====================================================
# 数据加载
# =====================================================

def load_task_data(project_path, task, sample_size):
    """加载并采样任务数据"""
    task_files = {
        "ci": "component_identification.jsonl",
        "ci_hard": "component_identification_hard.jsonl",
        "dp": "dependency_prediction.jsonl",
        "ar": "architecture_reconstruction.jsonl",
    }

    # 新逻辑：从固定样本清单读取行号，所有模型 / Level 使用完全相同的题目
    proj_name = next(p["name"] for p in PROJECTS if p["path"] == project_path)
    entry = MANIFEST["projects"][proj_name][task]

    if MANIFEST_VERSION == "v2" and task in ("dp", "ar"):
        filepath = os.path.join(HERE, entry["source_file"])
    else:
        filepath = os.path.join(project_path, task_files[task])
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    # 旧逻辑：运行时用全局随机流抽样，断点续跑/多模型连跑时样本会错位，已弃用
    # if len(data) > sample_size:
    #     data = random.sample(data, sample_size)

    assert entry["n_pool"] == len(data), f"{filepath}: pool size changed ({len(data)} vs {entry['n_pool']})"
    data = [data[i] for i in entry["indices"]]
    if task != "ar":
        assert [item_key(task, d) for d in data] == entry["keys"], f"{proj_name}/{task}: manifest key mismatch"

    return data


def load_modules(project_path):
    """加载模块列表（v2 下若 akg_v2/<name>/modules_index.json 存在则优先，目前仅 linux 新增 include/*）"""
    path = os.path.join(project_path, "modules_index.json")
    if MANIFEST_VERSION == "v2":
        proj_name = next(p["name"] for p in PROJECTS if p["path"] == project_path)
        v2_path = os.path.join(AKG_V2_DIR, proj_name, "modules_index.json")
        if os.path.exists(v2_path):
            path = v2_path
    with open(path, encoding="utf-8") as f:
        return json.load(f)["modules"]


def ar_ground_truth(proj_name, data):
    """AR 源模块顺序与真解。
    v1：真解 = 抽中的边里属于该源模块的目标（原口径）。
    v2：真解 = manifest true_targets（新图出边中隐藏的一半）；剔除无出边的源模块，其余顺序与 v1 相同。"""
    entry = MANIFEST["projects"][proj_name]["ar"]
    gt = defaultdict(set)
    for item in data:
        gt[item["source_module"]].add(item["target_module"])
    if MANIFEST_VERSION == "v1":
        sources = list(gt.keys())
        assert sources == entry["sources"], f"{proj_name}/ar: manifest source mismatch"
        return sources, gt
    dropped = set(entry.get("dropped_sources_no_outgoing_edge", []))
    sources = [s for s in gt if s not in dropped]
    assert sources == entry["sources"], f"{proj_name}/ar: manifest source mismatch"
    return sources, {s: set(entry["true_targets"][s]) for s in sources}


# =====================================================
# 解析模型输出
# =====================================================

def is_api_error(response):
    """API 调用失败（重试耗尽）的响应，不得进入解析"""
    return (not response) or response.startswith("ERROR:")


# ---- CI/DP/AR 解析：2026-10-07 升级 ----
# 旧规则（取全文第一个字母/yes/no 子串）在模型输出 CoT 推理文本时会把推理里的
# 字母或 not/no 当成答案（llama-4-maverick / llama-3.1-8b 的 CI 约 60% 响应为长文本）。
# 新规则：优先匹配明确的"answer is X"句式与行首/独立字母行，最后才回退旧规则。
# 对输出裸字母的合规模型，三种规则结果完全一致（恒等）。

_CI_ANSWER_RE = re.compile(
    r"(?:answer|option|choice)\s*(?:is|:|=)?\s*\**\s*\(?\**\s*([A-D])\b", re.IGNORECASE)
_CI_LETTER_LINE_RE = re.compile(
    r"^\s*\**\(?([A-D])\)?\**\s*[).:\]]", re.MULTILINE | re.IGNORECASE)
_CI_LONE_LETTER_RE = re.compile(
    r"^\s*\**\(?([A-D])\)?\**\s*[.:)]?\s*$", re.MULTILINE | re.IGNORECASE)

_DP_ANSWER_RE = re.compile(
    r"(?:answer|dependency|dependence|prediction|conclusion)\s*(?:is|:|=)?\s*\**\s*\(?\**\s*(yes|no)\b",
    re.IGNORECASE)
_DP_EDGE_LINE_RE = re.compile(
    r"^\s*(?:\d+[.):]\s*|[-*]\s*|\**)?(yes|no)\b[\s,.!]*$|\b(yes|no)\s*[.!]?\s*$",
    re.IGNORECASE | re.MULTILINE)


def parse_ci_response(response):
    """解析 CI 任务的模型输出，提取字母 A/B/C/D"""
    if is_api_error(response):
        return None
    text = response.strip()
    # 1) 明确的 "answer is X" / "option X" 句式，取最后一个
    ms = _CI_ANSWER_RE.findall(text)
    if ms:
        return ms[-1].upper()
    # 2) 行首字母行（"D) mis" / "A." / 独立字母行），取最后一个
    ms = _CI_LETTER_LINE_RE.findall(text) or _CI_LONE_LETTER_RE.findall(text)
    if ms:
        return ms[-1].upper()
    # 3) 回退：旧规则——全文第一个 A-D 字符
    for ch in text.upper():
        if ch in "ABCD":
            return ch
    return None


def parse_dp_response(response):
    """解析 DP 任务的模型输出，提取 Yes/No"""
    if is_api_error(response):
        return None
    text = response.strip().lower()
    # 1) 明确的 "answer is yes/no" 句式，取最后一个
    ms = _DP_ANSWER_RE.findall(text)
    if ms:
        return 1 if ms[-1] == "yes" else 0
    # 2) 行首/行尾独立的 yes/no（结论行），取最后一个
    ms = [a or b for a, b in _DP_EDGE_LINE_RE.findall(text)]
    if ms:
        return 1 if ms[-1] == "yes" else 0
    # 3) 回退：旧规则——第一个 yes/no 子串
    if "yes" in text:
        return 1
    elif "no" in text:
        return 0
    return None


def parse_ar_response(response, all_modules):
    """解析 AR 任务的模型输出，提取模块名列表

    2026-10-07 升级：CoT 长响应下旧行级模糊匹配会把推理文本里的模块名都吸进来
    （且产生重复）。新规则优先收集"整行恰好是模块名"的行（模型显式列出的清单），
    匹配不到任何整行时才回退旧的行级模糊匹配。结果去重保序。
    """
    predictions = []
    if is_api_error(response):
        return predictions
    module_set = set(all_modules)
    lines = [ln.strip().lstrip("0123456789.-) ").strip()
             for ln in response.strip().split("\n")]

    # 1) 整行恰好是模块名（显式清单）
    for line in lines:
        if line in module_set:
            predictions.append(line)

    # 2) 回退：行级模糊匹配（旧行为）
    if not predictions:
        for line in lines:
            if not line:
                continue
            if line in module_set:
                predictions.append(line)
            else:
                for m in all_modules:
                    if m in line or line in m:
                        predictions.append(m)
                        break

    # 去重保序，截断至前 5
    seen = set()
    deduped = [p for p in predictions if not (p in seen or seen.add(p))]
    return deduped[:5]


# =====================================================
# LLM 评估
# =====================================================

def evaluate_llm_ci(model_name, model_config, project, data):
    """评估 LLM 在 CI 任务上的表现"""
    correct = 0
    total = 0
    results = []

    for item in data:
        prompt, correct_letter, letter_map = build_ci_prompt(
            system=project["system"],
            domain=project["domain"],
            file_path=item["file"],
            candidates=item["candidates"],
            label=item["label"],
        )

        response = call_llm(model_config, prompt)
        predicted = parse_ci_response(response)

        is_correct = (predicted == correct_letter)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "file": item["file"],
            "label": item["label"],
            "predicted_letter": predicted,
            "predicted_module": letter_map.get(predicted, "PARSE_ERROR"),
            "correct": is_correct,
            "raw_response": response,
        })

        if total % 20 == 0:
            print(f"    [{model_name}] CI: {total}/{len(data)}, acc={correct/total:.3f}")

    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


def evaluate_llm_ci_hard(model_name, model_config, project, data):
    """评估 LLM 在 CI Hard 任务上的表现（只给文件名，不给路径）"""
    correct = 0
    total = 0
    results = []

    for item in data:
        prompt, correct_letter, letter_map = build_ci_hard_prompt(
            system=project["system"],
            domain=project["domain"],
            file_name=item["file"],
            candidates=item["candidates"],
            label=item["label"],
        )

        response = call_llm(model_config, prompt)
        predicted = parse_ci_response(response)

        is_correct = (predicted == correct_letter)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "file": item["file"],
            "file_full": item.get("file_full", ""),
            "label": item["label"],
            "predicted_letter": predicted,
            "predicted_module": letter_map.get(predicted, "PARSE_ERROR"),
            "correct": is_correct,
            "raw_response": response,
        })

        if total % 20 == 0:
            print(f"    [{model_name}] CI-Hard: {total}/{len(data)}, acc={correct/total:.3f}")

    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


def evaluate_llm_dp(model_name, model_config, project, data):
    """评估 LLM 在 DP 任务上的表现"""
    tp = fp = tn = fn = 0
    results = []

    for item in data:
        prompt = build_dp_prompt(
            system=project["system"],
            domain=project["domain"],
            module_a=item["module_A"],
            module_b=item["module_B"],
        )

        response = call_llm(model_config, prompt)
        predicted = parse_dp_response(response)
        label = item["label"]

        if predicted is None:
            predicted = 0  # 解析失败默认 No

        if predicted == 1 and label == 1: tp += 1
        elif predicted == 1 and label == 0: fp += 1
        elif predicted == 0 and label == 0: tn += 1
        elif predicted == 0 and label == 1: fn += 1

        results.append({
            "module_A": item["module_A"],
            "module_B": item["module_B"],
            "label": label,
            "predicted": predicted,
            "raw_response": response,
        })

        total = tp + fp + tn + fn
        if total % 20 == 0:
            acc = (tp + tn) / total if total > 0 else 0
            print(f"    [{model_name}] DP: {total}/{len(data)}, acc={acc:.3f}")

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total,
        "results": results,
    }


def evaluate_llm_ar(model_name, model_config, project, data, all_modules, dep_graph, module_freq):
    """评估 LLM 在 AR 任务上的表现"""
    # 按文件频率降序排列模块，高频模块优先作为候选
    sorted_modules = sorted(all_modules, key=lambda m: module_freq.get(m, 0), reverse=True)

    precision_sum = recall_sum = 0
    total = 0
    results = []

    # 构建 ground truth: source -> set of targets（口径随 MANIFEST_VERSION，见 ar_ground_truth）
    # 旧逻辑（100 条边最多 100 个 source，实际从未触发）
    # if len(unique_sources) > SAMPLE_SIZE_AR:
    #     unique_sources = random.sample(unique_sources, SAMPLE_SIZE_AR)
    unique_sources, gt = ar_ground_truth(project["name"], data)

    for source in unique_sources:
        prompt = build_ar_prompt(
            system=project["system"],
            domain=project["domain"],
            source_module=source,
            all_modules=sorted_modules,
        )

        response = call_llm(model_config, prompt)
        predicted = parse_ar_response(response, all_modules)

        true_targets = gt[source]
        predicted_set = set(predicted)

        hits = len(predicted_set & true_targets)
        p = hits / 5  # 标准 P@5：固定分母 5，短列表不被奖励
        r = hits / len(true_targets) if true_targets else 0

        precision_sum += p
        recall_sum += r
        total += 1

        results.append({
            "source_module": source,
            "true_targets": list(true_targets),
            "predicted": predicted,
            "precision": p,
            "recall": r,
            "raw_response": response,
        })

        if total % 10 == 0:
            print(f"    [{model_name}] AR: {total}/{len(unique_sources)}, P@5={precision_sum/total:.3f}")

    avg_precision = precision_sum / total if total > 0 else 0
    avg_recall = recall_sum / total if total > 0 else 0

    return {
        "precision_at_5": avg_precision, "recall_at_5": avg_recall,
        "total": total, "results": results,
    }


# =====================================================
# Baseline 评估
# =====================================================

def evaluate_baselines(project, project_path):
    """运行所有 baseline"""
    proj_data = load_project_data(project_path)
    all_modules = load_modules(project_path)
    results = {}

    # --- CI ---
    ci_data = load_task_data(project_path, "ci", SAMPLE_SIZE_CI)
    for bl_name, bl_fn in [
        ("random", lambda item: baseline_ci_random(item["candidates"])),
        ("frequency", lambda item: baseline_ci_frequency(item["candidates"], proj_data["module_freq"])),
        ("path_heuristic", lambda item: baseline_ci_path_heuristic(item["file"], item["candidates"])),
    ]:
        correct = sum(1 for item in ci_data if bl_fn(item) == item["label"])
        results[f"ci_{bl_name}"] = {"accuracy": correct / len(ci_data), "total": len(ci_data)}

    # --- DP ---
    dp_data = load_task_data(project_path, "dp", SAMPLE_SIZE_DP)
    for bl_name, bl_fn in [
        ("random", lambda item: baseline_dp_random()),
        ("frequency", lambda item: baseline_dp_frequency(item["module_A"], item["module_B"], proj_data["dep_graph"])),
        ("graph_neighbor", lambda item: baseline_dp_graph_neighbor(
            item["module_A"], item["module_B"], proj_data["dep_graph"],
            exclude_queried_edge=MANIFEST_VERSION == "v2")),
    ]:
        tp = fp = tn = fn = 0
        for item in dp_data:
            pred = bl_fn(item)
            label = item["label"]
            if pred == 1 and label == 1: tp += 1
            elif pred == 1 and label == 0: fp += 1
            elif pred == 0 and label == 0: tn += 1
            elif pred == 0 and label == 1: fn += 1
        total = tp + fp + tn + fn
        acc = (tp + tn) / total if total > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        results[f"dp_{bl_name}"] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # --- AR ---
    ar_data = load_task_data(project_path, "ar", SAMPLE_SIZE_AR)
    _, gt = ar_ground_truth(project["name"], ar_data)

    for bl_name, bl_fn in [
        ("random", lambda src: baseline_ar_random(all_modules, src)),
        ("frequency", lambda src: baseline_ar_frequency(all_modules, proj_data["module_freq"], src)),
        ("graph_neighbor", lambda src: baseline_ar_graph_neighbor(proj_data["dep_graph"], src)),
    ]:
        p_sum = r_sum = cnt = 0
        for src, targets in gt.items():
            predicted = set(bl_fn(src))
            if MANIFEST_VERSION == "v2" and bl_name == "graph_neighbor":
                # v2 的 AR 真解 = 新图完整出边，不遮蔽就等于直接把答案给基线；
                # 与 v1 的 masked 口径一致：去掉全部真解边，图为空 → 该基线退化为 0
                predicted -= set(targets)
            hits = len(predicted & targets)
            if MANIFEST_VERSION == "v2":
                p_sum += hits / 5  # 与 LLM 的 P@5 统一为固定分母 5
            else:
                p_sum += hits / len(predicted) if predicted else 0
            r_sum += hits / len(targets) if targets else 0
            cnt += 1
        results[f"ar_{bl_name}"] = {
            "precision_at_5": p_sum / cnt if cnt > 0 else 0,
            "recall_at_5": r_sum / cnt if cnt > 0 else 0,
        }

    return results


# =====================================================
# 主函数
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=["ci", "ci_hard", "dp", "ar", "all"])
    parser.add_argument("--model", default="all")
    parser.add_argument("--project", default="all")
    parser.add_argument("--baselines", action="store_true", help="Only run baselines")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 确定要跑的项目
    if args.project == "all":
        target_projects = PROJECTS
    else:
        target_projects = [p for p in PROJECTS if p["name"] == args.project]

    # -------- Baselines --------
    if args.baselines:
        print("\n=== Running Baselines ===\n")
        all_baseline_results = {}

        for proj in target_projects:
            print(f"  Project: {proj['name']}")
            bl_results = evaluate_baselines(proj, proj["path"])
            all_baseline_results[proj["name"]] = bl_results

            for k, v in bl_results.items():
                metric = v.get("accuracy", v.get("precision_at_5", "N/A"))
                print(f"    {k}: {metric:.4f}" if isinstance(metric, float) else f"    {k}: {metric}")

        out_path = os.path.join(args.output, "baselines.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_baseline_results, f, indent=2)
        print(f"\nBaseline results saved to {out_path}")
        return

    # -------- LLM Evaluation --------
    tasks = ["ci", "ci_hard", "dp", "ar"] if args.task == "all" else [args.task]
    models = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_name in models:
        model_config = MODELS[model_name]
        print(f"\n=== Model: {model_name} ===\n")

        for proj in target_projects:
            print(f"  Project: {proj['name']}")
            all_modules = load_modules(proj["path"])
            proj_data = load_project_data(proj["path"])

            for task in tasks:
                sample_sizes = {"ci": SAMPLE_SIZE_CI, "ci_hard": SAMPLE_SIZE_CI, "dp": SAMPLE_SIZE_DP, "ar": SAMPLE_SIZE_AR}
                out_path = os.path.join(args.output, f"{model_name}_{proj['name']}_{task}.json")
                if os.path.exists(out_path):
                    print(f"    Task: {task} -> skip (exists)")
                    continue

                data = load_task_data(proj["path"], task, sample_sizes[task])

                print(f"    Task: {task} ({len(data)} samples)")

                if task == "ci":
                    result = evaluate_llm_ci(model_name, model_config, proj, data)
                    print(f"    -> Accuracy: {result['accuracy']:.4f}")

                elif task == "ci_hard":
                    result = evaluate_llm_ci_hard(model_name, model_config, proj, data)
                    print(f"    -> Accuracy: {result['accuracy']:.4f}")

                elif task == "dp":
                    result = evaluate_llm_dp(model_name, model_config, proj, data)
                    print(f"    -> Acc={result['accuracy']:.4f} P={result['precision']:.4f} R={result['recall']:.4f} F1={result['f1']:.4f}")

                elif task == "ar":
                    result = evaluate_llm_ar(model_name, model_config, proj, data, all_modules, proj_data["dep_graph"], proj_data["module_freq"])
                    print(f"    -> P@5={result['precision_at_5']:.4f} R@5={result['recall_at_5']:.4f}")

                # 记录 API 失败数（审计用）
                result["n_api_errors"] = sum(
                    1 for r in result.get("results", []) if is_api_error(r.get("raw_response", ""))
                )
                if result["n_api_errors"]:
                    print(f"    !! API errors: {result['n_api_errors']}")

                # 保存结果
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
