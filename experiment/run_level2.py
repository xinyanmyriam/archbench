"""
Level 2 实验：Full-Context 模式

提供架构上下文（模块文件样例 / 依赖图），测试 LLM 在信息充分条件下的推理上限。

设计（2026-10 修订）：
- 12 系统 × {ci_hard, dp, ar}（ci 可选），题目全部来自 seed42 固定清单，与 L0/L1 逐题配对
- 判错与 L0 一致：API 失败（空 / "ERROR:"）不进入解析，保存 raw_response 并统计 n_api_errors
- CI / CI-Hard：候选模块的文件样例中排除被问文件本身（修复答案泄露）
- AR：完全确定性。隐藏边 = 清单 true_targets（与 L0/L1 评分目标相同），
      prompt 中只展示 source 的其余依赖 dep_graph[source] - true_targets，评分只针对隐藏边
- DP：label=1 时从上下文中删除 A->B 边（同旧版）

用法（在 experiment/ 目录下运行）：
    python run_level2.py --dry-run
    python run_level2.py --model gpt-6-sol --project nastran
    python run_level2.py --model claude-opus-5.5 --task ci_hard,dp
"""

import json
import os
import shutil
import sys
import argparse
from collections import defaultdict, Counter
from datetime import datetime

sys.path.insert(0, ".")

from config import MODELS, PROJECTS
from prompts_level2 import (
    build_ci_l2_prompt, build_ci_hard_l2_prompt,
    build_dp_l2_prompt, build_ar_l2_prompt,
)
from llm_client import call_llm_with_usage
from baselines import load_project_data
from run_experiment import (
    MANIFEST, MANIFEST_VERSION, load_task_data, load_modules, is_api_error,
    parse_ci_response, parse_dp_response, parse_ar_response, ar_ground_truth,
)

# =====================================================
# 配置
# =====================================================

# 旧配置（默认 deepseek / random.sample 抽样 / 输出 results/），保留备查
# DEFAULT_MODEL = "deepseek-v4-flash"
# TASKS = ["ci", "ci_hard", "dp", "ar"]
# OUTPUT_DIR = "results"

LEVEL2_SYSTEMS = [p["name"] for p in PROJECTS]          # 12 系统
LEVEL2_MODELS = ["gpt-6-sol", "claude-opus-5.5"]
LEVEL2_TASKS = ["ci_hard", "dp", "ar"]
ALL_TASKS = ["ci", "ci_hard", "dp", "ar"]

# v2 清单（akg_v2 标签）的结果写入 results_v2，避免覆盖已发布的 v1 结果
RESULTS_DIR = os.path.join("results_v2" if MANIFEST_VERSION == "v2" else "results_rerun", "level2")
LOGS_DIR = "logs"

PROJECT_INFO = {p["name"]: p for p in PROJECTS}

LOG_FILE = DETAIL_LOG = None


def log(msg):
    line = f"[{datetime.now().isoformat()}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_detail(entry):
    if not DETAIL_LOG:
        return
    entry["timestamp"] = datetime.now().isoformat()
    with open(DETAIL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# =====================================================
# 上下文加载
# =====================================================

def load_project_context(project_path):
    ctx = {}
    with open(os.path.join(project_path, "file_index.json"), encoding="utf-8") as f:
        fi = json.load(f)

    ctx["module_files"] = defaultdict(list)   # module -> 完整路径（file_index 顺序）
    for x in fi["files"]:
        ctx["module_files"][x["module"]].append(x["path"])
    ctx["all_paths"] = {x["path"] for x in fi["files"]}

    ctx["dep_graph"] = defaultdict(set)          # module -> targets
    ctx["reverse_dep_graph"] = defaultdict(set)  # module -> sources
    if MANIFEST_VERSION == "v2":
        # v2：用 akg_v2 模块图（load_project_data 已处理 v2 路径与 kubernetes 模块名归一化）
        for s, tg in load_project_data(project_path)["dep_graph"].items():
            for t in tg:
                ctx["dep_graph"][s].add(t)
                ctx["reverse_dep_graph"][t].add(s)
    else:
        with open(os.path.join(project_path, "module_dependency.json"), encoding="utf-8") as f:
            md = json.load(f)
        for d in md["dependencies"]:
            ctx["dep_graph"][d["source_module"]].add(d["target_module"])
            ctx["reverse_dep_graph"][d["target_module"]].add(d["source_module"])

    ctx["all_modules"] = load_modules(project_path)
    module_freq = load_project_data(project_path)["module_freq"]
    ctx["sorted_modules"] = sorted(ctx["all_modules"], key=lambda m: module_freq.get(m, 0), reverse=True)  # 与 L0 AR 相同
    return ctx


# =====================================================
# 构建题目
# =====================================================

def build_items(task, proj, ctx):
    name, system, domain = proj["name"], proj["system"], proj["domain"]
    items = []

    if task in ("ci", "ci_hard"):
        for d in load_task_data(proj["path"], task, None):
            asked = d["file"] if task == "ci" else d.get("file_full", "")
            builder = build_ci_l2_prompt if task == "ci" else build_ci_hard_l2_prompt
            prompt, letter, letter_map = builder(system, domain, d["file"], d["candidates"], d["label"],
                                                 ctx["module_files"], exclude_path=asked)
            meta = {"file": d["file"], "label": d["label"],
                    "excluded_from_examples": asked in ctx["all_paths"]}
            if task == "ci_hard":
                meta["file_full"] = asked
            items.append({"prompt": prompt, "gold": letter, "letter_map": letter_map, "meta": meta})

    elif task == "dp":
        for d in load_task_data(proj["path"], "dp", None):
            prompt = build_dp_l2_prompt(system, domain, d["module_A"], d["module_B"],
                                        ctx["dep_graph"], ctx["reverse_dep_graph"], d["label"])
            items.append({"prompt": prompt, "gold": d["label"],
                          "meta": {"module_A": d["module_A"], "module_B": d["module_B"], "label": d["label"]}})

    elif task == "ar":
        entry = MANIFEST["projects"][name]["ar"]
        sources, gt = ar_ground_truth(name, load_task_data(proj["path"], "ar", None))
        for s in sources:
            assert gt[s] == set(entry["true_targets"][s]), f"{name}/ar: true_targets mismatch for {s}"
            prompt, shown = build_ar_l2_prompt(system, domain, s, ctx["dep_graph"], gt[s], ctx["sorted_modules"])
            items.append({"prompt": prompt, "gold": gt[s],
                          "meta": {"source_module": s, "true_targets": sorted(gt[s]), "shown_deps": shown}})
    else:
        raise ValueError(f"unknown task {task}")

    return items


# =====================================================
# 调用 + 评分
# =====================================================

def run_task(model_name, model_config, proj, task, items, ctx, prev=None):
    """prev: 旧结果的 results 列表（--retry-errors），非 api_error 的题复用原回复重新判分。"""
    results = []
    usage_sum = Counter()
    correct = 0
    tp = fp = tn = fn = 0
    p_sum = r_sum = 0.0
    n_retried = 0

    for idx, it in enumerate(items, 1):
        old = prev[idx - 1] if prev else None
        if old is not None:
            for k, v in it["meta"].items():
                if isinstance(v, (str, int)):
                    assert old.get(k) == v, f"prev mismatch at {idx}: {k}"
        if old is not None and not is_api_error(old["raw_response"]):
            resp = {"text": old["raw_response"], "usage": old.get("usage", {}), "latency_ms": old.get("latency_ms", 0)}
        else:
            resp = call_llm_with_usage(model_config, it["prompt"])
            n_retried += old is not None
        text = resp["text"]
        for k, v in (resp.get("usage") or {}).items():
            usage_sum[k] += v or 0
        rec = dict(it["meta"])

        if task in ("ci", "ci_hard"):
            pred = parse_ci_response(text)
            ok = pred == it["gold"]
            correct += ok
            rec.update(predicted_letter=pred, predicted_module=it["letter_map"].get(pred, "PARSE_ERROR"),
                       correct=ok)
        elif task == "dp":
            pred = parse_dp_response(text)
            if pred is None:
                pred = 0  # 与 L0 一致：解析失败默认 No
            label = it["gold"]
            if pred == 1 and label == 1: tp += 1
            elif pred == 1 and label == 0: fp += 1
            elif pred == 0 and label == 0: tn += 1
            else: fn += 1
            ok = pred == label
            rec.update(predicted=pred)
        else:  # ar：只对隐藏边（true_targets）评分
            pred = parse_ar_response(text, ctx["all_modules"])
            ps, true = set(pred), it["gold"]
            hits = len(ps & true)
            p = hits / 5  # 标准 P@5：固定分母 5
            r = hits / len(true) if true else 0
            p_sum += p
            r_sum += r
            ok = None
            rec.update(predicted=pred, precision=p, recall=r)

        rec.update(raw_response=text, usage=resp.get("usage", {}), latency_ms=resp.get("latency_ms", 0))
        results.append(rec)
        log_detail({"experiment": "level2", "model": model_name, "project": proj["name"], "task": task,
                    "sample_idx": idx, "is_correct": ok, "api_error": is_api_error(text),
                    "response_raw": (text or "")[:200], "usage": resp.get("usage", {}),
                    "latency_ms": resp.get("latency_ms", 0)})
        if idx % 30 == 0:
            log(f"      {idx}/{len(items)}")

    total = len(results)
    if task in ("ci", "ci_hard"):
        out = {"accuracy": correct / total if total else 0, "correct": correct, "total": total}
    elif task == "dp":
        prec = tp / (tp + fp) if tp + fp else 0
        rec_ = tp / (tp + fn) if tp + fn else 0
        out = {"accuracy": (tp + tn) / total if total else 0, "precision": prec, "recall": rec_,
               "f1": 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0,
               "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total}
    else:
        out = {"precision_at_5": p_sum / total if total else 0, "recall_at_5": r_sum / total if total else 0,
               "total": total}

    out["n_api_errors"] = sum(1 for r in results if is_api_error(r["raw_response"]))
    if prev:
        out["n_retried"] = n_retried
    out["usage_total"] = dict(usage_sum)
    out["results"] = results
    return out


def headline(task, res):
    if task == "ar":
        return f"P@5={res['precision_at_5']:.4f} R@5={res['recall_at_5']:.4f}"
    if task == "dp":
        return f"Acc={res['accuracy']:.4f} F1={res['f1']:.4f}"
    return f"Acc={res['accuracy']:.4f}"


def get_result_path(model, project, task):
    return os.path.join(RESULTS_DIR, f"{model}_{project}_{task}_l2.json")


# =====================================================
# 主函数
# =====================================================

def main():
    global LOG_FILE, DETAIL_LOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=",".join(LEVEL2_MODELS), help="逗号分隔")
    parser.add_argument("--project", default="all", help="all 或逗号分隔")
    parser.add_argument("--task", default=",".join(LEVEL2_TASKS), help=f"逗号分隔，可选 {ALL_TASKS}")
    parser.add_argument("--dry-run", action="store_true", help="只构建 prompt 并统计，不调用 API")
    parser.add_argument("--retry-errors", action="store_true", help="已存在结果中只重跑 api_error 题（先备份）")
    args = parser.parse_args()

    models = args.model.split(",")
    projects = LEVEL2_SYSTEMS if args.project == "all" else args.project.split(",")
    tasks = args.task.split(",")
    for t in tasks:
        assert t in ALL_TASKS, f"unknown task {t}"

    if args.dry_run:
        n_all = chars_all = 0
        for pn in projects:
            proj = PROJECT_INFO[pn]
            ctx = load_project_context(proj["path"])
            for t in tasks:
                items = build_items(t, proj, ctx)
                chars = [len(i["prompt"]) for i in items]
                extra = ""
                if t in ("ci", "ci_hard"):
                    extra = f"  excluded={sum(i['meta']['excluded_from_examples'] for i in items)}"
                elif t == "ar":
                    extra = (f"  empty_shown={sum(not i['meta']['shown_deps'] for i in items)}"
                             f"  avg_shown={sum(len(i['meta']['shown_deps']) for i in items)/max(len(items),1):.1f}")
                print(f"{pn:13s} {t:8s} n={len(items):4d} avg_chars={sum(chars)/max(len(chars),1):7.0f} "
                      f"est_in_tok={sum(chars)//4:8d}{extra}")
                n_all += len(items)
                chars_all += sum(chars)
        print(f"\nTOTAL per model: calls={n_all}  est_input_tokens≈{chars_all//4:,} (chars/4)")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = "_".join(models) if len(models) == 1 else "multi"
    LOG_FILE = os.path.join(LOGS_DIR, f"level2_{tag}_{stamp}.log")
    DETAIL_LOG = os.path.join(LOGS_DIR, f"level2_detail_{tag}_{stamp}.jsonl")

    log(f"Level 2 START models={models} projects={projects} tasks={tasks}")
    for model_name in models:
        model_config = MODELS[model_name]
        log(f"Model: {model_name}")
        for pn in projects:
            proj = PROJECT_INFO[pn]
            ctx = None
            for t in tasks:
                out_path = get_result_path(model_name, pn, t)
                prev = None
                if os.path.exists(out_path):
                    if not args.retry_errors:
                        log(f"    {pn}/{t}: SKIP (exists)")
                        continue
                    with open(out_path, encoding="utf-8") as f:
                        old_res = json.load(f)
                    if old_res.get("n_api_errors", 0) == 0:
                        log(f"    {pn}/{t}: SKIP (exists, 0 api_errors)")
                        continue
                    prev = old_res["results"]
                    bdir = os.path.join(RESULTS_DIR, "_preretry_backup")
                    os.makedirs(bdir, exist_ok=True)
                    shutil.copy2(out_path, os.path.join(bdir, f"{os.path.basename(out_path)[:-5]}_{stamp}.json"))
                    log(f"    {pn}/{t}: RETRY {old_res['n_api_errors']} api_errors (backup saved)")
                ctx = ctx or load_project_context(proj["path"])
                items = build_items(t, proj, ctx)
                if prev is not None:
                    assert len(prev) == len(items), f"prev len {len(prev)} != {len(items)}"
                log(f"    {pn}/{t}: {len(items)} items")
                res = run_task(model_name, model_config, proj, t, items, ctx, prev)
                res.update(timestamp=datetime.now().isoformat(), model=model_name, project=pn, task=t,
                           level=2, manifest_seed=MANIFEST["seed"])
                tmp = out_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=2, ensure_ascii=False, default=str)
                os.replace(tmp, out_path)
                log(f"    -> {headline(t, res)}  api_errors={res['n_api_errors']}  usage={res['usage_total']}")
    log(f"DONE. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
