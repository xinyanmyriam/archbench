"""
Level 1 实验 (Code-Context / Externalization)

设计（2026-10 修订）：
- 12 系统 × {ci_hard, ci_obfuscated, dp, ar}（ci 可选），题目全部来自 seed42 固定清单
  （experiment/samples/manifest_seed42.json），与 Level 0 逐题配对：
    ci / ci_hard / dp / ar : 与 L0 相同的 load_task_data()
    ci_obfuscated          : 复用清单 ci_hard 的 indices（obfuscated.jsonl 与 hard.jsonl 按行号对齐），
                             并逐题校验 file_original / label_original 与 ci_hard keys 一致
    ar                     : 评分目标 = 清单 true_targets（与 L0 完全相同）；候选列表与 L0 相同
- 代码片段统一为前 CODE_SNIPPET_LINES 行
- 判错与 L0 一致：API 失败（空 / "ERROR:"）不进入解析，结果中保存 raw_response 并统计 n_api_errors
- 核心指标：Reasoning Gain (RG_L1) = L1 - L0

用法（在 experiment/ 目录下运行）：
    python run_level1_icse.py --dry-run                          # 只构建 prompt，统计规模/token
    python run_level1_icse.py --model gpt-6-sol --project nastran
    python run_level1_icse.py --model claude-opus-5.5 --task ci_hard,dp
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
from llm_client import call_llm_with_usage
from baselines import load_project_data
from run_experiment import (
    MANIFEST, MANIFEST_VERSION, load_task_data, load_modules, is_api_error,
    parse_ci_response, parse_dp_response, parse_ar_response, ar_ground_truth,
)
from prompts_level1 import (
    build_ci_l1_prompt, build_ci_hard_l1_prompt, build_ci_obf_l1_prompt,
    build_dp_l1_prompt, build_ar_l1_prompt,
)

# =====================================================
# 配置
# =====================================================

# 旧配置（5 系统 / 旧模型阵容 / random.sample 抽样），保留备查
# LEVEL1_SYSTEMS = ["kubernetes", "spring", "openfoam", "nastran", "hadoop"]
# LEVEL1_MODELS = ["deepseek-v4-flash", "claude-sonnet-4", "qwen3-8b", "llama-3.1-8b"]
# LEVEL1_TASKS = ["ci_hard", "ci_obfuscated", "dp"]
# SAMPLE_SIZE = 150
# RESULTS_DIR = "results/level1"

LEVEL1_SYSTEMS = [p["name"] for p in PROJECTS]          # 12 系统
LEVEL1_MODELS = ["gpt-6-sol", "claude-opus-5.5"]
LEVEL1_TASKS = ["ci_hard", "ci_obfuscated", "dp", "ar"]
ALL_TASKS = ["ci", "ci_hard", "ci_obfuscated", "dp", "ar"]
CODE_SNIPPET_LINES = 100
NO_SOURCE = "(no source)"

# v2 清单（akg_v2 标签）的结果写入 results_v2，避免覆盖已发布的 v1 结果
RESULTS_DIR = os.path.join("results_v2" if MANIFEST_VERSION == "v2" else "results_rerun", "level1")
LOGS_DIR = "logs"

PROJECT_INFO = {p["name"]: p for p in PROJECTS}

# 源码仓库路径（相对 experiment/）
REPO_PATHS = {
    "openfoam": "../openfoam/openfoam",
    "nastran": "../nastran/NASTRAN-95",
    "salome": "../salome/salome",
    "paraview": "../paraview/ParaView",
    "kubernetes": "../kubernetes/kubernetes",
    "linux": "../linux/linux",
    "hadoop": "../hadoop/hadoop",
    "spring": "../spring/spring-framework",
    "codeaster": "../codeaster/codeaster",
    "openmodelica": "../openmodelica/OpenModelica",
    "tensorflow": "../tensorflow/tensorflow",
    "pytorch": "../pytorch/pytorch",
}

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

def read_file_head(repo_path, rel_path, max_lines):
    """读取源码前 N 行；文件不存在返回 NO_SOURCE"""
    if not rel_path:
        return NO_SOURCE
    full = os.path.join(repo_path, rel_path)
    if not os.path.isfile(full):
        return NO_SOURCE
    lines = []
    with open(full, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            lines.append(line.rstrip())
    return "\n".join(lines)


def load_l1_context(proj):
    """模块文件列表、模块 import 目标（按出现次数降序、同次数按名称）、L0 同序候选模块"""
    path = proj["path"]
    with open(os.path.join(path, "file_index.json"), encoding="utf-8") as f:
        fi = json.load(f)
    file_to_module = {x["path"]: x["module"] for x in fi["files"]}
    module_files = defaultdict(list)
    for x in fi["files"]:
        module_files[x["module"]].append(os.path.basename(x["path"]))

    raw_path = os.path.join(path, "dependency_raw.json")
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            pairs = [(d["source"], d["target"]) for d in json.load(f)["dependencies"]]
        import_source = "dependency_raw"
    else:
        # salome 无 dependency_raw.json：用已解析的 file_dependency 目标文件名近似原始 include
        with open(os.path.join(path, "file_dependency.json"), encoding="utf-8") as f:
            pairs = [(d["source"], os.path.basename(d["target"])) for d in json.load(f)["dependencies"]]
        import_source = "file_dependency_basename"

    # 去掉模块自引用（include 目标文件名就是本模块内的文件），否则截断后的列表被自引用占满
    own_names = {m: set(fs) for m, fs in module_files.items()}
    counters = defaultdict(Counter)
    for s, t in pairs:
        m = file_to_module.get(s)
        if m and os.path.basename(t.replace("\\", "/")) not in own_names[m]:
            counters[m][t] += 1
    module_imports = {m: [t for t, _ in sorted(c.items(), key=lambda x: (-x[1], x[0]))]
                      for m, c in counters.items()}

    all_modules = load_modules(path)
    module_freq = load_project_data(path)["module_freq"]
    sorted_modules = sorted(all_modules, key=lambda m: module_freq.get(m, 0), reverse=True)  # 与 L0 AR 相同

    return {
        "module_files": module_files,
        "module_imports": module_imports,
        "import_source": import_source,
        "all_modules": all_modules,
        "sorted_modules": sorted_modules,
    }


def load_obf_data(proj):
    """ci_obfuscated：复用清单 ci_hard 的 indices，逐题校验与 ci_hard 配对"""
    entry = MANIFEST["projects"][proj["name"]]["ci_hard"]
    with open(os.path.join(proj["path"], "component_identification_obfuscated.jsonl"), encoding="utf-8") as f:
        pool = [json.loads(line) for line in f]
    assert len(pool) == entry["n_pool"], f"{proj['name']}/ci_obfuscated: pool size {len(pool)} vs ci_hard {entry['n_pool']}"
    data = [pool[i] for i in entry["indices"]]
    assert [[o["file_original"], o["label_original"]] for o in data] == entry["keys"], \
        f"{proj['name']}/ci_obfuscated: not aligned with ci_hard manifest"
    return data


# =====================================================
# 构建题目（prompt + 元数据），与调用/评分解耦，便于 dry-run
# =====================================================

def build_items(task, proj, ctx):
    name, system, domain = proj["name"], proj["system"], proj["domain"]
    repo = REPO_PATHS[name]
    n = CODE_SNIPPET_LINES
    items = []

    if task in ("ci", "ci_hard"):
        for d in load_task_data(proj["path"], task, None):
            rel = d["file"] if task == "ci" else d.get("file_full") or d["file"]
            snippet = read_file_head(repo, rel, n)
            builder = build_ci_l1_prompt if task == "ci" else build_ci_hard_l1_prompt
            prompt, letter, letter_map = builder(system, domain, d["file"], d["candidates"], d["label"], snippet, n)
            meta = {"file": d["file"], "label": d["label"], "has_snippet": snippet != NO_SOURCE}
            if task == "ci_hard":
                meta["file_full"] = d.get("file_full", "")
            items.append({"prompt": prompt, "gold": letter, "letter_map": letter_map, "meta": meta})

    elif task == "ci_obfuscated":
        for d in load_obf_data(proj):
            snippet = read_file_head(repo, d["file_original"], n)
            prompt, letter, letter_map = build_ci_obf_l1_prompt(
                system, domain, d["file"], d["candidates"], d["label"], snippet, n)
            meta = {"file": d["file"], "file_original": d["file_original"],
                    "label": d["label"], "label_original": d["label_original"],
                    "has_snippet": snippet != NO_SOURCE}
            items.append({"prompt": prompt, "gold": letter, "letter_map": letter_map, "meta": meta})

    elif task == "dp":
        for d in load_task_data(proj["path"], "dp", None):
            a, b = d["module_A"], d["module_B"]
            prompt = build_dp_l1_prompt(system, domain, a, b,
                                        ctx["module_files"].get(a, []), ctx["module_files"].get(b, []),
                                        ctx["module_imports"].get(a, []))
            items.append({"prompt": prompt, "gold": d["label"],
                          "meta": {"module_A": a, "module_B": b, "label": d["label"]}})

    elif task == "ar":
        entry = MANIFEST["projects"][name]["ar"]
        sources, gt = ar_ground_truth(name, load_task_data(proj["path"], "ar", None))
        for s in sources:
            assert gt[s] == set(entry["true_targets"][s]), f"{name}/ar: true_targets mismatch for {s}"
            prompt = build_ar_l1_prompt(system, domain, s, ctx["module_files"].get(s, []),
                                        ctx["module_imports"].get(s, []), ctx["sorted_modules"])
            items.append({"prompt": prompt, "gold": gt[s],
                          "meta": {"source_module": s, "true_targets": sorted(gt[s])}})
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

        if task in ("ci", "ci_hard", "ci_obfuscated"):
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
        else:  # ar
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
        log_detail({"experiment": "level1", "model": model_name, "project": proj["name"], "task": task,
                    "sample_idx": idx, "is_correct": ok, "api_error": is_api_error(text),
                    "response_raw": (text or "")[:200], "usage": resp.get("usage", {}),
                    "latency_ms": resp.get("latency_ms", 0)})
        if idx % 30 == 0:
            log(f"      {idx}/{len(items)}")

    total = len(results)
    if task in ("ci", "ci_hard", "ci_obfuscated"):
        out = {"accuracy": correct / total if total else 0, "correct": correct, "total": total,
               "n_with_snippet": sum(r["has_snippet"] for r in results)}
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
    return os.path.join(RESULTS_DIR, f"{model}_{project}_{task}_l1.json")


# =====================================================
# 主函数
# =====================================================

def main():
    global LOG_FILE, DETAIL_LOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=",".join(LEVEL1_MODELS), help="逗号分隔")
    parser.add_argument("--project", default="all", help="all 或逗号分隔")
    parser.add_argument("--task", default=",".join(LEVEL1_TASKS), help=f"逗号分隔，可选 {ALL_TASKS}")
    parser.add_argument("--dry-run", action="store_true", help="只构建 prompt 并统计，不调用 API")
    parser.add_argument("--retry-errors", action="store_true", help="已存在结果中只重跑 api_error 题（先备份）")
    args = parser.parse_args()

    models = args.model.split(",")
    projects = LEVEL1_SYSTEMS if args.project == "all" else args.project.split(",")
    tasks = args.task.split(",")
    for t in tasks:
        assert t in ALL_TASKS, f"unknown task {t}"

    if args.dry_run:
        stats = []
        for pn in projects:
            proj = PROJECT_INFO[pn]
            ctx = load_l1_context(proj)
            for t in tasks:
                items = build_items(t, proj, ctx)
                chars = [len(i["prompt"]) for i in items]
                snip = sum(i["meta"].get("has_snippet", True) for i in items)
                stats.append((pn, t, len(items), snip, sum(chars)))
                print(f"{pn:13s} {t:14s} n={len(items):4d} snippet={snip:4d} "
                      f"avg_chars={sum(chars)/max(len(chars),1):8.0f} est_in_tok={sum(chars)//4:9d}"
                      + (f"  [imports: {ctx['import_source']}]" if t in ("dp", "ar") else ""))
        n_all = sum(s[2] for s in stats)
        tok_all = sum(s[4] for s in stats) // 4
        print(f"\nTOTAL per model: calls={n_all}  est_input_tokens≈{tok_all:,} (chars/4)")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = "_".join(models) if len(models) == 1 else "multi"
    LOG_FILE = os.path.join(LOGS_DIR, f"level1_{tag}_{stamp}.log")
    DETAIL_LOG = os.path.join(LOGS_DIR, f"level1_detail_{tag}_{stamp}.jsonl")

    log(f"Level 1 START models={models} projects={projects} tasks={tasks} lines={CODE_SNIPPET_LINES}")
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
                ctx = ctx or load_l1_context(proj)
                items = build_items(t, proj, ctx)
                if prev is not None:
                    assert len(prev) == len(items), f"prev len {len(prev)} != {len(items)}"
                log(f"    {pn}/{t}: {len(items)} items")
                res = run_task(model_name, model_config, proj, t, items, ctx, prev)
                res.update(timestamp=datetime.now().isoformat(), model=model_name, project=pn, task=t,
                           level=1, code_snippet_lines=CODE_SNIPPET_LINES, manifest_seed=MANIFEST["seed"])
                if t in ("dp", "ar"):
                    res["import_source"] = ctx["import_source"]
                tmp = out_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=2, ensure_ascii=False, default=str)
                os.replace(tmp, out_path)
                log(f"    -> {headline(t, res)}  api_errors={res['n_api_errors']}  usage={res['usage_total']}")
    log(f"DONE. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
