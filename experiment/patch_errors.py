"""
只重跑结果中 API 出错（空响应 / "ERROR:" 开头）的样本，并在原结果中替换、重算指标。
- 默认（无参数）：deepseek-v4-flash，输入 results_rerun/_quarantine_4096trunc/（保持不动作为备份），
  输出 results_rerun/ 下同名文件（已存在则跳过）
- --model M --inplace：直接修补 results_rerun/ 中 M 的含错误文件，原文件先备份到
  results_rerun/_prepatch_backup/
- 未出错样本保持原结果；prompt 按原样本字段确定性重建（字母顺序与原运行一致）

用法:
    python patch_errors.py
    python patch_errors.py --model gpt-6-sol --inplace
"""
import argparse, json, os, datetime, shutil

from config import MODELS, PROJECTS
from prompts import build_ci_prompt, build_ci_hard_prompt, build_dp_prompt, build_ar_prompt
from llm_client import call_llm
from baselines import load_project_data
from run_experiment import (
    is_api_error, parse_ci_response, parse_dp_response, parse_ar_response, load_modules,
)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="deepseek-v4-flash")
ap.add_argument("--inplace", action="store_true", help="patch results_rerun/ files in place (backup first)")
args = ap.parse_args()

MODEL = args.model
DST_DIR = "results_rerun"
SRC_DIR = DST_DIR if args.inplace else os.path.join("results_rerun", "_quarantine_4096trunc")
BACKUP_DIR = os.path.join(DST_DIR, "_prepatch_backup")
cfg = MODELS[MODEL]
proj_by_name = {p["name"]: p for p in PROJECTS}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def patch_ci(proj, d):
    items = load_jsonl(os.path.join(proj["path"], "component_identification.jsonl"))
    idx = {}
    for it in items:
        idx.setdefault((it["file"], it["label"]), it)
    n = 0
    for r in d["results"]:
        if not is_api_error(r.get("raw_response", "")):
            continue
        it = idx[(r["file"], r["label"])]
        prompt, correct_letter, letter_map = build_ci_prompt(
            system=proj["system"], domain=proj["domain"],
            file_path=it["file"], candidates=it["candidates"], label=it["label"])
        resp = call_llm(cfg, prompt)
        pred = parse_ci_response(resp)
        r.update(predicted_letter=pred, predicted_module=letter_map.get(pred, "PARSE_ERROR"),
                 correct=(pred == correct_letter), raw_response=resp)
        n += 1
        print(f"      patched {n}: {r['file']} -> {pred} ({'OK' if r['correct'] else 'X'})")
    d["correct"] = sum(1 for r in d["results"] if r["correct"])
    d["total"] = len(d["results"])
    d["accuracy"] = d["correct"] / d["total"] if d["total"] else 0
    return n


def patch_ci_hard(proj, d):
    items = load_jsonl(os.path.join(proj["path"], "component_identification_hard.jsonl"))
    idx = {}
    for it in items:
        idx.setdefault((it["file"], it.get("file_full", ""), it["label"]), it)
    n = 0
    for r in d["results"]:
        if not is_api_error(r.get("raw_response", "")):
            continue
        it = idx[(r["file"], r.get("file_full", ""), r["label"])]
        prompt, correct_letter, letter_map = build_ci_hard_prompt(
            system=proj["system"], domain=proj["domain"],
            file_name=it["file"], candidates=it["candidates"], label=it["label"])
        resp = call_llm(cfg, prompt)
        pred = parse_ci_response(resp)
        r.update(predicted_letter=pred, predicted_module=letter_map.get(pred, "PARSE_ERROR"),
                 correct=(pred == correct_letter), raw_response=resp)
        n += 1
        print(f"      patched {n}: {r['file']} -> {pred} ({'OK' if r['correct'] else 'X'})")
    d["correct"] = sum(1 for r in d["results"] if r["correct"])
    d["total"] = len(d["results"])
    d["accuracy"] = d["correct"] / d["total"] if d["total"] else 0
    return n


def patch_dp(proj, d):
    n = 0
    for r in d["results"]:
        if not is_api_error(r.get("raw_response", "")):
            continue
        prompt = build_dp_prompt(system=proj["system"], domain=proj["domain"],
                                 module_a=r["module_A"], module_b=r["module_B"])
        resp = call_llm(cfg, prompt)
        pred = parse_dp_response(resp)
        r.update(predicted=0 if pred is None else pred, raw_response=resp)
        n += 1
        print(f"      patched {n}: {r['module_A']} -> {r['module_B']} = {r['predicted']} (label {r['label']})")
    tp = sum(1 for r in d["results"] if r["predicted"] == 1 and r["label"] == 1)
    fp = sum(1 for r in d["results"] if r["predicted"] == 1 and r["label"] == 0)
    tn = sum(1 for r in d["results"] if r["predicted"] == 0 and r["label"] == 0)
    fn = sum(1 for r in d["results"] if r["predicted"] == 0 and r["label"] == 1)
    total = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    d.update(accuracy=(tp + tn) / total if total else 0, precision=prec, recall=rec,
             f1=2 * prec * rec / (prec + rec) if prec + rec else 0,
             tp=tp, fp=fp, tn=tn, fn=fn, total=total)
    return n


def patch_ar(proj, d):
    all_modules = load_modules(proj["path"])
    freq = load_project_data(proj["path"])["module_freq"]
    sorted_modules = sorted(all_modules, key=lambda m: freq.get(m, 0), reverse=True)
    n = 0
    for r in d["results"]:
        if not is_api_error(r.get("raw_response", "")):
            continue
        prompt = build_ar_prompt(system=proj["system"], domain=proj["domain"],
                                 source_module=r["source_module"], all_modules=sorted_modules)
        resp = call_llm(cfg, prompt)
        pred = parse_ar_response(resp, all_modules)
        truth = set(r["true_targets"])
        hits = len(set(pred) & truth)
        r.update(predicted=pred, raw_response=resp,
                 precision=hits / len(set(pred)) if pred else 0,
                 recall=hits / len(truth) if truth else 0)
        n += 1
        print(f"      patched {n}: {r['source_module']} P={r['precision']:.2f} R={r['recall']:.2f}")
    t = len(d["results"])
    d["precision_at_5"] = sum(r["precision"] for r in d["results"]) / t if t else 0
    d["recall_at_5"] = sum(r["recall"] for r in d["results"]) / t if t else 0
    d["total"] = t
    return n


PATCHERS = {"ci": patch_ci, "ci_hard": patch_ci_hard, "dp": patch_dp, "ar": patch_ar}

for fname in sorted(os.listdir(SRC_DIR)):
    if not (fname.startswith(MODEL + "_") and fname.endswith(".json")):
        continue
    dst = os.path.join(DST_DIR, fname)
    src = os.path.join(SRC_DIR, fname)
    if not args.inplace and os.path.exists(dst):
        print(f"skip {fname} (already in {DST_DIR})")
        continue
    stem = fname[len(MODEL) + 1:-5]
    pname, task = stem.split("_", 1)
    proj = proj_by_name[pname]
    with open(src, encoding="utf-8") as f:
        d = json.load(f)
    before = sum(1 for r in d["results"] if is_api_error(r.get("raw_response", "")))
    if args.inplace:
        if not before:
            continue
        os.makedirs(BACKUP_DIR, exist_ok=True)
        backup = os.path.join(BACKUP_DIR, fname)
        if not os.path.exists(backup):
            shutil.copy2(src, backup)
        src = backup
    print(f"\n== {fname}: {before} errors to patch")
    n = PATCHERS[task](proj, d)
    d["n_api_errors"] = sum(1 for r in d["results"] if is_api_error(r.get("raw_response", "")))
    d["patch_info"] = {
        "n_patched": n,
        "reason": "API error responses (empty / ERROR:) re-queried",
        "patched_max_tokens": cfg.get("max_tokens"),
        "patched_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "original_file": src,
    }
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, default=str)
    m = d.get("accuracy", d.get("precision_at_5"))
    print(f"   -> saved {dst}: metric={m:.4f}, remaining errors={d['n_api_errors']}")
