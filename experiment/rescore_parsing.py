"""离线重打分脚本（2026-10-07）

背景：旧解析规则（CI 取全文第一个 A-D 字符、DP 取第一个 yes/no 子串、AR 行级模糊
匹配）在模型输出 CoT 长推理时产生系统性误判（llama-4-maverick / llama-3.1-8b 的
CI/DP、opus 的 L1 ci_obfuscated 受影响最重）。llm_client 修复后模型响应带推理文本
是常态，解析器已升级（见 run_experiment.py 2026-10-07 注释）。

本脚本用升级后的解析器对已存 raw_response 离线重算：
  - L0：除 qwen3-8b（DashScope 重跑中且响应全为裸字母，不受影响）外全部模型
  - L1/L2：claude-opus-5.5 与 gpt-6-sol 全部任务

原始文件先备份到 results_rerun/_prerescore_backup/{level}/（已存在则不覆盖，
保证始终保留最初状态），然后原地更新指标与逐题判定。

用法：
    python rescore_parsing.py            # 执行重打分
    python rescore_parsing.py --dry-run  # 只报告变化，不写文件
"""

import json
import os
import shutil
import argparse
from statistics import mean

from run_experiment import (
    PROJECTS, MANIFEST, load_task_data, load_modules,
    parse_ci_response, parse_dp_response, parse_ar_response,
)

L0_MODELS = ["claude-opus-5.5", "gpt-6-sol", "kimi-k3", "gpt-oss-120b",
             "qwen3", "llama-4-maverick", "llama-3.1-8b", "deepseek-v4-flash"]
L12_MODELS = ["claude-opus-5.5", "gpt-6-sol"]
SYSTEMS = [p["name"] for p in PROJECTS]
PROJ_BY_NAME = {p["name"]: p for p in PROJECTS}
BACKUP_ROOT = os.path.join("results_rerun", "_prerescore_backup")


def proj_path(name):
    return PROJ_BY_NAME[name]["path"]


def load_obfuscated_aligned(name):
    """ci_obfuscated：复用清单 ci_hard 的 indices（与 run_level1_icse 一致）"""
    path = proj_path(name)
    pool = []
    with open(os.path.join(path, "component_identification_obfuscated.jsonl"), encoding="utf-8") as f:
        for line in f:
            pool.append(json.loads(line))
    indices = MANIFEST["projects"][name]["ci_hard"]["indices"]
    return [pool[i] for i in indices]


def letter_map_of(item):
    return {chr(65 + i): c for i, c in enumerate(item["candidates"])}


def gold_letter(letter_map, label):
    return next((l for l, m in letter_map.items() if m == label), None)


def backup(file_path, level_key):
    dst_dir = os.path.join(BACKUP_ROOT, level_key)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(file_path))
    if not os.path.exists(dst):
        shutil.copy2(file_path, dst)


def rescore_ci_like(doc, data, dry):
    """ci / ci_hard / ci_obfuscated：重算 predicted_letter / correct / accuracy"""
    assert len(doc["results"]) == len(data), "results 与 data 长度不一致"
    n_changed = 0
    for rec, item in zip(doc["results"], data):
        lm = letter_map_of(item)
        gold = gold_letter(lm, item["label"])
        pred = parse_ci_response(rec.get("raw_response") or "")
        correct = (pred == gold)
        if rec.get("predicted_letter") != pred or bool(rec.get("correct")) != correct:
            n_changed += 1
        rec["predicted_letter"] = pred
        rec["predicted_module"] = lm.get(pred, "PARSE_ERROR")
        rec["correct"] = correct
    doc["accuracy"] = sum(1 for r in doc["results"] if r["correct"]) / len(doc["results"])
    doc["correct"] = sum(1 for r in doc["results"] if r["correct"])
    return n_changed, doc["accuracy"]


def rescore_dp(doc, data, dry):
    """dp：重算 predicted 与混淆矩阵"""
    assert len(doc["results"]) == len(data), "results 与 data 长度不一致"
    n_changed = 0
    tp = fp = tn = fn = 0
    for rec, item in zip(doc["results"], data):
        label = item["label"]
        pred = parse_dp_response(rec.get("raw_response") or "")
        if pred is None:
            pred = 0  # 与 evaluate_llm_dp 一致：解析失败记 No
        if rec.get("predicted") != pred:
            n_changed += 1
        rec["predicted"] = pred
        if pred == 1 and label == 1: tp += 1
        elif pred == 1 and label == 0: fp += 1
        elif pred == 0 and label == 0: tn += 1
        elif pred == 0 and label == 1: fn += 1
    total = tp + fp + tn + fn
    doc.update({
        "accuracy": (tp + tn) / total if total else 0,
        "precision": tp / (tp + fp) if (tp + fp) else 0,
        "recall": tp / (tp + fn) if (tp + fn) else 0,
        "f1": 0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total,
    })
    doc["f1"] = (2 * doc["precision"] * doc["recall"] / (doc["precision"] + doc["recall"])
                 if (doc["precision"] + doc["recall"]) else 0)
    return n_changed, doc["f1"]


def rescore_ar(doc, data, all_modules, dry):
    """ar：重算 predicted 列表与 P@5 / R@5"""
    gt = {}
    for item in data:
        gt.setdefault(item["source_module"], set()).add(item["target_module"])
    sources = list(gt.keys())
    assert len(doc["results"]) == len(sources), "results 与 sources 长度不一致"
    n_changed = 0
    ps, rs = [], []
    for rec, src in zip(doc["results"], sources):
        pred = parse_ar_response(rec.get("raw_response") or "", all_modules)
        true_set = gt[src]
        pred_set = set(pred)
        hits = len(pred_set & true_set)
        p = hits / len(pred_set) if pred_set else 0
        r = hits / len(true_set) if true_set else 0
        if rec.get("predicted") != pred:
            n_changed += 1
        rec["predicted"] = pred
        rec["precision"] = p
        rec["recall"] = r
        ps.append(p)
        rs.append(r)
    doc["precision_at_5"] = mean(ps) if ps else 0
    doc["recall_at_5"] = mean(rs) if rs else 0
    doc["total"] = len(doc["results"])
    return n_changed, (doc["precision_at_5"], doc["recall_at_5"])


def process(file_path, level_key, task, data, all_modules, dry, report):
    with open(file_path, encoding="utf-8") as f:
        doc = json.load(f)
    old = (doc.get("accuracy") if task.startswith("ci")
           else doc.get("f1") if task == "dp"
           else (doc.get("precision_at_5"), doc.get("recall_at_5")))
    try:
        if task in ("ci", "ci_hard", "ci_obfuscated"):
            n_changed, new = rescore_ci_like(doc, data, dry)
            metric = "acc"
        elif task == "dp":
            n_changed, new = rescore_dp(doc, data, dry)
            metric = "f1"
        else:
            n_changed, new = rescore_ar(doc, data, all_modules, dry)
            metric = "p@5/r@5"
    except AssertionError as e:
        report.append(f"  [SKIP] {file_path}: {e}")
        return

    if not dry:
        backup(file_path, level_key)
        tmp = file_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        os.replace(tmp, file_path)

    fmt = lambda v: (f"{v:.3f}" if isinstance(v, float) else
                     "/".join(f"{x:.3f}" for x in v) if isinstance(v, tuple) else str(v))
    report.append(f"  {os.path.basename(file_path):58s} {metric}: {fmt(old)} -> {fmt(new)}  ({n_changed} 题改变)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    report = []
    all_modules_cache = {}

    # ---- L0 ----
    report.append("== L0 (results_rerun/) ==")
    for model in L0_MODELS:
        for name in SYSTEMS:
            mods = None
            for task in ["ci", "ci_hard", "dp", "ar"]:
                fp = os.path.join("results_rerun", f"{model}_{name}_{task}.json")
                if not os.path.isfile(fp):
                    continue
                data = load_task_data(proj_path(name), task, None)
                if task == "ar" and mods is None:
                    mods = all_modules_cache.setdefault(name, load_modules(proj_path(name)))
                process(fp, "", task, data, mods, args.dry_run, report)

    # ---- L1 / L2 ----
    for level, suffix, tasks in [("level1", "_l1", ["ci_hard", "ci_obfuscated", "dp", "ar"]),
                                 ("level2", "_l2", ["ci_hard", "dp", "ar"])]:
        report.append(f"== {level.upper()} (results_rerun/{level}/) ==")
        for model in L12_MODELS:
            for name in SYSTEMS:
                mods = None
                for task in tasks:
                    fp = os.path.join("results_rerun", level, f"{model}_{name}_{task}{suffix}.json")
                    if not os.path.isfile(fp):
                        continue
                    if task == "ci_obfuscated":
                        data = load_obfuscated_aligned(name)
                    else:
                        src_task = "ci_hard" if task == "ci_hard" else task
                        data = load_task_data(proj_path(name), src_task, None)
                    if task == "ar" and mods is None:
                        mods = all_modules_cache.setdefault(name, load_modules(proj_path(name)))
                    process(fp, level, task, data, mods, args.dry_run, report)

    print("\n".join(report))
    n_skip = sum(1 for ln in report if "[SKIP]" in ln)
    print(f"\n完成。跳过 {n_skip} 个文件。" + ("（dry-run，未写入）" if args.dry_run else ""))


if __name__ == "__main__":
    main()
