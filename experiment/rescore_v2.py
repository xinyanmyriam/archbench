"""akg_v2 离线重打分（2026-10-20）

背景：DP/AR 标签改由 akg_v2（严格解析重建的模块图）给出，题目 indices 不变（linux 除外，
linux 已按新图重新采样，需要重新调 API，本脚本跳过 linux 的 DP/AR）。
本脚本对 results_rerun/ 中已存的 raw_response 用 v2 标签离线重算，不调 API、不改原文件：
  - DP：逐题标签换成 manifest_v2 的 keys[i][2]，重算 acc/P/R/F1
  - AR：按 source_module 对齐 v2 源模块（剔除无出边的源模块），真解 = 新图完整出边，
        P@5 = hits/5（固定分母，与 run_experiment / run_level1_icse 一致），R@5 = hits/|真解|
  - CI / CI-Hard / CI-obfuscated：与标签无关，原样复制
输出：results_v2/（L0）、results_v2/level1/（L1）、results_v2/rescore_v2_report.{json,md}

用法：
    python rescore_v2.py                       # 全部模型
    python rescore_v2.py --models qwen3-8b     # 只补某个模型（不覆盖全量报告）
"""

import os
os.environ["SAB_MANIFEST"] = "v2"  # 必须在 import run_experiment 之前

import argparse
import json
import shutil
from statistics import mean

from run_experiment import (
    PROJECTS, MANIFEST, MANIFEST_VERSION, MANIFEST_V1_PATH,
    load_task_data, load_modules, ar_ground_truth,
    parse_dp_response, parse_ar_response,
)
from baselines import load_project_data

assert MANIFEST_VERSION == "v2"

L0_MODELS = ["claude-opus-5.5", "gpt-6-sol", "kimi-k3", "gpt-oss-120b",
             "qwen3", "qwen3-8b", "llama-4-maverick", "llama-3.1-8b", "deepseek-v4-flash"]
L1_MODELS = ["claude-opus-5.5", "gpt-6-sol"]
SKIP_LABEL_TASKS = {"linux"}  # 新题，需重新调 API
PROJ = {p["name"]: p for p in PROJECTS}
SRC_ROOT = "results_rerun"
OUT_ROOT = "results_v2"

with open(MANIFEST_V1_PATH, encoding="utf-8") as f:
    MANIFEST_V1 = json.load(f)


def dp_metrics(preds, labels):
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    return {"accuracy": (tp + tn) / n if n else 0, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": n}


def rescore_dp(doc, name):
    data = load_task_data(PROJ[name]["path"], "dp", None)
    recs = doc["results"]
    assert len(recs) == len(data), f"len {len(recs)} vs {len(data)}"
    preds, old_labels, new_labels = [], [], []
    for rec, item in zip(recs, data):
        assert (rec["module_A"], rec["module_B"]) == (item["module_A"], item["module_B"]), "DP 题目错位"
        p = parse_dp_response(rec.get("raw_response") or "")
        p = 0 if p is None else p
        preds.append(p)
        old_labels.append(rec["label"])
        new_labels.append(item["label"])
        rec["label_v1"] = rec["label"]
        rec["label"] = item["label"]
        rec["predicted"] = p
    v1 = dp_metrics(preds, old_labels)
    v2 = dp_metrics(preds, new_labels)
    doc.update(v2)
    flips = sum(a != b for a, b in zip(old_labels, new_labels))
    return {"v1": v1, "v2": v2, "label_flips": flips, "n_pos_v2": sum(new_labels)}


def ar_candidates(name, source, sorted_modules):
    """与 build_ar_prompt 相同：去掉源模块后取前 50（L0 口径，用于估算召回上限）"""
    return set([m for m in sorted_modules if m != source][:50])


def rescore_ar(doc, name, all_modules, sorted_modules):
    v1_sources = MANIFEST_V1["projects"][name]["ar"]["sources"]
    recs = doc["results"]
    assert [r["source_module"] for r in recs] == v1_sources, "AR 结果与 v1 源模块顺序不一致"
    by_src = {r["source_module"]: r for r in recs}
    sources, gt = ar_ground_truth(name, load_task_data(PROJ[name]["path"], "ar", None))

    v1_p = [len(set(parse_ar_response(r.get("raw_response") or "", all_modules)) & set(r["true_targets"])) / 5
            for r in recs]
    new_recs, ps, rs, ceil = [], [], [], []
    for s in sources:
        r = dict(by_src[s])
        pred = parse_ar_response(r.get("raw_response") or "", all_modules)
        true = gt[s]
        hits = len(set(pred) & true)
        r["true_targets_v1"] = r["true_targets"]
        r["true_targets"] = sorted(true)
        r["predicted"] = pred
        r["precision"] = hits / 5
        r["recall"] = hits / len(true)
        ps.append(r["precision"])
        rs.append(r["recall"])
        ceil.append(len(true & ar_candidates(name, s, sorted_modules)) / len(true))
        new_recs.append(r)
    old = {"precision_at_5_stored": doc.get("precision_at_5"), "recall_at_5_stored": doc.get("recall_at_5"),
           "precision_at_5_fixed5": mean(v1_p), "n_sources": len(recs)}
    doc["results"] = new_recs
    doc["precision_at_5"] = mean(ps)
    doc["recall_at_5"] = mean(rs)
    doc["total"] = len(new_recs)
    new = {"precision_at_5": doc["precision_at_5"], "recall_at_5": doc["recall_at_5"],
           "n_sources": len(new_recs), "recall_ceiling_top50": mean(ceil)}
    return {"v1": old, "v2": new}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all", help="逗号分隔的模型名；all 表示全部")
    args = ap.parse_args()
    only = None if args.models == "all" else {s.strip() for s in args.models.split(",") if s.strip()}

    report = {"manifest": "manifest_v2.json", "levels": {}}
    mods_cache, sorted_cache = {}, {}
    errors = []

    jobs = []
    for m in L0_MODELS:
        if only and m not in only:
            continue
        jobs.append(("L0", m, "", "", ["ci", "ci_hard", "dp", "ar"]))
    for m in L1_MODELS:
        if only and m not in only:
            continue
        jobs.append(("L1", m, "level1", "_l1", ["ci", "ci_hard", "ci_obfuscated", "dp", "ar"]))

    for level, model, sub, suffix, tasks in jobs:
        os.makedirs(os.path.join(OUT_ROOT, sub), exist_ok=True)
        for name in PROJ:
            for task in tasks:
                src = os.path.join(SRC_ROOT, sub, f"{model}_{name}_{task}{suffix}.json")
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(OUT_ROOT, sub, os.path.basename(src))
                if task not in ("dp", "ar"):
                    shutil.copy2(src, dst)
                    continue
                if name in SKIP_LABEL_TASKS:
                    continue
                with open(src, encoding="utf-8") as f:
                    doc = json.load(f)
                try:
                    if task == "dp":
                        r = rescore_dp(doc, name)
                    else:
                        if name not in mods_cache:
                            mods_cache[name] = load_modules(PROJ[name]["path"])
                            freq = load_project_data(PROJ[name]["path"])["module_freq"]
                            sorted_cache[name] = sorted(mods_cache[name], key=lambda x: freq.get(x, 0), reverse=True)
                        r = rescore_ar(doc, name, mods_cache[name], sorted_cache[name])
                except AssertionError as e:
                    errors.append(f"{src}: {e}")
                    continue
                doc["manifest_version"] = "akg_v2"
                doc["rescored_from"] = src.replace("\\", "/")
                with open(dst, "w", encoding="utf-8") as f:
                    json.dump(doc, f, ensure_ascii=False, indent=1)
                report["levels"].setdefault(level, {}).setdefault(model, {}).setdefault(name, {})[task] = r

    report["errors"] = errors
    tag = "" if only is None else "_" + "+".join(sorted(only))
    with open(os.path.join(OUT_ROOT, f"rescore_v2_report{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    write_md(report, tag)
    print(f"done. errors={len(errors)}")
    for e in errors:
        print("  ", e)


def write_md(report, tag=""):
    L = ["# akg_v2 离线重打分（11 个系统，不含 linux DP/AR）", "",
         "AR P@5 一律为固定分母 5；“v1 存档” 列为 results_rerun 文件中记录的值。", ""]
    for level, models in report["levels"].items():
        L += [f"## {level} 各模型 11 系统均值", "",
              "| 模型 | DP Acc v1→v2 | DP F1 v1→v2 | AR P@5 v1存档 / v1(÷5) → v2 | AR R@5 v1→v2 | v2 R@5 上限(top50) |",
              "|---|---|---|---|---|---|"]
        for model, systems in models.items():
            dp = [s["dp"] for s in systems.values() if "dp" in s]
            ar = [s["ar"] for s in systems.values() if "ar" in s]
            f = lambda xs: f"{mean(xs):.3f}" if xs else "-"
            L.append(
                f"| {model} | {f([d['v1']['accuracy'] for d in dp])}→{f([d['v2']['accuracy'] for d in dp])} "
                f"| {f([d['v1']['f1'] for d in dp])}→{f([d['v2']['f1'] for d in dp])} "
                f"| {f([a['v1']['precision_at_5_stored'] for a in ar])} / {f([a['v1']['precision_at_5_fixed5'] for a in ar])}"
                f" → {f([a['v2']['precision_at_5'] for a in ar])} "
                f"| {f([a['v1']['recall_at_5_stored'] for a in ar])}→{f([a['v2']['recall_at_5'] for a in ar])} "
                f"| {f([a['v2']['recall_ceiling_top50'] for a in ar])} |")
        L.append("")
        L += [f"### {level} 逐系统（DP F1 / AR P@5 / AR R@5，v1→v2）", "",
              "| 模型 | 系统 | DP F1 | AR P@5(÷5) | AR R@5 |", "|---|---|---|---|---|"]
        for model, systems in models.items():
            for name, s in systems.items():
                dp, ar = s.get("dp"), s.get("ar")
                L.append(f"| {model} | {name} "
                         f"| {dp['v1']['f1']:.3f}→{dp['v2']['f1']:.3f} " if dp else f"| {model} | {name} | - ")
                L[-1] += (f"| {ar['v1']['precision_at_5_fixed5']:.3f}→{ar['v2']['precision_at_5']:.3f} "
                          f"| {ar['v1']['recall_at_5_stored']:.3f}→{ar['v2']['recall_at_5']:.3f} |" if ar else "| - | - |")
        L.append("")
    if report["errors"]:
        L += ["## 跳过的文件", ""] + [f"- {e}" for e in report["errors"]]
    with open(os.path.join(OUT_ROOT, f"rescore_v2_report{tag}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
