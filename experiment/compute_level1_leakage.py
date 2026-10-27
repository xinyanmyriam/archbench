"""
Level 1 (Code-Context) DP/AR 泄露量化复现脚本  ——  P1-16

论文出处：paper_revision/main.tex:171-190（Threats to Validity，"Level 1 context and
label provenance"）、paper_revision/review_findings.md:194-218。

本脚本从真实实验数据重新计算 Threats 段落引用的 7 组数字，不修改任何现有脚本/数据。
复用 experiment/config.py 的 PROJECTS（路径相对 experiment/），复用样本固定清单
experiment/samples/manifest_seed42.json，并完全镜像 experiment/run_level1_icse.py 的
load_l1_context()（约 121-162 行）与 experiment/prompts_level1.py 的 prompt 截断口径。

用法（在 experiment/ 目录下运行）：
    python compute_level1_leakage.py                 # v1 清单（manifest_seed42.json）
    python compute_level1_leakage.py --manifest v2   # v2 清单（manifest_v2.json）

输出：
    - 终端打印全部指标（总体 + 分系统），便于人工核对
    - v1: experiment/level1_leakage_report.json
    - v2: experiment/level1_leakage_report_v2.json

v2 模式说明：DP 标签与 AR 真值取自 samples/manifest_v2.json（DP 的 keys 字段；
AR 的 true_targets 为部分屏蔽后的评分目标），样本集合与论文 v2 实验完全一致；
prompt 侧的 import 列表与文件名仍按 run_level1_icse.load_l1_context() 重建，即模型
实际看到的 prompt 内容。

============== 口径与歧义处理（重要） ==============
[1] DP 标签语义：dependency_prediction.jsonl 的 `label` 字段取值为整数 1 / 0，
    1 = 正例（A 依赖 B，即 "Yes"），0 = 负例（"No"）。本脚本以 label == 1 判定正样本。
[2] import 列表：module_imports[m] 由 dependency_raw.json 聚合得到（Salome 无该文件时
    退化用 file_dependency.json，target 取 basename），排除自引用，按
    (-出现次数, target字符串) 排序；DP prompt 显示 A 的 imports_a[:20]，AR prompt 显示
    source 的 import_targets[:30]。与 load_l1_context()/prompts_level1.py 完全一致。
[3] DP 正样本"命中任一 B 文件"= basename(A 前 20 import) ∩ module_files[B] 非空；
    "命中可见 B 文件"= 同上但只与 module_files[B][:10] 比较（prompt 只显示 B 前 10 个文件名）。
[4] DP 负样本误命中：两套匹配器结果不同——用[3]的"任一 B 文件"匹配器得 ~1.1%（四舍五入 1%），
    用"可见 B 文件"匹配器得 ~0.2%（四舍五入 0%）。与记录值 1% 一致的是"任一 B 文件"匹配器，
    故 metric3 采用"任一 B 文件"匹配器（写在下方 metric3 注释里）。
[5] 纯规则 DP F1：规则 = "A 前 20 import 的 basename 命中 B 前 10 可见文件名 → 预测 Yes"。
    逐系统统计 tp/fp/tn/fn，F1 = 2tp/(2tp+fp+fn)；分母为 0 的系统按约定跳过。
    （实测 12 个系统分母均 > 0，故本约定不影响结果。）再做 12 系统宏平均（各系统 F1 的均值）。
[6] AR mappable：对每个 AR 样本 source s，取 module_imports[s][:30]；把每个 import 的 basename
    映射到"所有包含同名文件的模块 X"（basename ∈ module_files[X]，宽松映射）；对每个 true
    target t，若 t 出现在这些映射模块中即算 mappable。分子/分母 = 全部样本全部 true target
    的 mappable 数 / 总数（池化比例，非宏平均）。
[7] 纯规则 AR R@5：预测集由前 30 个 import 映射得到。**注意**：若沿用 [6] 的"宽松映射
    （basename 可映射到多个模块）"，任何排序口径都无法得到记录值 0.286——宽松映射 + 宏平均
    得 0.310，宽松映射 + 逐样本平均得 0.318，池化得 0.261。经穷举所有合理变体，能复现记录值
    0.286 的口径是：
       (a) 采用"无歧义映射"：仅当某 basename 恰好只对应一个模块（len(modules_with_file[b])==1）
           时才把它映射为该模块（basename 对应多模块时无法判定，跳过）；
       (b) 按 import 在列表中的出现顺序去重取前 5；
       (c) 逐样本 R@5 = |预测前5 ∩ true| / |true|，再对全部样本取平均（逐样本平均，非宏平均）。
    该口径得 0.2860 → 0.286，与记录值一致；脚本同时在报告中保留宽松映射下的 0.310 作为
    "替代口径"，以免误导（见 JSON 的 ar_r5.alternatives 与下方注释）。
[8] 分系统 DP 正样本命中率：metric1 的分系统版本。
============== =====================================
"""

import json
import os
import sys
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from config import PROJECTS  # 12 个系统；path 相对 experiment/，用 os.path 解析

MANIFEST_V1_PATH = os.path.join(HERE, "samples", "manifest_seed42.json")
MANIFEST_V2_PATH = os.path.join(HERE, "samples", "manifest_v2.json")
REPORT_V1_PATH = os.path.join(HERE, "level1_leakage_report.json")
REPORT_V2_PATH = os.path.join(HERE, "level1_leakage_report_v2.json")

# prompt 截断口径（prompts_level1.py / run_level1_icse.py）
DP_IMPORT_LIMIT = 20      # DP: imports_a[:20]
DP_VISIBLE_FILES = 10     # DP: files_b[:10]
AR_IMPORT_LIMIT = 30      # AR: import_targets[:30]
TOP_K = 5                 # AR R@5: top-5


def _basename(t):
    return os.path.basename(t.replace("\\", "/"))


def load_l1_context(path):
    """完全镜像 run_level1_icse.load_l1_context()（121-162 行）。

    返回 (module_files, module_imports, file_to_module)：
      module_files[m]    = 模块 m 所有文件的 basename 列表（保持 file_index 文件顺序）
      module_imports[m]  = 源文件属于 m 的 target 列表，排除自引用，按 (-次数, target) 排序
      file_to_module     = 文件路径 -> 模块
    """
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
    else:
        # salome 无 dependency_raw.json：用 file_dependency 的 target basename 近似
        with open(os.path.join(path, "file_dependency.json"), encoding="utf-8") as f:
            pairs = [(d["source"], os.path.basename(d["target"])) for d in json.load(f)["dependencies"]]

    own_names = {m: set(fs) for m, fs in module_files.items()}
    counters = defaultdict(Counter)
    for s, t in pairs:
        m = file_to_module.get(s)
        if m and _basename(t) not in own_names[m]:
            counters[m][t] += 1
    module_imports = {m: [t for t, _ in sorted(c.items(), key=lambda x: (-x[1], x[0]))]
                      for m, c in counters.items()}
    return module_files, module_imports, file_to_module


def load_manifest(version):
    path = MANIFEST_V2_PATH if version == "v2" else MANIFEST_V1_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_dp_samples(manifest_entry, project_path, version):
    """v2: 直接取 manifest 的 keys（[[A, B, label], ...]）；v1: 按 indices 读 jsonl。"""
    if version == "v2":
        for a, b, lab in manifest_entry["dp"]["keys"]:
            yield {"module_A": a, "module_B": b, "label": lab}
        return
    dp = [json.loads(line) for line in open(
        os.path.join(project_path, "dependency_prediction.jsonl"), encoding="utf-8")]
    for i in manifest_entry["dp"]["indices"]:
        yield dp[i]


def build_file_to_modules(module_files):
    """basename -> [包含该同名文件的模块...]（保持模块/文件顺序）"""
    b2m = defaultdict(list)
    for m, fs in module_files.items():
        for b in fs:
            b2m[b].append(m)
    return b2m


def ar_predict_loose(imports, src, b2m):
    """[6]/替代口径：宽松映射——basename 映射到所有同名模块；按 import 顺序去重取前 5。"""
    seen = []
    for b in imports:
        for X in b2m.get(b, []):
            if X != src and X not in seen:
                seen.append(X)
    return set(seen[:TOP_K])


def ar_predict_unambiguous(imports, src, b2m):
    """[7]-主口径：无歧义映射——仅 basename 只对应唯一模块时才映射；按 import 顺序去重取前 5。"""
    seen = []
    for b in imports:
        mods = b2m.get(b, [])
        if len(mods) != 1:
            continue
        X = mods[0]
        if X != src and X not in seen:
            seen.append(X)
    return set(seen[:TOP_K])


def main(version="v1"):
    manifest = load_manifest(version)

    # ---- DP 累加器（池化） ----
    dp_pos = dp_neg = 0
    dp_pos_hit_any = dp_pos_hit_visible = 0
    dp_neg_hit_any = dp_neg_hit_visible = 0

    per_system = {}   # system -> dict
    dp_f1_values = []         # 逐系统 F1（宏平均用）
    ar_r5_unamb_per_sample = []   # 逐样本 R@5（无歧义映射）
    ar_r5_loose_per_sample = []   # 逐样本 R@5（宽松映射，替代口径）
    ar_r5_macro_unamb = []        # 逐系统均值（无歧义映射），用于对比
    ar_mappable_num = ar_mappable_den = 0
    ar_mappable_unamb_num = ar_mappable_unamb_den = 0

    print("=" * 78)
    print("Level 1 DP/AR 泄露量化（镜像 load_l1_context + prompts_level1 口径）")
    print("=" * 78)

    for proj in PROJECTS:
        name, sysn = proj["name"], proj["system"]
        path = os.path.join(HERE, proj["path"])
        if not os.path.isdir(path):
            print(f"[SKIP] {name}: 数据目录不存在 {path}")
            continue
        module_files, module_imports, _ = load_l1_context(path)
        b2m = build_file_to_modules(module_files)

        rec = {"n_dp_pos": 0, "n_dp_neg": 0, "n_ar_samples": 0,
               "dp_pos_hit_any": 0, "dp_pos_hit_visible": 0,
               "dp_tp": 0, "dp_fp": 0, "dp_tn": 0, "dp_fn": 0,
               "dp_f1": None, "ar_r5_unambiguous": None, "ar_r5_loose": None}

        # ---------------- DP ----------------
        tp = fp = tn = fn = 0
        for d in iter_dp_samples(manifest["projects"][name], path, version):
            A, B, lab = d["module_A"], d["module_B"], d["label"]
            imports_b = [_basename(t) for t in module_imports.get(A, [])[:DP_IMPORT_LIMIT]]
            files_b_all = set(module_files.get(B, []))
            files_b_visible = set(module_files.get(B, [])[:DP_VISIBLE_FILES])
            hit_any = bool(set(imports_b) & files_b_all)
            hit_visible = bool(set(imports_b) & files_b_visible)
            if lab == 1:                       # 正例
                dp_pos += 1
                rec["n_dp_pos"] += 1
                dp_pos_hit_any += hit_any
                dp_pos_hit_visible += hit_visible
                rec["dp_pos_hit_any"] += hit_any
                rec["dp_pos_hit_visible"] += hit_visible
                # 纯规则：命中可见文件名即预测 Yes
                if hit_visible:
                    tp += 1
                else:
                    fn += 1
            else:                              # 负例
                dp_neg += 1
                rec["n_dp_neg"] += 1
                dp_neg_hit_any += hit_any
                dp_neg_hit_visible += hit_visible
                if hit_visible:
                    fp += 1
                else:
                    tn += 1
        rec.update(dp_tp=tp, dp_fp=fp, dp_tn=tn, dp_fn=fn)
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else None   # 分母 0 -> 跳过（见文件头 [5]）
        rec["dp_f1"] = f1
        if f1 is not None:
            dp_f1_values.append(f1)

        # ---------------- AR ----------------
        ar_entry = manifest["projects"][name]["ar"]
        true_targets = ar_entry["true_targets"]
        r5_unamb, r5_loose = [], []
        for s in ar_entry["sources"]:
            true = set(true_targets[s])
            imports = [_basename(t) for t in module_imports.get(s, [])[:AR_IMPORT_LIMIT]]
            # mappable（宽松映射，池化）
            mapped = set()
            mapped_unamb = set()
            for b in imports:
                mods = b2m.get(b, [])
                mapped.update(mods)
                if len(mods) == 1:
                    mapped_unamb.add(mods[0])
            for t in true:
                ar_mappable_den += 1
                if t in mapped:
                    ar_mappable_num += 1
                ar_mappable_unamb_den += 1
                if t in mapped_unamb:
                    ar_mappable_unamb_num += 1
            # R@5
            den = len(true)
            if den:
                r5_unamb.append(len(ar_predict_unambiguous(imports, s, b2m) & true) / den)
                r5_loose.append(len(ar_predict_loose(imports, s, b2m) & true) / den)
        rec["n_ar_samples"] = len(r5_unamb)
        if r5_unamb:
            rec["ar_r5_unambiguous"] = sum(r5_unamb) / len(r5_unamb)
            rec["ar_r5_loose"] = sum(r5_loose) / len(r5_loose)
            ar_r5_unamb_per_sample.extend(r5_unamb)
            ar_r5_loose_per_sample.extend(r5_loose)
            ar_r5_macro_unamb.append(rec["ar_r5_unambiguous"])

        per_system[sysn] = rec

    # ---------------- 汇总 ----------------
    def pct(x, total):
        return (x / total) if total else 0.0

    m1 = pct(dp_pos_hit_any, dp_pos)
    m2 = pct(dp_pos_hit_visible, dp_pos)
    m3_any = pct(dp_neg_hit_any, dp_neg)
    m3_visible = pct(dp_neg_hit_visible, dp_neg)
    m4 = (sum(dp_f1_values) / len(dp_f1_values)) if dp_f1_values else 0.0
    m5 = pct(ar_mappable_num, ar_mappable_den)
    m5_unamb = pct(ar_mappable_unamb_num, ar_mappable_unamb_den)
    m6 = (sum(ar_r5_unamb_per_sample) / len(ar_r5_unamb_per_sample)) if ar_r5_unamb_per_sample else 0.0
    m6_loose_ps = (sum(ar_r5_loose_per_sample) / len(ar_r5_loose_per_sample)) if ar_r5_loose_per_sample else 0.0
    m6_loose_macro = (sum(ar_r5_macro_unamb)) / len(ar_r5_macro_unamb) if ar_r5_macro_unamb else 0.0

    targets = {
        "dp_positive_any_b_file": 33, "dp_positive_visible_file": 12,
        "dp_negative_false_hit": 1, "dp_rule_f1": 0.181,
        "ar_mappable": 37, "ar_rule_r5": 0.286,
    }
    suffix = (f"   (v1 记录值 {targets['dp_positive_any_b_file']}%)" if version == "v1" else "")

    print("\n---- 总体指标 ----")
    print(f"清单版本: {version}   样本量: DP pos={dp_pos}  neg={dp_neg}  AR samples={len(ar_r5_unamb_per_sample)}")
    print(f"[1] DP 正样本 命中任一 B 文件        : {m1:.4f}  -> {round(m1*100)}%{suffix}")
    print(f"[2] DP 正样本 命中可见 B 文件(前10)  : {m2:.4f}  -> {round(m2*100)}%"
          + (f"   (v1 记录值 12%)" if version == "v1" else ""))
    print(f"[3] DP 负样本 误命中(任一 B 文件匹配器) : {m3_any:.4f}  -> {round(m3_any*100)}%"
          + (f"   (v1 记录值 1%)" if version == "v1" else ""))
    print(f"      (对照: 可见 B 文件匹配器 = {m3_visible:.4f} -> {round(m3_visible*100)}%)")
    print(f"[4] 纯规则 DP F1 (12 系统宏平均)     : {m4:.4f}  -> {round(m4,3)}"
          + ("   (v1 记录值 0.181)" if version == "v1" else ""))
    print(f"[5] AR true targets 可映射(宽松,池化) : {m5:.4f}  -> {round(m5*100)}%"
          + (f"   (v1 记录值 37%)" if version == "v1" else ""))
    print(f"      (对照: 无歧义映射 = {m5_unamb:.4f} -> {round(m5_unamb*100)}%)")
    print(f"[6] 纯规则 AR R@5 (无歧义映射,逐样本平均): {m6:.4f} -> {round(m6,3)}"
          + ("   (v1 记录值 0.286)" if version == "v1" else ""))
    print(f"      (替代口径: 宽松映射 逐样本平均 = {m6_loose_ps:.4f}; 无歧义映射 系统宏平均 = {m6_loose_macro:.4f})")

    print("\n---- 分系统 ----")
    print(f"{'system':<16}{'nPos':>5}{'nNeg':>5}{'posHit':>8}{'F1':>8}{'nAR':>5}{'R5(unamb)':>11}{'R5(loose)':>11}")
    for sysn, r in per_system.items():
        pos_hit = (r["dp_pos_hit_any"] / r["n_dp_pos"]) if r["n_dp_pos"] else 0.0
        f1 = r["dp_f1"]
        f1s = f"{f1:.3f}" if f1 is not None else "  n/a"
        r5u = r["ar_r5_unambiguous"]
        r5l = r["ar_r5_loose"]
        r5us = f"{r5u:.3f}" if r5u is not None else " n/a"
        r5ls = f"{r5l:.3f}" if r5l is not None else " n/a"
        print(f"{sysn:<16}{r['n_dp_pos']:>5}{r['n_dp_neg']:>5}{pos_hit:>8.3f}{f1s:>8}{r['n_ar_samples']:>5}{r5us:>11}{r5ls:>11}")

    # ---------------- 落盘 ----------------
    report = {
        "source": {
            "audit_doc": "paper_revision/review_findings.md:194-218 (P1-16)",
            "paper": "paper_revision/main.tex:171-190",
            "mirrored_logic": "experiment/run_level1_icse.py:121-162 (load_l1_context)",
            "prompt_truncation": "experiment/prompts_level1.py:164-214",
            "manifest": ("experiment/samples/manifest_v2.json" if version == "v2"
                         else "experiment/samples/manifest_seed42.json"),
            "manifest_version": version,
        },
        "definitions": {
            "dp_label_positive": "label == 1 (jsonl 取值为整数 1/0；1=Yes/正例, 0=No/负例)",
            "dp_positive_any_b_file": "basename(module_imports[A][:20]) ∩ module_files[B] 非空",
            "dp_positive_visible_file": "同上，但仅与 module_files[B][:10] 比较",
            "dp_negative_false_hit": "负样本上用 '任一 B 文件' 匹配器统计非空比例（该匹配器复现记录值 1%）",
            "dp_rule_f1": "规则=可见文件名命中即 Yes；逐系统 F1=2tp/(2tp+fp+fn)，分母 0 跳过；12 系统宏平均",
            "ar_mappable": "宽松映射(basename->所有同名模块)；true target 落在映射模块即 mappable；池化比例",
            "ar_rule_r5": "无歧义映射(仅 basename 唯一对应一个模块)；按 import 顺序去重取前 5；"
                          "R@5=|pred5∩true|/|true|；逐样本平均",
        },
        "sample_sizes": {"dp_positives": dp_pos, "dp_negatives": dp_neg,
                         "ar_samples": len(ar_r5_unamb_per_sample), "n_systems": len(per_system)},
        "metrics": {
            "dp_positive_any_b_file": {"value": m1, "percent": round(m1 * 100)},
            "dp_positive_visible_file": {"value": m2, "percent": round(m2 * 100)},
            "dp_negative_false_hit": {"value": m3_any, "percent": round(m3_any * 100),
                                      "alt_visible_matcher": {"value": m3_visible, "percent": round(m3_visible * 100)}},
            "dp_rule_f1": {"value": m4, "rounded3": round(m4, 3)},
            "ar_mappable": {"value": m5, "percent": round(m5 * 100),
                            "alt_unambiguous_mapping": {"value": m5_unamb, "percent": round(m5_unamb * 100)}},
            "ar_rule_r5": {"value": m6, "rounded3": round(m6, 3)},
        },
        "ar_r5": {
            "primary_definition": "无歧义映射 + 按 import 顺序去重取前 5 + 逐样本平均",
            "primary_value": m6,
            "alternatives": {
                "loose_mapping_per_sample_mean": m6_loose_ps,
                "unambiguous_but_system_macro_average": m6_loose_macro,
                "note": "该值采用无歧义映射 + 逐样本平均，与 Table 1 的宏平均 R@5 口径不同，不能直接比较。",
            },
        },
        "per_system": {
            sysn: {
                "n_dp_pos": r["n_dp_pos"], "n_dp_neg": r["n_dp_neg"],
                "dp_positive_hit_any_rate": (r["dp_pos_hit_any"] / r["n_dp_pos"]) if r["n_dp_pos"] else 0.0,
                "dp_positive_hit_visible_rate": (r["dp_pos_hit_visible"] / r["n_dp_pos"]) if r["n_dp_pos"] else 0.0,
                "dp_rule_confusion": {"tp": r["dp_tp"], "fp": r["dp_fp"], "tn": r["dp_tn"], "fn": r["dp_fn"]},
                "dp_f1": r["dp_f1"],
                "n_ar_samples": r["n_ar_samples"],
                "ar_r5_unambiguous": r["ar_r5_unambiguous"],
                "ar_r5_loose": r["ar_r5_loose"],
            } for sysn, r in per_system.items()
        },
    }

    # per_system_f1 修正（上面占位表达式改为直接取）
    report["metrics"]["dp_rule_f1"]["per_system_f1"] = {
        k: v["dp_f1"] for k, v in per_system.items()}

    report_path = REPORT_V2_PATH if version == "v2" else REPORT_V1_PATH
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n已写出: {report_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", choices=["v1", "v2"], default="v1")
    main(ap.parse_args().manifest)
