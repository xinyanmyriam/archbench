"""非 LLM 基线 v1（原始 AKG 标签）与 v2（akg_v2 重建标签 + AR 部分屏蔽）对比记录。

输入：results_rerun/baselines_fixed.json（v1）、results_v2/baselines_fixed.json（v2）
输出：results_v2/baselines_v1_vs_v2.{json,md}
用法：python compare_baselines.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(HERE, "results_rerun", "baselines_fixed.json")
V2 = os.path.join(HERE, "results_v2", "baselines_fixed.json")
OUT_MD = os.path.join(HERE, "results_v2", "baselines_v1_vs_v2.md")
OUT_JSON = os.path.join(HERE, "results_v2", "baselines_v1_vs_v2.json")

# (task, baseline, metric, 展示名, 是否百分比)
ROWS = [
    ("ci", "random", "accuracy", "CI  Random", True),
    ("ci", "frequency", "accuracy", "CI  Frequency", True),
    ("ci", "path_heuristic", "accuracy", "CI  Path Heuristic", True),
    ("ci_hard", "random", "accuracy", "CI-Hard  Random", True),
    ("ci_hard", "frequency", "accuracy", "CI-Hard  Frequency", True),
    ("ci_hard", "path_heuristic", "accuracy", "CI-Hard  Path Heuristic", True),
    ("dp", "random", "f1", "DP  Random  F1", True),
    ("dp", "frequency", "f1", "DP  Frequency  F1", True),
    ("dp", "graph_neighbor", "f1", "DP  Graph Neighbor  F1", True),
    ("ar", "random", "precision_at_5", "AR  Random  P@5", True),
    ("ar", "random", "recall_at_5", "AR  Random  R@5", True),
    ("ar", "frequency", "precision_at_5", "AR  Frequency  P@5", True),
    ("ar", "frequency", "recall_at_5", "AR  Frequency  R@5", True),
    ("ar", "graph_two_hop", "precision_at_5", "AR  Graph Two-Hop  P@5", True),
    ("ar", "graph_two_hop", "recall_at_5", "AR  Graph Two-Hop  R@5", True),
    ("ar", "graph_neighbor", "precision_at_5", "AR  Graph Neighbor  P@5", True),
    ("ar", "graph_neighbor", "recall_at_5", "AR  Graph Neighbor  R@5", True),
]


def get(d, task, bl, metric):
    return d["macro"][task][bl][metric]


def fmt(x, pct):
    return f"{x:.3f}" if pct else f"{x}"


def main():
    v1 = json.load(open(V1, encoding="utf-8"))
    v2 = json.load(open(V2, encoding="utf-8"))
    assert set(v1["per_system"]) == set(v2["per_system"])

    rows = []
    for task, bl, metric, label, pct in ROWS:
        a, b = get(v1, task, bl, metric), get(v2, task, bl, metric)
        rows.append({"task": task, "baseline": bl, "metric": metric, "label": label,
                     "v1": a, "v2": b, "delta": b - a})

    systems = list(v2["per_system"])
    per_system = {}
    for p in systems:
        per_system[p] = {
            t: {bl: {m: round(v2["per_system"][p][t][bl][m], 4)
                     for m in v2["per_system"][p][t][bl] if not m.startswith("n")}
                 for bl in v2["per_system"][p][t]}
            for t in ["dp", "ar"]
        }
        per_system[p]["_v1_dp_f1_gn"] = round(v1["per_system"][p]["dp"]["graph_neighbor"]["f1"], 4)
        per_system[p]["_v1_ar_p5_2hop"] = round(v1["per_system"][p]["ar"]["graph_two_hop"]["precision_at_5"], 4)

    L = [
        "# 非 LLM 基线 v1 → v2 对比",
        "",
        f"v1：`results_rerun/baselines_fixed.json`（清单 `{v1['manifest']}`，原始 AKG 标签）。",
        f"v2：`results_v2/baselines_fixed.json`（清单 `{v2['manifest']}`，akg_v2 重建标签；"
        "AR 真解为每个源模块隐藏的一半出边）。",
        "两版均由 `run_baselines_fixed.py` 生成（同一随机种子与评分口径），可直接比较。",
        "AR P@5 = hits/5，R@5 = hits/|真解|。DP Graph Neighbor、AR Graph Neighbor 均已屏蔽被查询边。",
        "",
        "## 12 个系统宏平均",
        "",
        "| 任务 / 基线 | v1 | v2 | Δ |",
        "|---|---|---|---|",
    ]
    for r in rows:
        L.append(f"| {r['label']} | {r['v1']:.3f} | {r['v2']:.3f} | {r['delta']:+.3f} |")

    L += ["", "## 逐系统：DP F1", "",
          "| 系统 | Random v1→v2 | Frequency v1→v2 | Graph Neighbor v1→v2 |", "|---|---|---|---|"]
    for p in systems:
        a, b = v1["per_system"][p]["dp"], v2["per_system"][p]["dp"]
        L.append(f"| {p} | {a['random']['f1']:.3f}→{b['random']['f1']:.3f} "
                 f"| {a['frequency']['f1']:.3f}→{b['frequency']['f1']:.3f} "
                 f"| {a['graph_neighbor']['f1']:.3f}→{b['graph_neighbor']['f1']:.3f} |")

    L += ["", "## 逐系统：AR P@5 / R@5", "",
          "| 系统 | Frequency v1→v2 | Graph Two-Hop v1→v2 |", "|---|---|---|"]
    for p in systems:
        a, b = v1["per_system"][p]["ar"], v2["per_system"][p]["ar"]
        L.append(f"| {p} "
                 f"| {a['frequency']['precision_at_5']:.3f}/{a['frequency']['recall_at_5']:.3f}"
                 f"→{b['frequency']['precision_at_5']:.3f}/{b['frequency']['recall_at_5']:.3f} "
                 f"| {a['graph_two_hop']['precision_at_5']:.3f}/{a['graph_two_hop']['recall_at_5']:.3f}"
                 f"→{b['graph_two_hop']['precision_at_5']:.3f}/{b['graph_two_hop']['recall_at_5']:.3f} |")

    L += ["", "## 备注", "",
          "- AR Graph Neighbor 在 v1/v2 均为 0 是必然结果：它只从「可见出边」中排序，而真解恰是被屏蔽的那些边，两者不相交。"
          "Graph Two-Hop 通过可见边的下游传播间接命中隐藏边，因此非零。",
          "- `*_unmasked` 变体未列入本表（使用了被查询边，存在标签泄露），仅作诊断，见两个 json 原始文件。",
          "- v2 下 CI / CI-Hard 与 v1 完全一致（标签仍来自 file_index，未受图重建影响）。", ""]

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"v1_source": os.path.relpath(V1, HERE), "v2_source": os.path.relpath(V2, HERE),
                   "v1_manifest": v1["manifest"], "v2_manifest": v2["manifest"],
                   "macro_compare": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                                     for r in rows],
                   "per_system_v2": per_system}, f, ensure_ascii=False, indent=1)
    print("\n".join(L))
    print(f"\nSaved: {OUT_MD}\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
