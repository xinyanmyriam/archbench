"""AR 部分屏蔽（方案 B）。

对 manifest_v2 中每个 AR 源模块：
  all_targets  = 新图完整出边（原 true_targets）
  true_targets = 按固定种子隐藏的 ceil(n/2) 条出边（L0/L1/L2 统一评分目标）
  其余出边由 L2 / 图基线作为可见依赖使用（dep_graph[src] - true_targets）。
幂等：已有 all_targets 时从 all_targets 重新抽取。首次运行备份为 manifest_v2_fullout.json。
"""
import json
import math
import os
import random
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "..", "experiment", "samples", "manifest_v2.json")
BACKUP = os.path.join(HERE, "..", "experiment", "samples", "manifest_v2_fullout.json")
SEED = 42


def main():
    if not os.path.exists(BACKUP):
        shutil.copy2(MANIFEST, BACKUP)
    with open(MANIFEST, encoding="utf-8") as f:
        m = json.load(f)

    report = {}
    for proj, ent in m["projects"].items():
        ar = ent["ar"]
        full = ar.get("all_targets") or ar["true_targets"]
        hidden = {}
        for s in ar["sources"]:
            tg = sorted(full[s])
            k = math.ceil(len(tg) / 2)
            rng = random.Random(f"{SEED}|{proj}|{s}")
            hidden[s] = sorted(rng.sample(tg, k))
        ar["all_targets"] = {s: sorted(full[s]) for s in ar["sources"]}
        ar["true_targets"] = hidden
        n = len(ar["sources"])
        report[proj] = {
            "sources": n,
            "avg_all": round(sum(len(v) for v in ar["all_targets"].values()) / n, 2),
            "avg_hidden": round(sum(len(v) for v in hidden.values()) / n, 2),
            "empty_shown": sum(len(v) == 1 for v in ar["all_targets"].values()),
        }

    m["ar_ground_truth"] = ("v2-partial-mask: true_targets = ceil(n/2) outgoing edges hidden per source "
                            f"(random.Random('{SEED}|<project>|<source>')); all_targets = full new-graph "
                            "outgoing edges; shown = all_targets - true_targets")
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "ar_partial_mask_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    for p, r in report.items():
        print(p, r)


if __name__ == "__main__":
    main()
