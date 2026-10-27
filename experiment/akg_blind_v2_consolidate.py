"""把盲审表按边合并成一行一条，便于在 Excel 中逐条审阅（不添加任何判定或推荐）。
用法：python akg_blind_v2_consolidate.py [--show 10]
输出：akg_ai_blind_v2_review.csv（200 行，label 列留空供填写）
"""
import csv
import sys
from collections import OrderedDict

SRC = "akg_ai_blind_v2.csv"
OUT = "akg_ai_blind_v2_review.csv"

rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))

edges = OrderedDict()
for r in rows:
    if r["pilot_id"] not in edges:
        edges[r["pilot_id"]] = []
    edges[r["pilot_id"]].append(r)

out_rows = []
for pid, rs in edges.items():
    e = rs[0]
    ev = []
    for r in rs:
        ev.append(
            f"{r['source_file']}:{r['line']} | {r['statement']} | resolved={r['resolved_target']} "
            f"| disk={r['resolved_on_disk']} | resolved_module={r['resolved_target_module']}"
        )
    out_rows.append({
        "pilot_id": pid,
        "system": e["system"],
        "source_module": e["source_module"],
        "target_module": e["target_module"],
        "weight": e["weight"],
        "n_examples": len(rs),
        "n_candidates": e["n_candidates"],
        "evidence": "\n".join(ev),
        "candidates_top15": e["candidates_top15"],
        "label_blind": "",
        "confidence_1to3": "",
        "comment": "",
    })

with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)
print(f"written {OUT}: {len(out_rows)} edges")

if "--show" in sys.argv:
    n = int(sys.argv[sys.argv.index("--show") + 1])
    for row in out_rows[:n]:
        print(f"\n[{row['pilot_id']}] {row['system']}  {row['source_module']} -> {row['target_module']}"
              f"  (weight={row['weight']}, ex={row['n_examples']}, candidates={row['n_candidates']})")
        for line in row["evidence"].split("\n"):
            print("   ", line)
