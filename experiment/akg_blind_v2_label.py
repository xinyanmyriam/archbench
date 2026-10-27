"""把人工盲审标签写回 akg_ai_blind_v2.csv（每条边只写第一行，续行保持 '^'）。
用法：
    python akg_blind_v2_label.py V001=C V002=C V008=C:注释...
    python akg_blind_v2_label.py --file labels.txt   # 每行 "V001 C [注释]"
"""
import csv, sys

CSV_PATH = "akg_ai_blind_v2.csv"
FIELDS = ["pilot_id", "sample_id", "system", "source_module", "target_module", "weight", "ex",
          "source_file", "line", "statement", "reference_string", "resolved_target",
          "resolved_on_disk", "resolved_target_module", "n_candidates", "candidates_top15",
          "label_blind", "confidence_1to3", "comment"]
VALID = {"C", "W", "S", "G", "U"}
ALIAS = {"Correct": "C", "Wrong-resolution": "W", "Spurious": "S", "Granularity": "G", "Unclear": "U"}


def parse_args(argv):
    pairs = []
    if argv and argv[0] == "--file":
        for line in open(argv[1], encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 2)
            pairs.append((parts[0], parts[1], parts[2] if len(parts) > 2 else ""))
    else:
        for a in argv:
            left, _, comment = a.partition(":")
            pid, _, label = left.partition("=")
            pairs.append((pid.strip(), label.strip(), comment.strip()))
    return pairs


pairs = parse_args(sys.argv[1:])
with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

seen = set()
unknown = []
for pid, label, comment in pairs:
    label = ALIAS.get(label, label)
    if label not in VALID:
        unknown.append((pid, label))
        continue
    hit = False
    for r in rows:
        if r["pilot_id"] == pid and r["label_blind"] != "^":
            r["label_blind"] = label
            if comment:
                r["comment"] = comment
            hit = True
            seen.add(pid)
    if not hit:
        unknown.append((pid, "row-not-found"))

with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

filled = {r["pilot_id"] for r in rows if r["label_blind"] not in ("", "^")}
print(f"updated {len(seen)} edges; total labeled {len(filled)}/200")
if unknown:
    print("skipped:", unknown)
from collections import Counter
print(dict(Counter(r["label_blind"] for r in rows if r["label_blind"] not in ("", "^"))))
