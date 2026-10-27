#!/usr/bin/env python3
"""
Trial fix for the Linux AKG include resolver (does NOT overwrite any data).

Original rule (pipeline/generate_all.py): resolve '#include <a/b.h>' by
basename only, picking the candidate with the longest common path prefix.
Because include/ is excluded from the index, '<linux/kernel.h>' ends up at
'arch/sparc/kernel/kernel.h' etc.

Stricter rule tried here (module set and file index unchanged):
  * path-qualified include (contains '/'): candidate must equal the include
    string or end with '/' + include string; otherwise unresolved (external
    or not indexed). If several candidates tie on common prefix with the
    source, drop it as ambiguous (e.g. <asm/io.h> from a driver).
  * bare include ('x.h'): prefer the file in the source's own directory,
    otherwise fall back to the original basename heuristic (counted).

Reports how the module graph and the existing DP/AR task labels change.
Output: experiment/akg_fix_trial_linux.json
"""

import json
import os
from collections import Counter, defaultdict

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP)
SYS = os.path.join(ROOT, "linux")
TASKS = os.path.join(EXP, "remote_experiment", "linux")
OUT = os.path.join(EXP, "akg_fix_trial_linux.json")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cpl(a, b):
    n = 0
    for x, y in zip(a.split("/"), b.split("/")):
        if x != y:
            break
        n += 1
    return n


def module_edges(fdeps, f2m):
    c = Counter()
    for s, t in fdeps:
        a, b = f2m.get(s), f2m.get(t)
        if a and b and a != b:
            c[(a, b)] += 1
    return c


def main():
    files = load(os.path.join(SYS, "file_index.json"))["files"]
    f2m = {f["path"]: f["module"] for f in files}
    by_base = defaultdict(list)
    for p in f2m:
        by_base[os.path.basename(p)].append(p)
    raws = load(os.path.join(SYS, "dependency_raw.json"))["dependencies"]
    old_f = [(d["source"], d["target"]) for d in load(os.path.join(SYS, "file_dependency.json"))["dependencies"]]

    new_f, why = [], Counter()
    for r in raws:
        src, inc = r["source"], r["target"]
        cands = by_base.get(os.path.basename(inc), [])
        res = None
        if "/" in inc:
            ok = [c for c in cands if c == inc or c.endswith("/" + inc)]
            if not ok:
                why["path_include_unresolved"] += 1
            else:
                best = max(cpl(src, c) for c in ok)
                top = [c for c in ok if cpl(src, c) == best]
                if len(top) == 1:
                    res = top[0]
                    why["path_include_resolved"] += 1
                else:
                    why["path_include_ambiguous_dropped"] += 1
        else:
            same = os.path.join(os.path.dirname(src), inc).replace("\\", "/")
            if same in f2m:
                res = same
                why["bare_include_same_dir"] += 1
            elif cands:
                res = max(cands, key=lambda c: cpl(src, c))
                why["bare_include_basename_fallback"] += 1
            else:
                why["bare_include_unresolved"] += 1
        if res and res != src:
            new_f.append((src, res))

    old_m, new_m = module_edges(old_f, f2m), module_edges(new_f, f2m)
    # sanity: old aggregation must reproduce module_dependency.json
    md = {(d["source_module"], d["target_module"]): d["weight"]
          for d in load(os.path.join(SYS, "module_dependency.json"))["dependencies"]}
    assert dict(old_m) == md, "old aggregation does not reproduce module_dependency.json"

    old_e, new_e = set(old_m), set(new_m)
    removed, added = old_e - new_e, new_e - old_e

    # validation sample edges (linux)
    import csv
    with open(os.path.join(EXP, "akg_validation_sample.csv"), encoding="utf-8") as f:
        samp = [(r["source_module"], r["target_module"]) for r in csv.DictReader(f) if r["system"] == "linux"]

    dp = jsonl(os.path.join(TASKS, "dependency_prediction.jsonl"))
    flips = Counter()
    for t in dp:
        new_label = int((t["module_A"], t["module_B"]) in new_e)
        flips[f"{t['label']}->{new_label}"] += 1

    ar = jsonl(os.path.join(TASKS, "architecture_reconstruction.jsonl"))
    ar_srcs = sorted({t["source_module"] for t in ar})
    out_old, out_new = defaultdict(set), defaultdict(set)
    for a, b in old_e:
        out_old[a].add(b)
    for a, b in new_e:
        out_new[a].add(b)
    jac = [len(out_old[s] & out_new[s]) / len(out_old[s] | out_new[s]) if out_old[s] | out_new[s] else 1.0
           for s in ar_srcs]
    ar_edge_kept = sum((t["source_module"], t["target_module"]) in new_e for t in ar)

    top_removed = sorted(((old_m[e], e) for e in removed), reverse=True)[:15]
    top_added = sorted(((new_m[e], e) for e in added), reverse=True)[:15]
    res = {
        "raw_includes": len(raws),
        "resolution": dict(why),
        "file_links_old": len(old_f), "file_links_new": len(new_f),
        "module_edges_old": len(old_e), "module_edges_new": len(new_e),
        "module_edges_kept": len(old_e & new_e),
        "module_edges_removed": len(removed), "module_edges_added": len(added),
        "weight_old": sum(old_m.values()), "weight_new": sum(new_m.values()),
        "validation_sample_linux": len(samp),
        "validation_sample_survive": sum(e in new_e for e in samp),
        "dp_tasks": len(dp), "dp_label_flips": dict(flips),
        "ar_tasks": len(ar), "ar_task_edges_still_present": ar_edge_kept,
        "ar_source_modules": len(ar_srcs),
        "ar_target_set_jaccard_mean": round(sum(jac) / len(jac), 3),
        "top_removed": [f"{a} -> {b} ({w})" for w, (a, b) in top_removed],
        "top_added": [f"{a} -> {b} ({w})" for w, (a, b) in top_added],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
