#!/usr/bin/env python3
"""
Build experiment/samples/manifest_v2.json (manifest_seed42.json is left untouched).

Rules (decided 2026-10-20):
  * CI / CI-Hard: copied verbatim from v1.
  * DP (11 systems): same indices / items, labels taken from akg_v2 relabel.
  * AR (11 systems): same sampled sources; sources with no outgoing edge in the
    v2 module graph are dropped; ground truth of every remaining source is its
    FULL outgoing target set in the v2 graph (v1 used only the sampled edges).
  * linux: DP / AR come from akg_v2/linux/manifest_linux_v2.json (resampled,
    stratified negatives); AR ground truth rewritten with the same full-set rule.
"""
import copy
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SAMPLES = os.path.join(ROOT, "experiment", "samples")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def rel(p):
    return os.path.relpath(p, os.path.join(ROOT, "experiment")).replace(os.sep, "/")


def main():
    v1 = load(os.path.join(SAMPLES, "manifest_seed42.json"))
    v2 = {"seed": v1["seed"], "sizes": v1["sizes"], "version": "akg_v2",
          "derived_from": "manifest_seed42.json",
          "ar_ground_truth": "full outgoing target set in the akg_v2 module graph",
          "projects": {}}
    report = {}
    for sys_, p1 in v1["projects"].items():
        p2 = {"ci": copy.deepcopy(p1["ci"]), "ci_hard": copy.deepcopy(p1["ci_hard"])}
        d = os.path.join(HERE, sys_)
        if sys_ == "linux":
            lm = load(os.path.join(d, "manifest_linux_v2.json"))
            dp_src = os.path.join(d, "dependency_prediction_resampled.jsonl")
            ar_src = os.path.join(d, "architecture_reconstruction_resampled.jsonl")
            p2["dp"] = {"source_file": rel(dp_src), "n_pool": lm["dp"]["n_pool"],
                        "indices": lm["dp"]["indices"], "keys": lm["dp"]["keys"]}
            out = defaultdict(set)
            for e in load(os.path.join(d, "module_dependency.json"))["dependencies"]:
                if e["source_module"] != e["target_module"]:
                    out[e["source_module"]].add(e["target_module"])
            sources = lm["ar"]["sources"]
            full = {s: out[s] for s in sources}
            ar_indices, ar_pool = lm["ar"]["indices"], lm["ar"]["n_pool"]
            dp_flips = None
        else:
            dp_src = os.path.join(d, "dependency_prediction.jsonl")
            ar_src = os.path.join(d, "architecture_reconstruction.jsonl")
            dp = jl(dp_src)
            assert len(dp) == p1["dp"]["n_pool"], sys_
            keys, flips = [], defaultdict(int)
            for i, (a, b, l) in zip(p1["dp"]["indices"], p1["dp"]["keys"]):
                it = dp[i]
                assert (it["module_A"], it["module_B"]) == (a, b), (sys_, i)
                keys.append([a, b, it["label"]])
                if it["label"] != l:
                    flips[f"{l}->{it['label']}"] += 1
            p2["dp"] = {"source_file": rel(dp_src), "n_pool": len(dp),
                        "indices": p1["dp"]["indices"], "keys": keys}
            ar = jl(ar_src)
            assert len(ar) == p1["ar"]["n_pool"], sys_
            nts = {r["source_module"]: set(r.get("new_target_set") or []) for r in ar}
            sources = p1["ar"]["sources"]
            full = {s: nts[s] for s in sources}
            ar_indices, ar_pool = p1["ar"]["indices"], len(ar)
            dp_flips = dict(flips)
        kept = [s for s in sources if full[s]]
        dropped = [s for s in sources if not full[s]]
        p2["ar"] = {"source_file": rel(ar_src), "n_pool": ar_pool, "indices": ar_indices,
                    "sources": kept, "dropped_sources_no_outgoing_edge": dropped,
                    "true_targets": {s: sorted(full[s]) for s in kept}}
        v2["projects"][sys_] = p2
        report[sys_] = {
            "dp_items": len(p2["dp"]["keys"]),
            "dp_pos": sum(k[2] for k in p2["dp"]["keys"]),
            "dp_flips": dp_flips,
            "ar_sources_v1": len(p1["ar"]["sources"]), "ar_sources_v2": len(kept),
            "ar_dropped": len(dropped),
            "ar_avg_gt_v1": round(sum(len(v) for v in p1["ar"]["true_targets"].values())
                                  / len(p1["ar"]["sources"]), 2),
            "ar_avg_gt_v2": round(sum(len(full[s]) for s in kept) / len(kept), 2),
        }
    out_p = os.path.join(SAMPLES, "manifest_v2.json")
    if os.path.exists(out_p):
        os.replace(out_p, out_p + ".bak")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(v2, f, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "manifest_v2_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    tot = {k: sum(r[k] for r in report.values()) for k in ("dp_items", "dp_pos", "ar_sources_v1", "ar_sources_v2", "ar_dropped")}
    for s, r in report.items():
        print(f"{s:13s} DP {r['dp_items']:3d} pos {r['dp_pos']:3d}  AR {r['ar_sources_v1']:3d}->{r['ar_sources_v2']:3d}"
              f"  |GT| {r['ar_avg_gt_v1']:6.2f}->{r['ar_avg_gt_v2']:6.2f}  flips {r['dp_flips']}")
    print("TOTAL", tot)


if __name__ == "__main__":
    main()
