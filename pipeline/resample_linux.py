#!/usr/bin/env python3
"""
Linux: resample DP / AR task pools on the v2 module graph (seed 42), and build
a Linux-only evaluation manifest. Other systems keep their original items and
manifest_seed42.json entries (their manifest RNG stream is shared, so a global
rebuild would change them).

Sampling logic mirrors pipeline/generate_all.py::generate_benchmark_tasks
(DP: up to 2000 positives + equal hard negatives; AR: up to 2000 edges),
and experiment/build_sample_manifest.py (DP 200 items, AR up to 100 sources).
Edge / module lists are sorted first so the result is reproducible
(the original iterated a Python set).

Out (akg_v2/linux/):
  modules_index.json                         original modules + include/* modules
  dependency_prediction_resampled.jsonl
  architecture_reconstruction_resampled.jsonl
  manifest_linux_v2.json
  resample_stats.json
"""
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OD = os.path.join(HERE, "linux")
SEED = 42
SIZE_DP, SIZE_AR = 200, 100


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def dump(o, p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=1)


def dump_jsonl(rows, p):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def is_new(m):
    return m.startswith("include/")


def main():
    old_mods = load(os.path.join(ROOT, "linux", "modules_index.json"))["modules"]
    new_inc = load(os.path.join(OD, "stats.json"))["linux_new_include_modules"]
    modules_list = sorted(set(old_mods) | set(new_inc))
    dump({"system": "linux", "total": len(modules_list), "modules": modules_list},
         os.path.join(OD, "modules_index.json"))

    md = load(os.path.join(OD, "module_dependency.json"))["dependencies"]
    dep_pairs = {(d["source_module"], d["target_module"]) for d in md
                 if d["source_module"] != d["target_module"]}
    assert all(a in modules_list and b in modules_list for a, b in dep_pairs)
    outgoing, incoming = defaultdict(set), defaultdict(set)
    for a, b in dep_pairs:
        outgoing[a].add(b)
        incoming[b].add(a)
    dep_list = sorted(dep_pairs)
    mset = set(modules_list)

    random.seed(SEED)
    # ---- DP ----
    pos = random.sample(dep_list, min(2000, len(dep_list)))
    out_keys, in_keys = sorted(outgoing), sorted(incoming)

    # Stratified hard negatives: each positive's stratum is
    # (src is include/*, tgt is include/*); the negative is drawn from the same
    # stratum with the original three strategies, so "touches include/*" carries
    # no label signal.
    by_type = {True: sorted(m for m in modules_list if is_new(m)),
               False: sorted(m for m in modules_list if not is_new(m))}
    out_by = {t: sorted(k for k in out_keys if is_new(k) == t) for t in (True, False)}
    in_by = {t: sorted(k for k in in_keys if is_new(k) == t) for t in (True, False)}

    def hard_negative(sa, sb):
        s = random.choice([1, 2, 3])
        if s == 1 and out_by[sa]:
            src = random.choice(out_by[sa])
            pool = [m for m in by_type[sb] if m not in outgoing[src] and m != src]
            if pool:
                return (src, random.choice(pool))
        if s == 2 and in_by[sb]:
            tgt = random.choice(in_by[sb])
            pool = [m for m in by_type[sa] if m not in incoming[tgt] and m != tgt]
            if pool:
                return (random.choice(pool), tgt)
        a, b = random.choice(by_type[sa]), random.choice(by_type[sb])
        if a != b and (a, b) not in dep_pairs:
            return (a, b)
        return None

    neg, neg_order = set(), []
    for a, b in pos:
        sa, sb = is_new(a), is_new(b)
        for _ in range(10000):
            p = hard_negative(sa, sb)
            if p and p not in dep_pairs and p not in neg:
                break
        else:
            raise RuntimeError(f"no negative for stratum {(sa, sb)}")
        neg.add(p)
        neg_order.append(p)
    dp = [{"module_A": a, "module_B": b, "label": 1} for a, b in pos] + \
         [{"module_A": a, "module_B": b, "label": 0} for a, b in neg_order]
    # ---- AR ----
    ar = [{"source_module": a, "target_module": b}
          for a, b in random.sample(dep_list, min(2000, len(dep_list)))]
    dump_jsonl(dp, os.path.join(OD, "dependency_prediction_resampled.jsonl"))
    dump_jsonl(ar, os.path.join(OD, "architecture_reconstruction_resampled.jsonl"))

    # ---- Linux-only manifest (separate RNG stream) ----
    rng = random.Random(SEED)
    dp_idx = rng.sample(range(len(dp)), SIZE_DP) if len(dp) > SIZE_DP else list(range(len(dp)))
    ar_idx = rng.sample(range(len(ar)), SIZE_AR) if len(ar) > SIZE_AR else list(range(len(ar)))
    gt = defaultdict(set)
    for i in ar_idx:
        gt[ar[i]["source_module"]].add(ar[i]["target_module"])
    sources = list(gt)
    if len(sources) > SIZE_AR:
        sources = rng.sample(sources, SIZE_AR)
    manifest = {
        "seed": SEED, "project": "linux", "version": "akg_v2",
        "note": "Linux DP/AR resampled on v2 graph; CI/CI-Hard keep manifest_seed42.json",
        "dp": {"source_file": "akg_v2/linux/dependency_prediction_resampled.jsonl",
               "n_pool": len(dp), "indices": dp_idx,
               "keys": [[dp[i]["module_A"], dp[i]["module_B"], dp[i]["label"]] for i in dp_idx]},
        "ar": {"source_file": "akg_v2/linux/architecture_reconstruction_resampled.jsonl",
               "n_pool": len(ar), "indices": ar_idx, "sources": sources,
               "true_targets": {s: sorted(gt[s]) for s in sources}},
    }
    dump(manifest, os.path.join(OD, "manifest_linux_v2.json"))

    # ---- stats: how much of the sample touches the new include/* modules ----
    def touches(a, b):
        return is_new(a) or is_new(b)
    dps = [dp[i] for i in dp_idx]
    tgt_all = [t for s in sources for t in gt[s]]
    stats = {
        "modules_total": len(modules_list), "modules_original": len(old_mods),
        "modules_include_new": len(new_inc),
        "module_edges": len(dep_list),
        "module_edges_between_original_modules": sum(1 for a, b in dep_list if not touches(a, b)),
        "dp_pool": len(dp), "dp_pool_pos": len(pos), "ar_pool": len(ar),
        "dp_negative_sampling": "stratified by (src is include/*, tgt is include/*), matched 1:1 to positives",
        "dp_pool_strata_pos": {str(k): sum(1 for a, b in pos if (is_new(a), is_new(b)) == k)
                               for k in [(False, False), (False, True), (True, False), (True, True)]},
        "dp_pool_strata_neg": {str(k): sum(1 for a, b in neg_order if (is_new(a), is_new(b)) == k)
                               for k in [(False, False), (False, True), (True, False), (True, True)]},
        "dp_sample": len(dps),
        "dp_sample_pos": sum(t["label"] for t in dps),
        "dp_sample_pos_touching_include": sum(1 for t in dps if t["label"] and touches(t["module_A"], t["module_B"])),
        "dp_sample_neg_touching_include": sum(1 for t in dps if not t["label"] and touches(t["module_A"], t["module_B"])),
        "ar_sources": len(sources),
        "ar_sources_that_are_include": sum(1 for s in sources if is_new(s)),
        "ar_true_targets_total": len(tgt_all),
        "ar_true_targets_include": sum(1 for t in tgt_all if is_new(t)),
        "ar_avg_targets_per_source": round(len(tgt_all) / len(sources), 2) if sources else None,
    }
    dump(stats, os.path.join(OD, "resample_stats.json"))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
