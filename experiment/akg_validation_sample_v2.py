#!/usr/bin/env python3
"""
AKG v2 sampling validation - R2-5 / R3-2 / R2-10

Draw a new stratified sample of module-level dependencies from the REBUILT
(akg_v2) dependency graphs, replacing the 2026-10-09 sample that was drawn
from the superseded v1 graph.

For every sampled edge, attach up to 3 file-level links that produce it,
taken from akg_v2/<system>/file_dependency.json, so the annotator can check
both endpoints on disk.

Output:
  experiment/akg_validation_sample_v2.csv
  experiment/akg_validation_sample_v2_summary.json
"""

import csv
import json
import os
import random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AKG2 = os.path.join(ROOT, "systems")

SYSTEMS = ["nastran", "openfoam", "salome", "paraview", "kubernetes", "linux",
           "hadoop", "spring", "codeaster", "openmodelica", "tensorflow", "pytorch"]
TARGET_SAMPLE_SIZE = 200
RANDOM_SEED = 42
OUT_CSV = os.path.join(HERE, "akg_validation_sample_v2.csv")
OUT_JSON = os.path.join(HERE, "akg_validation_sample_v2_summary.json")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dataset_dir(system):
    base = os.path.join(ROOT, system)
    if system == "salome":
        base = os.path.join(base, "unified")
    return base


def file_to_module(system):
    """file path -> module, matching the index akg_v2 used for this system."""
    return {f["path"]: f["module"]
            for f in load(os.path.join(dataset_dir(system), "file_index.json"))["files"]}


def linux_extra_mapping(f2m, system):
    """Add the linux include-directory modules that akg_v2 introduced."""
    modules = set(load(os.path.join(ROOT, system, "modules_index.json"))["modules"])
    extra = {}
    for p in f2m:
        parts = p.split("/")
        if parts[0] == "include":
            extra[p] = "include/" + parts[1] if len(parts) >= 3 else "include"
        elif "include" in parts[:-1]:
            m = "/".join(parts[:2])
            if m in modules:
                extra[p] = m
    return extra


def k8s_norm(m):
    """Module normalization used for kubernetes in the evaluation scripts."""
    parts = m.split("/")
    if len(parts) >= 4 and parts[:3] == ["staging", "src", "k8s.io"]:
        return "/".join(parts[:4])
    return "/".join(parts[:2])


def module_of_path(system, path, f2m, modules):
    """Resolve a file / import path to the module named in akg_v2's graph."""
    if system == "kubernetes":
        return k8s_norm(path)
    if path in f2m:
        return f2m[path]
    if system != "linux":
        return None
    parts = path.split("/")
    if parts[0] == "include":
        return "include/" + parts[1] if len(parts) >= 3 else "include"
    if "include" in parts[:-1]:
        m = "/".join(parts[:2])
        if m in modules:
            return m
    return None


def main():
    random.seed(RANDOM_SEED)

    edges = {}
    links_by_pair = {}
    counts = {}

    for system in SYSTEMS:
        md = load(os.path.join(AKG2, system, "module_dependency.json"))
        deps = md["dependencies"]
        edges[system] = deps
        counts[system] = len(deps)

        f2m = file_to_module(system)
        modules = set(load(os.path.join(dataset_dir(system), "modules_index.json"))["modules"])

        if system == "kubernetes":
            seen, norm = set(), []
            for d in deps:
                a, b = k8s_norm(d["source_module"]), k8s_norm(d["target_module"])
                if a == b or (a, b) in seen:
                    continue
                seen.add((a, b))
                n = dict(d)
                n["source_module"], n["target_module"] = a, b
                norm.append(n)
            deps = edges[system] = norm
            counts[system] = len(deps)

        links = load(os.path.join(AKG2, system, "file_dependency.json"))["dependencies"]
        cross = os.path.join(AKG2, system, "cross_file_dependency.json")
        if os.path.exists(cross):
            links = links + load(cross)["dependencies"]

        pair_links = defaultdict(list)
        for link in links:
            a = module_of_path(system, link["source"], f2m, modules)
            b = module_of_path(system, link["target"], f2m, modules)
            if not a or not b or a == b:
                continue
            key = (a, b)
            if len(pair_links[key]) < 3:
                pair_links[key].append(link)
        links_by_pair[system] = pair_links

        covered = sum(1 for d in deps
                      if (d["source_module"], d["target_module"]) in pair_links)
        print(f"{system:14s} edges={len(deps):5d}  with_file_links={covered:5d}")

    total = sum(counts.values())
    alloc = {}
    for s in SYSTEMS:
        alloc[s] = max(1, round(counts[s] / total * TARGET_SAMPLE_SIZE))
    diff = TARGET_SAMPLE_SIZE - sum(alloc.values())
    if diff:
        largest = max(SYSTEMS, key=lambda s: counts.get(s, 0))
        alloc[largest] += diff

    rows = []
    sid = 0
    method_counts = Counter()
    for system in SYSTEMS:
        deps = edges[system]
        n = min(alloc[system], len(deps))
        for dep in random.sample(deps, n):
            sid += 1
            a, b = dep["source_module"], dep["target_module"]
            ex = links_by_pair[system].get((a, b), [])
            if ex:
                status = "file_link"
            else:
                status = "no_file_link"
            method_counts[status] += 1
            rows.append([
                sid, system, a, b, dep.get("weight", ""),
                " | ".join(f'{e["source"]} -> {e["target"]} ({e.get("type", "")})'
                           for e in ex),
                status,
                "",  # label - to be filled by annotators
                "",  # comment
            ])

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "system", "source_module", "target_module",
                    "weight", "file_level_examples", "evidence_status",
                    "label", "comment"])
        w.writerows(rows)

    summary = {
        "manifest": "akg_v2",
        "seed": RANDOM_SEED,
        "target_sample_size": TARGET_SAMPLE_SIZE,
        "actual_sample_size": len(rows),
        "per_system": {s: {"edges": counts[s], "sampled": alloc[s]} for s in SYSTEMS},
        "evidence": dict(method_counts),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nsampled {len(rows)} edges -> {OUT_CSV}")
    print(f"evidence status: {dict(method_counts)}")


if __name__ == "__main__":
    main()
