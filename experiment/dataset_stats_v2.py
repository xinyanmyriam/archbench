"""Recompute the SoftArchBench dataset statistics reported in the paper.

Every value printed here is derived from files on disk so it can be traced back
to a concrete artifact:

  * module index      -> akg_v2/<system>/modules_index.json, falling back to
                         dataset_v1/<system>/modules_index.json
  * source files      -> dataset_v1/<system>/file_index.json
  * task instances    -> dataset_v1/<system>/*.jsonl (line counts)
  * module-level deps -> akg_v2/<system>/module_dependency.json ("total")
  * file-level deps   -> akg_v2/<system>/file_dependency.json    ("total")
  * graph endpoints   -> union of source_module/target_module in akg_v2

salome stores its raw data under unified/, so the loader accepts both the flat
and the nested layouts.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "systems")
AKG = os.path.join(ROOT, "systems")

# CI-Hard reuses the CI instances with the directory path masked, so it is a
# variant of CI rather than an additional set of benchmark instances.
TASKS = [
    ("ci", "component_identification.jsonl"),
    ("ci_hard", "component_identification_hard.jsonl"),
    ("dp", "dependency_prediction.jsonl"),
    ("ar", "architecture_reconstruction.jsonl"),
]


def _find(system_dir, filename):
    for sub in ("", "unified", "reposanalysis"):
        path = os.path.join(system_dir, sub, filename)
        if os.path.isfile(path):
            return path
    return None


def _load_json(path):
    if path is None or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _count_lines(path):
    if path is None:
        return 0
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return sum(1 for _ in fh)


def systems():
    return sorted(
        s for s in os.listdir(DATASET)
        if os.path.isdir(os.path.join(DATASET, s))
        and _find(os.path.join(DATASET, s), "modules_index.json")
    )


def module_index(system):
    """Set of module names: prefer the rebuilt (v2) index, fall back to v1."""
    for path in (os.path.join(AKG, system, "modules_index.json"),
                 _find(os.path.join(DATASET, system), "modules_index.json")):
        entry = _load_json(path)
        if entry is not None:
            return set(entry["modules"] if isinstance(entry, dict) else entry)
    return set()


def file_count(entry):
    if entry is None:
        return 0
    if isinstance(entry, list):
        return len(entry)
    for key in ("files", "file_index"):
        if key in entry:
            return len(entry[key])
    return len(entry)


def main():
    print("%-14s %6s %9s %7s %10s %8s %8s %7s %7s" % (
        "system", "mods", "endpoints", "edges", "both_in_idx", "ci", "ci_hard", "dp", "ar"))
    print("-" * 88)
    total = dict.fromkeys(
        ("mods", "endpoints", "edges", "both_in_idx", "ci", "ci_hard", "dp", "ar", "files"), 0)

    for system in systems():
        sdir = os.path.join(DATASET, system)
        index = module_index(system)
        deps_entry = _load_json(os.path.join(AKG, system, "module_dependency.json"))
        deps = deps_entry["dependencies"] if deps_entry else []
        endpoints = set()
        both = 0
        for edge in deps:
            src, tgt = edge["source_module"], edge["target_module"]
            endpoints.update((src, tgt))
            if src in index and tgt in index:
                both += 1
        counts = {task: _count_lines(_find(sdir, name)) for task, name in TASKS}
        files = file_count(_load_json(_find(sdir, "file_index.json")))

        row = {
            "mods": len(index), "endpoints": len(endpoints), "edges": len(deps),
            "both_in_idx": both, "ci": counts["ci"], "ci_hard": counts["ci_hard"],
            "dp": counts["dp"], "ar": counts["ar"], "files": files,
        }
        for key, value in row.items():
            total[key] += value
        print("%-14s %6d %9d %7d %10d %8d %8d %7d %7d" % (
            system, row["mods"], row["endpoints"], row["edges"],
            row["both_in_idx"], row["ci"], row["ci_hard"], row["dp"], row["ar"]))

    print("-" * 88)
    print("%-14s %6d %9d %7d %10d %8d %8d %7d %7d" % (
        "TOTAL", total["mods"], total["endpoints"], total["edges"],
        total["both_in_idx"], total["ci"], total["ci_hard"], total["dp"], total["ar"]))

    mod_deps = file_deps = 0
    for system in systems():
        md = _load_json(os.path.join(AKG, system, "module_dependency.json"))
        fd = _load_json(os.path.join(AKG, system, "file_dependency.json"))
        if md:
            mod_deps += md.get("total", len(md["dependencies"]))
        if fd:
            file_deps += fd.get("total", len(fd["dependencies"]))

    unique_instances = total["ci"] + total["dp"] + total["ar"]
    all_files_instances = unique_instances + total["ci_hard"]

    print()
    print("--- values reported in the paper ---")
    print("modules (rebuilt index)          : %d" % total["mods"])
    print("modules (graph endpoints)        : %d" % total["endpoints"])
    print("source files                     : %d" % total["files"])
    print("module-level dependencies        : %d" % mod_deps)
    print("file-level dependencies          : %d" % file_deps)
    print("benchmark instances (CI+DP+AR)   : %d   <- paper value" % unique_instances)
    print("  CI                             : %d" % total["ci"])
    print("  DP                             : %d" % total["dp"])
    print("  AR                             : %d" % total["ar"])
    print("instances if CI-Hard counted     : %d" % all_files_instances)
    print("edges between indexed modules    : %d" % total["both_in_idx"])


if __name__ == "__main__":
    main()
