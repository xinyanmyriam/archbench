#!/usr/bin/env python3
"""
AKG (Architecture Knowledge Graph) Sampling Validation - R2-5, R3-2

Performs stratified random sampling of ~200 module-level dependencies across
12 systems' module_dependency.json files. For each sampled dependency, tries to
find 2-3 example file-level dependencies from dependency_raw.json.

Exports CSV: akg_validation_sample.csv for manual labeling.

Reviewer request: "conduct manual validation or comparison with documented
architecture information" and "Provide a quantification or manual sample check
of the automated Architecture Knowledge Graph extraction pipeline to strengthen
ground-truth reliability."
"""

import json
import csv
import os
import random
from collections import defaultdict

# ============================================================================
# Configuration
# ============================================================================

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_EXP_DIR = os.path.join(EXPERIMENT_DIR, "remote_experiment")

SYSTEMS = [
    "nastran", "openfoam", "salome", "paraview", "kubernetes", "linux",
    "hadoop", "spring", "codeaster", "openmodelica", "tensorflow", "pytorch"
]

TARGET_SAMPLE_SIZE = 200
OUTPUT_CSV = os.path.join(EXPERIMENT_DIR, "akg_validation_sample.csv")
RANDOM_SEED = 42
TARGET_AWARE_METHODS = {
    "direct_file_lookup", "python_dotted_path", "python_dotted_exact",
    "java_class_suffix", "path_suffix_match",
}


# ============================================================================
# Helper Functions
# ============================================================================

def load_json(filepath):
    """Safely load a JSON file, returning None on failure."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  [WARN] Could not load {filepath}: {e}")
        return None


def load_file_index(system_dir):
    """Load file_index.json and build file_path -> module mapping."""
    filepath = os.path.join(system_dir, "file_index.json")
    data = load_json(filepath)
    if data is None:
        return {}, {}

    file_to_mod = {}
    mod_to_files = defaultdict(set)

    for f in data.get("files", []):
        path = f.get("path", "")
        mod = f.get("module", "")
        if path and mod:
            file_to_mod[path] = mod
            mod_to_files[mod].add(path)

    return file_to_mod, dict(mod_to_files)


def load_raw_dependencies(system_dir):
    """Load dependency_raw.json, returning the dependencies list."""
    filepath = os.path.join(system_dir, "dependency_raw.json")
    data = load_json(filepath)
    if data is None:
        return None
    return data.get("dependencies", [])


def find_file_level_examples(src_mod, tgt_mod, raw_deps, file_to_mod,
                             mod_to_files, max_examples=3):
    """
    Find 2-3 example file-level dependencies that contributed to the
    module-level edge (src_mod -> tgt_mod).

    Uses only target-aware matching strategies:
    1. Direct file_to_mod lookup for both source and target
    2. Dotted path to slash path conversion (Python imports)
    3. Java class name to file suffix matching

    If the target cannot be verified, returns no examples rather than unrelated
    source-only evidence.
    """
    if not raw_deps:
        return [], "no_target_verified_evidence"

    tgt_files = mod_to_files.get(tgt_mod, set())
    src_files = mod_to_files.get(src_mod, set())

    # Precompute target file suffixes for Java matching
    # e.g., for file spring-beans/src/.../BeanFactory.java -> suffix = src/.../BeanFactory.java
    tgt_suffix_set = set()
    for tf in tgt_files:
        if tf.startswith(tgt_mod + "/"):
            suffix = tf[len(tgt_mod) + 1:]
            tgt_suffix_set.add(suffix)

    # Also build a set of dotted target paths for Python matching
    # e.g., tensorflow/python/module -> tensorflow.python.module
    tgt_dotted = tgt_mod.replace("/", ".")
    tgt_dotted_prefix = tgt_dotted + "."

    examples = []
    search_method = "direct_match"

    # First pass: try exact matching (both source and target resolvable)
    for dep in raw_deps:
        s = dep.get("source", "")
        t = dep.get("target", "")

        # Source must be in src_mod
        s_mod = file_to_mod.get(s)
        if s_mod != src_mod:
            continue

        # Try to resolve target to tgt_mod
        matched = False

        # Strategy 1: Direct file_to_mod lookup
        t_mod = file_to_mod.get(t)
        if t_mod == tgt_mod:
            matched = True
            search_method = "direct_file_lookup"

        # Strategy 2: Dotted path prefix match (Python imports)
        if not matched:
            t_slash = t.replace(".", "/")
            if t_slash == tgt_mod or t_slash.startswith(tgt_mod + "/"):
                matched = True
                search_method = "python_dotted_path"

        # Strategy 3: Dotted path exact match (target IS the module's dotted form)
        if not matched:
            if t == tgt_dotted or t.startswith(tgt_dotted_prefix):
                matched = True
                search_method = "python_dotted_exact"

        # Strategy 4: Java class name suffix match
        if not matched:
            t_path = t.replace(".", "/")
            # Check if any target file ends with this path + .java
            for suffix in tgt_suffix_set:
                if suffix.endswith(t_path + ".java") or suffix == t_path + ".java":
                    matched = True
                    search_method = "java_class_suffix"
                    break

        # Strategy 5: Check if target path is a prefix of any target file
        if not matched:
            for tf in tgt_files:
                if tf.endswith("/" + t.replace(".", "/")):
                    matched = True
                    search_method = "path_suffix_match"
                    break

        if matched:
            examples.append(dep)
            if len(examples) >= max_examples:
                break

    if not examples:
        return [], "no_target_verified_evidence"
    return examples, search_method


def format_examples(examples):
    """Format example file-level deps as a readable string for CSV."""
    if not examples:
        return ""
    parts = []
    for ex in examples:
        s = ex.get("source", "")
        t = ex.get("target", "")
        dep_type = ex.get("type", "")
        parts.append(f"{s} -> {t} ({dep_type})")
    return " | ".join(parts)


def has_target_aware_evidence(row):
    """Return whether a CSV row contains evidence verified at both endpoints."""
    method = row[7].split("search_method=", 1)[-1]
    return bool(row[5]) and method in TARGET_AWARE_METHODS


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("AKG Sampling Validation - R2-5, R3-2")
    print("Stratified random sample of module-level dependencies for manual labeling")
    print("=" * 80)

    random.seed(RANDOM_SEED)

    # Step 1: Load all systems' module dependencies and count
    print("\nStep 1: Loading module dependencies for all 12 systems...")

    system_deps = {}  # system -> list of dependency dicts
    system_counts = {}  # system -> total dependency count
    total_deps_all = 0

    for system in SYSTEMS:
        system_dir = os.path.join(REMOTE_EXP_DIR, system)
        md_file = os.path.join(system_dir, "module_dependency.json")
        data = load_json(md_file)
        if data is None:
            print(f"  [SKIP] {system}: no module_dependency.json")
            system_deps[system] = []
            system_counts[system] = 0
            continue

        deps = data.get("dependencies", [])
        system_deps[system] = deps
        system_counts[system] = len(deps)
        total_deps_all += len(deps)
        sys_name = data.get("system", system)
        print(f"  {system} (sys={sys_name}): {len(deps)} dependencies")

    print(f"\n  Total dependencies across all systems: {total_deps_all}")

    # Step 2: Stratified sampling
    print(f"\nStep 2: Stratified sampling (target={TARGET_SAMPLE_SIZE})...")

    # Calculate proportional sample sizes
    sample_sizes = {}
    allocated = 0
    for system in SYSTEMS:
        count = system_counts.get(system, 0)
        if total_deps_all > 0:
            n = max(1, round(count / total_deps_all * TARGET_SAMPLE_SIZE))
        else:
            n = 0
        sample_sizes[system] = n
        allocated += n

    # Adjust for rounding to get as close to TARGET_SAMPLE_SIZE as possible
    # Remove allocation from the largest system if over, add to largest if under
    diff = TARGET_SAMPLE_SIZE - allocated
    if diff != 0:
        # Find system with most deps to adjust
        largest = max(SYSTEMS, key=lambda s: system_counts.get(s, 0))
        sample_sizes[largest] = max(0, sample_sizes[largest] + diff)

    print(f"\n  {'System':<16} {'Total Deps':>10} {'Sample Size':>12} {'%':>6}")
    print(f"  {'-'*16} {'-'*10} {'-'*12} {'-'*6}")

    total_sample = 0
    for system in SYSTEMS:
        total = system_counts.get(system, 0)
        sample = sample_sizes.get(system, 0)
        pct = f"{sample/sum(sample_sizes.values())*100:.1f}%" if sum(sample_sizes.values()) > 0 else "0%"
        print(f"  {system:<16} {total:>10} {sample:>12} {pct:>6}")
        total_sample += sample

    print(f"  {'-'*16} {'-'*10} {'-'*12} {'-'*6}")
    print(f"  {'TOTAL':<16} {total_deps_all:>10} {total_sample:>12}")

    # Step 3: Sample and find file-level examples
    print(f"\nStep 3: Sampling dependencies and finding file-level examples...")

    csv_rows = []
    sample_id = 0

    for system in SYSTEMS:
        deps = system_deps.get(system, [])
        n = sample_sizes.get(system, 0)
        if n == 0 or len(deps) == 0:
            continue

        system_dir = os.path.join(REMOTE_EXP_DIR, system)

        # Load file_index for this system
        file_to_mod, mod_to_files = load_file_index(system_dir)
        print(f"\n  {system}: file_index loaded ({len(file_to_mod)} files, {len(mod_to_files)} modules)")

        # Load raw dependencies for this system
        raw_deps = load_raw_dependencies(system_dir)
        if raw_deps is None:
            print(f"    [WARN] No dependency_raw.json - examples will be unavailable")
        else:
            print(f"    dependency_raw.json loaded ({len(raw_deps)} file-level deps)")

        # Sample
        actual_n = min(n, len(deps))
        sampled = random.sample(deps, actual_n)

        method_counts = defaultdict(int)

        for dep in sampled:
            sample_id += 1
            src_mod = dep.get("source_module", "")
            tgt_mod = dep.get("target_module", "")
            weight = dep.get("weight", "")

            # Find file-level examples
            examples, method = find_file_level_examples(
                src_mod, tgt_mod, raw_deps, file_to_mod, mod_to_files,
                max_examples=3
            )

            method_counts[method] += 1

            example_str = format_examples(examples)

            notes = f"search_method={method}"

            csv_rows.append([
                sample_id,
                system,
                src_mod,
                tgt_mod,
                weight,
                example_str,
                "",  # label - for human to fill
                notes,
            ])

        print(f"    Sampled {actual_n} dependencies")
        print(f"    Search methods: {dict(method_counts)}")

    # Step 4: Write CSV
    print(f"\nStep 4: Writing CSV to {OUTPUT_CSV}...")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "system",
            "source_module",
            "target_module",
            "weight",
            "example_files",
            "label",  # empty - for human to fill (1=correct, 0=incorrect)
            "notes",
        ])
        writer.writerows(csv_rows)

    print(f"  CSV written with {len(csv_rows)} rows")

    # Step 5: Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\n{'System':<16} {'Total Deps':>10} {'Sample':>7} {'With Examples':>14} {'No Examples':>12}")
    print(f"{'-'*16} {'-'*10} {'-'*7} {'-'*14} {'-'*12}")

    for system in SYSTEMS:
        total = system_counts.get(system, 0)
        sample = sample_sizes.get(system, 0)
        system_rows = [r for r in csv_rows if r[1] == system]
        with_ex = sum(1 for r in system_rows if has_target_aware_evidence(r))
        without_ex = len(system_rows) - with_ex
        print(f"{system:<16} {total:>10} {sample:>7} {with_ex:>14} {without_ex:>12}")

    total_with = sum(1 for r in csv_rows if has_target_aware_evidence(r))
    total_without = len(csv_rows) - total_with
    print(f"{'-'*16} {'-'*10} {'-'*7} {'-'*14} {'-'*12}")
    print(f"{'TOTAL':<16} {total_deps_all:>10} {len(csv_rows):>7} {total_with:>14} {total_without:>12}")

    # Weight distribution
    print(f"\nWeight distribution of sampled dependencies:")
    weights = [r[4] for r in csv_rows if isinstance(r[4], (int, float))]
    if weights:
        print(f"  Min weight:    {min(weights)}")
        print(f"  Max weight:    {max(weights)}")
        print(f"  Mean weight:   {sum(weights)/len(weights):.1f}")
        print(f"  Median weight: {sorted(weights)[len(weights)//2]}")

    print(f"\nSearch method distribution:")
    method_stats = defaultdict(int)
    for r in csv_rows:
        notes = r[7]
        if "search_method=" in notes:
            method = notes.split("search_method=", 1)[1]
            method_stats[method] += 1
    for method, count in sorted(method_stats.items(), key=lambda x: -x[1]):
        print(f"  {method}: {count}")
    print(f"  Target-aware evidence samples: {total_with}")
    print(f"  No target-verified evidence: {total_without}")

    print(f"\n{'='*80}")
    print(f"CSV file saved to: {OUTPUT_CSV}")
    print(f"Total samples: {len(csv_rows)}")
    print(f"Instructions for manual labeling:")
    print(f"  - Fill the 'label' column with 1 (correct dependency) or 0 (incorrect)")
    print(f"  - Use the 'example_files' column to verify each dependency")
    print(f"  - The 'notes' column shows the search method used")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
