#!/usr/bin/env python3
"""
Qualitative Error Analysis for AR (Architecture Reconstruction) - R2-8

Classifies each incorrect top-5 prediction item (not each query) into three categories:
1. Direction reversed: model predicted A->B, but actual dependency is B->A
2. Lexically or path related but structurally unrelated
3. Completely unrelated: no informative path/naming overlap, no reversed dependency

Reads AR results for claude-opus-5.5 and gpt-6-sol across 12 systems.
Exports one row per incorrect top-5 prediction item to ar_error_analysis.csv
"""

import json
import csv
import os
import re
from collections import defaultdict

# ============================================================================
# Configuration
# ============================================================================

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
# SAB_MANIFEST=v1（默认）：原始 AKG 依赖图与 results_rerun；v2：akg_v2 重建图与 results_v2
MANIFEST_VERSION = os.environ.get("SAB_MANIFEST", "v1")
if MANIFEST_VERSION == "v2":
    RESULTS_RERUN_DIR = os.path.join(EXPERIMENT_DIR, "results_v2")
    REMOTE_EXP_DIR = os.path.join(EXPERIMENT_DIR, "..", "systems")
    OUTPUT_CSV = os.path.join(EXPERIMENT_DIR, "ar_error_analysis_v2.csv")
else:
    RESULTS_RERUN_DIR = os.path.join(EXPERIMENT_DIR, "results_rerun")
    REMOTE_EXP_DIR = os.path.join(EXPERIMENT_DIR, "remote_experiment")
    OUTPUT_CSV = os.path.join(EXPERIMENT_DIR, "ar_error_analysis.csv")

MODELS = ["claude-opus-5.5", "gpt-6-sol"]
SYSTEMS = [
    "nastran", "openfoam", "salome", "paraview", "kubernetes", "linux",
    "hadoop", "spring", "codeaster", "openmodelica", "tensorflow", "pytorch"
]

GENERIC_PATH_TOKENS = {
    "src", "source", "applications", "application", "app", "apps", "pkg",
    "packages", "lib", "libs", "modules", "module", "drivers", "include",
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


def build_dependency_graph(system):
    """Build a set of (source, target) edges from module_dependency.json."""
    filepath = os.path.join(REMOTE_EXP_DIR, system, "module_dependency.json")
    data = load_json(filepath)
    if data is None:
        return set()

    edges = set()
    for dep in data.get("dependencies", []):
        src = dep.get("source_module", "")
        tgt = dep.get("target_module", "")
        if src and tgt:
            edges.add((src, tgt))
    return edges


def get_common_path_prefix(mod1, mod2):
    """Return a shared leading path only when it has an informative component."""
    parts1 = mod1.split("/")
    parts2 = mod2.split("/")

    shared_parts = []
    for p1, p2 in zip(parts1, parts2):
        if p1 != p2:
            break
        shared_parts.append(p1)

    informative = [p for p in shared_parts if p.lower() not in GENERIC_PATH_TOKENS]
    if informative:
        return len(shared_parts), "/".join(shared_parts)
    return 0, ""


def get_common_token_prefix(mod1, mod2):
    """Return a naming match when the first informative tokens are shared."""
    tokens1 = [
        token for token in re.split(r"[/\\-_.]+", mod1)
        if token and token.lower() not in GENERIC_PATH_TOKENS
    ]
    tokens2 = [
        token for token in re.split(r"[/\\-_.]+", mod2)
        if token and token.lower() not in GENERIC_PATH_TOKENS
    ]

    if tokens1 and tokens2 and tokens1[0].lower() == tokens2[0].lower():
        return 1, tokens1[0]
    return 0, ""


def is_lexically_or_path_related(source_module, predicted):
    """Check for informative shared path components or leading naming tokens."""
    path_count, path_shared = get_common_path_prefix(source_module, predicted)
    if path_count:
        return True, f"Shared informative path prefix: '{path_shared}' ({path_count} components)"

    token_count, token_shared = get_common_token_prefix(source_module, predicted)
    if token_count:
        return True, f"Shared first informative naming token: '{token_shared}'"

    return False, ""


def classify_error(source_module, predicted, dep_graph):
    """Classify one incorrect top-5 prediction item."""
    if (predicted, source_module) in dep_graph:
        return "direction_reversed", (
            f"Model predicted {source_module}->{predicted}, "
            f"but actual edge is {predicted}->{source_module}"
        )

    is_related, notes = is_lexically_or_path_related(source_module, predicted)
    if is_related:
        return "lexically_or_path_related", notes

    return "completely_unrelated", (
        "No informative path/naming overlap and no reversed dependency"
    )


def normalize_module_name(name):
    """Strip whitespace and normalize module name."""
    if not name:
        return ""
    return name.strip().strip("/").strip()


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_model_system(model, system, dep_graph):
    """Analyze incorrect top-5 prediction items for one model/system pair."""
    ar_file = os.path.join(RESULTS_RERUN_DIR, f"{model}_{system}_ar.json")
    data = load_json(ar_file)
    if data is None:
        return None

    results = data.get("results", [])
    error_counts = defaultdict(int)
    total_predictions = 0
    error_rows = []

    for entry in results:
        source_module = normalize_module_name(entry.get("source_module", ""))
        true_targets = entry.get("true_targets", [])
        true_set = set(normalize_module_name(t) for t in true_targets)

        for pred in entry.get("predicted", []):
            pred_norm = normalize_module_name(pred)
            if not pred_norm:
                continue
            total_predictions += 1
            if pred_norm in true_set:
                continue

            error_type, notes = classify_error(source_module, pred_norm, dep_graph)
            error_counts[error_type] += 1
            error_rows.append([
                model,
                system,
                source_module,
                pred_norm,
                "; ".join(true_targets),
                error_type,
                notes,
            ])

    return error_counts, total_predictions, error_rows, len(results)


def main():
    print("=" * 80)
    print("Qualitative Error Analysis for Architecture Reconstruction (AR)")
    print("Unit: each incorrect top-5 prediction item (not each query)")
    print("R2-8: Direction reversal vs. lexical/path relatedness vs. unrelated")
    print("=" * 80)

    csv_rows = []
    all_stats = {}
    per_system = defaultdict(lambda: defaultdict(int))
    grand_totals = {
        "total_predictions": 0,
        "total_errors": 0,
    }

    for model in MODELS:
        model_stats = defaultdict(int)
        model_total_preds = 0
        model_total_errors = 0

        for system in SYSTEMS:
            print(f"\nProcessing: {model} / {system} ...")
            dep_graph = build_dependency_graph(system)
            print(f"  Dependency graph: {len(dep_graph)} edges")

            analysis = analyze_model_system(model, system, dep_graph)
            if analysis is None:
                print(f"  [SKIP] No AR results found for {model}/{system}")
                continue

            system_counts, total_predictions, error_rows, result_count = analysis
            total_errors = len(error_rows)
            print(f"  AR results: {result_count} queries")
            print(f"  Incorrect top-5 prediction items: {dict(system_counts)}")

            model_total_preds += total_predictions
            model_total_errors += total_errors
            grand_totals["total_predictions"] += total_predictions
            grand_totals["total_errors"] += total_errors
            csv_rows.extend(error_rows)

            for error_type, count in system_counts.items():
                model_stats[error_type] += count
                per_system[system][error_type] += count
                per_system[system]["_total"] += count

        all_stats[model] = model_stats
        all_stats[model]["_total_predictions"] = model_total_preds
        all_stats[model]["_total_errors"] = model_total_errors

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "system", "source_module", "predicted",
            "true_targets", "error_type", "notes"
        ])
        writer.writerows(csv_rows)

    print(f"\n{'=' * 80}")
    print(f"CSV exported: {OUTPUT_CSV}")
    print(f"Total rows: {len(csv_rows)}")
    print(f"{'=' * 80}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Error Classification by Model")
    print("=" * 80)

    header = (
        f"{'Model':<20} "
        f"{'Top-5 Items':>12} "
        f"{'Wrong Items':>13} "
        f"{'Dir Reversed':>14} "
        f"{'Lex/Path Rel':>13} "
        f"{'Unrelated':>11} "
        f"{'Error Rate':>11}"
    )
    print(header)
    print("-" * len(header))

    for model in MODELS:
        stats = all_stats.get(model, {})
        total_preds = stats.get("_total_predictions", 0)
        total_errors = stats.get("_total_errors", 0)
        dir_rev = stats.get("direction_reversed", 0)
        related = stats.get("lexically_or_path_related", 0)
        unrel = stats.get("completely_unrelated", 0)

        error_rate = f"{total_errors/total_preds*100:.1f}%" if total_preds > 0 else "N/A"

        print(
            f"{model:<20} "
            f"{total_preds:>12} "
            f"{total_errors:>13} "
            f"{dir_rev:>14} "
            f"{related:>13} "
            f"{unrel:>11} "
            f"{error_rate:>11}"
        )

    # Print percentage breakdown
    print("\n" + "=" * 80)
    print("SUMMARY: Error Type Distribution (%)")
    print("=" * 80)

    header2 = (
        f"{'Model':<20} "
        f"{'Dir Reversed %':>16} "
        f"{'Lex/Path Rel %':>15} "
        f"{'Unrelated %':>13}"
    )
    print(header2)
    print("-" * len(header2))

    for model in MODELS:
        stats = all_stats.get(model, {})
        total_errors = stats.get("_total_errors", 0)
        dir_rev = stats.get("direction_reversed", 0)
        related = stats.get("lexically_or_path_related", 0)
        unrel = stats.get("completely_unrelated", 0)

        if total_errors > 0:
            dr_pct = f"{dir_rev/total_errors*100:.1f}%"
            sr_pct = f"{related/total_errors*100:.1f}%"
            ur_pct = f"{unrel/total_errors*100:.1f}%"
        else:
            dr_pct = sr_pct = ur_pct = "N/A"

        print(
            f"{model:<20} "
            f"{dr_pct:>16} "
            f"{sr_pct:>15} "
            f"{ur_pct:>13}"
        )

    # Combined stats across both models
    print("\n" + "=" * 80)
    print("COMBINED: Both Models Together")
    print("=" * 80)

    total_preds = grand_totals["total_predictions"]
    total_errors = grand_totals["total_errors"]
    combined_dir = sum(all_stats.get(m, {}).get("direction_reversed", 0) for m in MODELS)
    combined_related = sum(
        all_stats.get(m, {}).get("lexically_or_path_related", 0) for m in MODELS
    )
    combined_unr = sum(all_stats.get(m, {}).get("completely_unrelated", 0) for m in MODELS)

    print(f"Total top-5 prediction items: {total_preds}")
    print(f"Total incorrect top-5 prediction items: {total_errors}")
    print(f"  Direction reversed:          {combined_dir} ({combined_dir/total_errors*100:.1f}%)" if total_errors else "")
    print(f"  Lexically or path related:   {combined_related} ({combined_related/total_errors*100:.1f}%)" if total_errors else "")
    print(f"  Completely unrelated:        {combined_unr} ({combined_unr/total_errors*100:.1f}%)" if total_errors else "")

    print("\n" + "=" * 80)
    print("PER-SYSTEM: Incorrect Top-5 Prediction Item Distribution")
    print("=" * 80)

    header3 = (
        f"{'System':<16} "
        f"{'Errors':>7} "
        f"{'Dir Rev':>8} "
        f"{'%':>6} "
        f"{'Lex/Path':>8} "
        f"{'%':>6} "
        f"{'Unrel':>6} "
        f"{'%':>6}"
    )
    print(header3)
    print("-" * len(header3))

    for system in SYSTEMS:
        stats = per_system.get(system, {})
        total = stats.get("_total", 0)
        dr = stats.get("direction_reversed", 0)
        sr = stats.get("lexically_or_path_related", 0)
        ur = stats.get("completely_unrelated", 0)
        if total > 0:
            dr_p = f"{dr/total*100:.0f}%"
            sr_p = f"{sr/total*100:.0f}%"
            ur_p = f"{ur/total*100:.0f}%"
        else:
            dr_p = sr_p = ur_p = "-"
        print(
            f"{system:<16} "
            f"{total:>7} "
            f"{dr:>8} "
            f"{dr_p:>6} "
            f"{sr:>8} "
            f"{sr_p:>6} "
            f"{ur:>6} "
            f"{ur_p:>6}"
        )

    print("\n" + "=" * 80)
    print("Done. CSV file saved to:", OUTPUT_CSV)
    print("=" * 80)


if __name__ == "__main__":
    main()
