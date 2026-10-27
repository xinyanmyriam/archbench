"""
深度分析 baseline 结果，评估数据集质量和论文可行性
"""

import json

with open("results/baselines.json") as f:
    baselines = json.load(f)

print("=" * 80)
print("DATASET QUALITY & PUBLISHABILITY ANALYSIS")
print("=" * 80)

# =====================================================
# 1. CI 任务分析
# =====================================================
print("\n[1] Component Identification Analysis")
print("-" * 60)

ci_path_scores = {}
ci_freq_scores = {}
for proj, data in baselines.items():
    ci_path_scores[proj] = data["ci_path_heuristic"]["accuracy"]
    ci_freq_scores[proj] = data["ci_frequency"]["accuracy"]

print("\nPath Heuristic accuracy (should NOT be 1.0 for all):")
perfect = [p for p, s in ci_path_scores.items() if s >= 0.99]
imperfect = [p for p, s in ci_path_scores.items() if s < 0.99]
print(f"  Perfect (>=0.99): {len(perfect)}/12 -> {perfect}")
print(f"  Imperfect (<0.99): {len(imperfect)}/12 -> {imperfect}")

if len(perfect) >= 10:
    print("\n  ⚠️  WARNING: Path Heuristic achieves ~100% on most projects.")
    print("  This means CI task is TOO EASY for path-based methods.")
    print("  The file path directly reveals the module name.")
    print("  LLMs will likely also achieve ~100% on CI.")
    print("  RECOMMENDATION: Need harder CI variants (e.g., obfuscated paths,")
    print("  or cross-module file classification without path hints).")

# =====================================================
# 2. DP 任务分析
# =====================================================
print("\n\n[2] Dependency Prediction Analysis")
print("-" * 60)

print("\nGraph Neighbor F1 (upper bound for structure-based methods):")
for proj, data in baselines.items():
    gn_f1 = data["dp_graph_neighbor"]["f1"]
    rand_f1 = data["dp_random"]["f1"]
    gap = gn_f1 - rand_f1
    marker = "✓" if 0.2 < gap < 0.6 else "⚠️"
    print(f"  {proj:18s} GN_F1={gn_f1:.4f}  Random_F1={rand_f1:.4f}  Gap={gap:.4f} {marker}")

print("\n  ✓ = Good difficulty range (LLMs have room to differentiate)")
print("  ⚠️ = Too easy or too hard")

# =====================================================
# 3. AR 任务分析
# =====================================================
print("\n\n[3] Architecture Reconstruction Analysis")
print("-" * 60)

print("\nGraph Neighbor P@5 (oracle-like upper bound):")
for proj, data in baselines.items():
    gn = data["ar_graph_neighbor"]["precision_at_5"]
    rand = data["ar_random"]["precision_at_5"]
    print(f"  {proj:18s} GN_P@5={gn:.4f}  Random_P@5={rand:.4f}")

# =====================================================
# 4. 数据集规模问题
# =====================================================
print("\n\n[4] Dataset Scale Issues")
print("-" * 60)

small_projects = []
for proj, data in baselines.items():
    dp_total = data["dp_random"].get("total", 0)
    if dp_total and data["dp_random"]["accuracy"] == 0.5:
        # 检查是否样本太少
        pass

# 从 modules_index 检查
import os
for proj_name in baselines.keys():
    paths = {
        "nastran": "../nastran", "openmodelica": "../openmodelica",
        "openfoam": "../openfoam", "salome": "../salome/unified",
        "paraview": "../paraview", "kubernetes": "../kubernetes",
        "linux": "../linux", "hadoop": "../hadoop", "spring": "../spring",
        "codeaster": "../codeaster", "tensorflow": "../tensorflow",
        "pytorch": "../pytorch",
    }
    try:
        with open(os.path.join(paths[proj_name], "module_dependency.json")) as f:
            md = json.load(f)
        n_deps = md.get("total", len(md.get("dependencies", [])))
        if n_deps < 30:
            small_projects.append((proj_name, n_deps))
    except:
        pass

if small_projects:
    print("Projects with very few module dependencies:")
    for name, n in small_projects:
        print(f"  {name}: only {n} module dependencies")
    print("  RECOMMENDATION: These projects may need to be flagged in the paper")
    print("  or used only for CI task.")

# =====================================================
# 5. 总体评估
# =====================================================
print("\n\n" + "=" * 80)
print("OVERALL PUBLISHABILITY ASSESSMENT")
print("=" * 80)

print("""
STRENGTHS:
  + 12 industrial systems across 5 domains and 8 languages - unique coverage
  + 68K+ benchmark samples - sufficient scale
  + Three complementary tasks testing different architecture understanding aspects
  + Industrial simulation software (NASTRAN, Code_Aster, SALOME) rarely studied
  + Unified pipeline with consistent schema across all projects

CRITICAL ISSUES TO ADDRESS BEFORE SUBMISSION:

  1. CI Task is Too Easy (Path Heuristic ~100%)
     The file path directly encodes the module name. A simple string match
     achieves near-perfect accuracy. LLMs will also trivially solve this.
     
     FIX OPTIONS:
     a) Add a "hard" CI variant: strip directory prefixes, show only filename
     b) Add cross-project transfer: train on one system, test on another
     c) Add a "reasoning" variant: provide code snippet instead of file path

  2. NASTRAN & OpenModelica Too Small for DP/AR
     Only 4 and 25 module dependencies respectively.
     
     FIX: Report them separately or exclude from DP/AR analysis.

  3. Need LLM Experiments (the main contribution)
     Baselines alone are not enough. The core story is:
     "How well do LLMs understand industrial software architecture?"
     
     MINIMUM: Run GPT-4o + DeepSeek-V3 + one open-source model on all tasks.
     This gives you the empirical findings for the paper.

  4. Need Error Analysis
     After LLM experiments, analyze:
     - Which domains are hardest? (likely industrial simulation)
     - Which languages are hardest? (likely Fortran, mixed)
     - What types of dependencies do LLMs miss?
     - Do LLMs show bias toward popular projects (K8s, TF) vs niche (NASTRAN)?

TARGET VENUES (realistic assessment):
  - SANER/ICSME (CCF-B): HIGH chance if you have LLM experiments + analysis
  - ICSE/FSE dataset track (CCF-A): MEDIUM chance, needs stronger novelty
  - JSS/IST (SCI Q1): HIGH chance as empirical study
  - EMSE (SCI Q1): MEDIUM-HIGH, good fit for empirical software engineering
  - MSR (CCF-C but prestigious): HIGH chance for dataset + mining paper
""")
