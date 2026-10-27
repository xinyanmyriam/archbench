"""
生成论文所需的完整结果表格和图表数据
"""
import json, os
from collections import defaultdict

results_dir = "results"
projects_order = ["openfoam","nastran","salome","paraview","kubernetes","linux",
                  "hadoop","spring","codeaster","openmodelica","tensorflow","pytorch"]
models_order = ["gpt-5","claude-sonnet-4","deepseek-v4-flash","qwen3","llama-4-maverick",
                "gpt-oss-120b","kimi-k2.5","qwen3-8b","llama-3.1-8b"]
tasks = ["ci","ci_hard","dp","ar"]

# 加载所有结果
all_results = {}
for f in os.listdir(results_dir):
    if not f.endswith(".json") or f.startswith("baselines") or f.startswith("quick") or f.startswith("log"):
        continue
    for t in ["ci_hard", "ci", "dp", "ar"]:
        suffix = f"_{t}.json"
        if f.endswith(suffix):
            rest = f[:-len(suffix)]
            for p in projects_order:
                if rest.endswith("_" + p):
                    model = rest[:-(len(p)+1)]
                    with open(os.path.join(results_dir, f)) as fh:
                        data = json.load(fh)
                    all_results[(model, p, t)] = data
                    break
            break

# 加载 baselines
baselines = json.load(open(os.path.join(results_dir, "baselines.json")))

# =====================================================
# Table: Main results (all models x all projects x CI-Hard)
# =====================================================
print("=" * 120)
print("TABLE: CI-Hard Accuracy (all models)")
print("=" * 120)
header = f"{'Project':16s}"
for m in models_order:
    header += f" {m:>12s}"
print(header)
print("-" * len(header))

for p in projects_order:
    row = f"{p:16s}"
    for m in models_order:
        key = (m, p, "ci_hard")
        if key in all_results:
            row += f" {all_results[key]['accuracy']:12.3f}"
        else:
            row += f" {'--':>12s}"
    print(row)

# Average
row = f"{'AVERAGE':16s}"
for m in models_order:
    vals = [all_results[(m,p,"ci_hard")]["accuracy"] for p in projects_order if (m,p,"ci_hard") in all_results]
    avg = sum(vals)/len(vals) if vals else 0
    row += f" {avg:12.3f}"
print(row)

# =====================================================
# Table: DP Accuracy
# =====================================================
print("\n" + "=" * 120)
print("TABLE: DP Accuracy (all models)")
print("=" * 120)
header = f"{'Project':16s}"
for m in models_order:
    header += f" {m:>12s}"
print(header)
print("-" * len(header))

for p in projects_order:
    row = f"{p:16s}"
    for m in models_order:
        key = (m, p, "dp")
        if key in all_results:
            row += f" {all_results[key]['accuracy']:12.3f}"
        else:
            row += f" {'--':>12s}"
    print(row)

row = f"{'AVERAGE':16s}"
for m in models_order:
    vals = [all_results[(m,p,"dp")]["accuracy"] for p in projects_order if (m,p,"dp") in all_results]
    avg = sum(vals)/len(vals) if vals else 0
    row += f" {avg:12.3f}"
print(row)

# =====================================================
# Table: AR Precision@5
# =====================================================
print("\n" + "=" * 120)
print("TABLE: AR Precision@5 (all models)")
print("=" * 120)
header = f"{'Project':16s}"
for m in models_order:
    header += f" {m:>12s}"
print(header)
print("-" * len(header))

for p in projects_order:
    row = f"{p:16s}"
    for m in models_order:
        key = (m, p, "ar")
        if key in all_results:
            row += f" {all_results[key]['precision_at_5']:12.3f}"
        else:
            row += f" {'--':>12s}"
    print(row)

row = f"{'AVERAGE':16s}"
for m in models_order:
    vals = [all_results[(m,p,"ar")]["precision_at_5"] for p in projects_order if (m,p,"ar") in all_results]
    avg = sum(vals)/len(vals) if vals else 0
    row += f" {avg:12.3f}"
print(row)

# =====================================================
# Domain-level summary
# =====================================================
print("\n" + "=" * 80)
print("TABLE: Domain-level Average (DeepSeek-V4-Flash)")
print("=" * 80)

domains = {
    "Ind. Simulation": ["openfoam","nastran","salome","paraview"],
    "Ind. System": ["kubernetes","linux","hadoop"],
    "Enterprise": ["spring"],
    "Ind. Software": ["codeaster","openmodelica"],
    "ML Infra": ["tensorflow","pytorch"],
}

m = "deepseek-v4-flash"
print(f"{'Domain':20s} {'CI':>7s} {'CI-H':>7s} {'DP':>7s} {'AR':>7s}")
print("-" * 50)
for domain, projs in domains.items():
    ci = sum(all_results[(m,p,"ci")]["accuracy"] for p in projs) / len(projs)
    ch = sum(all_results[(m,p,"ci_hard")]["accuracy"] for p in projs) / len(projs)
    dp = sum(all_results[(m,p,"dp")]["accuracy"] for p in projs) / len(projs)
    ar = sum(all_results[(m,p,"ar")]["precision_at_5"] for p in projs) / len(projs)
    print(f"{domain:20s} {ci:7.3f} {ch:7.3f} {dp:7.3f} {ar:7.3f}")

# =====================================================
# Scale comparison
# =====================================================
print("\n" + "=" * 80)
print("TABLE: Scale Comparison (CI-Hard Average)")
print("=" * 80)

scale_pairs = [
    ("Qwen3-235B", "qwen3", "Qwen3-8B", "qwen3-8b"),
    ("Llama-4-Maverick", "llama-4-maverick", "Llama-3.1-8B", "llama-3.1-8b"),
]

for large_name, large_key, small_name, small_key in scale_pairs:
    large_vals = [all_results[(large_key,p,"ci_hard")]["accuracy"] for p in projects_order if (large_key,p,"ci_hard") in all_results]
    small_vals = [all_results[(small_key,p,"ci_hard")]["accuracy"] for p in projects_order if (small_key,p,"ci_hard") in all_results]
    large_avg = sum(large_vals)/len(large_vals) if large_vals else 0
    small_avg = sum(small_vals)/len(small_vals) if small_vals else 0
    print(f"  {large_name:20s} avg CI-Hard = {large_avg:.3f}")
    print(f"  {small_name:20s} avg CI-Hard = {small_avg:.3f}")
    print(f"  Gap: {large_avg - small_avg:+.3f}")
    print()

# =====================================================
# Save JSON for plotting
# =====================================================
plot_data = {
    "models": models_order,
    "projects": projects_order,
    "ci_hard": {},
    "dp": {},
    "ar": {},
}

for m in models_order:
    plot_data["ci_hard"][m] = [all_results.get((m,p,"ci_hard"),{}).get("accuracy",None) for p in projects_order]
    plot_data["dp"][m] = [all_results.get((m,p,"dp"),{}).get("accuracy",None) for p in projects_order]
    plot_data["ar"][m] = [all_results.get((m,p,"ar"),{}).get("precision_at_5",None) for p in projects_order]

with open("results/plot_data.json", "w") as f:
    json.dump(plot_data, f, indent=2)

print("\nPlot data saved to results/plot_data.json")
