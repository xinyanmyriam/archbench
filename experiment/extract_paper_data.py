"""提取论文所需的完整数据"""
import json, os

results_dir = "results"
baselines = json.load(open(os.path.join(results_dir, "baselines.json")))

projects = ["openfoam","nastran","salome","paraview","kubernetes","linux",
            "hadoop","spring","codeaster","openmodelica","tensorflow","pytorch"]

print("=== DeepSeek-V4-Flash Complete Results ===\n")
print(f"{'Project':16s} {'CI':>6s} {'CI-H':>6s} {'DP-A':>6s} {'DP-F1':>6s} {'AR':>6s} | {'BL-Rnd':>7s} {'BL-Frq':>7s} {'BL-Pth':>7s} {'BL-GN':>6s}")
print("-" * 95)

for p in projects:
    ci = json.load(open(os.path.join(results_dir, f"deepseek-v4-flash_{p}_ci.json")))
    ch = json.load(open(os.path.join(results_dir, f"deepseek-v4-flash_{p}_ci_hard.json")))
    dp = json.load(open(os.path.join(results_dir, f"deepseek-v4-flash_{p}_dp.json")))
    ar = json.load(open(os.path.join(results_dir, f"deepseek-v4-flash_{p}_ar.json")))
    bl = baselines.get(p, {})

    print(f"{p:16s} {ci['accuracy']:6.3f} {ch['accuracy']:6.3f} {dp['accuracy']:6.3f} {dp['f1']:6.3f} {ar['precision_at_5']:6.3f} | {bl.get('ci_random',{}).get('accuracy',0):7.3f} {bl.get('ci_frequency',{}).get('accuracy',0):7.3f} {bl.get('ci_path_heuristic',{}).get('accuracy',0):7.3f} {bl.get('dp_graph_neighbor',{}).get('f1',0):6.3f}")

# 按领域分组统计
print("\n=== Domain-level Analysis ===\n")
domains = {
    "Industrial Simulation": ["openfoam", "nastran", "salome", "paraview"],
    "Industrial System": ["kubernetes", "linux", "hadoop"],
    "Enterprise System": ["spring"],
    "Industrial Software": ["codeaster", "openmodelica"],
    "ML Infrastructure": ["tensorflow", "pytorch"],
}

for domain, projs in domains.items():
    ci_avg = ch_avg = dp_avg = ar_avg = 0
    n = len(projs)
    for p in projs:
        ci_avg += json.load(open(os.path.join(results_dir, f"deepseek-v4-flash_{p}_ci.json")))["accuracy"]
        ch_avg += json.load(open(os.path.join(results_dir, f"deepseek-v4-flash_{p}_ci_hard.json")))["accuracy"]
        dp_avg += json.load(open(os.path.join(results_dir, f"deepseek-v4-flash_{p}_dp.json")))["accuracy"]
        ar_avg += json.load(open(os.path.join(results_dir, f"deepseek-v4-flash_{p}_ar.json")))["precision_at_5"]
    print(f"{domain:25s} CI={ci_avg/n:.3f} CI-H={ch_avg/n:.3f} DP={dp_avg/n:.3f} AR={ar_avg/n:.3f}")
