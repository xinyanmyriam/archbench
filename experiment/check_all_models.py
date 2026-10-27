import os, json
from collections import defaultdict

results_dir = "results"
projects = ["openfoam","nastran","salome","paraview","kubernetes","linux",
            "hadoop","spring","codeaster","openmodelica","tensorflow","pytorch"]
tasks = ["ci","ci_hard","dp","ar"]

model_counts = defaultdict(int)
model_tasks = defaultdict(set)

for f in os.listdir(results_dir):
    if not f.endswith(".json") or f.startswith("baselines") or f.startswith("quick") or f.startswith("log"):
        continue
    for t in ["ci_hard", "ci", "dp", "ar"]:
        suffix = f"_{t}.json"
        if f.endswith(suffix):
            model = f[:-len(suffix)]
            for p in projects:
                if model.endswith("_" + p):
                    model_name = model[:-(len(p)+1)]
                    model_counts[model_name] += 1
                    model_tasks[model_name].add((p, t))
                    break
            break

for model in sorted(model_counts.keys()):
    count = model_counts[model]
    missing = []
    for p in projects:
        for t in tasks:
            if (p, t) not in model_tasks[model]:
                missing.append(f"{p}/{t}")
    status = "COMPLETE" if count == 48 else f"{count}/48"
    print(f"{model:25s} {status}")
    if missing and len(missing) <= 10:
        print(f"  missing: {', '.join(missing)}")
    elif missing:
        print(f"  missing: {len(missing)} tasks ({missing[0]} ... {missing[-1]})")
