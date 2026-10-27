import json, os

results = {}
for f in sorted(os.listdir("results")):
    if not f.startswith("deepseek-v4-flash_") or not f.endswith(".json"):
        continue
    with open(os.path.join("results", f)) as fh:
        data = json.load(fh)

    name = f.replace(".json", "").replace("deepseek-v4-flash_", "")
    if "_ci_hard" in name:
        proj, task = name.replace("_ci_hard", ""), "ci_hard"
    elif "_ci" in name:
        proj, task = name.replace("_ci", ""), "ci"
    elif "_dp" in name:
        proj, task = name.replace("_dp", ""), "dp"
    elif "_ar" in name:
        proj, task = name.replace("_ar", ""), "ar"
    else:
        continue

    if proj not in results:
        results[proj] = {}

    if task in ("ci", "ci_hard"):
        results[proj][task] = data["accuracy"]
    elif task == "dp":
        results[proj][task] = data["accuracy"]
    elif task == "ar":
        results[proj][task] = data["precision_at_5"]

header = f"{'Project':16s} {'CI':>7s} {'CI-Hard':>8s} {'DP':>7s} {'AR(P@5)':>8s}"
print(header)
print("-" * len(header))

order = ["openfoam", "nastran", "salome", "paraview", "kubernetes", "linux",
         "hadoop", "spring", "codeaster", "openmodelica", "tensorflow", "pytorch"]

for proj in order:
    if proj not in results:
        continue
    r = results[proj]
    ci = f"{r['ci']:.3f}" if "ci" in r else "--"
    ch = f"{r['ci_hard']:.3f}" if "ci_hard" in r else "--"
    dp = f"{r['dp']:.3f}" if "dp" in r else "--"
    ar = f"{r['ar']:.3f}" if "ar" in r else "--"
    print(f"{proj:16s} {ci:>7s} {ch:>8s} {dp:>7s} {ar:>8s}")

done = sum(len(v) for v in results.values())
print(f"\nCompleted: {done}/48 tasks")
