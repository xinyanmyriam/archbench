"""检查所有模型的完成情况和结果正确性"""
import json, os
from collections import defaultdict

results_dir = "results"
models = defaultdict(dict)

for f in sorted(os.listdir(results_dir)):
    if not f.endswith(".json") or f.startswith("baselines") or f.startswith("quick") or f.startswith("log"):
        continue

    # 解析文件名: model_project_task.json
    name = f.replace(".json", "")

    # 处理 ci_hard 和 _l1 后缀
    if "_ci_hard_l1" in name:
        task = "ci_hard_l1"
        rest = name.replace("_ci_hard_l1", "")
    elif "_ci_hard" in name:
        task = "ci_hard"
        rest = name.replace("_ci_hard", "")
    elif "_ci_l1" in name:
        task = "ci_l1"
        rest = name.replace("_ci_l1", "")
    elif "_ci" in name:
        task = "ci"
        rest = name.replace("_ci", "")
    elif "_dp_l1" in name:
        task = "dp_l1"
        rest = name.replace("_dp_l1", "")
    elif "_dp" in name:
        task = "dp"
        rest = name.replace("_dp", "")
    elif "_ar_l1" in name:
        task = "ar_l1"
        rest = name.replace("_ar_l1", "")
    elif "_ar" in name:
        task = "ar"
        rest = name.replace("_ar", "")
    else:
        continue

    # rest = model_project, 需要找到分割点
    # 已知项目名列表
    projects = ["openfoam", "nastran", "salome", "paraview", "kubernetes",
                "linux", "hadoop", "spring", "codeaster", "openmodelica",
                "tensorflow", "pytorch"]

    model = None
    project = None
    for p in projects:
        if rest.endswith("_" + p):
            model = rest[:-(len(p)+1)]
            project = p
            break

    if not model or not project:
        continue

    with open(os.path.join(results_dir, f)) as fh:
        data = json.load(fh)

    models[model][(project, task)] = data

# 汇总
expected_tasks = ["ci", "ci_hard", "dp", "ar"]
expected_projects = ["openfoam", "nastran", "salome", "paraview", "kubernetes",
                     "linux", "hadoop", "spring", "codeaster", "openmodelica",
                     "tensorflow", "pytorch"]

print("=" * 80)
print("COMPLETENESS CHECK")
print("=" * 80)

for model in sorted(models.keys()):
    results = models[model]
    done = len(results)
    expected = len(expected_tasks) * len(expected_projects)
    print(f"\n{model}: {done}/{expected} tasks")

    missing = []
    for p in expected_projects:
        for t in expected_tasks:
            if (p, t) not in results:
                missing.append(f"{p}/{t}")

    if missing:
        print(f"  MISSING: {', '.join(missing)}")
    else:
        print(f"  ✅ ALL COMPLETE")

    # 抽查结果合理性
    issues = []
    for (p, t), data in results.items():
        if t in ("ci", "ci_hard"):
            acc = data.get("accuracy", -1)
            total = data.get("total", 0)
            if total == 0:
                issues.append(f"{p}/{t}: total=0")
            elif acc < 0:
                issues.append(f"{p}/{t}: no accuracy field")
        elif t == "dp":
            acc = data.get("accuracy", -1)
            total = data.get("total", 0)
            if total == 0:
                issues.append(f"{p}/{t}: total=0")
        elif t == "ar":
            p5 = data.get("precision_at_5", -1)
            total = data.get("total", 0)
            if total == 0:
                issues.append(f"{p}/{t}: total=0")

    if issues:
        print(f"  ⚠️  ISSUES: {', '.join(issues)}")
    else:
        print(f"  ✅ All results valid")

# 打印完整结果表
print("\n" + "=" * 80)
print("FULL RESULTS TABLE")
print("=" * 80)

for model in sorted(models.keys()):
    results = models[model]
    print(f"\n--- {model} ---")
    print(f"{'Project':16s} {'CI':>7s} {'CI-Hard':>8s} {'DP':>7s} {'AR(P@5)':>8s}")
    print("-" * 50)

    for p in expected_projects:
        ci = results.get((p, "ci"), {}).get("accuracy", None)
        ch = results.get((p, "ci_hard"), {}).get("accuracy", None)
        dp = results.get((p, "dp"), {}).get("accuracy", None)
        ar = results.get((p, "ar"), {}).get("precision_at_5", None)

        ci_s = f"{ci:.3f}" if ci is not None else "--"
        ch_s = f"{ch:.3f}" if ch is not None else "--"
        dp_s = f"{dp:.3f}" if dp is not None else "--"
        ar_s = f"{ar:.3f}" if ar is not None else "--"
        print(f"{p:16s} {ci_s:>7s} {ch_s:>8s} {dp_s:>7s} {ar_s:>8s}")
