"""
审计所有实验结果的完整性和时间戳

检查：
1. 每个结果文件是否有效（能解析 JSON）
2. 文件修改时间（作为实验执行时间的近似）
3. 结果是否包含必要字段
4. 生成完整的审计报告
"""

import os
import json
import time
from datetime import datetime

print("=" * 80)
print(f"EXPERIMENT AUDIT REPORT")
print(f"Generated: {datetime.now().isoformat()}")
print("=" * 80)

# =====================================================
# 1. Level 0 API 实验结果
# =====================================================
print("\n[1] Level 0 API Results (experiment/results/)")
results_dir = "results"
api_files = [f for f in os.listdir(results_dir)
             if f.endswith(".json") and not f.startswith(("baselines", "quick", "log", "plot", "rq"))]

models_found = set()
valid = 0
invalid = 0
earliest = None
latest = None

for f in sorted(api_files):
    path = os.path.join(results_dir, f)
    mtime = os.path.getmtime(path)
    dt = datetime.fromtimestamp(mtime)

    if earliest is None or mtime < earliest:
        earliest = mtime
    if latest is None or mtime > latest:
        latest = mtime

    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if "accuracy" in data or "precision_at_5" in data:
            valid += 1
        else:
            invalid += 1
            print(f"  WARNING: {f} missing key metrics")
    except:
        invalid += 1
        print(f"  ERROR: {f} invalid JSON")

    # 提取模型名
    for t in ["_ci_hard", "_ci", "_dp", "_ar"]:
        if f.endswith(t + ".json"):
            model = f[:-(len(t) + 5)]  # remove _project_task.json
            break

print(f"  Total files: {len(api_files)}")
print(f"  Valid: {valid}, Invalid: {invalid}")
if earliest:
    print(f"  Time range: {datetime.fromtimestamp(earliest).isoformat()} to {datetime.fromtimestamp(latest).isoformat()}")

# =====================================================
# 2. Probing 结果
# =====================================================
print("\n[2] Probing Results (experiment/probing/results/)")
probing_dir = "probing/results"
if os.path.exists(probing_dir):
    probing_files = [f for f in os.listdir(probing_dir) if f.endswith(".json")]
    p_valid = 0
    p_earliest = p_latest = None

    for f in probing_files:
        path = os.path.join(probing_dir, f)
        mtime = os.path.getmtime(path)
        if p_earliest is None or mtime < p_earliest:
            p_earliest = mtime
        if p_latest is None or mtime > p_latest:
            p_latest = mtime
        try:
            with open(path, encoding="utf-8") as fh:
                json.load(fh)
            p_valid += 1
        except:
            print(f"  ERROR: {f}")

    print(f"  Total files: {len(probing_files)}")
    print(f"  Valid: {p_valid}")
    if p_earliest:
        print(f"  Time range: {datetime.fromtimestamp(p_earliest).isoformat()} to {datetime.fromtimestamp(p_latest).isoformat()}")

# =====================================================
# 3. RSA 结果
# =====================================================
print("\n[3] RSA Results (experiment/probing/rsa_results/)")
rsa_dir = "probing/rsa_results"
if os.path.exists(rsa_dir):
    rsa_files = os.listdir(rsa_dir)
    for f in sorted(rsa_files):
        path = os.path.join(rsa_dir, f)
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path) / 1024
        print(f"  {f:40s} {size:8.1f} KB  {datetime.fromtimestamp(mtime).isoformat()}")

# =====================================================
# 4. RQ1.3 结果
# =====================================================
print("\n[4] RQ1.3 Results (experiment/results/rq1_3/)")
rq13_dir = "results/rq1_3"
if os.path.exists(rq13_dir):
    rq13_files = os.listdir(rq13_dir)
    for f in sorted(rq13_files):
        path = os.path.join(rq13_dir, f)
        mtime = os.path.getmtime(path)
        print(f"  {f:50s} {datetime.fromtimestamp(mtime).isoformat()}")
    if not rq13_files:
        print("  (empty - experiment still running)")
else:
    print("  (directory not found - experiment not started)")

# =====================================================
# 5. 完整性检查
# =====================================================
print("\n[5] Completeness Check")

expected_models = ["gpt-5", "claude-sonnet-4", "deepseek-v4-flash", "qwen3",
                   "llama-4-maverick", "gpt-oss-120b", "kimi-k2.5", "qwen3-8b", "llama-3.1-8b"]
expected_projects = ["openfoam", "nastran", "salome", "paraview", "kubernetes",
                     "linux", "hadoop", "spring", "codeaster", "openmodelica",
                     "tensorflow", "pytorch"]
expected_tasks = ["ci", "ci_hard", "dp", "ar"]

missing = []
for m in expected_models:
    for p in expected_projects:
        for t in expected_tasks:
            fname = f"{m}_{p}_{t}.json"
            if fname not in api_files:
                missing.append(fname)

if missing:
    print(f"  Missing Level 0 results: {len(missing)}")
    for f in missing[:5]:
        print(f"    {f}")
    if len(missing) > 5:
        print(f"    ... and {len(missing)-5} more")
else:
    print(f"  Level 0: ALL COMPLETE ({len(expected_models)}×{len(expected_projects)}×{len(expected_tasks)} = {len(expected_models)*len(expected_projects)*len(expected_tasks)})")

print(f"\n  Probing: {len(probing_files) if os.path.exists(probing_dir) else 0} files")
print(f"  RSA: {len(rsa_files) if os.path.exists(rsa_dir) else 0} files")
print(f"  RQ1.3: {len(rq13_files) if os.path.exists(rq13_dir) else 0} files (in progress)")
