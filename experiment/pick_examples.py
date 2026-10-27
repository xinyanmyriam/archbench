"""
从每种任务中挑选有代表性的示例用于论文正文
选择标准：
- 来自不同领域的项目
- 有一定难度（不是太简单也不是太难）
- 能展示任务设计的巧妙之处
"""
import json, os, random
random.seed(42)

projects = {
    "kubernetes": ("../kubernetes", "Kubernetes", "container orchestration"),
    "openfoam": ("../openfoam", "OpenFOAM", "CFD simulation"),
    "spring": ("../spring", "Spring Framework", "enterprise Java framework"),
    "linux": ("../linux", "Linux Kernel", "operating system kernel"),
    "codeaster": ("../codeaster", "Code_Aster", "finite element analysis"),
    "tensorflow": ("../tensorflow", "TensorFlow", "machine learning framework"),
}

print("=" * 80)
print("TASK EXAMPLES FOR PAPER")
print("=" * 80)

# =====================================================
# CI Easy - 选一个路径明显的
# =====================================================
print("\n--- Task 1: Component Identification (CI Easy) ---")
print("Example from Kubernetes:\n")
with open("../kubernetes/component_identification.jsonl") as f:
    data = [json.loads(l) for l in f]
# 找一个路径明显包含模块名的
for item in data:
    if "kube-scheduler" in item["file"] and item["label"] == "cmd/kube-scheduler":
        print(f"  File: {item['file']}")
        print(f"  Candidates: {item['candidates']}")
        print(f"  Label: {item['label']}")
        break

# =====================================================
# CI Hard - 选一个文件名模糊的
# =====================================================
print("\n--- Task 2: Component Identification Hard (CI-Hard) ---")
print("Example from Code_Aster:\n")
with open("../codeaster/component_identification_hard.jsonl") as f:
    data = [json.loads(l) for l in f]
# 找一个有领域含义的文件名
for item in data:
    if "Crack" in item["file"] or "Thermal" in item["file"] or "Mesh" in item["file"]:
        print(f"  File name: {item['file']}")
        print(f"  Full path: {item.get('file_full', 'N/A')}")
        print(f"  Candidates: {item['candidates']}")
        print(f"  Label: {item['label']}")
        break

print("\nExample from Linux Kernel:\n")
with open("../linux/component_identification_hard.jsonl") as f:
    data = [json.loads(l) for l in f]
for item in data:
    if item["file"] == "core.c" or item["file"] == "main.c":
        print(f"  File name: {item['file']}")
        print(f"  Full path: {item.get('file_full', 'N/A')}")
        print(f"  Candidates: {item['candidates']}")
        print(f"  Label: {item['label']}")
        break

# =====================================================
# DP - 选一个正样本和一个负样本
# =====================================================
print("\n--- Task 3: Dependency Prediction (DP) ---")
print("Positive example from Spring Framework:\n")
with open("../spring/dependency_prediction.jsonl") as f:
    data = [json.loads(l) for l in f]
pos = [d for d in data if d["label"] == 1]
neg = [d for d in data if d["label"] == 0]
# 找一个语义上合理的正样本
for item in pos:
    if "spring-web" in item["module_A"] and "spring-core" in item["module_B"]:
        print(f"  Module A: {item['module_A']}")
        print(f"  Module B: {item['module_B']}")
        print(f"  Label: Yes (dependency exists)")
        break
else:
    item = pos[0]
    print(f"  Module A: {item['module_A']}")
    print(f"  Module B: {item['module_B']}")
    print(f"  Label: Yes (dependency exists)")

print("\nNegative example from Spring Framework:\n")
for item in neg:
    if "spring-jdbc" in item["module_A"] or "spring-websocket" in item["module_A"]:
        print(f"  Module A: {item['module_A']}")
        print(f"  Module B: {item['module_B']}")
        print(f"  Label: No (no dependency)")
        break
else:
    item = neg[0]
    print(f"  Module A: {item['module_A']}")
    print(f"  Module B: {item['module_B']}")
    print(f"  Label: No (no dependency)")

# =====================================================
# AR - 选一个有多个依赖的模块
# =====================================================
print("\n--- Task 4: Architecture Reconstruction (AR) ---")
print("Example from TensorFlow:\n")
with open("../tensorflow/architecture_reconstruction.jsonl") as f:
    data = [json.loads(l) for l in f]
# 收集同一个 source 的所有 target
from collections import defaultdict
gt = defaultdict(set)
for item in data:
    gt[item["source_module"]].add(item["target_module"])
# 找一个有 3-5 个依赖的
for src, targets in gt.items():
    if 3 <= len(targets) <= 6 and "python" in src:
        print(f"  Source module: {src}")
        print(f"  Known dependencies ({len(targets)}):")
        for t in sorted(targets):
            print(f"    -> {t}")
        print(f"  Task: Predict top-5 dependency targets from candidate list")
        break

print("\nExample from Kubernetes:\n")
with open("../kubernetes/architecture_reconstruction.jsonl") as f:
    data = [json.loads(l) for l in f]
gt2 = defaultdict(set)
for item in data:
    gt2[item["source_module"]].add(item["target_module"])
for src, targets in gt2.items():
    if 3 <= len(targets) <= 5 and "cmd/" in src:
        print(f"  Source module: {src}")
        print(f"  Known dependencies ({len(targets)}):")
        for t in sorted(targets):
            print(f"    -> {t}")
        print(f"  Task: Predict top-5 dependency targets from candidate list")
        break
