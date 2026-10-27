"""
Multi-hop Reasoning Tasks for Journal Extension

新增任务:
1. Transitive Dependency: A→B→C, 给定 A 和 C, 问是否存在传递依赖
2. Change Impact Analysis: 修改模块 A, 预测哪些模块会受影响
3. Architectural Smell Detection: 识别循环依赖、god module 等
4. Layer Inference: 给定依赖方向, 推断模块的架构层级

用法:
    python generate_multihop_tasks.py --project all
    python generate_multihop_tasks.py --project openfoam --task transitive
"""

import json
import os
import sys
import random
from collections import defaultdict, deque

random.seed(42)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROJECTS


# =====================================================
# 工具函数
# =====================================================

def load_dependency_graph(project_path):
    """加载模块依赖图"""
    with open(os.path.join(project_path, "module_dependency.json"), encoding="utf-8") as f:
        dep_data = json.load(f)
    with open(os.path.join(project_path, "modules_index.json"), encoding="utf-8") as f:
        modules = json.load(f)["modules"]

    # 构建有向图
    graph = defaultdict(set)       # source -> set of targets
    reverse_graph = defaultdict(set)  # target -> set of sources
    weights = defaultdict(int)

    for dep in dep_data.get("dependencies", []):
        src = dep["source_module"]
        tgt = dep["target_module"]
        w = dep.get("weight", 1)
        graph[src].add(tgt)
        reverse_graph[tgt].add(src)
        weights[(src, tgt)] = w

    return modules, graph, reverse_graph, weights


def find_transitive_paths(graph, max_hops=3):
    """找到所有 2-hop 和 3-hop 的传递依赖路径"""
    paths = []

    for src in graph:
        # BFS 找 2-hop 路径
        for mid in graph[src]:
            for tgt in graph.get(mid, set()):
                if tgt != src and tgt not in graph[src]:
                    # src→mid→tgt 存在, 但 src→tgt 不直接存在
                    paths.append({
                        "source": src,
                        "target": tgt,
                        "path": [src, mid, tgt],
                        "hops": 2,
                        "has_transitive_dep": True,
                    })

    return paths


# =====================================================
# Task 1: Transitive Dependency Prediction
# =====================================================

def generate_transitive_dependency(project_path, output_path, sample_size=200):
    """
    生成传递依赖预测任务。

    正例: A→B→C 存在 (2-hop), 但 A→C 不直接存在
    负例: A 和 C 之间既无直接也无传递依赖
    """
    modules, graph, reverse_graph, weights = load_dependency_graph(project_path)

    # 找正例: 2-hop 传递路径
    positive_paths = find_transitive_paths(graph)

    if len(positive_paths) < 10:
        return 0

    # 采样正例
    if len(positive_paths) > sample_size // 2:
        positive_samples = random.sample(positive_paths, sample_size // 2)
    else:
        positive_samples = positive_paths

    # 生成负例: 随机模块对, 确保无直接和传递依赖
    all_reachable = {}
    for src in graph:
        reachable = set()
        queue = deque([src])
        visited = {src}
        while queue:
            node = queue.popleft()
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    reachable.add(neighbor)
                    queue.append(neighbor)
        all_reachable[src] = reachable

    negative_samples = []
    attempts = 0
    while len(negative_samples) < len(positive_samples) and attempts < 10000:
        src = random.choice(modules)
        tgt = random.choice(modules)
        if src != tgt and tgt not in all_reachable.get(src, set()):
            negative_samples.append({
                "source": src,
                "target": tgt,
                "path": None,
                "hops": -1,
                "has_transitive_dep": False,
            })
        attempts += 1

    # 合并并打乱
    data = []
    for item in positive_samples:
        data.append({
            "source_module": item["source"],
            "target_module": item["target"],
            "label": 1,
            "hops": item["hops"],
            "intermediate_path": item["path"],
        })
    for item in negative_samples:
        data.append({
            "source_module": item["source"],
            "target_module": item["target"],
            "label": 0,
            "hops": -1,
            "intermediate_path": None,
        })

    random.shuffle(data)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return len(data)


# =====================================================
# Task 2: Change Impact Analysis
# =====================================================

def generate_change_impact(project_path, output_path, sample_size=200):
    """
    生成变更影响分析任务。

    给定一个被修改的模块, 预测哪些模块会受到影响。
    受影响 = 直接或间接依赖于被修改模块的模块。
    """
    modules, graph, reverse_graph, weights = load_dependency_graph(project_path)

    data = []
    source_modules = [m for m in modules if len(reverse_graph.get(m, set())) > 0]

    if len(source_modules) < 5:
        return 0

    for modified_module in random.sample(source_modules, min(sample_size, len(source_modules))):
        # 找所有直接和间接依赖于 modified_module 的模块 (反向 BFS)
        impacted = set()
        queue = deque([modified_module])
        visited = {modified_module}
        while queue:
            node = queue.popleft()
            for dependent in reverse_graph.get(node, set()):
                if dependent not in visited:
                    visited.add(dependent)
                    impacted.add(dependent)
                    queue.append(dependent)

        # 直接受影响的
        direct_impact = reverse_graph.get(modified_module, set())

        if len(impacted) == 0:
            continue

        # 选一些不受影响的模块作为干扰项
        not_impacted = [m for m in modules if m != modified_module and m not in impacted]

        data.append({
            "modified_module": modified_module,
            "direct_impact": list(direct_impact),
            "total_impact": list(impacted),
            "impact_size": len(impacted),
            "not_impacted_sample": random.sample(not_impacted, min(5, len(not_impacted))),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return len(data)


# =====================================================
# Task 3: Architectural Smell Detection
# =====================================================

def generate_smell_detection(project_path, output_path):
    """
    生成架构坏味道检测任务。

    检测:
    1. Cyclic dependencies: A→B→C→A
    2. God modules: 模块被过多其他模块依赖 (fan-in > 阈值)
    3. Unstable modules: 高 fan-out + 高 fan-in
    """
    modules, graph, reverse_graph, weights = load_dependency_graph(project_path)

    smells = []

    # 1. 检测循环依赖
    def find_cycles(graph, max_length=4):
        cycles = []
        visited_global = set()

        for start in graph:
            if start in visited_global:
                continue
            # DFS 找环
            stack = [(start, [start])]
            visited_local = set()

            while stack:
                node, path = stack.pop()
                if len(path) > max_length:
                    continue

                for neighbor in graph.get(node, set()):
                    if neighbor == start and len(path) >= 2:
                        cycles.append(list(path) + [start])
                    elif neighbor not in visited_local and len(path) < max_length:
                        visited_local.add(neighbor)
                        stack.append((neighbor, path + [neighbor]))

            visited_global.add(start)

        return cycles

    cycles = find_cycles(graph, max_length=4)
    for cycle in cycles[:50]:  # 限制数量
        smells.append({
            "smell_type": "cyclic_dependency",
            "modules_involved": cycle,
            "description": f"Cyclic dependency: {' → '.join(cycle)}",
        })

    # 2. 检测 God Modules (fan-in > 中位数 * 3)
    fan_in = {m: len(reverse_graph.get(m, set())) for m in modules}
    if fan_in:
        median_fan_in = sorted(fan_in.values())[len(fan_in) // 2]
        threshold = max(median_fan_in * 3, 5)
        for m, fi in fan_in.items():
            if fi >= threshold:
                smells.append({
                    "smell_type": "god_module",
                    "modules_involved": [m],
                    "fan_in": fi,
                    "threshold": threshold,
                    "description": f"God module: {m} (fan-in={fi}, threshold={threshold})",
                })

    # 3. 检测 Unstable Modules (高 fan-out + 高 fan-in)
    fan_out = {m: len(graph.get(m, set())) for m in modules}
    for m in modules:
        fi = fan_in.get(m, 0)
        fo = fan_out.get(m, 0)
        if fi >= 3 and fo >= 3:
            instability = fo / (fi + fo) if (fi + fo) > 0 else 0
            if 0.3 < instability < 0.7:  # 不稳定区间
                smells.append({
                    "smell_type": "unstable_module",
                    "modules_involved": [m],
                    "fan_in": fi,
                    "fan_out": fo,
                    "instability": round(instability, 3),
                    "description": f"Unstable module: {m} (fan-in={fi}, fan-out={fo}, I={instability:.3f})",
                })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in smells:
            f.write(json.dumps(item) + "\n")

    return len(smells)


# =====================================================
# Task 4: Architectural Layer Inference
# =====================================================

def generate_layer_inference(project_path, output_path):
    """
    生成架构层级推断任务。

    基于依赖方向推断模块的架构层级:
    - 被很多模块依赖但自身依赖少的 → 底层 (infrastructure)
    - 依赖很多模块但很少被依赖的 → 上层 (application)
    - 中间的 → 中间层 (service/domain)
    """
    modules, graph, reverse_graph, weights = load_dependency_graph(project_path)

    fan_in = {m: len(reverse_graph.get(m, set())) for m in modules}
    fan_out = {m: len(graph.get(m, set())) for m in modules}

    data = []
    for m in modules:
        fi = fan_in.get(m, 0)
        fo = fan_out.get(m, 0)
        total = fi + fo

        if total == 0:
            layer = "isolated"
        elif fi > 0 and fo == 0:
            layer = "infrastructure"  # 只被依赖, 不依赖别人
        elif fo > 0 and fi == 0:
            layer = "application"     # 只依赖别人, 不被依赖
        elif fi >= fo * 2:
            layer = "infrastructure"  # 被依赖远多于依赖别人
        elif fo >= fi * 2:
            layer = "application"     # 依赖别人远多于被依赖
        else:
            layer = "service"         # 中间层

        data.append({
            "module": m,
            "fan_in": fi,
            "fan_out": fo,
            "layer": layer,
            "dependencies": list(graph.get(m, set())),
            "dependents": list(reverse_graph.get(m, set())),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return len(data)


# =====================================================
# 主函数
# =====================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="all")
    parser.add_argument("--task", default="all",
                        choices=["transitive", "impact", "smell", "layer", "all"])
    args = parser.parse_args()

    if args.project == "all":
        projects = PROJECTS
    else:
        projects = [p for p in PROJECTS if p["name"] == args.project]

    tasks = ["transitive", "impact", "smell", "layer"] if args.task == "all" else [args.task]

    for proj in projects:
        print(f"\nProject: {proj['name']}")

        if "transitive" in tasks:
            out = os.path.join(proj["path"], "transitive_dependency.jsonl")
            n = generate_transitive_dependency(proj["path"], out)
            print(f"  Transitive Dependency: {n} samples")

        if "impact" in tasks:
            out = os.path.join(proj["path"], "change_impact.jsonl")
            n = generate_change_impact(proj["path"], out)
            print(f"  Change Impact: {n} samples")

        if "smell" in tasks:
            out = os.path.join(proj["path"], "architectural_smells.jsonl")
            n = generate_smell_detection(proj["path"], out)
            print(f"  Architectural Smells: {n} samples")

        if "layer" in tasks:
            out = os.path.join(proj["path"], "layer_inference.jsonl")
            n = generate_layer_inference(proj["path"], out)
            print(f"  Layer Inference: {n} samples")

    print("\nDone.")


if __name__ == "__main__":
    main()
