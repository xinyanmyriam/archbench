"""
Baseline 方法实现

1. Random: 随机选择
2. Frequency: 选出现频率最高的模块
3. Path Heuristic: 基于路径字符串匹配
4. Graph Neighbor: 基于图结构的邻居信息
"""

import json
import random
import os
from collections import Counter, defaultdict

random.seed(42)

# 与 run_experiment.MANIFEST_VERSION 相同的开关（此处不能 import run_experiment，会循环导入）
MANIFEST_VERSION = os.environ.get("SAB_MANIFEST", "v1")
AKG_V2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "systems")


def _k8s_normalize(m):
    """与 kubernetes/generate_benchmark_tasks.py 的 normalize_module 相同；
    kubernetes 的 module_dependency.json 用的是未归一化模块名，题目与 modules_index 是归一化后的"""
    parts = m.split("/")
    if m.startswith("staging/src/k8s.io/"):
        return "staging/src/k8s.io/" + parts[3] if len(parts) > 3 else m
    return parts[0] + "/" + parts[1] if len(parts) > 1 else m


def _project_name(project_path):
    from config import PROJECTS
    return next(p["name"] for p in PROJECTS if p["path"] == project_path)


def load_project_data(project_path):
    """加载项目的所有数据文件"""
    data = {}

    with open(os.path.join(project_path, "file_index.json"), encoding="utf-8") as f:
        data["file_index"] = json.load(f)

    name = _project_name(project_path) if MANIFEST_VERSION == "v2" else None
    md_path = (os.path.join(AKG_V2_DIR, name, "module_dependency.json") if name
               else os.path.join(project_path, "module_dependency.json"))
    with open(md_path, encoding="utf-8") as f:
        data["module_dep"] = json.load(f)

    # 模块频率统计
    module_counter = Counter()
    for fi in data["file_index"]["files"]:
        module_counter[fi["module"]] += 1
    if name == "linux":
        # v2 新增的 include/* 模块不在 file_index 里；按 akg_v2 文件依赖中出现的 include/<x>/ 下不同文件数计频
        with open(os.path.join(AKG_V2_DIR, "linux", "file_dependency.json"), encoding="utf-8") as f:
            fd = json.load(f)["dependencies"]
        inc_files = {p for d in fd for p in (d["source"], d["target"]) if p.startswith("include/")}
        for p in inc_files:
            parts = p.split("/")
            if len(parts) > 2:
                module_counter["include/" + parts[1]] += 1
    data["module_freq"] = module_counter

    # 模块依赖图
    dep_graph = defaultdict(dict)
    for d in data["module_dep"]["dependencies"]:
        src = d["source_module"]
        tgt = d["target_module"]
        if name == "kubernetes":
            src, tgt = _k8s_normalize(src), _k8s_normalize(tgt)
            if src == tgt:
                continue
        w = d.get("weight", 1)
        dep_graph[src][tgt] = dep_graph[src].get(tgt, 0) + w
    data["dep_graph"] = dep_graph

    return data


# =====================================================
# Task 1: Component Identification Baselines
# =====================================================

def baseline_ci_random(candidates):
    """随机选择"""
    return random.choice(candidates)


def baseline_ci_frequency(candidates, module_freq):
    """选出现频率最高的候选模块"""
    return max(candidates, key=lambda m: module_freq.get(m, 0))


def baseline_ci_path_heuristic(file_path, candidates):
    """基于路径字符串匹配：选与文件路径重叠最多的模块"""
    file_parts = set(file_path.lower().replace("\\", "/").split("/"))

    best = None
    best_score = -1

    for c in candidates:
        module_parts = set(c.lower().replace("\\", "/").split("/"))
        overlap = len(file_parts & module_parts)
        if overlap > best_score:
            best_score = overlap
            best = c

    if best_score == 0:
        return random.choice(candidates)

    return best


# =====================================================
# Task 2: Dependency Prediction Baselines
# =====================================================

def baseline_dp_random():
    """随机预测"""
    return random.choice([0, 1])


def baseline_dp_frequency(module_a, module_b, dep_graph):
    """如果 module_a 有很多出边，倾向于预测 Yes"""
    out_degree = len(dep_graph.get(module_a, {}))
    # 简单阈值：出度 > 中位数则预测 Yes
    return 1 if out_degree > 3 else 0


def baseline_dp_graph_neighbor(module_a, module_b, dep_graph, exclude_queried_edge=False):
    """基于二跳邻居：如果 A 和 B 有共同邻居，预测 Yes

    exclude_queried_edge=True 时先删掉 A→B 这条被查询的边，避免标签泄露
    （v2 的 DP 标签直接取自同一张图，不删就等于直接查答案）。
    """
    neighbors_a = set(dep_graph.get(module_a, {}).keys())
    if exclude_queried_edge:
        neighbors_a.discard(module_b)
    neighbors_b = set(dep_graph.get(module_b, {}).keys())

    # 直接依赖
    if module_b in neighbors_a:
        return 1

    # 共同邻居
    common = neighbors_a & neighbors_b
    return 1 if len(common) > 0 else 0


# =====================================================
# Task 3: Architecture Reconstruction Baselines
# =====================================================

def baseline_ar_random(all_modules, source_module, k=5):
    """随机选 k 个模块"""
    candidates = [m for m in all_modules if m != source_module]
    return random.sample(candidates, min(k, len(candidates)))


def baseline_ar_frequency(all_modules, module_freq, source_module, k=5):
    """选文件数最多的 k 个模块"""
    candidates = [m for m in all_modules if m != source_module]
    candidates.sort(key=lambda m: module_freq.get(m, 0), reverse=True)
    return candidates[:k]


def baseline_ar_graph_neighbor(dep_graph, source_module, k=5):
    """选已知邻居中权重最高的 k 个"""
    neighbors = dep_graph.get(source_module, {})
    sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
    return [n[0] for n in sorted_neighbors[:k]]
