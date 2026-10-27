"""
Level 2 Prompt 模板：Full-Context 模式

与 Level 0/1 的区别：提供完整的架构上下文
- CI: 提供完整的模块列表 + 每个候选模块的文件样例
- CI Hard: 同上
- DP: 提供模块 A 的完整依赖列表（去掉待预测的 B）+ 模块 B 的被依赖列表
- AR: 提供系统的完整模块依赖图（去掉待预测的边）

这测试的是：当 LLM 拥有几乎完整的架构信息时，能否正确推理出缺失的部分？
"""


# =====================================================
# Level 2: CI with full architecture context
# =====================================================

PROMPT_CI_L2 = """You are a software architecture expert analyzing {system}, an open-source {domain} system.

Here is the module structure of this system. Each module contains the listed example files:

{module_file_examples}

Given this architecture, which module does the following file belong to?

File path: {file_path}

Candidate modules:
A) {candidate_a}
B) {candidate_b}
C) {candidate_c}
D) {candidate_d}

Answer with ONLY the letter (A, B, C, or D)."""


# =====================================================
# Level 2: CI Hard with full architecture context
# =====================================================

PROMPT_CI_HARD_L2 = """You are a software architecture expert analyzing {system}, an open-source {domain} system.

Here is the module structure of this system. Each module contains the listed example files:

{module_file_examples}

Given this architecture, which module does the following file (name only) belong to?

File name: {file_name}

Candidate modules:
A) {candidate_a}
B) {candidate_b}
C) {candidate_c}
D) {candidate_d}

Based on the file naming patterns in each module, answer with ONLY the letter (A, B, C, or D)."""


# =====================================================
# Level 2: DP with dependency graph context
# =====================================================

PROMPT_DP_L2 = """You are a software architecture expert analyzing {system}, an open-source {domain} system.

Here is the known dependency structure of this system:

Modules that {module_a} depends on:
{deps_of_a}

Modules that depend on {module_b}:
{dependents_of_b}

Based on this dependency graph and the architectural roles of these modules, does {module_a} also depend on {module_b}?

A dependency means {module_a} imports, includes, or calls code defined in {module_b}.

Answer with ONLY "Yes" or "No"."""


# =====================================================
# Level 2: AR with full dependency graph
# =====================================================

PROMPT_AR_L2 = """You are a software architecture expert analyzing {system}, an open-source {domain} system.

Here is the known module dependency graph of this system (source -> targets):

{dependency_graph}

Module "{source_module}" has the dependencies listed above (if any). Based on the overall dependency patterns in this system, predict which additional modules {source_module} most likely depends on.

Available modules:
{module_list}

List the top 5 most likely additional dependency targets for {source_module}.

Answer with ONLY the module names, one per line. Do not include modules already listed as known dependencies."""


# =====================================================
# 构造函数
# =====================================================

import os


def _get_module_file_examples(module_files, candidates, exclude_path=None, max_files_per_module=5):
    """为每个候选模块生成文件样例；exclude_path（被问文件）不出现在样例中，避免泄露答案"""
    lines = []
    for module in candidates:
        files = [f for f in module_files.get(module, []) if f != exclude_path]
        sample = files[:max_files_per_module]
        file_names = [os.path.basename(f) for f in sample]
        if len(files) > max_files_per_module:
            file_names.append(f"... ({len(files)} files total)")
        lines.append(f"  {module}: {', '.join(file_names) if file_names else '(no other files)'}")
    return "\n".join(lines)


def build_ci_l2_prompt(system, domain, file_path, candidates, label, module_files, exclude_path=None):
    letter_map = {}
    for i, c in enumerate(candidates):
        letter_map[chr(65 + i)] = c

    examples = _get_module_file_examples(module_files, candidates, exclude_path)

    prompt = PROMPT_CI_L2.format(
        system=system,
        domain=domain,
        module_file_examples=examples,
        file_path=file_path,
        candidate_a=candidates[0],
        candidate_b=candidates[1],
        candidate_c=candidates[2],
        candidate_d=candidates[3],
    )

    correct_letter = None
    for letter, module in letter_map.items():
        if module == label:
            correct_letter = letter
            break

    return prompt, correct_letter, letter_map


def build_ci_hard_l2_prompt(system, domain, file_name, candidates, label, module_files, exclude_path=None):
    letter_map = {}
    for i, c in enumerate(candidates):
        letter_map[chr(65 + i)] = c

    examples = _get_module_file_examples(module_files, candidates, exclude_path)

    prompt = PROMPT_CI_HARD_L2.format(
        system=system,
        domain=domain,
        module_file_examples=examples,
        file_name=file_name,
        candidate_a=candidates[0],
        candidate_b=candidates[1],
        candidate_c=candidates[2],
        candidate_d=candidates[3],
    )

    correct_letter = None
    for letter, module in letter_map.items():
        if module == label:
            correct_letter = letter
            break

    return prompt, correct_letter, letter_map


def build_dp_l2_prompt(system, domain, module_a, module_b,
                        dep_graph, reverse_dep_graph, label):
    """
    dep_graph: module -> set of targets (已知的依赖)
    reverse_dep_graph: module -> set of sources (被谁依赖)

    关键：如果 label=1（真实存在依赖），从 dep_graph 中去掉这条边，
    避免直接泄露答案。
    """
    deps_a = set(dep_graph.get(module_a, set()))
    if label == 1:
        deps_a = deps_a - {module_b}  # 去掉待预测的边

    dependents_b = set(reverse_dep_graph.get(module_b, set()))
    if label == 1:
        dependents_b = dependents_b - {module_a}

    deps_a_str = ", ".join(sorted(deps_a)[:15]) if deps_a else "(none known)"
    if len(deps_a) > 15:
        deps_a_str += f" ... ({len(deps_a)} total)"

    dependents_b_str = ", ".join(sorted(dependents_b)[:15]) if dependents_b else "(none known)"
    if len(dependents_b) > 15:
        dependents_b_str += f" ... ({len(dependents_b)} total)"

    prompt = PROMPT_DP_L2.format(
        system=system,
        domain=domain,
        module_a=module_a,
        module_b=module_b,
        deps_of_a=deps_a_str,
        dependents_of_b=dependents_b_str,
    )
    return prompt


def build_ar_l2_prompt(system, domain, source_module, dep_graph,
                        true_targets, all_modules, max_candidates=50,
                        max_other_modules=20, max_targets_per_module=10):
    """
    提供依赖图上下文，让模型预测 source_module 被隐藏的依赖。完全确定性（无随机）。

    - 隐藏边 = true_targets（seed42 清单中的 AR 目标，与 L0/L1 评分目标相同）
    - source 的已知边 shown = dep_graph[source] - true_targets（排序）
    - 其他模块：按 all_modules 顺序（L0 的 freq 降序）取前 max_other_modules 个有依赖的模块，
      每个模块的 targets 排序后取前 max_targets_per_module 个
    - 候选：all_modules 去掉 source 与 shown 后取前 max_candidates 个（与 L0 候选同序）

    返回 (prompt, shown_deps)
    """
    hidden = set(true_targets)
    shown_deps = sorted(set(dep_graph.get(source_module, ())) - hidden)

    graph_lines = []
    if shown_deps:
        graph_lines.append(f"  {source_module} -> {', '.join(shown_deps)}")

    n_other = 0
    for m in all_modules:
        if n_other >= max_other_modules:
            break
        if m == source_module:
            continue
        targets = sorted(dep_graph.get(m, ()))[:max_targets_per_module]
        if targets:
            graph_lines.append(f"  {m} -> {', '.join(targets)}")
            n_other += 1

    graph_text = "\n".join(graph_lines) if graph_lines else "  (no dependencies known)"

    shown_set = set(shown_deps)
    candidates = [m for m in all_modules if m != source_module and m not in shown_set][:max_candidates]
    module_list = "\n".join(f"- {m}" for m in candidates)

    prompt = PROMPT_AR_L2.format(
        system=system,
        domain=domain,
        source_module=source_module,
        dependency_graph=graph_text,
        module_list=module_list,
    )
    return prompt, shown_deps
