"""
Prompt 模板设计

设计原则：
1. Zero-shot：不给示例，测试模型的真实架构理解能力
2. 提供系统背景信息（项目名、领域），模拟真实开发者场景
3. 要求结构化输出，方便自动解析
"""


# =====================================================
# Task 1: Component Identification
# =====================================================

PROMPT_CI = """You are a software architecture expert. You are analyzing the architecture of {system}, an open-source {domain} system.

Given a file path from this project, determine which module (component) it most likely belongs to.

File path: {file_path}

Candidate modules:
A) {candidate_a}
B) {candidate_b}
C) {candidate_c}
D) {candidate_d}

Based on your understanding of {system}'s architecture and the file path's naming conventions, directory structure, and typical code organization patterns, which module does this file belong to?

Answer with ONLY the letter (A, B, C, or D)."""


# =====================================================
# Task 2: Dependency Prediction
# =====================================================

PROMPT_DP = """You are a software architecture expert. You are analyzing the architecture of {system}, an open-source {domain} system.

Given two modules from this project, determine whether Module A depends on Module B (i.e., Module A imports, includes, or calls code defined in Module B). A dependency means that Module A cannot compile or function correctly without Module B.

Module A: {module_a}
Module B: {module_b}

Based on your understanding of {system}'s architecture, the typical responsibilities of these modules, and common dependency patterns in {domain} systems, does Module A depend on Module B?

Answer with ONLY "Yes" or "No"."""


# =====================================================
# Task 3: Architecture Reconstruction
# =====================================================

PROMPT_AR = """You are a software architecture expert. You are analyzing the architecture of {system}, an open-source {domain} system.

Given a source module, predict which other modules it most likely depends on (i.e., imports/uses code from).

Source module: {source_module}

Available modules in this system:
{module_list}

Based on your understanding of {system}'s architecture and the typical role of the source module, list the top 5 modules that {source_module} most likely depends on.

Answer with ONLY the module names, one per line, ranked by likelihood. Do not include explanations."""


# =====================================================
# Prompt 构造函数
# =====================================================

def build_ci_prompt(system, domain, file_path, candidates, label):
    """构造 Component Identification prompt"""
    # 随机分配 ABCD（candidates 已经 shuffle 过）
    letter_map = {}
    for i, c in enumerate(candidates):
        letter = chr(65 + i)  # A, B, C, D
        letter_map[letter] = c

    prompt = PROMPT_CI.format(
        system=system,
        domain=domain,
        file_path=file_path,
        candidate_a=candidates[0],
        candidate_b=candidates[1],
        candidate_c=candidates[2],
        candidate_d=candidates[3],
    )

    # 找到正确答案的字母
    correct_letter = None
    for letter, module in letter_map.items():
        if module == label:
            correct_letter = letter
            break

    return prompt, correct_letter, letter_map


def build_dp_prompt(system, domain, module_a, module_b):
    """构造 Dependency Prediction prompt"""
    prompt = PROMPT_DP.format(
        system=system,
        domain=domain,
        module_a=module_a,
        module_b=module_b,
    )
    return prompt


def build_ar_prompt(system, domain, source_module, all_modules, max_candidates=50):
    """构造 Architecture Reconstruction prompt

    当模块数超过 max_candidates 时，只保留出现频率最高的模块作为候选，
    避免 prompt 过长导致 token 浪费和模型性能下降。
    """
    candidates = [m for m in all_modules if m != source_module]

    if len(candidates) > max_candidates:
        # 保留前 max_candidates 个（modules_index 已按字母排序，
        # 但这里不做额外排序，调用方可传入按频率排序的列表）
        candidates = candidates[:max_candidates]

    module_list = "\n".join(f"- {m}" for m in candidates)
    prompt = PROMPT_AR.format(
        system=system,
        domain=domain,
        source_module=source_module,
        module_list=module_list,
    )
    return prompt


# =====================================================
# Task 1-Hard: Component Identification (Hard Mode)
# =====================================================

PROMPT_CI_HARD = """You are a software architecture expert. You are analyzing the architecture of {system}, an open-source {domain} system.

Given only a source file name (without its directory path), determine which module (component) it most likely belongs to based on the file's naming conventions and your knowledge of {system}'s architecture.

File name: {file_name}

Candidate modules:
A) {candidate_a}
B) {candidate_b}
C) {candidate_c}
D) {candidate_d}

Answer with ONLY the letter (A, B, C, or D)."""


def build_ci_hard_prompt(system, domain, file_name, candidates, label):
    """构造 Hard CI prompt"""
    letter_map = {}
    for i, c in enumerate(candidates):
        letter_map[chr(65 + i)] = c

    prompt = PROMPT_CI_HARD.format(
        system=system,
        domain=domain,
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
