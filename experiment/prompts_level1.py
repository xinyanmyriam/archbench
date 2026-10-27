"""
Level 1 Prompt 模板：Code-Context 模式

与 Level 0 的区别：在 prompt 中注入真实代码上下文
- CI: 提供文件的前 N 行代码（N = CODE_SNIPPET_LINES，默认 100，包含 import/include 语句）
- CI Hard: 同上，但仍然隐藏文件路径
- CI Obfuscated: 同 CI Hard，但文件路径与模块名均已匿名化，代码内容为真实代码
- DP: 提供两个模块各自的文件列表 + 模块 A 的 import 目标
- AR: 提供 source module 的文件列表 + 所有 import/include 目标
"""


# =====================================================
# Level 1: CI with code context
# =====================================================

PROMPT_CI_L1 = """You are a software architecture expert. You are analyzing the architecture of {system}, an open-source {domain} system.

Given a file path and the first lines of its source code, determine which module (component) it belongs to.

File path: {file_path}

Source code (first {n_lines} lines):
```
{code_snippet}
```

Candidate modules:
A) {candidate_a}
B) {candidate_b}
C) {candidate_c}
D) {candidate_d}

Answer with ONLY the letter (A, B, C, or D)."""


# =====================================================
# Level 1: CI Hard with code context
# =====================================================

PROMPT_CI_HARD_L1 = """You are a software architecture expert. You are analyzing the architecture of {system}, an open-source {domain} system.

Given only a file name (without its directory path) and the first lines of its source code, determine which module (component) it belongs to.

File name: {file_name}

Source code (first {n_lines} lines):
```
{code_snippet}
```

Candidate modules:
A) {candidate_a}
B) {candidate_b}
C) {candidate_c}
D) {candidate_d}

Based on the import/include statements and code content, which module does this file belong to?

Answer with ONLY the letter (A, B, C, or D)."""


# =====================================================
# Level 1: CI Obfuscated with code context
# =====================================================

PROMPT_CI_OBF_L1 = """You are a software architecture expert. You are analyzing the architecture of {system}, an open-source {domain} system.

Given an obfuscated file path and the first lines of its source code, determine which module it belongs to. The file and module names have been anonymized, but the code content is real.

Obfuscated file path: {file_path}

Source code (first {n_lines} lines):
```
{code_snippet}
```

Candidate modules:
A) {candidate_a}
B) {candidate_b}
C) {candidate_c}
D) {candidate_d}

Based on the import/include statements and code content (not the obfuscated names), which module does this file belong to?

Answer with ONLY the letter (A, B, C, or D)."""


# =====================================================
# Level 1: DP with dependency context
# =====================================================

PROMPT_DP_L1 = """You are a software architecture expert. You are analyzing the architecture of {system}, an open-source {domain} system.

Given two modules and their internal structure, determine whether Module A depends on Module B.

Module A: {module_a}
  Files in this module: {files_a}
  Import/include targets from this module: {imports_a}

Module B: {module_b}
  Files in this module: {files_b}

A dependency means Module A imports, includes, or calls code defined in Module B, such that Module A cannot compile or function correctly without Module B.

Does Module A depend on Module B?

Answer with ONLY "Yes" or "No"."""


# =====================================================
# Level 1: AR with dependency context
# =====================================================

PROMPT_AR_L1 = """You are a software architecture expert. You are analyzing the architecture of {system}, an open-source {domain} system.

Given a source module and its internal import/include targets, predict which other modules it depends on.

Source module: {source_module}
  Files: {source_files}
  All import/include targets from this module:
{import_list}

Available modules in this system:
{module_list}

Based on the import targets and module names, list the top 5 modules that {source_module} most likely depends on.

Answer with ONLY the module names, one per line, ranked by likelihood. Do not include explanations."""


# =====================================================
# 构造函数
# =====================================================

def _mc_prompt(template, candidates, label, **fields):
    """四选一模板通用构造：返回 (prompt, correct_letter, letter_map)"""
    letter_map = {chr(65 + i): c for i, c in enumerate(candidates)}
    prompt = template.format(
        candidate_a=candidates[0],
        candidate_b=candidates[1],
        candidate_c=candidates[2],
        candidate_d=candidates[3],
        **fields,
    )
    correct_letter = next((l for l, m in letter_map.items() if m == label), None)
    return prompt, correct_letter, letter_map


def build_ci_l1_prompt(system, domain, file_path, candidates, label, code_snippet, n_lines):
    return _mc_prompt(PROMPT_CI_L1, candidates, label, system=system, domain=domain,
                      file_path=file_path, code_snippet=code_snippet, n_lines=n_lines)


def build_ci_hard_l1_prompt(system, domain, file_name, candidates, label, code_snippet, n_lines):
    return _mc_prompt(PROMPT_CI_HARD_L1, candidates, label, system=system, domain=domain,
                      file_name=file_name, code_snippet=code_snippet, n_lines=n_lines)


def build_ci_obf_l1_prompt(system, domain, file_path, candidates, label, code_snippet, n_lines):
    return _mc_prompt(PROMPT_CI_OBF_L1, candidates, label, system=system, domain=domain,
                      file_path=file_path, code_snippet=code_snippet, n_lines=n_lines)


def build_dp_l1_prompt(system, domain, module_a, module_b,
                        files_a, files_b, imports_a):
    # 截断文件列表，避免 prompt 过长
    files_a_str = ", ".join(files_a[:10])
    if len(files_a) > 10:
        files_a_str += f" ... ({len(files_a)} files total)"

    files_b_str = ", ".join(files_b[:10])
    if len(files_b) > 10:
        files_b_str += f" ... ({len(files_b)} files total)"

    imports_a_str = ", ".join(imports_a[:20]) or "(none)"
    if len(imports_a) > 20:
        imports_a_str += f" ... ({len(imports_a)} targets total)"

    prompt = PROMPT_DP_L1.format(
        system=system,
        domain=domain,
        module_a=module_a,
        module_b=module_b,
        files_a=files_a_str,
        files_b=files_b_str,
        imports_a=imports_a_str,
    )
    return prompt


def build_ar_l1_prompt(system, domain, source_module, source_files,
                        import_targets, all_modules, max_candidates=50):
    source_files_str = ", ".join(source_files[:15])
    if len(source_files) > 15:
        source_files_str += f" ... ({len(source_files)} files total)"

    import_list = "\n".join(f"  - {t}" for t in import_targets[:30]) or "  (none)"
    if len(import_targets) > 30:
        import_list += f"\n  ... ({len(import_targets)} targets total)"

    candidates = [m for m in all_modules if m != source_module]
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
    module_list = "\n".join(f"- {m}" for m in candidates)

    prompt = PROMPT_AR_L1.format(
        system=system,
        domain=domain,
        source_module=source_module,
        source_files=source_files_str,
        import_list=import_list,
        module_list=module_list,
    )
    return prompt
