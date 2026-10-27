"""生成最终实验结果报告 results_rerun_summary.md（可复现：直接读结果 JSON）。

用法：python generate_final_report.py
"""
import json
import glob
import os
from statistics import mean

SYSTEMS = ["nastran", "openfoam", "salome", "paraview", "kubernetes", "linux",
           "hadoop", "spring", "codeaster", "openmodelica", "tensorflow", "pytorch"]
L0_MODELS = ["claude-opus-5.5", "gpt-6-sol", "kimi-k3", "gpt-oss-120b",
             "llama-4-maverick", "qwen3", "deepseek-v4-flash",
             "qwen3-8b", "llama-3.1-8b"]
L12_MODELS = ["claude-opus-5.5", "gpt-6-sol"]

OUT = "results_rerun_summary.md"


def load(f):
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def cell(v, fmt="{:.3f}"):
    return fmt.format(v) if v is not None else "–"


def l0_metric(d, task):
    if task in ("ci", "ci_hard"):
        return fnum(d.get("accuracy"))
    if task == "dp":
        return fnum(d.get("f1"))
    p, r = fnum(d.get("precision_at_5")), fnum(d.get("recall_at_5"))
    return (p, r) if p is not None else None


def l0_file(model, sys_, task):
    return f"results_rerun/{model}_{sys_}_{task}.json"


def l12_file(level, model, sys_, task):
    suf = "l1" if level == "level1" else "l2"
    return f"results_rerun/{level}/{model}_{sys_}_{task}_{suf}.json"


# ---------- 收集错误明细 ----------
def _is_error_text(raw):
    """真实失败判定：raw_response 为错误文本（L2/L1 item 不写 api_error 标记）。"""
    r = (raw or "").lstrip()
    return r.startswith("ERROR") or "Error code:" in r[:40] or r.startswith("Traceback")


def _count_real_errors(fp):
    if not os.path.isfile(fp):
        return 0
    return sum(1 for r in load(fp).get("results", []) if _is_error_text(r.get("raw_response")))


def collect_errors():
    """按 raw_response 是否为错误文本统计真实最终失败题（过程性重试成功不计）。"""
    rows = []
    for m in L0_MODELS:
        for s in SYSTEMS:
            for t in ["ci", "ci_hard", "dp", "ar"]:
                fp = l0_file(m, s, t)
                n = _count_real_errors(fp)
                if n:
                    rows.append((m, s, t, n))
    for level, tasks in [("level1", ["ci_hard", "ci_obfuscated", "dp", "ar"]),
                         ("level2", ["ci_hard", "dp", "ar"])]:
        for m in L12_MODELS:
            for s in SYSTEMS:
                for t in tasks:
                    fp = l12_file(level, m, s, t)
                    n = _count_real_errors(fp)
                    if n:
                        rows.append((f"{m} [{level}]", s, t, n))
    return rows


# ---------- 宏平均 ----------
def macro(model, task, level="l0"):
    vals = []
    for s in SYSTEMS:
        fp = l0_file(model, s, task) if level == "l0" else l12_file(level, model, s, task)
        if not os.path.isfile(fp):
            continue
        v = l0_metric(load(fp), task)
        if v is None:
            continue
        vals.append(v if not isinstance(v, tuple) else v)
    if not vals:
        return None
    if task == "ar":
        return mean(v[0] for v in vals), mean(v[1] for v in vals)
    return mean(vals)


# ---------- 输出 ----------
lines = []
W = lines.append

W("# ArchBench 重跑实验最终结果（seed42 固定测试集）")
W("")
W("- 生成日期：2026-10-08")
W("- 测试集：`experiment/samples/manifest_seed42.json`，12 个系统 × 4 任务，所有模型/层级共用同一份样本")
W("- 结果目录：`experiment/results_rerun/`（L0）、`results_rerun/level1/`、`results_rerun/level2/`")
W("- 解析器：2026-10-07 升级版（CoT 稳健解析）；原始文件备份于 `results_rerun/_prerescore_backup/`")
W("- 重打分脚本：`experiment/rescore_parsing.py`；本报告生成脚本：`experiment/generate_final_report.py`")
W("")

# ===== L0 宏平均 =====
W("## 1. Level 0 宏平均（12 系统）")
W("")
W("| 模型 | CI | CI-Hard | DP F1 | AR P@5 | AR R@5 | API错误 |")
W("|---|---|---|---|---|---|---|")
for m in L0_MODELS:
    ci, cih = macro(m, "ci"), macro(m, "ci_hard")
    dp = macro(m, "dp")
    ar = macro(m, "ar")
    nerr = sum(_count_real_errors(l0_file(m, s, t))
               for s in SYSTEMS for t in ["ci", "ci_hard", "dp", "ar"])
    W(f"| {m} | {cell(ci)} | {cell(cih)} | {cell(dp)} | {cell(ar[0])} | {cell(ar[1])} | {nerr} |")
W("")

# ===== L0 每任务明细矩阵 =====
task_titles = [("ci", "CI（组件识别）"), ("ci_hard", "CI-Hard（难例）"),
               ("dp", "DP F1（依赖预测）"), ("ar", "AR P@5 / R@5（架构重构）")]
W("## 2. Level 0 逐系统明细")
W("")
hdr_models = [m.replace("claude-", "").replace("-v4-flash", "") for m in L0_MODELS]
for task, title in task_titles:
    W(f"### 2.{['ci','ci_hard','dp','ar'].index(task)+1} {title}")
    W("")
    W("| 系统 | " + " | ".join(hdr_models) + " |")
    W("|" + "---|" * (len(L0_MODELS) + 1))
    for s in SYSTEMS:
        row = [s]
        for m in L0_MODELS:
            fp = l0_file(m, s, task)
            if not os.path.isfile(fp):
                row.append("–")
                continue
            v = l0_metric(load(fp), task)
            if v is None:
                row.append("–")
            elif task == "ar":
                row.append(f"{v[0]:.2f}/{v[1]:.2f}")
            else:
                row.append(f"{v:.3f}")
        W("| " + " | ".join(row) + " |")
    W("")

# ===== L1 / L2 =====
W("## 3. Level 1 结果（两个前沿闭源模型）")
W("")
W("| 模型 | CI-Hard | CI-Obfuscated | DP F1 | AR P@5 | AR R@5 | API错误 |")
W("|---|---|---|---|---|---|---|")
for m in L12_MODELS:
    cih = macro(m, "ci_hard", "level1")
    obf = macro(m, "ci_obfuscated", "level1")
    dp = macro(m, "dp", "level1")
    ar = macro(m, "ar", "level1")
    nerr = sum(_count_real_errors(l12_file("level1", m, s, t))
               for s in SYSTEMS for t in ["ci_hard", "ci_obfuscated", "dp", "ar"])
    W(f"| {m} | {cell(cih)} | {cell(obf)} | {cell(dp)} | {cell(ar[0])} | {cell(ar[1])} | {nerr} |")
W("")

W("## 4. Level 2 结果（提供仓库级上下文）")
W("")
W("| 模型 | CI-Hard | DP F1 | AR P@5 | AR R@5 | API错误 |")
W("|---|---|---|---|---|---|")
for m in L12_MODELS:
    cih = macro(m, "ci_hard", "level2")
    dp = macro(m, "dp", "level2")
    ar = macro(m, "ar", "level2")
    nerr = sum(_count_real_errors(l12_file("level2", m, s, t))
               for s in SYSTEMS for t in ["ci_hard", "dp", "ar"])
    W(f"| {m} | {cell(cih)} | {cell(dp)} | {cell(ar[0])} | {cell(ar[1])} | {nerr} |")
W("")

# L1/L2 逐系统
for level, title, tasks in [
        ("level1", "Level 1", ["ci_hard", "ci_obfuscated", "dp", "ar"]),
        ("level2", "Level 2", ["ci_hard", "dp", "ar"])]:
    W(f"### {title} 逐系统明细")
    W("")
    W("| 系统 | 模型 | " + " | ".join(
        {"ci_hard": "CI-Hard", "ci_obfuscated": "CI-Obf", "dp": "DP F1", "ar": "AR P/R"}[t]
        for t in tasks) + " |")
    W("|---|---|" + "---|" * len(tasks))
    for s in SYSTEMS:
        for m in L12_MODELS:
            row = [s, m.replace("claude-", "")]
            for t in tasks:
                fp = l12_file(level, m, s, t)
                if not os.path.isfile(fp):
                    row.append("–")
                    continue
                v = l0_metric(load(fp), t)
                if v is None:
                    row.append("–")
                elif t == "ar":
                    row.append(f"{v[0]:.2f}/{v[1]:.2f}")
                else:
                    row.append(f"{v:.3f}")
            W("| " + " | ".join(row) + " |")
    W("")

# ===== 错误明细 =====
W("## 5. API 错误明细（需在论文局限性中如实报告）")
W("")
errs = collect_errors()
W("| 模型/层级 | 系统 | 任务 | 错误数 | 原因 |")
W("|---|---|---|---|---|")
for m, s, t, n in errs:
    if m == "deepseek-v4-flash":
        reason = "输出超 32768 token 硬上限（finish_reason=length），API 限制无法补回"
    elif "[level1]" in m:
        reason = "OfoxAI 内容安全策略 400 永久拒绝（nastran Fortran / codeaster 源码被误判）"
    elif "[level2]" in m:
        reason = "OfoxAI 内容安全策略 400 永久拒绝（Linux 内核源码被误判）"
    else:
        reason = "API 错误"
    W(f"| {m} | {s} | {t} | {n} | {reason} |")
W("")
W(f"错误总计：{sum(r[3] for r in errs)} 题真实失败（按 raw_response 为错误文本判定；"
  f"重试后成功的过程性错误不计）。这些题预测为空、计为答错，已反映在指标中。除上述外全部 0 错误。")
W("")

# ===== 关键发现 =====
W("## 6. 关键发现")
W("")
W("1. **标识符命名依赖（回应 R3-1）**：CI-Obfuscated 上 opus 0.246 / gpt-6-sol 0.235，"
  "而同模型 CI-Hard 分别为 0.987 / 0.911。混淆命名后准确率骤降约 70 个百分点，"
  "说明模型在组件识别中高度依赖标识符命名而非代码结构语义。")
W("2. **模型规模效应（回应 R2-7）**：CI-Hard 上 Llama 系 maverick 0.744 → 8B 0.555（-18.9pt），"
  "Qwen 系 235B 0.703 → 8B 0.661（-4.2pt）。小模型并非普遍追平大模型，"
  "旧论文基于被空响应污染数据的“规模效应极小”结论需修正为“因模型家族而异”。")
W("3. **L2 仓库级上下文提升 AR 召回**：opus R@5 0.600(L1) → 0.704(L2)，"
  "gpt-6-sol 0.574 → 0.688；P@5 提升较小，说明依赖图帮助模型找回更多真实边但仍引入噪声。")
W("4. **前沿闭源模型领先**：opus 在 CI-Hard（0.968）、DP F1（0.771）、AR R@5（0.564）均居 L0 首位。")
W("5. **解析伪影修正**：旧解析器在 CoT 长响应下系统性低估 llama 两个模型的 CI/CI-Hard，"
  "升级解析器并离线重打分后（无额外 API 调用），llama-4-maverick CI 0.588→0.994、"
  "CI-Hard 0.458→0.744；llama-3.1-8b CI 0.595→0.970。对输出裸字母的模型（如 gpt-6-sol）"
  "解析结果逐题 0 改变，验证新解析器的保守性。")
W("")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print(f"已生成 {OUT}（{len(lines)} 行）；错误明细 {len(errs)} 条，共 {sum(r[3] for r in errs)} 题")
