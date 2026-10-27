#!/usr/bin/env python3
"""
Run the two annotation models (config.ANNOTATION_MODELS) on the pilot evidence pack.

The pack (akg_ai_pilot_evidence.jsonl) contains repository facts only: the edge, the
source statements, whether the resolved target exists on disk, and the candidate
paths. It carries no AKG verdict, no audit category and no auto_check column, so the
models judge without seeing what the tool concluded.

Output is written to disk and NOT printed, so the human annotator can keep filling
akg_ai_pilot_blind.csv without being anchored by the models. Compare the two only
after the blind pass is finished.

Outputs (experiment/):
  akg_ai_pilot_model_labels.csv   pilot_id, model, label, confidence, reason, parse_ok
  akg_ai_pilot_model_raw.jsonl    one line per call, raw text kept for auditing
"""

import csv
import json
import os
import re
import sys

import config
import llm_client

EXP = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(EXP, "akg_ai_pilot_evidence.jsonl")
OUT_LABELS = os.path.join(EXP, "akg_ai_pilot_model_labels.csv")
OUT_RAW = os.path.join(EXP, "akg_ai_pilot_model_raw.jsonl")

LABELS = ("Correct", "Wrong-resolution", "Spurious", "Granularity", "Unclear")

GUIDE = """你是一名软件架构依赖分析领域的审稿助手，任务是复核自动化工具 AKG 输出的一条模块级依赖边。

AKG 从源码中抽取 include / import / call 语句，把它们解析到具体的文件，再按文件所属模块聚合出模块级依赖边。你的工作是判断这条边是否成立。

## 判定口径（5 类，每条边给一个标签）
- Correct：至少一条示例链接确实把 source_module 中的文件静态依赖到 target_module 中的文件，且解析目标正确。
- Wrong-resolution：依赖语句真实存在，但解析到了错误的文件或模块（例如同名头文件、同名类）。
- Spurious：找不到真实的静态依赖语句，或者目标在项目外（标准库、第三方库、vendor），却被映射到了项目内。
- Granularity：依赖本身正确，但边端点的粒度不一致，例如父子模块，或 target 是包目录而 source 是模块。
- Unclear：信息不足、生成代码、条件编译等，无法确定。

## 判定要点
- C/C++：比较 #include 的路径前缀与解析目标是否一致；仅文件名相同不算正确。内核项目里 <linux/*.h> 应对应 include/linux/*.h。
- Java：比较 import 的全限定名与解析目标路径推导出的全限定名。
- Python：指向顶层包或标准库的导入（如 import torch、import logging）被解析到项目内的同名文件时，判 Spurious。
- Go：外部包被映射到 staging/src/... 下不存在的目录时判 Spurious；目录映射错位判 Wrong-resolution。
- Fortran：CALL X 对应 X.f 中的子程序，判 Correct。

## 纪律
1. 只能使用下面给出的仓库事实（语句原文、磁盘存在性、候选路径清单）。不得假设清单之外的任何文件或路径存在。
2. 候选路径清单列出的是该引用串在仓库中所有可能的目标；据此判断解析目标选得对不对。
3. 示例链接只是该边全部支撑链接的一个随机子集（最多 3 条），只能依据这些链接下结论。
4. 证据不足时判 Unclear，并在 reason 里说明缺什么。

## 输出格式
只输出一个 JSON 对象，不要任何解释文字、不要代码块标记：
{"label": "五类之一", "confidence": 1到3的整数, "reason": "一句话理由"}
"""


def build_prompt(rec):
    brief = {k: v for k, v in rec.items() if k != "pilot_id"}
    return (GUIDE + "\n## 待审边（仓库事实）\n"
            + json.dumps(brief, ensure_ascii=False, indent=1)
            + "\n\n请输出 JSON。")


def parse_label(text):
    """(label, confidence, reason, ok) — never treat an unparsable answer as a label."""
    if not text:
        return "", "", "", False
    s = text.strip()
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            d = None
        if isinstance(d, dict):
            lab = str(d.get("label", "")).strip()
            for L in LABELS:                       # tolerate case/spacing differences
                if lab.lower() == L.lower():
                    lab = L
                    break
            try:
                conf = int(d.get("confidence", 0))
            except (TypeError, ValueError):
                conf = 0
            return lab, conf, str(d.get("reason", "")).strip()[:300], lab in LABELS
    for L in LABELS:                           # fallback: bare label somewhere in the text
        if re.search(r"\b%s\b" % re.escape(L), s):
            return L, 0, s[:300], True
    return "", "", s[:300], False


def main():
    with open(PACK, encoding="utf-8") as f:
        packs = [json.loads(l) for l in f if l.strip()]
    models = list(config.ANNOTATION_MODELS.items())
    print(f"edges={len(packs)} models={[m for m, _ in models]}", flush=True)

    rows, raw = [], []
    for name, cfg in models:
        ok = fail = 0
        for rec in packs:
            pid, prompt = rec["pilot_id"], build_prompt(rec)
            try:
                res = llm_client.call_llm_with_usage(cfg, prompt, max_retries=3)
                text = res.get("text", "")
            except Exception as e:                        # API error: keep, do not parse
                text, res = "", {"error": f"{type(e).__name__}: {e}"}
            lab, conf, reason, parsed = parse_label(text)
            ok, fail = (ok + 1, fail) if parsed else (ok, fail + 1)
            rows.append([pid, name, lab, conf or "", reason, int(parsed)])
            raw.append({"pilot_id": pid, "model": name, "raw": text,
                        "usage": res.get("usage"), "error": res.get("error"),
                        "parsed": parsed})
            print(f"  {name} {pid} {'ok' if parsed else 'FAIL'}", flush=True)
        print(f"{name}: ok={ok} parse_failed={fail}", flush=True)

    with open(OUT_LABELS, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pilot_id", "model", "label", "confidence", "reason", "parse_ok"])
        w.writerows(rows)
    with open(OUT_RAW, "w", encoding="utf-8") as f:
        for r in raw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_LABELS} ({len(rows)} rows), {OUT_RAW}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
