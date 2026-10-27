# -*- coding: utf-8 -*-
"""akg_ai_pilot_agreement.py

AKG 标签验证试跑：人工金标 vs 双 LLM 辅助标注模型的对比与统计。

输入（均在 experiment/ 下）:
  akg_ai_pilot_blind.csv          人工盲审表（每条边多行示例，标签只记在 ex1 行）
  akg_ai_pilot_model_labels.csv   两个模型的独立标注（gemini-4-argon / claude-sonnet-5.5）
  akg_ai_pilot_strata.csv         分层信息（system / audit_category / link_verdicts）
  akg_ai_pilot_exclude.json       排除清单；仅从「模型 vs 人工」一致性计算中剔除，金标精确率与分层统计仍计入
  akg_ai_edge_category.json       200 条候选边池（category / system），用作后分层加权的总体参考

输出:
  akg_ai_pilot_agreement.json     机器可读结果
  akg_ai_pilot_agreement.md       人类可读报告

统计口径依据 akg_annotation_guide.md 第 3 节:
  边精确率 = Correct / (Correct + Wrong-resolution + Spurious)；Granularity / Unclear 单列、不进分母；
  95% CI 用 Wilson；同时按系统和按审计类别报告。试跑按「审计类别 x 系统」分层，故另给后分层加权估计。

用法:
  python experiment/akg_ai_pilot_agreement.py
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 五类标签（akg_annotation_guide.md 第 1 节）
LABELS = ["Correct", "Wrong-resolution", "Spurious", "Granularity", "Unclear"]
# 边精确率的参与类别：Granularity / Unclear 单列，不计入分母
PRECISION_LABELS = ["Correct", "Wrong-resolution", "Spurious"]

Z95 = 1.959963984540054
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson 95% 置信区间。"""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def cohen_kappa(pairs: list[tuple[str, str]], labels: list[str]) -> float:
    """两标注者在给定标签集合上的 Cohen's kappa。"""
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca.get(l, 0) / n) * (cb.get(l, 0) / n) for l in labels)
    if abs(1.0 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def precision_of(golds: list[str]) -> dict:
    """边精确率 + Wilson CI；同时给出 Granularity / Unclear 单列计数。"""
    counts = Counter(golds)
    denom = sum(counts.get(l, 0) for l in PRECISION_LABELS)
    k = counts.get("Correct", 0)
    lo, hi = wilson_ci(k, denom)
    out = {
        "n": len(golds),
        "counts": {l: counts.get(l, 0) for l in LABELS},
        "denominator": denom,
        "precision": (k / denom) if denom else float("nan"),
        "wilson95": [lo, hi],
    }
    return out


def pct(x: float) -> str:
    return "n/a" if (x is None or math.isnan(x)) else f"{100 * x:.1f}%"


def rate_ci(k: int, n: int) -> str:
    lo, hi = wilson_ci(k, n)
    if math.isnan(lo):
        return "n/a"
    return f"{100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def main() -> None:
    # ---------------- 读入 ----------------
    blind = read_csv(HERE / "akg_ai_pilot_blind.csv")
    model_rows = read_csv(HERE / "akg_ai_pilot_model_labels.csv")
    strata_rows = read_csv(HERE / "akg_ai_pilot_strata.csv")
    exclude = json.loads((HERE / "akg_ai_pilot_exclude.json").read_text(encoding="utf-8"))
    pool = json.loads((HERE / "akg_ai_edge_category.json").read_text(encoding="utf-8"))

    excluded_pids = set(exclude.get("exclude_from_agreement", []))

    # 人工金标：每个边只在 ex1 行记录一次标签
    gold: dict[str, str] = {}
    gold_conf: dict[str, str] = {}
    gold_comment: dict[str, str] = {}
    gold_weight: dict[str, str] = {}
    multi_labeled = []
    for row in blind:
        pid = (row.get("pilot_id") or "").strip()
        if not pid:
            continue
        label = (row.get("label_blind") or "").strip()
        if not label:
            continue
        if pid in gold:
            multi_labeled.append(pid)
            continue
        gold[pid] = label
        gold_conf[pid] = (row.get("confidence_1to3") or "").strip()
        gold_comment[pid] = (row.get("comment") or "").strip()
        gold_weight[pid] = (row.get("weight") or "").strip()

    # 模型标注
    model_labels: dict[tuple[str, str], str] = {}
    model_parse_fail: list[tuple[str, str]] = []
    models: list[str] = []
    for row in model_rows:
        pid = (row.get("pilot_id") or "").strip()
        model = (row.get("model") or "").strip()
        if not pid or not model:
            continue
        if model not in models:
            models.append(model)
        if (row.get("parse_ok") or "").strip() == "1":
            model_labels[(pid, model)] = (row.get("label") or "").strip()
        else:
            model_parse_fail.append((pid, model))

    # 分层信息
    strata: dict[str, dict] = {}
    for row in strata_rows:
        pid = (row.get("pilot_id") or "").strip()
        if not pid:
            continue
        strata[pid] = {
            "sample_id": (row.get("sample_id") or "").strip(),
            "system": (row.get("system") or "").strip(),
            "audit_category": (row.get("audit_category") or "").strip(),
            "link_verdicts": json.loads(row.get("link_verdicts") or "{}"),
        }

    all_pids = sorted(gold)
    # 排除清单只作用于「模型 vs 人工」的一致性计算（见 akg_ai_pilot_exclude.json 的键名
    # exclude_from_agreement）；金标分布、边精确率与分层统计仍使用全部试跑边。
    included = list(all_pids)
    agreement_pids = [p for p in all_pids if p not in excluded_pids]
    pool_cat_counts = Counter(v.get("category", "") for v in pool.values())
    pool_n = sum(pool_cat_counts.values())

    # ---------------- 人工金标统计 ----------------
    human = precision_of([gold[p] for p in included])
    # 同一批边剔除排除样本后的对照值（供论文按更保守的口径引用）
    human_agree_subset = precision_of([gold[p] for p in agreement_pids])

    by_system: dict[str, list[str]] = defaultdict(list)
    by_category: dict[str, list[str]] = defaultdict(list)
    pilot_cat_counts: Counter = Counter()
    for p in included:
        st = strata.get(p, {})
        by_system[st.get("system", "?")].append(gold[p])
        by_category[st.get("audit_category", "?")].append(gold[p])
        pilot_cat_counts[st.get("audit_category", "?")] += 1

    system_stats = {s: precision_of(v) for s, v in sorted(by_system.items())}
    category_stats = {c: precision_of(v) for c, v in sorted(by_category.items())}

    # 后分层加权（按审计类别）: 权重取 200 边候选池的类别占比
    strata_weighted = []
    covered = 0
    for cat, stats in category_stats.items():
        w = pool_cat_counts.get(cat, 0) / pool_n if pool_n else 0.0
        if stats["denominator"] > 0 and w > 0:
            strata_weighted.append((w, stats["precision"]))
            covered += w
    post_strat = (
        sum(w * pr for w, pr in strata_weighted) / covered if covered else float("nan")
    )

    # 后分层估计的 bootstrap CI：在每个审计类别层内重采样
    cat_members = {c: [gold[p] for p in included if strata.get(p, {}).get("audit_category") == c]
                   for c in category_stats}
    rng = random.Random(BOOTSTRAP_SEED)
    boots = []
    for _ in range(BOOTSTRAP_N):
        vals = []
        for cat, w in ((c, pool_cat_counts.get(c, 0) / pool_n) for c in category_stats):
            if w <= 0 or not cat_members[cat]:
                continue
            sample = [rng.choice(cat_members[cat]) for _ in cat_members[cat]]
            pr = precision_of(sample)["precision"]
            if math.isnan(pr):
                # 重采样后该层全是 Granularity/Unclear，分母为 0；跳过而非污染均值
                continue
            vals.append((w, pr))
        num = sum(w * pr for w, pr in vals)
        den = sum(w for w, _ in vals)
        if den:
            boots.append(num / den)
    boots.sort()
    boot_ci = (
        (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]) if boots else (float("nan"), float("nan"))
    )

    # ---------------- 模型 vs 人工 ----------------
    model_stats = {}
    for model in models:
        pairs = [(gold[p], model_labels[(p, model)]) for p in agreement_pids if (p, model) in model_labels]
        agree = sum(1 for a, b in pairs if a == b)
        conf = Counter()
        for a, b in pairs:
            conf[(a, b)] += 1
        model_stats[model] = {
            "n": len(pairs),
            "agree": agree,
            "agree_rate": agree / len(pairs) if pairs else float("nan"),
            "wilson95": list(wilson_ci(agree, len(pairs))),
            "kappa": cohen_kappa(pairs, LABELS),
            "confusion": {f"{a} -> {b}": c for (a, b), c in sorted(conf.items())},
            "mismatches": sorted(
                p for p in agreement_pids
                if (p, model) in model_labels and gold[p] != model_labels[(p, model)]
            ),
        }

    # 双模型一致性
    pair_pairs = [
        (model_labels[(p, models[0])], model_labels[(p, models[1])])
        for p in agreement_pids
        if (p, models[0]) in model_labels and (p, models[1]) in model_labels
    ]
    pair_agree = sum(1 for a, b in pair_pairs if a == b)
    model_model = {
        "models": models,
        "n": len(pair_pairs),
        "agree": pair_agree,
        "agree_rate": pair_agree / len(pair_pairs) if pair_pairs else float("nan"),
        "wilson95": list(wilson_ci(pair_agree, len(pair_pairs))),
        "kappa": cohen_kappa(pair_pairs, LABELS),
    }

    # ---------------- 输出 ----------------
    result = {
        "n_pilot": len(all_pids),
        "n_gold_scored": len(included),
        "n_agreement": len(agreement_pids),
        "excluded_from_agreement": sorted(excluded_pids),
        "exclude_reason": exclude.get("reason", ""),
        "models": models,
        "model_parse_failures": model_parse_fail,
        "missing_gold_confidence": [p for p in all_pids if not gold_conf.get(p)],
        "human_gold": human,
        "human_gold_excluding_agreement_outliers": human_agree_subset,
        "human_gold_by_system": system_stats,
        "human_gold_by_audit_category": category_stats,
        "post_stratified_precision": {
            "estimate": post_strat,
            "bootstrap95": list(boot_ci),
            "bootstrap_degenerate": bool(boot_ci[0] == boot_ci[1]),
            "weights_from": "akg_ai_edge_category.json (200-edge candidate pool)",
            "stratum_weights": {c: pool_cat_counts.get(c, 0) / pool_n for c in category_stats},
            "pilot_share": {c: pilot_cat_counts[c] / len(included) for c in category_stats},
        },
        "model_vs_human": model_stats,
        "model_vs_model": model_model,
    }

    (HERE / "akg_ai_pilot_agreement.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # markdown 报告
    lines: list[str] = []
    a = lines.append
    a("# AKG 标签验证试跑：人工金标 vs 双模型对比与统计\n")
    a(f"- 试跑边数：{len(all_pids)}；进入金标精确率/分层统计：{len(included)}；"
      f"进入「模型 vs 人工」一致性统计：{len(agreement_pids)}"
      f"（一致性统计排除 {', '.join(sorted(excluded_pids)) or '无'}）")
    a(f"- 模型：{', '.join(models)}；解析失败：{len(model_parse_fail)}")
    a(f"- 缺失置信度：{', '.join(p for p in all_pids if not gold_conf.get(p)) or '无'}")
    a(f"- 排除理由：{exclude.get('reason', '')}\n")

    a("## 1. 人工金标分布与边精确率\n")
    a("| 标签 | 数 |")
    a("|---|---|")
    for l in LABELS:
        a(f"| {l} | {human['counts'][l]} |")
    a(f"\n边精确率 = Correct / (Correct + Wrong-resolution + Spurious) = "
      f"{human['counts']['Correct']}/{human['denominator']} = {pct(human['precision'])} "
      f"(Wilson 95% CI [{pct(human['wilson95'][0])}, {pct(human['wilson95'][1])}])")
    a(f"Granularity / Unclear 单列：{human['counts']['Granularity']} / {human['counts']['Unclear']}（不进分母）")
    a(f"对照（剔除一致性排除样本 {', '.join(sorted(excluded_pids)) or '无'} 后）："
      f"{human_agree_subset['counts']['Correct']}/{human_agree_subset['denominator']} = "
      f"{pct(human_agree_subset['precision'])} "
      f"(Wilson 95% CI [{pct(human_agree_subset['wilson95'][0])}, {pct(human_agree_subset['wilson95'][1])}])\n")

    a("## 2. 按系统\n")
    a("| 系统 | n | Correct | Wrong-res | Spurious | Granularity | 边精确率 |")
    a("|---|---|---|---|---|---|---|")
    for s, st in system_stats.items():
        a(f"| {s} | {st['n']} | {st['counts']['Correct']} | {st['counts']['Wrong-resolution']} "
          f"| {st['counts']['Spurious']} | {st['counts']['Granularity']} | {pct(st['precision'])} |")

    a("\n## 3. 按审计类别（后分层加权）\n")
    a("| 审计类别 | 试跑 n | 池占比 | Correct | 边精确率 |")
    a("|---|---|---|---|---|")
    for c, st in category_stats.items():
        w = pool_cat_counts.get(c, 0) / pool_n * 100
        a(f"| {c} | {st['n']} | {w:.1f}% | {st['counts']['Correct']}/{st['denominator']} | {pct(st['precision'])} |")
    a(f"\n后分层加权精确率（按审计类别、权重取自 200 边候选池）= **{pct(post_strat)}** "
      f"(bootstrap 95% CI [{pct(boot_ci[0])}, {pct(boot_ci[1])}])")
    if boot_ci[0] == boot_ci[1]:
        a("> 注：本次试跑各审计类别层内边精确率在重采样下无变异（clear_error 层全为非 Correct，"
          "其余三层全为 Correct，Granularity 不计入分母），bootstrap CI 退化，只能报点估计；"
          "该加权值的区间需待全量样本才能给出。")
    a("> 试跑是等量分层抽样（每类 5 条），与候选池的真实类别占比差异很大，故池化值需用后分层加权。\n")

    a("## 4. 模型 vs 人工金标\n")
    a("| 模型 | n | 一致 | 一致率 (Wilson 95%) | Cohen's κ |")
    a("|---|---|---|---|---|")
    for m, st in model_stats.items():
        a(f"| {m} | {st['n']} | {st['agree']} | {rate_ci(st['agree'], st['n'])} | {st['kappa']:.3f} |")
    a("")
    for m, st in model_stats.items():
        a(f"### {m} 混淆矩阵（人工 → 模型）\n")
        a("| | " + " | ".join(LABELS) + " |")
        a("|---|" + "---|" * len(LABELS))
        for hl in LABELS:
            cells = [str(st["confusion"].get(f"{hl} -> {ml}", 0)) for ml in LABELS]
            a(f"| {hl} | " + " | ".join(cells) + " |")
        a("")

    a("## 5. 双模型一致性\n")
    a(f"- {models[0]} vs {models[1]}：{model_model['agree']}/{model_model['n']} = "
      f"{pct(model_model['agree_rate'])} "
      f"(Wilson 95% CI [{pct(model_model['wilson95'][0])}, {pct(model_model['wilson95'][1])}])，"
      f"κ = {model_model['kappa']:.3f}\n")

    a("## 6. 不一致明细（仅一致性统计所覆盖的边）\n")
    a("| 边 | 系统 | 审计类别 | 人工 | " + " | ".join(models) + " |")
    a("|---|---|---|---|" + "---|" * len(models))
    for p in agreement_pids:
        ml = [model_labels.get((p, m), "-") for m in models]
        if gold[p] != ml[0] or (len(ml) > 1 and gold[p] != ml[1]) or ml[0] != ml[1]:
            st = strata.get(p, {})
            a(f"| {p} | {st.get('system', '?')} | {st.get('audit_category', '?')} | {gold[p]} | " + " | ".join(ml) + " |")
    a("")

    a("## 7. 数据质量备注\n")
    if multi_labeled:
        a(f"- ⚠ 同一边被标注多次：{', '.join(sorted(set(multi_labeled)))}")
    if model_parse_fail:
        a(f"- ⚠ 模型解析失败：{model_parse_fail}")
    miss = [p for p in all_pids if not gold_conf.get(p)]
    if miss:
        a(f"- ⚠ 缺置信度：{', '.join(miss)}")
    a(f"- 盲审表 `weight` 列为该边在审计分组内的链接条数（如 P06=209、P10=1104），非抽样权重；本统计为边级、不加权。")
    a(f"- 排除样本（{', '.join(sorted(excluded_pids)) or '无'}）仅从一致性计算中剔除，其金标仍计入精确率与分层统计；"
      f"若论文采用更保守口径（也剔除该样本），边精确率取第 1 节的对照值。")
    a(f"- 结论：以上金标统计基于 {len(included)} 条边、一致性统计基于 {len(agreement_pids)} 条边；"
      f"样本量小，按系统的 CI 仅作方向参考。\n")

    (HERE / "akg_ai_pilot_agreement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
