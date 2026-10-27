#!/usr/bin/env python3
"""
Aggregate the akg_v2 results (results_v2/) into tables used by the paper.

Inputs (all under results_v2/):
  {model}_{project}_{task}.json          Level 0  (ci, ci_hard, dp, ar)
  level1/{model}_{project}_{task}_l1.json Level 1  (ci_hard, ci_obfuscated, dp, ar)
  level2/{model}_{project}_{task}_l2.json Level 2  (ci_hard, dp, ar)
  baselines_fixed.json                   non-LLM baselines (per_system + macro)

Outputs (results_v2/summary/):
  summary_long.csv        one row per (level, model, project, task)
  summary_macro.csv       macro average across the 12 systems
  summary_domain.csv      macro average per domain
  summary_deltas.csv      L1/L2 minus L0 for the two frontier models
  summary_v2.md           human-readable digest

Macro averaging uses the same rule as the paper: unweighted mean over systems
with a result file, so systems with fewer AR items (e.g. NASTRAN) count once.
"""

import csv
import glob
import json
import os
import statistics

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(EXP, "results_v2")
OUT = os.path.join(ROOT, "summary")

PROJECTS = [
    "openfoam", "nastran", "salome", "paraview", "kubernetes", "linux",
    "hadoop", "spring", "codeaster", "openmodelica", "tensorflow", "pytorch",
]

SYSTEM_NAMES = {
    "openfoam": "OpenFOAM", "nastran": "NASTRAN-95", "salome": "SALOME",
    "paraview": "ParaView", "kubernetes": "Kubernetes", "linux": "Linux Kernel",
    "hadoop": "Apache Hadoop", "spring": "Spring Framework",
    "codeaster": "Code_Aster", "openmodelica": "OpenModelica",
    "tensorflow": "TensorFlow", "pytorch": "PyTorch",
}

MODEL_NAMES = {
    "gpt-6-sol": "GPT-6 Sol", "claude-opus-5.5": "Claude Opus 5.5",
    "deepseek-v4-flash": "DeepSeek-V4-Flash", "qwen3": "Qwen3-235B-A22B",
    "llama-4-maverick": "Llama-4-Maverick", "gpt-oss-120b": "GPT-OSS-120B",
    "kimi-k3": "Kimi-K3", "llama-3.1-8b": "Llama-3.1-8B",
    "qwen3.8-27b": "Qwen3.8-27B",
}
MODELS = list(MODEL_NAMES)

DOMAINS = {
    "Industrial Simulation": ["openfoam", "nastran", "salome", "paraview",
                              "codeaster", "openmodelica"],
    "Infrastructure Systems": ["kubernetes", "linux", "hadoop"],
    "Enterprise": ["spring"],
    "ML Infrastructure": ["tensorflow", "pytorch"],
}

TASKS = {
    0: ["ci", "ci_hard", "dp", "ar"],
    1: ["ci_hard", "ci_obfuscated", "dp", "ar"],
    2: ["ci_hard", "dp", "ar"],
}

# metric reported per task, plus secondary metrics
PRIMARY = {"ci": "accuracy", "ci_hard": "accuracy",
           "ci_obfuscated": "accuracy", "dp": "f1", "ar": "precision_at_5"}
SECONDARY = {"dp": ["accuracy", "precision", "recall"],
             "ar": ["recall_at_5"]}


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def scan(level):
    """{(model, project, task): record} for one level."""
    sub = {0: "", 1: "level1", 2: "level2"}[level]
    suffix = {0: "", 1: "_l1", 2: "_l2"}[level]
    out = {}
    for f in glob.glob(os.path.join(ROOT, sub, "*.json")):
        base = os.path.basename(f)[:-5]
        if suffix:
            if not base.endswith(suffix):
                continue
            base = base[:-len(suffix)]
        model = next((m for m in MODELS if base.startswith(m + "_")), None)
        if model is None:
            continue
        rest = base[len(model) + 1:]
        task = next((t for t in sorted(TASKS[level], key=len, reverse=True)
                     if rest.endswith("_" + t)), None)
        if task is None:
            continue
        project = rest[:-len(task) - 1]
        if project not in PROJECTS:
            continue
        out[(model, project, task)] = read_json(f)
    return out


def metrics(rec):
    m = {}
    for k in ("accuracy", "precision", "recall", "f1",
              "precision_at_5", "recall_at_5", "correct", "total",
              "n_api_errors", "n_retried"):
        if k in rec:
            m[k] = rec[k]
    return m


def macro(values):
    return statistics.mean(values) if values else None


def collect():
    rows = []
    for level in (0, 1, 2):
        data = scan(level)
        for (model, project, task), rec in sorted(data.items()):
            m = metrics(rec)
            rows.append({
                "level": level, "model": model, "project": project,
                "system": SYSTEM_NAMES[project], "task": task,
                "primary_metric": PRIMARY[task],
                "primary": m.get(PRIMARY[task]),
                **{k: v for k, v in m.items() if k != PRIMARY[task]},
                "n_systems": 1,
            })
    return rows


def add_baselines(rows):
    """Non-LLM baselines from baselines_fixed.json (per-system records)."""
    path = os.path.join(ROOT, "baselines_fixed.json")
    if not os.path.exists(path):
        return rows
    per = read_json(path).get("per_system", {})
    for project, tasks in per.items():
        for task, methods in tasks.items():
            if task not in PRIMARY:
                continue
            for method, rec in methods.items():
                if method.endswith("_unmasked"):     # leaky, not reported
                    continue
                if task == "ar":
                    # one-hop neighbours cannot contain a held-out target (scores 0
                    # by construction); the reported AR baseline is the two-hop rule
                    if method != "graph_two_hop" and method != "frequency" \
                            and method != "random":
                        continue
                    if method == "graph_two_hop":
                        method = "graph_neighbor"
                rows.append({
                    "level": "baseline", "model": method, "project": project,
                    "system": SYSTEM_NAMES.get(project, project), "task": task,
                    "primary_metric": PRIMARY[task],
                    "primary": rec.get(PRIMARY[task]), "n_systems": 1,
                    "total": rec.get("n"),
                    "recall_at_5": rec.get("recall_at_5") if task == "ar" else None,
                })
    return rows


def aggregate(rows, key_fields):
    groups = {}
    for r in rows:
        k = tuple(r[f] for f in key_fields)
        groups.setdefault(k, []).append(r)
    out = []
    for k, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        prim = [r["primary"] for r in rs if r.get("primary") is not None]
        api = sum(r.get("n_api_errors") or 0 for r in rs)
        retried = sum(r.get("n_retried") or 0 for r in rs)
        totals = [r.get("total") for r in rs if r.get("total")]
        row = dict(zip(key_fields, k))
        row.update({
            "macro": round(macro(prim), 4) if prim else None,
            "n_systems": len(rs),
            "n_items": sum(totals) if totals else None,
            "api_errors": api,
            "n_retried": retried,
        })
        for extra in ("accuracy", "precision", "recall", "f1",
                      "recall_at_5"):
            vals = [r[extra] for r in rs if r.get(extra) is not None]
            if vals:
                row[extra] = round(macro(vals), 4)
        if totals and all(r.get("correct") is not None for r in rs):
            row["pooled"] = round(
                sum(r["correct"] for r in rs) / sum(r["total"] for r in rs), 4)
        out.append(row)
    return out


def write_csv(path, rows, fields=None):
    if not rows:
        return
    if fields is None:                      # union of keys, first-seen order
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fmt(v, pct=False):
    if v is None:
        return "--"
    return f"{v * 100:.1f}" if pct else f"{v:.3f}"


def fmt_signed(v):
    """差值列统一用 pt、一位小数并带符号，避免与 0-1 区间的原始分数混淆"""
    if v is None:
        return "--"
    return f"{v:+.1f}"


def md_table(headers, body):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = add_baselines(collect())
    llm = [r for r in rows if r["level"] != "baseline"]

    macro_rows = aggregate(rows, ["level", "model", "task"])
    domain_rows = aggregate(
        [dict(r, domain=next(d for d, ps in DOMAINS.items()
                             if r["project"] in ps)) for r in llm],
        ["level", "model", "domain", "task"])

    write_csv(os.path.join(OUT, "summary_long.csv"), rows)
    write_csv(os.path.join(OUT, "summary_macro.csv"), macro_rows)
    write_csv(os.path.join(OUT, "summary_domain.csv"), domain_rows)

    # L1 / L2 deltas against L0, per model and task
    l0 = {(r["model"], r["task"]): r["macro"] for r in macro_rows
          if r["level"] == 0}
    deltas = []
    for r in macro_rows:
        if r["level"] in (1, 2):
            ref = l0.get((r["model"], r["task"]))
            deltas.append({
                "model": r["model"], "task": r["task"], "level": r["level"],
                "macro": r["macro"], "L0_macro": ref,
                "delta_pt": round((r["macro"] - ref) * 100, 2)
                if r["macro"] is not None and ref is not None else None,
            })
    write_csv(os.path.join(OUT, "summary_deltas.csv"), deltas)

    # human-readable digest
    md = ["# akg_v2 result summary", "",
          "Generated by `experiment/summarize_results.py`; macro = unweighted "
          "mean over systems with a result file.", ""]

    md += ["## Coverage and API errors", "",
           "`Items` is the total number of scored items at that level across all "
           "tasks present (L0/L1: CI + CI-Hard + DP + AR, hence 7357 per model; "
           "L2: CI-Hard + DP + AR, hence 4957).", ""]
    body = []
    for level in (0, 1, 2):
        for model in MODELS:
            rs = [r for r in llm if r["level"] == level and r["model"] == model]
            if not rs:
                continue
            body.append([level, MODEL_NAMES[model], len(rs),
                         sum(r.get("total") or 0 for r in rs),
                         sum(r.get("n_api_errors") or 0 for r in rs),
                         sum(r.get("n_retried") or 0 for r in rs)])
    md += [md_table(["Level", "Model", "Files", "Items", "API errors",
                     "Retried"], body), ""]

    md += ["## Level 0 macro average", ""]
    body = []
    for model in MODELS + ["random", "frequency", "path_heuristic",
                           "graph_neighbor"]:
        sel = {r["task"]: r for r in macro_rows
               if str(r["level"]) == "0" and r["model"] == model}
        bsel = {r["task"]: r for r in macro_rows
                if str(r["level"]) == "baseline" and r["model"] == model}
        sel = sel or bsel
        if not sel:
            continue
        name = MODEL_NAMES.get(model, model)
        body.append([name,
                     fmt(sel.get("ci", {}).get("macro"), True),
                     fmt(sel.get("ci_hard", {}).get("macro"), True),
                     fmt(sel.get("dp", {}).get("macro")),
                     fmt(sel.get("ar", {}).get("macro"), True),
                     fmt(sel.get("ar", {}).get("recall_at_5"), True)])
    md += [md_table(["Model", "CI %", "CI-Hard %", "DP F1", "AR P@5 %",
                     "AR R@5 %"], body), ""]

    md += ["## Level 1 macro average (two frontier models)", ""]
    body = []
    for model in ("claude-opus-5.5", "gpt-6-sol"):
        sel = {r["task"]: r for r in macro_rows
               if str(r["level"]) == "1" and r["model"] == model}
        body.append([MODEL_NAMES[model],
                     fmt(sel.get("ci_hard", {}).get("macro"), True),
                     fmt(sel.get("ci_obfuscated", {}).get("macro"), True),
                     fmt(sel.get("dp", {}).get("macro")),
                     fmt(sel.get("ar", {}).get("macro"), True),
                     fmt(sel.get("ar", {}).get("recall_at_5"), True)])
    md += [md_table(["Model", "CI-Hard %", "CI-Obf. %", "DP F1", "AR P@5 %",
                     "AR R@5 %"], body), ""]

    md += ["## Level 2 macro average (three models)", ""]
    body = []
    for model in ("claude-opus-5.5", "gpt-6-sol", "deepseek-v4-flash"):
        sel = {r["task"]: r for r in macro_rows
               if str(r["level"]) == "2" and r["model"] == model}
        body.append([MODEL_NAMES[model],
                     fmt(sel.get("ci_hard", {}).get("macro"), True),
                     fmt(sel.get("dp", {}).get("macro")),
                     fmt(sel.get("ar", {}).get("macro"), True),
                     fmt(sel.get("ar", {}).get("recall_at_5"), True)])
    md += [md_table(["Model", "CI-Hard %", "DP F1", "AR P@5 %", "AR R@5 %"],
                    body), ""]

    md += ["## Level 1 / Level 2 minus Level 0 (macro, percentage points)", ""]
    body = []
    for r in deltas:
        body.append([MODEL_NAMES.get(r["model"], r["model"]),
                     f"Level {r['level']}", r["task"],
                     fmt(r["L0_macro"]), fmt(r["macro"]),
                     fmt_signed(r["delta_pt"])])
    md += [md_table(["Model", "Level", "Task", "L0", "L1/L2", "Δ (pt)"],
                    body), ""]

    md += ["## Domain macro average (Level 0)", ""]
    body = []
    for domain in DOMAINS:
        vals = {}
        for task in ("ci", "ci_hard", "dp", "ar"):
            xs = [r["macro"] for r in domain_rows
                  if str(r["level"]) == "0" and r["domain"] == domain
                  and r["task"] == task and r.get("macro") is not None]
            vals[task] = macro(xs)
        body.append([domain, len(DOMAINS[domain]),
                     fmt(vals.get("ci"), True), fmt(vals.get("ci_hard"), True),
                     fmt(vals.get("dp")), fmt(vals.get("ar"), True)])
    md += [md_table(["Domain", "Systems", "CI %", "CI-Hard %", "DP F1",
                     "AR P@5 %"], body), ""]

    with open(os.path.join(OUT, "summary_v2.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
