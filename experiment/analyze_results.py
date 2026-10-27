"""
结果分析与汇总脚本

用法:
    python analyze_results.py
"""

import json
import os
from collections import defaultdict

RESULTS_DIR = "results"


def load_baselines():
    path = os.path.join(RESULTS_DIR, "baselines.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_llm_results():
    """加载所有 LLM 结果文件"""
    results = defaultdict(lambda: defaultdict(dict))

    for fname in os.listdir(RESULTS_DIR):
        if fname == "baselines.json" or not fname.endswith(".json"):
            continue

        parts = fname.replace(".json", "").split("_")
        # model_project_task.json -> 需要解析
        # 格式: gpt-4o_openfoam_ci.json
        task = parts[-1]
        project = parts[-2]
        model = "_".join(parts[:-2])

        with open(os.path.join(RESULTS_DIR, fname), encoding="utf-8") as f:
            data = json.load(f)

        results[model][project][task] = data

    return results


def print_baseline_table(baselines):
    """打印 baseline 结果表"""
    print("\n" + "=" * 100)
    print("BASELINE RESULTS")
    print("=" * 100)

    # CI
    print("\n--- Component Identification (Accuracy) ---")
    print(f"{'Project':18s} {'Random':>8s} {'Frequency':>10s} {'PathHeur':>10s}")
    print("-" * 50)
    for proj, data in baselines.items():
        r = data.get("ci_random", {}).get("accuracy", 0)
        f = data.get("ci_frequency", {}).get("accuracy", 0)
        h = data.get("ci_path_heuristic", {}).get("accuracy", 0)
        print(f"{proj:18s} {r:8.4f} {f:10.4f} {h:10.4f}")

    # DP
    print("\n--- Dependency Prediction (F1) ---")
    print(f"{'Project':18s} {'Random':>8s} {'Frequency':>10s} {'GraphNbr':>10s}")
    print("-" * 50)
    for proj, data in baselines.items():
        r = data.get("dp_random", {}).get("f1", 0)
        f = data.get("dp_frequency", {}).get("f1", 0)
        g = data.get("dp_graph_neighbor", {}).get("f1", 0)
        print(f"{proj:18s} {r:8.4f} {f:10.4f} {g:10.4f}")

    # AR
    print("\n--- Architecture Reconstruction (P@5) ---")
    print(f"{'Project':18s} {'Random':>8s} {'Frequency':>10s} {'GraphNbr':>10s}")
    print("-" * 50)
    for proj, data in baselines.items():
        r = data.get("ar_random", {}).get("precision_at_5", 0)
        f = data.get("ar_frequency", {}).get("precision_at_5", 0)
        g = data.get("ar_graph_neighbor", {}).get("precision_at_5", 0)
        print(f"{proj:18s} {r:8.4f} {f:10.4f} {g:10.4f}")


def print_llm_table(llm_results, baselines):
    """打印 LLM vs Baseline 对比表"""
    if not llm_results:
        print("\nNo LLM results found. Run experiments first.")
        return

    print("\n" + "=" * 100)
    print("LLM vs BASELINE COMPARISON")
    print("=" * 100)

    models = sorted(llm_results.keys())
    projects = sorted(set(p for m in llm_results.values() for p in m.keys()))

    # CI comparison
    print("\n--- Component Identification (Accuracy) ---")
    header = f"{'Project':18s} {'PathHeur':>10s}"
    for m in models:
        header += f" {m:>14s}"
    print(header)
    print("-" * len(header))

    for proj in projects:
        bl = baselines.get(proj, {}).get("ci_path_heuristic", {}).get("accuracy", 0)
        row = f"{proj:18s} {bl:10.4f}"
        for m in models:
            acc = llm_results[m].get(proj, {}).get("ci", {}).get("accuracy", -1)
            row += f" {acc:14.4f}" if acc >= 0 else f" {'N/A':>14s}"
        print(row)


def generate_latex_table(baselines, llm_results):
    """生成 LaTeX 表格（论文用）"""
    print("\n" + "=" * 100)
    print("LATEX TABLE (for paper)")
    print("=" * 100)

    print(r"""
\begin{table*}[t]
\centering
\caption{Component Identification Accuracy across 12 industrial systems}
\label{tab:ci_results}
\begin{tabular}{l|ccc|ccccc}
\toprule
\textbf{System} & \textbf{Random} & \textbf{Freq.} & \textbf{Path} & \textbf{GPT-4o} & \textbf{Claude} & \textbf{DeepSeek} & \textbf{Llama} & \textbf{Qwen} \\
\midrule""")

    models = ["gpt-4o", "claude-3.5-sonnet", "deepseek-v3", "llama-3.1-70b", "qwen2.5-72b"]

    for proj, data in baselines.items():
        r = data.get("ci_random", {}).get("accuracy", 0)
        f = data.get("ci_frequency", {}).get("accuracy", 0)
        h = data.get("ci_path_heuristic", {}).get("accuracy", 0)

        row = f"{proj} & {r:.2f} & {f:.2f} & {h:.2f}"
        for m in models:
            acc = llm_results.get(m, {}).get(proj, {}).get("ci", {}).get("accuracy", -1)
            row += f" & {acc:.2f}" if acc >= 0 else " & --"
        row += r" \\"
        print(row)

    print(r"""\bottomrule
\end{tabular}
\end{table*}""")


def main():
    baselines = load_baselines()
    llm_results = load_llm_results()

    print_baseline_table(baselines)
    print_llm_table(llm_results, baselines)
    generate_latex_table(baselines, llm_results)


if __name__ == "__main__":
    main()
