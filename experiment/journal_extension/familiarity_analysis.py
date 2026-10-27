"""
Familiarity Analysis: 量化 LLM 表现与系统"知名度"的相关性

假设: LLM 在架构任务上的表现与该系统在训练数据中的出现频率正相关。
我们用 GitHub 指标和 StackOverflow 讨论量作为"知名度"的代理指标。

分析:
1. 收集每个系统的知名度指标
2. 计算知名度与模型表现的 Spearman 相关系数
3. 可视化相关性

用法:
    python familiarity_analysis.py
    python familiarity_analysis.py --update-metrics  # 更新 GitHub 指标
"""

import json
import os
import sys
import argparse

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROJECTS


# =====================================================
# 系统知名度指标 (手动收集或通过 API)
# =====================================================

# 这些数据需要手动更新或通过 GitHub API 获取
# 格式: {project_name: {metric: value}}
FAMILIARITY_METRICS = {
    "openfoam": {
        "github_stars": 1800,
        "github_forks": 900,
        "stackoverflow_tags": 2500,
        "google_scholar_hits": 50000,
        "first_release_year": 2004,
        "primary_language": "C++",
    },
    "nastran": {
        "github_stars": 200,
        "github_forks": 100,
        "stackoverflow_tags": 50,
        "google_scholar_hits": 30000,
        "first_release_year": 1968,
        "primary_language": "Fortran",
    },
    "salome": {
        "github_stars": 300,
        "github_forks": 150,
        "stackoverflow_tags": 200,
        "google_scholar_hits": 5000,
        "first_release_year": 2001,
        "primary_language": "C++",
    },
    "paraview": {
        "github_stars": 1500,
        "github_forks": 500,
        "stackoverflow_tags": 1500,
        "google_scholar_hits": 20000,
        "first_release_year": 2002,
        "primary_language": "C++",
    },
    "kubernetes": {
        "github_stars": 110000,
        "github_forks": 40000,
        "stackoverflow_tags": 80000,
        "google_scholar_hits": 15000,
        "first_release_year": 2014,
        "primary_language": "Go",
    },
    "linux": {
        "github_stars": 180000,
        "github_forks": 55000,
        "stackoverflow_tags": 200000,
        "google_scholar_hits": 500000,
        "first_release_year": 1991,
        "primary_language": "C",
    },
    "hadoop": {
        "github_stars": 14000,
        "github_forks": 8000,
        "stackoverflow_tags": 50000,
        "google_scholar_hits": 100000,
        "first_release_year": 2006,
        "primary_language": "Java",
    },
    "spring": {
        "github_stars": 56000,
        "github_forks": 38000,
        "stackoverflow_tags": 200000,
        "google_scholar_hits": 50000,
        "first_release_year": 2003,
        "primary_language": "Java",
    },
    "codeaster": {
        "github_stars": 100,
        "github_forks": 50,
        "stackoverflow_tags": 30,
        "google_scholar_hits": 3000,
        "first_release_year": 1989,
        "primary_language": "Fortran/C++",
    },
    "openmodelica": {
        "github_stars": 800,
        "github_forks": 300,
        "stackoverflow_tags": 500,
        "google_scholar_hits": 8000,
        "first_release_year": 1997,
        "primary_language": "Modelica/C++",
    },
    "tensorflow": {
        "github_stars": 185000,
        "github_forks": 74000,
        "stackoverflow_tags": 150000,
        "google_scholar_hits": 200000,
        "first_release_year": 2015,
        "primary_language": "Python/C++",
    },
    "pytorch": {
        "github_stars": 82000,
        "github_forks": 22000,
        "stackoverflow_tags": 80000,
        "google_scholar_hits": 150000,
        "first_release_year": 2016,
        "primary_language": "Python/C++",
    },
}


# =====================================================
# 分析函数
# =====================================================

def load_model_performance(results_dir, model_name):
    """加载模型在各项目上的表现"""
    performance = {}

    for proj in PROJECTS:
        proj_perf = {}

        # CI-Hard
        path = os.path.join(results_dir, f"{model_name}_{proj['name']}_ci_hard.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            proj_perf["ci_hard"] = data.get("accuracy", 0)

        # DP
        path = os.path.join(results_dir, f"{model_name}_{proj['name']}_dp.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            proj_perf["dp"] = data.get("accuracy", 0)

        # AR
        path = os.path.join(results_dir, f"{model_name}_{proj['name']}_ar.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            proj_perf["ar"] = data.get("precision_at_5", 0)

        if proj_perf:
            performance[proj["name"]] = proj_perf

    return performance


def compute_correlations(performance, metric_name="github_stars"):
    """计算知名度指标与模型表现的 Spearman 相关系数"""
    results = {}

    for task in ["ci_hard", "dp", "ar"]:
        familiarity_values = []
        performance_values = []

        for proj_name, perf in performance.items():
            if proj_name in FAMILIARITY_METRICS and task in perf:
                familiarity_values.append(
                    FAMILIARITY_METRICS[proj_name].get(metric_name, 0)
                )
                performance_values.append(perf[task])

        if len(familiarity_values) >= 5:
            rho, p_value = spearmanr(familiarity_values, performance_values)
            results[task] = {
                "spearman_rho": float(rho),
                "p_value": float(p_value),
                "n_systems": len(familiarity_values),
                "significant": bool(p_value < 0.05),
            }

    return results


def run_analysis(results_dir, models):
    """对多个模型运行相关性分析"""
    all_results = {}

    metrics_to_test = ["github_stars", "stackoverflow_tags", "google_scholar_hits"]

    for model in models:
        print(f"\nModel: {model}")
        performance = load_model_performance(results_dir, model)

        if not performance:
            print(f"  No results found, skipping")
            continue

        model_results = {}
        for metric in metrics_to_test:
            correlations = compute_correlations(performance, metric)
            model_results[metric] = correlations

            for task, corr in correlations.items():
                sig = "*" if corr["significant"] else "ns"
                print(f"  {metric:25s} x {task:8s}: "
                      f"rho={corr['spearman_rho']:+.3f} "
                      f"p={corr['p_value']:.4f} {sig}")

        all_results[model] = model_results

    return all_results


# =====================================================
# 主函数
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--models", nargs="+",
                        default=["deepseek-v4-flash", "claude-sonnet-4", "qwen3"])
    parser.add_argument("--output", default="familiarity_results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 80)
    print("FAMILIARITY BIAS ANALYSIS")
    print("=" * 80)
    print(f"\nHypothesis: LLM performance correlates with system popularity")
    print(f"Metric: Spearman rank correlation")

    all_results = run_analysis(args.results_dir, args.models)

    # 保存
    out_path = os.path.join(args.output, "familiarity_correlations.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {out_path}")

    # 汇总
    print("\n" + "=" * 80)
    print("SUMMARY: Is there a familiarity bias?")
    print("=" * 80)
    print("\nIf rho > 0 and p < 0.05: YES, more popular systems -> better performance")
    print("If rho ~ 0 or p > 0.05: NO significant correlation\n")


if __name__ == "__main__":
    main()
