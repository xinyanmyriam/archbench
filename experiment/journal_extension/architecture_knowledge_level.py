"""
Architecture Knowledge Level (AKL): 形式化定义与计算

形式化框架:
  我们定义 Architecture Knowledge Level (AKL) 来量化 LLM 对一个软件系统
  架构知识的掌握程度。

  AKL(M, S) = weighted combination of task-specific scores

  其中:
  - M = model
  - S = system
  - 各任务权重反映其对架构理解的贡献度

Level 定义:
  AKL-0 (No Knowledge):     所有任务接近随机基线
  AKL-1 (Surface Knowledge): CI-Easy 高, 但 CI-Hard/DP/AR 低
  AKL-2 (Partial Knowledge): CI-Hard 和 DP 中等, AR 低
  AKL-3 (Deep Knowledge):    所有任务都显著高于基线
  AKL-4 (Expert Knowledge):  接近 Graph Neighbor baseline 或超越

用法:
    python architecture_knowledge_level.py --results-dir ../results/
"""

import json
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROJECTS


# =====================================================
# AKL 形式化定义
# =====================================================

# 随机基线
RANDOM_BASELINES = {
    "ci": 0.25,       # 4-choice
    "ci_hard": 0.25,  # 4-choice
    "dp": 0.50,       # binary
    "ar": 0.05,       # open-ended, ~5% random
}

# 任务权重 (反映对架构理解的贡献度)
TASK_WEIGHTS = {
    "ci_hard": 0.30,   # 无路径提示的组件识别 - 核心能力
    "dp": 0.35,        # 依赖预测 - 最重要的架构推理
    "ar": 0.35,        # 架构重建 - 最难的综合能力
}

# AKL 等级阈值
AKL_THRESHOLDS = {
    0: 0.0,    # No Knowledge: normalized score < 0.1
    1: 0.1,    # Surface Knowledge: 0.1 <= score < 0.3
    2: 0.3,    # Partial Knowledge: 0.3 <= score < 0.6
    3: 0.6,    # Deep Knowledge: 0.6 <= score < 0.85
    4: 0.85,   # Expert Knowledge: score >= 0.85
}


def normalize_score(raw_score, task, baseline_score=None):
    """
    将原始分数归一化到 [0, 1]。

    归一化公式: (raw - random_baseline) / (1 - random_baseline)
    这样随机表现 = 0, 完美表现 = 1。
    """
    baseline = baseline_score if baseline_score is not None else RANDOM_BASELINES[task]
    if raw_score <= baseline:
        return 0.0
    normalized = (raw_score - baseline) / (1.0 - baseline)
    return min(max(normalized, 0.0), 1.0)


def compute_akl_score(ci_hard_acc, dp_acc, ar_p5):
    """
    计算 AKL 综合分数。

    AKL(M, S) = Σ w_t × normalize(score_t)
    """
    scores = {
        "ci_hard": normalize_score(ci_hard_acc, "ci_hard"),
        "dp": normalize_score(dp_acc, "dp"),
        "ar": normalize_score(ar_p5, "ar"),
    }

    weighted_sum = sum(
        TASK_WEIGHTS[task] * scores[task]
        for task in TASK_WEIGHTS
    )

    return weighted_sum, scores


def determine_akl_level(akl_score):
    """根据 AKL 分数确定等级"""
    for level in sorted(AKL_THRESHOLDS.keys(), reverse=True):
        if akl_score >= AKL_THRESHOLDS[level]:
            return level
    return 0


def get_level_description(level):
    """获取等级描述"""
    descriptions = {
        0: "No Knowledge - Performance at or below random baseline",
        1: "Surface Knowledge - Relies on superficial cues (paths, names)",
        2: "Partial Knowledge - Some architectural reasoning, limited to familiar patterns",
        3: "Deep Knowledge - Robust architectural understanding across tasks",
        4: "Expert Knowledge - Near-perfect, approaching structural oracle",
    }
    return descriptions.get(level, "Unknown")


# =====================================================
# 分析函数
# =====================================================

def load_results(results_dir, model_name, project_name):
    """加载一个模型-项目的所有任务结果"""
    results = {}

    # CI-Hard
    path = os.path.join(results_dir, f"{model_name}_{project_name}_ci_hard.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        results["ci_hard"] = data.get("accuracy", 0)

    # DP
    path = os.path.join(results_dir, f"{model_name}_{project_name}_dp.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        results["dp"] = data.get("accuracy", 0)

    # AR
    path = os.path.join(results_dir, f"{model_name}_{project_name}_ar.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        results["ar"] = data.get("precision_at_5", 0)

    return results


def compute_all_akl(results_dir, models):
    """计算所有模型-项目的 AKL"""
    all_akl = {}

    for model in models:
        model_akl = {}
        for proj in PROJECTS:
            results = load_results(results_dir, model, proj["name"])

            if len(results) >= 3:
                akl_score, component_scores = compute_akl_score(
                    results.get("ci_hard", 0),
                    results.get("dp", 0),
                    results.get("ar", 0),
                )
                level = determine_akl_level(akl_score)

                model_akl[proj["name"]] = {
                    "akl_score": round(akl_score, 4),
                    "akl_level": level,
                    "level_description": get_level_description(level),
                    "component_scores": {k: round(v, 4) for k, v in component_scores.items()},
                    "raw_scores": results,
                }

        if model_akl:
            # 计算模型平均 AKL
            avg_score = np.mean([v["akl_score"] for v in model_akl.values()])
            avg_level = determine_akl_level(avg_score)
            model_akl["_average"] = {
                "akl_score": round(float(avg_score), 4),
                "akl_level": avg_level,
                "level_description": get_level_description(avg_level),
            }
            all_akl[model] = model_akl

    return all_akl


# =====================================================
# 主函数
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--models", nargs="+",
                        default=["deepseek-v4-flash", "claude-sonnet-4", "qwen3",
                                 "llama-4-maverick", "gpt-oss-120b", "qwen3-8b",
                                 "llama-3.1-8b"])
    parser.add_argument("--output", default="akl_results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 90)
    print("ARCHITECTURE KNOWLEDGE LEVEL (AKL) ANALYSIS")
    print("=" * 90)
    print(f"\nFormula: AKL(M,S) = 0.30×CI-Hard_norm + 0.35×DP_norm + 0.35×AR_norm")
    print(f"Normalization: (raw - random_baseline) / (1 - random_baseline)")
    print(f"\nLevels: 0=None, 1=Surface, 2=Partial, 3=Deep, 4=Expert")

    all_akl = compute_all_akl(args.results_dir, args.models)

    # 打印汇总表
    print(f"\n{'Model':20s} {'Avg AKL':>8s} {'Level':>6s} | ", end="")
    print(" ".join(f"{p['name'][:8]:>8s}" for p in PROJECTS[:6]))
    print("-" * 90)

    for model, akl_data in all_akl.items():
        avg = akl_data.get("_average", {})
        row = f"{model:20s} {avg.get('akl_score', 0):8.3f} {'L' + str(avg.get('akl_level', 0)):>6s} | "
        for proj in PROJECTS[:6]:
            proj_akl = akl_data.get(proj["name"], {})
            score = proj_akl.get("akl_score", 0)
            row += f"{score:8.3f} "
        print(row)

    # 保存
    out_path = os.path.join(args.output, "akl_all_models.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_akl, f, indent=2)

    print(f"\nResults saved to {out_path}")

    # 打印 AKL 分布
    print("\n--- AKL Level Distribution ---")
    level_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for model, akl_data in all_akl.items():
        for proj_name, proj_akl in akl_data.items():
            if proj_name.startswith("_"):
                continue
            level = proj_akl.get("akl_level", 0)
            level_counts[level] += 1

    total = sum(level_counts.values())
    for level, count in sorted(level_counts.items()):
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  Level {level} ({get_level_description(level)[:20]:20s}): "
              f"{count:4d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
