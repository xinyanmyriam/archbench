"""
Error Analysis: 深入分析 LLM 在架构任务上的错误模式

分析维度:
1. 错误类型分类 (Error Taxonomy)
2. 按系统特征分析 (系统规模、语言、领域)
3. 按模块特征分析 (fan-in/fan-out、层级、命名模式)
4. 混淆矩阵分析 (哪些模块容易被混淆)

用法:
    python error_analysis.py --results-dir ../results/ --model deepseek-v4-flash
"""

import json
import os
import sys
import argparse
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROJECTS


# =====================================================
# Error Taxonomy
# =====================================================

class ErrorTaxonomy:
    """架构任务错误分类体系"""

    # CI/CI-Hard 错误类型
    CI_ERRORS = {
        "sibling_confusion": "预测了同一父模块下的兄弟模块",
        "parent_child_confusion": "预测了父模块或子模块",
        "cross_domain_confusion": "预测了完全不同领域的模块",
        "naming_similarity": "被相似命名误导",
        "parse_error": "模型输出无法解析",
    }

    # DP 错误类型
    DP_ERRORS = {
        "false_positive_popular": "对热门模块过度预测依赖 (false positive)",
        "false_negative_indirect": "漏掉间接/隐式依赖 (false negative)",
        "direction_confusion": "依赖方向判断错误",
        "cross_layer_miss": "跨架构层的依赖漏判",
    }

    @staticmethod
    def classify_ci_error(file_path, true_module, predicted_module, all_modules):
        """分类 CI 错误"""
        if predicted_module == "PARSE_ERROR" or predicted_module is None:
            return "parse_error"

        # 检查是否是兄弟模块
        true_parts = true_module.split("/")
        pred_parts = predicted_module.split("/")

        if len(true_parts) > 1 and len(pred_parts) > 1:
            if true_parts[:-1] == pred_parts[:-1]:
                return "sibling_confusion"

        # 检查父子关系
        if true_module.startswith(predicted_module + "/"):
            return "parent_child_confusion"
        if predicted_module.startswith(true_module + "/"):
            return "parent_child_confusion"

        # 检查命名相似性
        true_name = true_parts[-1].lower() if true_parts else ""
        pred_name = pred_parts[-1].lower() if pred_parts else ""
        file_name = os.path.basename(file_path).lower().split(".")[0]

        if file_name in pred_name or pred_name in file_name:
            return "naming_similarity"

        # 检查是否跨顶层目录
        if true_parts and pred_parts and true_parts[0] != pred_parts[0]:
            return "cross_domain_confusion"

        return "sibling_confusion"  # 默认

    @staticmethod
    def classify_dp_error(module_a, module_b, label, predicted, graph):
        """分类 DP 错误"""
        if label == 0 and predicted == 1:
            # False positive
            # 检查是否是热门模块
            fan_in_b = len([k for k, v in graph.items() if module_b in v])
            if fan_in_b > 5:
                return "false_positive_popular"
            return "false_positive_popular"

        elif label == 1 and predicted == 0:
            # False negative
            # 检查是否跨层
            a_parts = module_a.split("/")
            b_parts = module_b.split("/")
            if a_parts and b_parts and a_parts[0] != b_parts[0]:
                return "cross_layer_miss"
            return "false_negative_indirect"

        return "unknown"


# =====================================================
# 分析函数
# =====================================================

def analyze_ci_errors(results_path, project_path, project_name):
    """分析 CI-Hard 的错误模式"""
    if not os.path.exists(results_path):
        return None

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    with open(os.path.join(project_path, "modules_index.json"), encoding="utf-8") as f:
        modules = json.load(f)["modules"]

    errors = []
    error_types = Counter()

    for item in data.get("results", []):
        if not item.get("correct", True):
            error_type = ErrorTaxonomy.classify_ci_error(
                item.get("file", ""),
                item.get("label", ""),
                item.get("predicted_module", ""),
                modules,
            )
            error_types[error_type] += 1
            errors.append({
                "file": item.get("file", ""),
                "true_module": item.get("label", ""),
                "predicted_module": item.get("predicted_module", ""),
                "error_type": error_type,
            })

    total_errors = len(errors)
    total_samples = data.get("total", 1)

    return {
        "project": project_name,
        "total_samples": total_samples,
        "total_errors": total_errors,
        "error_rate": total_errors / total_samples if total_samples > 0 else 0,
        "error_type_distribution": dict(error_types),
        "error_type_percentages": {
            k: v / total_errors if total_errors > 0 else 0
            for k, v in error_types.items()
        },
        "sample_errors": errors[:20],  # 保留前20个示例
    }


def analyze_dp_errors(results_path, project_path, project_name):
    """分析 DP 的错误模式"""
    if not os.path.exists(results_path):
        return None

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    # 加载依赖图
    with open(os.path.join(project_path, "module_dependency.json"), encoding="utf-8") as f:
        dep_data = json.load(f)

    graph = defaultdict(set)
    for dep in dep_data.get("dependencies", []):
        graph[dep["source_module"]].add(dep["target_module"])

    errors = []
    error_types = Counter()

    for item in data.get("results", []):
        label = item.get("label", 0)
        predicted = item.get("predicted", 0)

        if label != predicted:
            error_type = ErrorTaxonomy.classify_dp_error(
                item.get("module_A", ""),
                item.get("module_B", ""),
                label, predicted, graph,
            )
            error_types[error_type] += 1
            errors.append({
                "module_A": item.get("module_A", ""),
                "module_B": item.get("module_B", ""),
                "label": label,
                "predicted": predicted,
                "error_type": error_type,
            })

    total_errors = len(errors)
    total_samples = data.get("total", 1)

    return {
        "project": project_name,
        "total_samples": total_samples,
        "total_errors": total_errors,
        "error_rate": total_errors / total_samples if total_samples > 0 else 0,
        "error_type_distribution": dict(error_types),
        "sample_errors": errors[:20],
    }


def generate_report(all_ci_errors, all_dp_errors):
    """生成汇总报告"""
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS REPORT")
    print("=" * 80)

    # CI-Hard 错误汇总
    print("\n--- CI-Hard Error Type Distribution (across all projects) ---")
    total_ci_errors = Counter()
    for result in all_ci_errors.values():
        if result:
            for k, v in result["error_type_distribution"].items():
                total_ci_errors[k] += v

    total = sum(total_ci_errors.values())
    for error_type, count in total_ci_errors.most_common():
        pct = count / total * 100 if total > 0 else 0
        desc = ErrorTaxonomy.CI_ERRORS.get(error_type, "")
        print(f"  {error_type:25s} {count:5d} ({pct:5.1f}%)  {desc}")

    # DP 错误汇总
    print("\n--- DP Error Type Distribution (across all projects) ---")
    total_dp_errors = Counter()
    for result in all_dp_errors.values():
        if result:
            for k, v in result["error_type_distribution"].items():
                total_dp_errors[k] += v

    total = sum(total_dp_errors.values())
    for error_type, count in total_dp_errors.most_common():
        pct = count / total * 100 if total > 0 else 0
        desc = ErrorTaxonomy.DP_ERRORS.get(error_type, "")
        print(f"  {error_type:30s} {count:5d} ({pct:5.1f}%)  {desc}")

    # 按项目的错误率
    print("\n--- Error Rate by Project ---")
    print(f"{'Project':18s} {'CI-Hard Err%':>12s} {'DP Err%':>10s}")
    for proj in PROJECTS:
        ci_err = all_ci_errors.get(proj["name"], {})
        dp_err = all_dp_errors.get(proj["name"], {})
        ci_rate = ci_err.get("error_rate", 0) * 100 if ci_err else 0
        dp_rate = dp_err.get("error_rate", 0) * 100 if dp_err else 0
        print(f"  {proj['name']:18s} {ci_rate:10.1f}% {dp_rate:8.1f}%")


# =====================================================
# 主函数
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--output", default="error_analysis_results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    all_ci_errors = {}
    all_dp_errors = {}

    for proj in PROJECTS:
        # CI-Hard 错误分析
        ci_path = os.path.join(args.results_dir, f"{args.model}_{proj['name']}_ci_hard.json")
        ci_result = analyze_ci_errors(ci_path, proj["path"], proj["name"])
        if ci_result:
            all_ci_errors[proj["name"]] = ci_result

        # DP 错误分析
        dp_path = os.path.join(args.results_dir, f"{args.model}_{proj['name']}_dp.json")
        dp_result = analyze_dp_errors(dp_path, proj["path"], proj["name"])
        if dp_result:
            all_dp_errors[proj["name"]] = dp_result

    # 保存
    out_path = os.path.join(args.output, f"error_analysis_{args.model}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"ci_hard": all_ci_errors, "dp": all_dp_errors}, f, indent=2)

    # 生成报告
    generate_report(all_ci_errors, all_dp_errors)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
