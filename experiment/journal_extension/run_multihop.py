"""
Multi-hop 任务的 LLM 评估脚本

用法:
    python run_multihop.py --task transitive --model deepseek-v4-flash --project openfoam
    python run_multihop.py --task all --model claude-sonnet-4 --project all
"""

import json
import os
import sys
import random
import argparse

random.seed(42)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODELS, PROJECTS
from llm_client import call_llm


# =====================================================
# Prompt 模板
# =====================================================

PROMPT_TRANSITIVE = """You are a software architecture expert analyzing {system}, a {domain} system.

Given two modules, determine whether Module A has a transitive dependency on Module C.
A transitive dependency means there exists a chain: A depends on some module B, and B depends on C (or longer chains).

Module A: {module_a}
Module C: {module_c}

Based on your understanding of {system}'s architecture, does Module A transitively depend on Module C?

Answer with ONLY "Yes" or "No"."""

PROMPT_IMPACT = """You are a software architecture expert analyzing {system}, a {domain} system.

If module "{modified_module}" is modified (e.g., its API changes), which of the following modules would be directly or indirectly affected?

Candidate modules:
{candidates}

List ONLY the module names that would be affected, one per line. If none would be affected, answer "None"."""

PROMPT_SMELL = """You are a software architecture expert analyzing {system}, a {domain} system.

Given the following set of module dependencies, identify any architectural smells (design problems):

Dependencies:
{dependencies}

Possible smells to look for:
- Cyclic dependencies (A→B→C→A)
- God modules (one module depended on by too many others)
- Unstable modules (high coupling in both directions)

Describe any architectural smells you find. If none, answer "None found"."""

PROMPT_LAYER = """You are a software architecture expert analyzing {system}, a {domain} system.

Given the following module and its dependency information, infer its architectural layer.

Module: {module}
Depends on: {dependencies}
Depended on by: {dependents}

Architectural layers:
- infrastructure: Core/foundational modules (heavily depended upon, few outgoing deps)
- service: Middle-layer modules (moderate incoming and outgoing deps)
- application: Top-level modules (many outgoing deps, few incoming)
- isolated: No dependencies in either direction

Answer with ONLY one of: infrastructure, service, application, isolated"""


# =====================================================
# 评估函数
# =====================================================

def evaluate_transitive(model_config, project, sample_size=100):
    """评估传递依赖预测"""
    filepath = os.path.join(project["path"], "transitive_dependency.jsonl")
    if not os.path.exists(filepath):
        return None

    data = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    if len(data) > sample_size:
        data = random.sample(data, sample_size)

    tp = fp = tn = fn = 0
    results = []

    for item in data:
        prompt = PROMPT_TRANSITIVE.format(
            system=project["system"],
            domain=project["domain"],
            module_a=item["source_module"],
            module_c=item["target_module"],
        )

        response = call_llm(model_config, prompt)
        predicted = 1 if "yes" in response.strip().lower() else 0
        label = item["label"]

        if predicted == 1 and label == 1: tp += 1
        elif predicted == 1 and label == 0: fp += 1
        elif predicted == 0 and label == 0: tn += 1
        elif predicted == 0 and label == 1: fn += 1

        results.append({
            "source": item["source_module"],
            "target": item["target_module"],
            "label": label,
            "predicted": predicted,
            "hops": item.get("hops", -1),
        })

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total,
        "results": results,
    }


def evaluate_layer_inference(model_config, project, sample_size=100):
    """评估架构层级推断"""
    filepath = os.path.join(project["path"], "layer_inference.jsonl")
    if not os.path.exists(filepath):
        return None

    data = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["layer"] != "isolated":  # 跳过孤立模块
                data.append(item)

    if len(data) > sample_size:
        data = random.sample(data, sample_size)

    correct = 0
    results = []

    for item in data:
        deps_str = ", ".join(item["dependencies"][:10]) if item["dependencies"] else "None"
        dependents_str = ", ".join(item["dependents"][:10]) if item["dependents"] else "None"

        prompt = PROMPT_LAYER.format(
            system=project["system"],
            domain=project["domain"],
            module=item["module"],
            dependencies=deps_str,
            dependents=dependents_str,
        )

        response = call_llm(model_config, prompt).strip().lower()

        # 解析预测
        predicted = "unknown"
        for layer in ["infrastructure", "service", "application", "isolated"]:
            if layer in response:
                predicted = layer
                break

        is_correct = (predicted == item["layer"])
        if is_correct:
            correct += 1

        results.append({
            "module": item["module"],
            "true_layer": item["layer"],
            "predicted_layer": predicted,
            "correct": is_correct,
        })

    accuracy = correct / len(data) if data else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(data),
        "results": results,
    }


# =====================================================
# 主函数
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all",
                        choices=["transitive", "layer", "all"])
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--project", default="all")
    parser.add_argument("--output", default="results_multihop")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model_config = MODELS[args.model]

    if args.project == "all":
        projects = PROJECTS
    else:
        projects = [p for p in PROJECTS if p["name"] == args.project]

    tasks = ["transitive", "layer"] if args.task == "all" else [args.task]

    for proj in projects:
        print(f"\nProject: {proj['name']}")

        for task in tasks:
            print(f"  Task: {task}")

            if task == "transitive":
                result = evaluate_transitive(model_config, proj)
            elif task == "layer":
                result = evaluate_layer_inference(model_config, proj)
            else:
                continue

            if result is None:
                print(f"    No data found, skipping")
                continue

            if "accuracy" in result:
                print(f"    Accuracy: {result['accuracy']:.4f}")

            out_path = os.path.join(args.output, f"{args.model}_{proj['name']}_{task}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
