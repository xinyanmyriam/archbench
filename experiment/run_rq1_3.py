"""
RQ1.3: Are representations robust and transferable?

实验1: 扰动测试 (Perturbation Test)
  - Random perturbation: 随机打乱模块名中的字符
  - Structure-preserving perturbation: 保留路径结构但替换具体名称

实验2: 虚构系统测试 (Fictitious System Test)
  - 构造一个虚构的软件系统（模型训练数据中不可能见过）
  - 测试模型能否基于结构线索推理出正确的架构关系

用法:
    python run_rq1_3.py
"""

import json
import os
import sys
import random
import string
from collections import defaultdict

random.seed(42)
sys.path.insert(0, ".")

from config import MODELS
from llm_client import call_llm

# 使用的模型
MODEL_NAMES = ["deepseek-v4-flash", "claude-sonnet-4"]

# 选择的项目（覆盖不同领域）
PROJECTS = [
    {"name": "kubernetes", "path": "../kubernetes", "system": "Kubernetes", "domain": "container orchestration"},
    {"name": "spring", "path": "../spring", "system": "Spring Framework", "domain": "enterprise Java framework"},
    {"name": "openfoam", "path": "../openfoam", "system": "OpenFOAM", "domain": "CFD simulation"},
    {"name": "tensorflow", "path": "../tensorflow", "system": "TensorFlow", "domain": "machine learning framework"},
    {"name": "codeaster", "path": "../codeaster", "system": "Code_Aster", "domain": "finite element analysis"},
]

SAMPLE_SIZE = 100
os.makedirs("results/rq1_3", exist_ok=True)


def parse_ci_response(response):
    for ch in response.strip().upper():
        if ch in "ABCD":
            return ch
    return None


def parse_dp_response(response):
    r = response.strip().lower()
    if "yes" in r:
        return 1
    elif "no" in r:
        return 0
    return None


# =====================================================
# 实验1: 扰动测试
# =====================================================

def random_scramble(name):
    """随机打乱模块名（保留路径分隔符）"""
    parts = name.split("/")
    scrambled = []
    for part in parts:
        chars = list(part)
        random.shuffle(chars)
        scrambled.append("".join(chars))
    return "/".join(scrambled)


def structure_preserving_rename(name, mapping):
    """保留路径结构但替换为无意义名称"""
    parts = name.split("/")
    renamed = []
    for part in parts:
        if part not in mapping:
            # 生成同长度的随机字符串
            mapping[part] = "".join(random.choices(string.ascii_lowercase, k=max(3, len(part))))
        renamed.append(mapping[part])
    return "/".join(renamed)


PROMPT_CI_PERTURBED = """You are a software architecture expert analyzing a software system called "{system}".

Given a file path, determine which module it belongs to.

File path: {file_path}

Candidate modules:
A) {candidate_a}
B) {candidate_b}
C) {candidate_c}
D) {candidate_d}

Answer with ONLY the letter (A, B, C, or D)."""


def run_perturbation_test(model_name, model_config, project, perturbation_type):
    """运行扰动测试"""
    print(f"    Perturbation ({perturbation_type}): {model_name} / {project['name']}")

    # 加载 CI 数据
    data = []
    with open(os.path.join(project["path"], "component_identification.jsonl"), encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    data = random.sample(data, min(SAMPLE_SIZE, len(data)))

    # 构建名称映射（structure-preserving 用）
    name_mapping = {}

    correct = 0
    total = 0
    results = []

    for item in data:
        file_path = item["file"]
        candidates = item["candidates"]
        label = item["label"]

        # 应用扰动
        if perturbation_type == "random":
            file_path_p = random_scramble(file_path)
            candidates_p = [random_scramble(c) for c in candidates]
            label_p = random_scramble(label)
            system_name = random_scramble(project["system"])
        elif perturbation_type == "structure_preserving":
            file_path_p = structure_preserving_rename(file_path, name_mapping)
            candidates_p = [structure_preserving_rename(c, name_mapping) for c in candidates]
            label_p = structure_preserving_rename(label, name_mapping)
            system_name = "SystemX"
        else:
            raise ValueError(f"Unknown perturbation: {perturbation_type}")

        # 找正确答案的位置
        correct_idx = candidates.index(label)
        correct_letter = chr(65 + correct_idx)

        prompt = PROMPT_CI_PERTURBED.format(
            system=system_name,
            file_path=file_path_p,
            candidate_a=candidates_p[0],
            candidate_b=candidates_p[1],
            candidate_c=candidates_p[2],
            candidate_d=candidates_p[3],
        )

        response = call_llm(model_config, prompt)
        predicted = parse_ci_response(response)
        is_correct = (predicted == correct_letter)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "original_file": item["file"],
            "perturbed_file": file_path_p,
            "label": label,
            "predicted_letter": predicted,
            "correct": is_correct,
        })

        if total % 20 == 0:
            print(f"      {total}/{len(data)} acc={correct/total:.3f}")

    accuracy = correct / total if total > 0 else 0
    print(f"      -> Accuracy: {accuracy:.4f}")
    return {"accuracy": accuracy, "total": total, "correct": correct, "results": results}


# =====================================================
# 实验2: 虚构系统测试
# =====================================================

FICTITIOUS_SYSTEM = {
    "name": "ZephyrOS",
    "domain": "real-time operating system for autonomous drones",
    "modules": [
        "core/scheduler",
        "core/memory",
        "core/ipc",
        "drivers/sensors/imu",
        "drivers/sensors/gps",
        "drivers/actuators/motor",
        "drivers/actuators/servo",
        "drivers/comm/radio",
        "drivers/comm/wifi",
        "navigation/pathplanner",
        "navigation/obstacle_avoidance",
        "navigation/localization",
        "flight/controller",
        "flight/stabilizer",
        "flight/autopilot",
        "telemetry/logger",
        "telemetry/transmitter",
        "config/params",
        "utils/math",
        "utils/filters",
    ],
    # 合理的依赖关系（基于常识设计）
    "dependencies": [
        ("flight/controller", "core/scheduler"),
        ("flight/controller", "drivers/sensors/imu"),
        ("flight/controller", "drivers/actuators/motor"),
        ("flight/stabilizer", "drivers/sensors/imu"),
        ("flight/stabilizer", "utils/filters"),
        ("flight/autopilot", "navigation/pathplanner"),
        ("flight/autopilot", "flight/controller"),
        ("navigation/pathplanner", "navigation/localization"),
        ("navigation/pathplanner", "navigation/obstacle_avoidance"),
        ("navigation/localization", "drivers/sensors/gps"),
        ("navigation/localization", "drivers/sensors/imu"),
        ("navigation/obstacle_avoidance", "drivers/sensors/imu"),
        ("telemetry/logger", "core/memory"),
        ("telemetry/transmitter", "drivers/comm/radio"),
        ("drivers/comm/radio", "core/ipc"),
        ("drivers/comm/wifi", "core/ipc"),
        ("config/params", "core/memory"),
    ],
}


def generate_fictitious_ci_tasks():
    """生成虚构系统的 CI 任务"""
    modules = FICTITIOUS_SYSTEM["modules"]
    tasks = []

    # 为每个模块生成合理的文件名
    file_templates = {
        "core": ["init.c", "main.c", "api.h", "types.h"],
        "drivers": ["driver.c", "hal.c", "interface.h", "config.h"],
        "navigation": ["algorithm.cpp", "planner.cpp", "state.h", "types.h"],
        "flight": ["control_loop.cpp", "pid.cpp", "state_machine.cpp", "params.h"],
        "telemetry": ["protocol.c", "buffer.c", "format.h"],
        "config": ["parser.c", "defaults.h", "schema.h"],
        "utils": ["impl.c", "common.h"],
    }

    for module in modules:
        top_level = module.split("/")[0]
        files = file_templates.get(top_level, ["main.c", "types.h"])

        for fname in files[:2]:
            file_path = f"{module}/{fname}"
            candidates = [module]
            while len(candidates) < 4:
                c = random.choice(modules)
                if c not in candidates:
                    candidates.append(c)
            random.shuffle(candidates)
            tasks.append({
                "file": file_path,
                "candidates": candidates,
                "label": module,
            })

    return tasks[:SAMPLE_SIZE]


def generate_fictitious_dp_tasks():
    """生成虚构系统的 DP 任务"""
    modules = FICTITIOUS_SYSTEM["modules"]
    dep_set = set(FICTITIOUS_SYSTEM["dependencies"])
    tasks = []

    # 正样本
    for src, tgt in FICTITIOUS_SYSTEM["dependencies"]:
        tasks.append({"module_A": src, "module_B": tgt, "label": 1})

    # 负样本（等量）
    neg_count = 0
    while neg_count < len(FICTITIOUS_SYSTEM["dependencies"]):
        a = random.choice(modules)
        b = random.choice(modules)
        if a != b and (a, b) not in dep_set:
            tasks.append({"module_A": a, "module_B": b, "label": 0})
            neg_count += 1

    random.shuffle(tasks)
    return tasks


PROMPT_FICTITIOUS_CI = """You are a software architecture expert. You are analyzing a system called "{system}", which is a {domain}.

The system has the following modules:
{module_list}

Given a file path, determine which module it belongs to.

File path: {file_path}

Candidate modules:
A) {candidate_a}
B) {candidate_b}
C) {candidate_c}
D) {candidate_d}

Answer with ONLY the letter (A, B, C, or D)."""


PROMPT_FICTITIOUS_DP = """You are a software architecture expert. You are analyzing a system called "{system}", which is a {domain}.

The system has the following modules:
{module_list}

Based on the module names and typical architecture patterns for {domain} systems, does Module A depend on Module B?

Module A: {module_a}
Module B: {module_b}

Answer with ONLY "Yes" or "No"."""


def run_fictitious_test(model_name, model_config):
    """运行虚构系统测试"""
    print(f"    Fictitious system: {model_name}")
    sys_info = FICTITIOUS_SYSTEM
    module_list = "\n".join(f"  - {m}" for m in sys_info["modules"])

    # CI 测试
    ci_tasks = generate_fictitious_ci_tasks()
    ci_correct = 0
    ci_total = 0

    for item in ci_tasks:
        correct_idx = item["candidates"].index(item["label"])
        correct_letter = chr(65 + correct_idx)

        prompt = PROMPT_FICTITIOUS_CI.format(
            system=sys_info["name"],
            domain=sys_info["domain"],
            module_list=module_list,
            file_path=item["file"],
            candidate_a=item["candidates"][0],
            candidate_b=item["candidates"][1],
            candidate_c=item["candidates"][2],
            candidate_d=item["candidates"][3],
        )

        response = call_llm(model_config, prompt)
        predicted = parse_ci_response(response)
        if predicted == correct_letter:
            ci_correct += 1
        ci_total += 1

    ci_acc = ci_correct / ci_total if ci_total > 0 else 0
    print(f"      CI accuracy: {ci_acc:.4f} ({ci_correct}/{ci_total})")

    # DP 测试
    dp_tasks = generate_fictitious_dp_tasks()
    tp = fp = tn = fn = 0

    for item in dp_tasks:
        prompt = PROMPT_FICTITIOUS_DP.format(
            system=sys_info["name"],
            domain=sys_info["domain"],
            module_list=module_list,
            module_a=item["module_A"],
            module_b=item["module_B"],
        )

        response = call_llm(model_config, prompt)
        predicted = parse_dp_response(response)
        if predicted is None:
            predicted = 0
        label = item["label"]

        if predicted == 1 and label == 1: tp += 1
        elif predicted == 1 and label == 0: fp += 1
        elif predicted == 0 and label == 0: tn += 1
        elif predicted == 0 and label == 1: fn += 1

    dp_total = tp + fp + tn + fn
    dp_acc = (tp + tn) / dp_total if dp_total > 0 else 0
    dp_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    dp_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    dp_f1 = 2 * dp_prec * dp_rec / (dp_prec + dp_rec) if (dp_prec + dp_rec) > 0 else 0
    print(f"      DP: acc={dp_acc:.4f} P={dp_prec:.4f} R={dp_rec:.4f} F1={dp_f1:.4f}")

    return {
        "ci": {"accuracy": ci_acc, "total": ci_total},
        "dp": {"accuracy": dp_acc, "precision": dp_prec, "recall": dp_rec, "f1": dp_f1, "total": dp_total},
    }


# =====================================================
# 主函数
# =====================================================

def main():
    print("=" * 70)
    print("RQ1.3: Robustness and Transferability")
    print("=" * 70)

    all_results = {}

    for model_name in MODEL_NAMES:
        model_config = MODELS[model_name]
        print(f"\n  Model: {model_name}")
        all_results[model_name] = {}

        # --- 实验1: 扰动测试 ---
        print(f"\n  [Experiment 1: Perturbation Test]")
        for proj in PROJECTS:
            for ptype in ["random", "structure_preserving"]:
                key = f"perturbation_{ptype}_{proj['name']}"
                result = run_perturbation_test(model_name, model_config, proj, ptype)
                all_results[model_name][key] = result

                out_path = f"results/rq1_3/{model_name}_{key}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, default=str)

        # --- 实验2: 虚构系统测试 ---
        print(f"\n  [Experiment 2: Fictitious System Test]")
        result = run_fictitious_test(model_name, model_config)
        all_results[model_name]["fictitious"] = result

        out_path = f"results/rq1_3/{model_name}_fictitious.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

    # --- 汇总 ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for model_name in MODEL_NAMES:
        print(f"\n  {model_name}:")
        r = all_results[model_name]

        # 扰动测试汇总
        for ptype in ["random", "structure_preserving"]:
            accs = [r[f"perturbation_{ptype}_{p['name']}"]["accuracy"] for p in PROJECTS]
            avg = sum(accs) / len(accs)
            print(f"    Perturbation ({ptype:22s}): avg_acc={avg:.4f} ({[f'{a:.2f}' for a in accs]})")

        # 虚构系统
        fict = r["fictitious"]
        print(f"    Fictitious CI: {fict['ci']['accuracy']:.4f}")
        print(f"    Fictitious DP: acc={fict['dp']['accuracy']:.4f} F1={fict['dp']['f1']:.4f}")

    # 保存完整结果
    with open("results/rq1_3/summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\nAll results saved to results/rq1_3/")


if __name__ == "__main__":
    main()
