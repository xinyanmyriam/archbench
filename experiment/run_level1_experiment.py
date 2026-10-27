#!/usr/bin/env python3
"""
Level 1 实验运行脚本

使用已存在的 prompts_level1.py 构建 prompt

用法:
    python run_level1_experiment.py --model qwen3-8b --project kubernetes --task ci
"""

import argparse
import os
import sys
import json
import random

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts_level1 import (
    build_ci_l1_prompt, build_ci_hard_l1_prompt, 
    build_dp_l1_prompt, build_ar_l1_prompt
)
from llm_client import call_api

# 配置
SYSTEMS = {
    'kubernetes': {'domain': 'cloud computing'},
    'spring': {'domain': 'web framework'},
    'openfoam': {'domain': 'CFD simulation'},
    'nastran': {'domain': 'FEA software'},
    'hadoop': {'domain': 'big data'},
}

SAMPLE_SIZE = 150

def load_code_snippet(file_path, project_path, max_lines=100):
    """
    加载代码文件的前 N 行作为 snippet
    
    Args:
        file_path: 文件相对路径，如 "cmd/clicheck/check_cli_conventions.go"
        project_path: 项目根目录
        max_lines: 最大行数
    
    Returns:
        代码片段字符串
    """
    # 构建完整路径
    # 尝试多种可能的路径格式
    possible_paths = [
        os.path.join(project_path, file_path),
        os.path.join(project_path, file_path + '.go'),
        os.path.join(project_path, file_path + '.java'),
        os.path.join(project_path, file_path + '.py'),
        os.path.join(project_path, file_path + '.cpp'),
        os.path.join(project_path, file_path + '.c'),
        os.path.join(project_path, file_path + '.h'),
    ]
    
    for full_path in possible_paths:
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line.rstrip())
                return '\n'.join(lines)
            except Exception as e:
                print(f"  Warning: Failed to read {full_path}: {e}")
                continue
    
    # 文件不存在，返回空
    return "# Source file not found"
    """加载 CI 数据"""
    if obfuscated:
        filepath = os.path.join(project_path, 'component_identification_obfuscated.jsonl')
    else:
        filepath = os.path.join(project_path, 'component_identification.jsonl')
    
    if not os.path.exists(filepath):
        return []
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # 采样
    if len(data) > sample_size:
        random.seed(42)
        data = random.sample(data, sample_size)
    
    return data


def load_dp_data(project_path, sample_size):
    """加载 DP 数据"""
    filepath = os.path.join(project_path, 'dependency_prediction.jsonl')
    
    if not os.path.exists(filepath):
        return []
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    if len(data) > sample_size:
        random.seed(42)
        data = random.sample(data, sample_size)
    
    return data


def run_level1_experiment(model, project, task):
    """运行单个 Level 1 实验"""
    
    project_path = project
    system_info = SYSTEMS.get(project, {'domain': 'software system'})
    
    print(f"\n{'='*60}")
    print(f"Level 1: {model} | {project} | {task}")
    print(f"{'='*60}")
    
    # 加载数据
    if task == 'ci':
        raw_data = load_ci_data(project_path, SAMPLE_SIZE, obfuscated=False)
    elif task == 'ci_hard':
        # CI-Hard 用原始文件但隐藏路径
        raw_data = load_ci_data(project_path, SAMPLE_SIZE, obfuscated=False)
    elif task == 'ci_obfuscated':
        raw_data = load_ci_data(project_path, SAMPLE_SIZE, obfuscated=True)
    elif task == 'dp':
        raw_data = load_dp_data(project_path, SAMPLE_SIZE)
    else:
        print(f"Unknown task: {task}")
        return None
    
    if not raw_data:
        print(f"No data found for {project}")
        return None
    
    print(f"Loaded {len(raw_data)} samples")
    
    results = []
    for i, item in enumerate(raw_data):
        # 读取代码文件获取 snippet
        file_path = item.get('file', '')
        code_snippet = load_code_snippet(file_path, project_path, max_lines=100)
        
        # 获取 candidates 和 label
        candidates = item.get('candidates', [])
        label = item.get('label', '')
        
        # 确保有4个 candidates（不够则补空）
        while len(candidates) < 4:
            candidates.append('unknown')
        
        # 构建 prompt（根据任务类型）
        if task == 'ci':
            prompt, correct, options = build_ci_l1_prompt(
                system=project,
                domain=system_info['domain'],
                file_path=file_path,
                candidates=candidates[:4],
                label=label,
                code_snippet=code_snippet,
            )
        elif task == 'ci_hard':
            # CI-Hard: 隐藏路径，只给文件名
            file_name = os.path.basename(file_path)
            prompt, correct, options = build_ci_hard_l1_prompt(
                system=project,
                domain=system_info['domain'],
                file_name=file_name,
                candidates=candidates[:4],
                label=label,
                code_snippet=code_snippet,
            )
        elif task == 'ci_obfuscated':
            # CI-Obfuscated: 用匿名化路径
            obfuscated_path = item.get('file', file_path)
            prompt, correct, options = build_ci_l1_prompt(
                system=project,
                domain=system_info['domain'],
                file_path=obfuscated_path,
                candidates=candidates[:4],
                label=label,
                code_snippet=code_snippet,
            )
        elif task == 'dp':
            # DP: 两个模块的依赖判断
            module_a = item.get('module_A', '')
            module_b = item.get('module_B', '')
            # DP 任务的 code snippet 应该是两个模块的文件列表
            dp_snippet = f"Module A: {module_a}\nModule B: {module_b}"
            prompt = build_dp_l1_prompt(
                system=project,
                domain=system_info['domain'],
                module_a=module_a,
                module_b=module_b,
                files_a=[],  # TODO: 获取文件列表
                files_b=[],  # TODO: 获取文件列表
                imports_a=[],  # TODO: 获取 import 列表
            )
            correct = str(item.get('label', 0))
            options = {'Yes': '1', 'No': '0'}
        # ... 其他任务类似
        
        # 调用 API
        try:
            response = call_api(model, prompt)
            results.append({
                'prompt': prompt,
                'response': response,
                'correct': correct,
            })
        except Exception as e:
            print(f"Error at {i}: {e}")
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(raw_data)}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    
    results = run_level1_experiment(args.model, args.project, args.task)
    
    if results:
        # 保存结果
        output_dir = f'../paperB/rq3_externalization/level1'
        os.makedirs(output_dir, exist_ok=True)
        output_file = f'{output_dir}/{args.model}_{args.project}_{args.task}.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()