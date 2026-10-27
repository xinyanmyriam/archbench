#!/usr/bin/env python3
"""
生成 Level 1 任务列表

Level 1: 给 LLM 提供 code context (include + code snippet)
对比 Level 0 (只有模块名)

用法:
    python generate_level1_tasks.py
"""

import os
import json
import random

# 配置
LEVEL1_SYSTEMS = ['kubernetes', 'spring', 'openfoam', 'nastran', 'hadoop']
LEVEL1_TASKS = ['ci', 'ci_hard', 'ci_obfuscated', 'dp']
LEVEL1_MODELS = ['qwen3-8b', 'llama-3.1-8b', 'deepseek-v4-flash', 'claude-sonnet-4']

SAMPLE_SIZE = 150  # 每任务150样本

def load_data_with_context(project, task):
    """
    加载 Level 1 数据（包含 code context）
    
    返回: list of {"text": prompt_with_context, "label": ...}
    """
    base_path = project
    
    # 根据任务加载数据
    if task == 'ci':
        filepath = os.path.join(base_path, 'component_identification.jsonl')
    elif task == 'ci_hard':
        filepath = os.path.join(base_path, 'component_identification_hard.jsonl')
    elif task == 'ci_obfuscated':
        filepath = os.path.join(base_path, 'component_identification_obfuscated.jsonl')
    elif task == 'dp':
        filepath = os.path.join(base_path, 'dependency_prediction.jsonl')
    else:
        return []
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return []
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            
            if task in ['ci', 'ci_hard', 'ci_obfuscated']:
                # 构建 Level 1 prompt（包含 code context）
                file_path = item.get('file', '')
                label = item.get('label', '')
                
                # 模拟：实际应该读取文件内容
                # 这里只是示例结构
                prompt = f"""Given the module name: {label}

Context information:
- File path: {file_path}
- Include files: {item.get('includes', 'N/A')}
- Code snippet (first 100 lines): [file content would be inserted here]

Question: Which module does this file belong to?
"""
                data.append({
                    'text': prompt,
                    'label': label,
                    'file': file_path,
                })
            else:  # dp
                module_a = item.get('module_A', '')
                module_b = item.get('module_B', '')
                label = item.get('label', 0)
                
                prompt = f"""Given two modules: {module_a} and {module_b}

Context: These modules are in the same system.

Question: Is there a dependency from {module_a} to {module_b}?
"""
                data.append({
                    'text': prompt,
                    'label': str(label),
                })
    
    # 采样
    if len(data) > SAMPLE_SIZE:
        data = random.sample(data, SAMPLE_SIZE)
    
    return data


def generate_tasks_file():
    """生成任务列表文件"""
    tasks = []
    
    for model in LEVEL1_MODELS:
        for system in LEVEL1_SYSTEMS:
            for task in LEVEL1_TASKS:
                tasks.append({
                    'model': model,
                    'project': system,
                    'task': task,
                    'level': 1,  # Level 1
                })
    
    # 保存任务列表
    output_file = 'level1_tasks.txt'
    with open(output_file, 'w') as f:
        for t in tasks:
            cmd = f"python run_level1_experiment.py --model {t['model']} --project {t['project']} --task {t['task']} --level 1"
            f.write(cmd + '\n')
    
    print(f"Generated {len(tasks)} tasks")
    print(f"Saved to {output_file}")
    print()
    print("Run with:")
    print(f"  parallel -j 4 < {output_file}")
    
    return tasks


if __name__ == '__main__':
    tasks = generate_tasks_file()
    
    # 打印任务统计
    print("\n" + "=" * 60)
    print("Level 1 任务统计")
    print("=" * 60)
    print(f"系统数: {len(LEVEL1_SYSTEMS)}")
    print(f"任务数: {len(LEVEL1_TASKS)}")
    print(f"模型数: {len(LEVEL1_MODELS)}")
    print(f"总任务: {len(tasks)}")