#!/usr/bin/env python3
"""
生成 Level 1 任务列表并创建运行脚本

用法:
    python prepare_level1.py
    
    然后在服务器上运行:
    parallel -j 4 < level1_tasks.sh
"""

import os

# Level 1 配置
LEVEL1_SYSTEMS = ['kubernetes', 'spring', 'openfoam', 'nastran', 'hadoop']
LEVEL1_TASKS = ['ci', 'ci_hard', 'ci_obfuscated', 'dp']
LEVEL1_MODELS = ['qwen3-8b', 'llama-3.1-8b', 'deepseek-v4-flash', 'claude-sonnet-4']

# 生成任务列表
tasks = []
for model in LEVEL1_MODELS:
    for system in LEVEL1_SYSTEMS:
        for task in LEVEL1_TASKS:
            tasks.append((model, system, task))

print(f"生成 {len(tasks)} 个 Level 1 任务")

# 保存为 shell 脚本
with open('level1_tasks.sh', 'w') as f:
    for model, system, task in tasks:
        cmd = f"python run_level1_experiment.py --model {model} --project {system} --task {task}"
        f.write(cmd + '\n')

print("已保存到 level1_tasks.sh")
print("\n服务器运行命令:")
print("  parallel -j 4 < level1_tasks.sh")
print("\n或者用 nohup 后台运行:")
print("  nohup parallel -j 4 < level1_tasks.sh > level1_output.log 2>&1 &")