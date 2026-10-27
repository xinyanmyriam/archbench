#!/usr/bin/env python3
"""
使用 Python multiprocessing 并行运行 probing 任务

用法:
    python run_probing_parallel.py
"""
import os
import sys
import subprocess
from multiprocessing import Pool, cpu_count

# 项目列表
PROJECTS = [
    'codeaster', 'hadoop', 'kubernetes', 'linux', 
    'nastran', 'openfoam', 'openmodelica', 'paraview', 
    'salome', 'spring', 'tensorflow', 'pipeline'
]

# 任务列表
TASKS = ['ci', 'ci_hard', 'ci_obfuscated', 'ci_content', 'dp']

# 模型列表
MODELS = ['qwen3-1.7b', 'qwen3-8b', 'qwen3-14b', 'qwen3-32b']

# 结果目录
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'probing', 'results')

def result_exists(model, project, task):
    """检查结果文件是否已存在"""
    filename = f"probing_{model}_{project}_{task}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    return os.path.exists(filepath)

def run_task(args):
    """运行单个任务"""
    model, project, task = args
    
    # 检查是否已有结果
    if result_exists(model, project, task):
        return f"SKIP {model} | {project} | {task}"
    
    cmd = f"python probing/probe_layers.py --model {model} --project {project} --task {task}"
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            return f"OK {model} | {project} | {task}"
        else:
            return f"FAIL {model} | {project} | {task}\n{result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT {model} | {project} | {task}"
    except Exception as e:
        return f"ERROR {model} | {project} | {task}: {e}"

def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 生成任务列表
    tasks = []
    for model in MODELS:
        for project in PROJECTS:
            for task in TASKS:
                tasks.append((model, project, task))
    
    # 过滤掉已有结果的任务
    tasks_to_run = [t for t in tasks if not result_exists(*t)]
    
    print(f"总任务数: {len(tasks)}")
    print(f"需要运行: {len(tasks_to_run)}")
    print(f"已跳过: {len(tasks) - len(tasks_to_run)}")
    print(f"使用 {cpu_count()} CPU 核心")
    
    # 并行运行
    with Pool(processes=min(32, cpu_count())) as pool:
        results = pool.map(run_task, tasks_to_run)
    
    # 统计结果
    success = sum(1 for r in results if r.startswith("OK"))
    failed = sum(1 for r in results if r.startswith("FAIL"))
    skipped = len(tasks) - len(tasks_to_run)
    
    print(f"\n=== 完成 ===")
    print(f"成功: {success}, 失败: {failed}, 跳过: {skipped}")
    
    # 输出失败的任务
    for r in results:
        if r.startswith("FAIL") or r.startswith("ERROR"):
            print(r)

if __name__ == '__main__':
    main()