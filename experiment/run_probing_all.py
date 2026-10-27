"""
运行所有 qwen 模型的完整 probing 实验
支持断点续跑：如果结果文件已存在则跳过
"""
import os
import sys
import subprocess

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

def run_probe(model, project, task):
    """运行单个 probing 任务"""
    # 检查是否已有结果
    if result_exists(model, project, task):
        print(f"  [SKIP] {model} | {project} | {task} (结果已存在)")
        return True
    
    # 设置环境变量
    env = os.environ.copy()
    pytorch_path = r'E:\pytorch_env'
    
    # 设置 PATH，让系统能找到正确的 python 和 DLL
    env['PATH'] = pytorch_path + os.pathsep + env.get('PATH', '')
    # 设置 PYTHONPATH
    env['PYTHONPATH'] = pytorch_path + os.pathsep + env.get('PYTHONPATH', '')
    
    cmd = 'python probing/probe_layers.py --model {} --project {} --task {}'.format(
        model, project, task
    )
    print(f"\n{'='*60}")
    print(f"Running: {model} | {project} | {task}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, env=env)
    return result.returncode == 0

def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    total = len(MODELS) * len(PROJECTS) * len(TASKS)
    current = 0
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for model in MODELS:
        print(f"\n\n{'#'*80}")
        print(f"# MODEL: {model}")
        print(f"{'#'*80}\n")
        
        for project in PROJECTS:
            for task in TASKS:
                current += 1
                
                # 检查是否跳过
                if result_exists(model, project, task):
                    skip_count += 1
                    print(f"[SKIP] [{current}/{total}] {model} | {project} | {task}")
                    continue
                
                print(f"\n[{current}/{total}] {model} | {project} | {task}")
                
                success = run_probe(model, project, task)
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    print(f"WARNING: Failed - {model} | {project} | {task}")
                
                # 每完成一个任务后保存进度
                print(f"\nProgress: {current}/{total} ({current*100/total:.1f}%)")
                print(f"Success: {success_count}, Failed: {fail_count}, Skipped: {skip_count}")
    
    print(f"\n\n{'#'*80}")
    print(f"# ALL PROBING COMPLETED!")
    print(f"# Total: {total}, Success: {success_count}, Failed: {fail_count}, Skipped: {skip_count}")
    print(f"{'#'*80}")

if __name__ == '__main__':
    main()