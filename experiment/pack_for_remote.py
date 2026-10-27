"""
打包最小文件集，用于在有代理的电脑上跑闭源模型实验。
运行后会生成 remote_experiment/ 目录，只需拷贝这个目录。
"""
import os
import shutil
import json

OUT = "remote_experiment"

# 清理
if os.path.exists(OUT):
    shutil.rmtree(OUT)

# 1. 拷贝 experiment 脚本
scripts = [
    "config.py", "prompts.py", "prompts_level1.py",
    "llm_client.py", "baselines.py",
    "run_experiment.py", "run_level1.py",
    "show_progress.py", "check_all_models.py",
    "analyze_results.py",
]
os.makedirs(OUT, exist_ok=True)
for s in scripts:
    if os.path.exists(s):
        shutil.copy(s, os.path.join(OUT, s))

# 2. 拷贝各项目的 benchmark 数据（只要 .jsonl 和关键 .json）
projects = {
    "openfoam": "../openfoam",
    "nastran": "../nastran",
    "salome": "../salome/unified",
    "paraview": "../paraview",
    "kubernetes": "../kubernetes",
    "linux": "../linux",
    "hadoop": "../hadoop",
    "spring": "../spring",
    "codeaster": "../codeaster",
    "openmodelica": "../openmodelica",
    "tensorflow": "../tensorflow",
    "pytorch": "../pytorch",
}

needed_files = [
    "component_identification.jsonl",
    "component_identification_hard.jsonl",
    "dependency_prediction.jsonl",
    "architecture_reconstruction.jsonl",
    "modules_index.json",
    "file_index.json",
    "module_dependency.json",
    "dependency_raw.json",
]

total_size = 0
for name, path in projects.items():
    dest = os.path.join(OUT, name)
    os.makedirs(dest, exist_ok=True)
    for f in needed_files:
        src = os.path.join(path, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dest, f))
            total_size += os.path.getsize(src)

# 3. 创建 results 目录
os.makedirs(os.path.join(OUT, "results"), exist_ok=True)

# 4. 修改 config.py 中的路径（从 ../project 改为 ./project）
config_path = os.path.join(OUT, "config.py")
with open(config_path, "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace('"../openfoam"', '"./openfoam"')
content = content.replace('"../nastran"', '"./nastran"')
content = content.replace('"../salome/unified"', '"./salome"')
content = content.replace('"../paraview"', '"./paraview"')
content = content.replace('"../kubernetes"', '"./kubernetes"')
content = content.replace('"../linux"', '"./linux"')
content = content.replace('"../hadoop"', '"./hadoop"')
content = content.replace('"../spring"', '"./spring"')
content = content.replace('"../codeaster"', '"./codeaster"')
content = content.replace('"../openmodelica"', '"./openmodelica"')
content = content.replace('"../tensorflow"', '"./tensorflow"')
content = content.replace('"../pytorch"', '"./pytorch"')

# 添加闭源模型
closed_models = '''
    # --- 闭源模型（需要代理才能调用）---
    "gpt-5": {
        "provider": "openai",
        "model": "openai/gpt-5",
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
    },
    "claude-sonnet-4": {
        "provider": "openai",
        "model": "anthropic/claude-sonnet-4",
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
    },
    "gemini-2.5-pro": {
        "provider": "openai",
        "model": "google/gemini-2.5-pro",
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
    },
'''
# 在 MODELS = { 后面插入
content = content.replace("MODELS = {", "MODELS = {" + closed_models, 1)

with open(config_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Packed to {OUT}/")
print(f"Total data size: {total_size / 1024 / 1024:.1f} MB")
print(f"\nOn the remote machine:")
print(f"  1. pip install openai")
print(f"  2. set HTTPS_PROXY=http://127.0.0.1:7890  (adjust port)")
print(f"  3. cd {OUT}")
print(f"  4. python run_experiment.py --model gpt-5 --project openfoam --task ci  (test first)")
print(f"  5. python run_experiment.py --model gpt-5 --project all --task all")
print(f"  6. python run_experiment.py --model claude-sonnet-4 --project all --task all")
print(f"  7. python run_experiment.py --model gemini-2.5-pro --project all --task all")
