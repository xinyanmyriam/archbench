"""
生成 Hard 模式的 Component Identification 任务

Hard 模式：只给文件名（不含目录路径），让模型根据文件名的语义来判断模块归属。
这样 Path Heuristic 就无法直接匹配了。

例如：
  Easy: file="cmd/kube-apiserver/app/server.go", candidates=[cmd/kube-apiserver, ...]
  Hard: file="server.go", candidates=[cmd/kube-apiserver, ...]
"""

import json
import os
import random

random.seed(42)

PROJECTS = [
    ("openfoam", "../openfoam"),
    ("nastran", "../nastran"),
    ("salome", "../salome/unified"),
    ("paraview", "../paraview"),
    ("kubernetes", "../kubernetes"),
    ("linux", "../linux"),
    ("hadoop", "../hadoop"),
    ("spring", "../spring"),
    ("codeaster", "../codeaster"),
    ("openmodelica", "../openmodelica"),
    ("tensorflow", "../tensorflow"),
    ("pytorch", "../pytorch"),
]

for name, path in PROJECTS:
    input_path = os.path.join(path, "component_identification.jsonl")
    output_path = os.path.join(path, "component_identification_hard.jsonl")

    if not os.path.exists(input_path):
        print(f"Skip {name}: no CI data")
        continue

    data = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    hard_data = []
    for item in data:
        hard_item = {
            "file": os.path.basename(item["file"]),           # 只保留文件名
            "file_full": item["file"],                         # 保留完整路径（用于验证）
            "candidates": item["candidates"],
            "label": item["label"],
        }
        hard_data.append(hard_item)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in hard_data:
            f.write(json.dumps(item) + "\n")

    print(f"{name}: {len(hard_data)} hard CI tasks generated")

print("\nDone. Hard CI tasks saved as component_identification_hard.jsonl")
