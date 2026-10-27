"""
Run probing analysis for qwen3-1.7b
"""
import sys
import subprocess
import os

# Add pytorch_env to path
sys.path.insert(0, r'E:\pytorch_env')

# Import and verify torch
import torch
print(f"PyTorch version: {torch.__version__}")

# Set working directory
os.chdir('experiment')

# Run commands
commands = [
    ["python", "probing/probe_layers.py", "--model", "qwen3-1.7b", "--project", "all", "--task", "all"],
]

for cmd in commands:
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        sys.exit(1)

print("\nProbing analysis for qwen3-1.7b completed!")
