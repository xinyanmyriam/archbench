"""
Run probing analysis for qwen3-1.7b and qwen3-8b
"""
import sys
import subprocess

# Add pytorch_env to path
sys.path.insert(0, r'E:\pytorch_env')

# Import and verify torch
import torch
print(f"PyTorch version: {torch.__version__}")

# Run commands
commands = [
    ["python", "probing/probe_layers.py", "--model", "qwen3-1.7b", "--project", "all", "--task", "all"],
    ["python", "probing/probe_layers.py", "--model", "qwen3-8b", "--project", "all", "--task", "all"],
    ["python", "probing/rsa_analysis_v2.py", "--model", "qwen3-1.7b", "--project", "all"],
    ["python", "probing/rsa_analysis_v2.py", "--model", "qwen3-8b", "--project", "all"],
]

for cmd in commands:
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    result = subprocess.run(cmd, cwd="experiment")
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        sys.exit(1)

print("\nAll probing analysis completed!")
