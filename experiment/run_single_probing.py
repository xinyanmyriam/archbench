"""
Run single probing analysis for qwen3-1.7b on openfoam/ci
"""
import sys
import subprocess
import os

# Add pytorch_env to path FIRST
sys.path.insert(0, r'E:\pytorch_env')

# Set working directory
os.chdir('experiment')

# Import and verify torch AFTER setting path
import torch
print(f"PyTorch version: {torch.__version__}")

# Run single task
env = os.environ.copy()
env['PYTHONPATH'] = r'E:\pytorch_env;' + env.get('PYTHONPATH', '')

cmd = ["python", "probing/probe_layers.py", "--model", "qwen3-1.7b", "--project", "openfoam", "--task", "ci"]

print(f"\n{'='*80}")
print(f"Running: {' '.join(cmd)}")
print(f"{'='*80}\n")
result = subprocess.run(cmd, env=env)

if result.returncode == 0:
    print("\nProbing analysis completed!")
else:
    print(f"Command failed: {' '.join(cmd)}")
    sys.exit(1)
