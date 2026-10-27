"""
Run all qwen model probing experiments
"""
import sys
import subprocess
import os
import importlib.util

# Add pytorch_env to path FIRST
sys.path.insert(0, r'E:\pytorch_env')

# Set working directory
os.chdir('experiment')

# Import and verify torch AFTER setting path
import torch
print(f"PyTorch version: {torch.__version__}")

# Set environment
env = os.environ.copy()
env['PYTHONPATH'] = r'E:\pytorch_env;' + env.get('PYTHONPATH', '')

# Run commands
commands = [
    # qwen3-8b probing
    ["python", "-c", """
import sys
sys.path.insert(0, r'E:\\\\pytorch_env')
sys.path.insert(0, r'.')
sys.path.insert(0, r'./probing')
from probe_layers import main
import sys as sys_mod
sys_mod.argv = ['probe_layers.py', '--model', 'qwen3-8b', '--project', 'all', '--task', 'all']
main()
print('\\n=== qwen3-8b probing completed ===')
"""],
    # qwen3-1.7b RSA
    ["python", "-c", """
import sys
sys.path.insert(0, r'E:\\\\pytorch_env')
sys.path.insert(0, r'.')
sys.path.insert(0, r'./probing')
from rsa_analysis_v2 import main
import sys as sys_mod
sys_mod.argv = ['rsa_analysis_v2.py', '--model', 'qwen3-1.7b', '--project', 'all']
main()
print('\\n=== qwen3-1.7b RSA completed ===')
"""],
    # qwen3-8b RSA
    ["python", "-c", """
import sys
sys.path.insert(0, r'E:\\\\pytorch_env')
sys.path.insert(0, r'.')
sys.path.insert(0, r'./probing')
from rsa_analysis_v2 import main
import sys as sys_mod
sys_mod.argv = ['rsa_analysis_v2.py', '--model', 'qwen3-8b', '--project', 'all']
main()
print('\\n=== qwen3-8b RSA completed ===')
"""],
]

for i, cmd in enumerate(commands):
    print(f"\n{'='*80}")
    print(f"Running command {i+1}/{len(commands)}")
    print(f"{'='*80}\n")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"Command {i+1} failed with return code {result.returncode}")
        # Continue with next command
        continue

print("\nAll experiments completed!")