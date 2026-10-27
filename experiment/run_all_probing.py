"""
Run all qwen model probing + RSA experiments
Models: qwen3-1.7b, qwen3-8b, qwen3-14b, qwen3-32b
Projects: all 12 projects
Tasks: ci, ci_hard, ci_obfuscated, ci_content, dp
"""
import sys
import subprocess
import os

# Set working directory
os.chdir('experiment')

# Add pytorch_env to path
sys.path.insert(0, r'E:\pytorch_env')

# Import torch to verify
import torch
print(f"PyTorch version: {torch.__version__}")

# Set environment
env = os.environ.copy()
env['PYTHONPATH'] = r'E:\pytorch_env;' + env.get('PYTHONPATH', '')

# Models to run
models = ["qwen3-1.7b", "qwen3-8b", "qwen3-14b", "qwen3-32b"]

# Build command list
commands = []

# Step 1: Probing for each model
for model in models:
    cmd = f"""
import sys
sys.path.insert(0, r'E:\\\\pytorch_env')
sys.path.insert(0, r'.')
sys.path.insert(0, r'./probing')
from probe_layers import main
import sys as sys_mod
sys_mod.argv = ['probe_layers.py', '--model', '{model}', '--project', 'all', '--task', 'all']
main()
print('\\n=== {model} probing completed ===')
"""
    commands.append(["python", "-c", cmd])
    print(f"Added probing command for {model}")

# Step 2: RSA for each model
for model in models:
    cmd = f"""
import sys
sys.path.insert(0, r'E:\\\\pytorch_env')
sys.path.insert(0, r'.')
sys.path.insert(0, r'./probing')
from rsa_analysis_v2 import main
import sys as sys_mod
sys_mod.argv = ['rsa_analysis_v2.py', '--model', '{model}', '--project', 'all']
main()
print('\\n=== {model} RSA completed ===')
"""
    commands.append(["python", "-c", cmd])
    print(f"Added RSA command for {model}")

print(f"\nTotal commands: {len(commands)}")
print("="*80)

# Run all commands
for i, cmd in enumerate(commands):
    print(f"\n[{i+1}/{len(commands)}] Running: {cmd[2][:100]}...")
    print("="*80)
    
    result = subprocess.run(cmd, env=env)
    
    if result.returncode != 0:
        print(f"WARNING: Command {i+1} failed with return code {result.returncode}")
        # Continue with next command
    
    # Check results after each model completes
    if (i+1) % 6 == 0:  # After each model's probing + RSA
        print(f"\n*** Progress: {(i+1)/len(commands)*100:.1f}% completed ***")

print("\n" + "="*80)
print("ALL EXPERIMENTS COMPLETED!")
print("="*80)