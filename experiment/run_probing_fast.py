"""
Run fast probing analysis - only test a few layers
"""
import sys
import subprocess
import os
import importlib.util

# Add pytorch_env to path FIRST
sys.path.insert(0, r'E:\pytorch_env')

# Load probing_config and patch it
spec = importlib.util.spec_from_file_location("probing_config", "experiment/probing/probing_config.py")
probing_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probing_config)

# Patch LAYERS_TO_PROBE
probing_config.LAYERS_TO_PROBE = [0, 14, 28]  # Only probe 3 layers
print(f"Patched LAYERS_TO_PROBE to {probing_config.LAYERS_TO_PROBE}")

# Set environment
env = os.environ.copy()
env['PYTHONPATH'] = r'E:\pytorch_env;' + env.get('PYTHONPATH', '')

# Run single task with patched config
cmd = ["python", "-c", f"""
import sys
sys.path.insert(0, r'E:\\\\pytorch_env')
sys.path.insert(0, r'.')
sys.path.insert(0, r'./probing')

# Patch LAYERS_TO_PROBE
from probing import probing_config
probing_config.LAYERS_TO_PROBE = [0, 14, 28]
print(f'Using LAYERS_TO_PROBE: {{probing_config.LAYERS_TO_PROBE}}')

from probe_layers import main
import sys as sys_mod
sys_mod.argv = ['probe_layers.py', '--model', 'qwen3-1.7b', '--project', 'openfoam', '--task', 'ci']
main()
print('\\nProbing analysis completed!')
"""]

# Set working directory
os.chdir('experiment')

print(f"\n{'='*80}")
print(f"Running: Fast probing for qwen3-1.7b on openfoam/ci")
print(f"{'='*80}\n")
result = subprocess.run(cmd, env=env)

if result.returncode == 0:
    print("\nProbing analysis completed!")
else:
    print(f"Command failed with return code {result.returncode}")
    sys.exit(1)