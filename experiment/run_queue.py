"""
模型排队执行器

逻辑：依次跑 queue 中的模型，每个跑完再启动下一个。
同时 Qwen3 和 Llama-4 还在后台跑着，这个脚本占第三个"槽位"。

队列顺序：
1. deepseek-r1（推理型，和通用型对比）
2. kimi-k2.5（Moonshot 开源）
3. deepseek-v3.2（旧版本，消融对比）
4. deepseek-v4-pro（最强，放最后）
"""

import subprocess
import sys
import time

QUEUE = [
    "deepseek-r1",
    "kimi-k2.5",
    "deepseek-v3.2",
    "deepseek-v4-pro",
]

for i, model in enumerate(QUEUE):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(QUEUE)}] Starting: {model}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, "run_experiment.py",
         "--model", model, "--project", "all", "--task", "all"],
        cwd=".",
    )

    if result.returncode == 0:
        print(f"\n[OK] {model} completed successfully.")
    else:
        print(f"\n[WARN] {model} exited with code {result.returncode}.")

    # 短暂等待，避免限流
    if i < len(QUEUE) - 1:
        print("Waiting 30s before next model...")
        time.sleep(30)

print("\n" + "="*60)
print("ALL MODELS IN QUEUE COMPLETED")
print("="*60)
