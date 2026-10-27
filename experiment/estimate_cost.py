"""
精确估算 API 调用成本

计算逻辑：
- 每条 prompt 的 token 数 = 系统描述 + 任务描述 + 数据内容
- 每条 response 的 token 数 = 模型输出（CI/DP 很短，AR 稍长）
- 总 token × 单价 = 成本
"""

# =====================================================
# 每条 prompt 的估算 token 数
# =====================================================

# CI Easy: ~120 tokens (prompt) + ~5 tokens (response)
# CI Hard: ~110 tokens (prompt) + ~5 tokens (response)
# DP:      ~100 tokens (prompt) + ~5 tokens (response)
# AR:      ~800 tokens (prompt, 含50个模块列表) + ~50 tokens (response)

TOKENS_PER_SAMPLE = {
    "ci_easy": {"input": 120, "output": 5},
    "ci_hard": {"input": 110, "output": 5},
    "dp":      {"input": 100, "output": 5},
    "ar":      {"input": 800, "output": 50},
}

# =====================================================
# 采样数量（每项目）
# =====================================================

SAMPLES_PER_PROJECT = {
    "ci_easy": 200,
    "ci_hard": 200,
    "dp":      200,
    "ar":      100,
}

NUM_PROJECTS = 12

# =====================================================
# 模型定价（每百万 token，OpenRouter 价格，2026年4月）
# =====================================================

MODEL_PRICING = {
    "gpt-5":           {"input": 2.50,  "output": 15.00},
    "claude-sonnet-4": {"input": 3.00,  "output": 15.00},
    "gemini-2.5-pro":  {"input": 2.00,  "output": 12.00},
    "qwen3":           {"input": 0.50,  "output": 2.00},
    "kimi-k2.5":       {"input": 0.50,  "output": 2.00},
    "deepseek-v3":     {"input": 0.27,  "output": 1.10},
}

# =====================================================
# 计算
# =====================================================

print("=" * 75)
print("API COST ESTIMATION")
print("=" * 75)

# 每个任务的总 token 数
print("\n--- Per-task token totals (all 12 projects) ---")
task_totals = {}
for task, tokens in TOKENS_PER_SAMPLE.items():
    n = SAMPLES_PER_PROJECT[task] * NUM_PROJECTS
    total_input = n * tokens["input"]
    total_output = n * tokens["output"]
    task_totals[task] = {"input": total_input, "output": total_output, "samples": n}
    print(f"  {task:10s}: {n:5d} samples, {total_input:>10,d} input tokens, {total_output:>8,d} output tokens")

# 所有任务合计
all_input = sum(t["input"] for t in task_totals.values())
all_output = sum(t["output"] for t in task_totals.values())
all_samples = sum(t["samples"] for t in task_totals.values())
print(f"  {'TOTAL':10s}: {all_samples:5d} samples, {all_input:>10,d} input tokens, {all_output:>8,d} output tokens")

# 每个模型的成本
print("\n--- Per-model cost ---")
print(f"  {'Model':20s} {'Input Cost':>12s} {'Output Cost':>12s} {'Total':>10s}")
print("  " + "-" * 58)

grand_total = 0
for model, pricing in MODEL_PRICING.items():
    input_cost = all_input / 1_000_000 * pricing["input"]
    output_cost = all_output / 1_000_000 * pricing["output"]
    total = input_cost + output_cost
    grand_total += total
    print(f"  {model:20s} ${input_cost:>10.2f} ${output_cost:>10.2f} ${total:>8.2f}")

print("  " + "-" * 58)
print(f"  {'ALL 6 MODELS':20s} {'':>12s} {'':>12s} ${grand_total:>8.2f}")

print(f"\n--- Summary ---")
print(f"  Total API calls: {all_samples * len(MODEL_PRICING):,d}")
print(f"  Total cost: ${grand_total:.2f}")
print(f"  Cheapest single model (DeepSeek-V3): ${sum(t['input'] for t in task_totals.values()) / 1e6 * 0.27 + sum(t['output'] for t in task_totals.values()) / 1e6 * 1.10:.2f}")
print(f"  Most expensive single model (Claude): ${sum(t['input'] for t in task_totals.values()) / 1e6 * 3.0 + sum(t['output'] for t in task_totals.values()) / 1e6 * 15.0:.2f}")

print(f"\n--- Budget recommendations ---")
print(f"  Minimum (3 models: GPT-5 + Claude + DeepSeek):  ~${(all_input/1e6*2.5 + all_output/1e6*15) + (all_input/1e6*3 + all_output/1e6*15) + (all_input/1e6*0.27 + all_output/1e6*1.1):.0f}")
print(f"  Full (all 6 models):                             ~${grand_total:.0f}")
print(f"  Full + buffer for retries/errors (+30%):         ~${grand_total * 1.3:.0f}")
