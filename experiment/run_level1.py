"""
Level 1 实验设计 (RQ3: Externalization)

目标：验证 reasoning evidence，而不是大规模 benchmark

设计原则：
- 4-5 个系统（类型覆盖：Popular, Industrial, Middle-state）
- 配对实验：L0 vs L1（同一个sample的对比）
- 核心指标：Reasoning Gain = L1 - L0

系统选择：
- Kubernetes (Popular, 依赖 memorization?)
- Spring (Popular)
- OpenFOAM (Industrial, 依赖 reasoning?)
- Nastran (Industrial)
- Hadoop (Middle-state)

模型选择：
- Qwen3-8B (Full)
- Llama-3.1-8B (Full)
- Claude Sonnet 4 (API)
- DeepSeek-V4-Flash (API)
- 可选: Qwen3-32B (少量)

任务：
- CI-Easy: 100-200 samples
- CI-Hard: 100-200 samples  
- CI-Obfuscated: 100-200 samples (关键！检验是否依赖 lexical cues)
- DP: 100-200 samples

Context设计：
- include/import: 全部保留
- code snippet: 50-100 lines
"""

# =====================================================
# Level 1 实验配置
# =====================================================

# 选择的系统（4-5个，形成对比）
LEVEL1_SYSTEMS = [
    'kubernetes',   # Popular - 可能更多依赖 memorization
    'spring',       # Popular
    'openfoam',     # Industrial - 可能更多依赖 reasoning
    'nastran',      # Industrial
    'hadoop',       # Middle-state
]

# 每个系统的采样数
LEVEL1_SAMPLES = {
    'ci': 150,          # CI-Easy
    'ci_hard': 150,     # CI-Hard
    'ci_obfuscated': 150,  # CI-Obfuscated (关键！)
    'dp': 150,          # Dependency Prediction
}

# 使用的模型
LEVEL1_MODELS = [
    'qwen3-8b',           # API
    'llama-3.1-8b',       # API
    'deepseek-v4-flash',  # API
    'claude-sonnet-4',    # API
]

# Context 配置
LEVEL1_CONTEXT = {
    'include_lines': 'all',     # 全部 include/import
    'code_snippet_lines': 100,  # 100 行代码
}

print("Level 1 实验设计:")
print("=" * 60)
print(f"系统: {LEVEL1_SYSTEMS}")
print(f"每任务采样数: {LEVEL1_SAMPLES}")
print(f"模型: {LEVEL1_MODELS}")
print(f"Context: include={LEVEL1_CONTEXT['include_lines']}, code={LEVEL1_CONTEXT['code_snippet_lines']} lines")
print()
print("总样本数:")
total = len(LEVEL1_SYSTEMS) * sum(LEVEL1_SAMPLES.values()) * len(LEVEL1_MODELS)
print(f"  {len(LEVEL1_SYSTEMS)} 系统 × {sum(LEVEL1_SAMPLES.values())} 任务 × {len(LEVEL1_MODELS)} 模型 = {total}")
print()
print("核心分析指标:")
print("  Reasoning Gain (RG) = Performance(L1) - Performance(L0)")
print("  按系统分析 RG，发现 memorization vs reasoning 模式")