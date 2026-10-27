"""
实验配置

使用 OpenRouter (https://openrouter.ai) 作为统一 API 网关。

注意：GPT/Claude/Gemini 在中国区不可用（403），
已替换为中国区可用的同级别模型。
"""

# =====================================================
# 采样配置
# =====================================================

SAMPLE_SIZE_CI = 200   # Component Identification（每项目）
SAMPLE_SIZE_DP = 200   # Dependency Prediction（每项目）
SAMPLE_SIZE_AR = 100   # Architecture Reconstruction（每项目）

# =====================================================
# OpenRouter API Key
# =====================================================

OPENROUTER_API_KEY = ""  # 发布前已移除；使用前填入自己的密钥

# api2d 中转站（已弃用，wildcard 余额耗尽且返回 403；密钥已移除）

# OfoxAI（https://app.ofox.io 获取 API Key）
# 支持 GPT 和 Claude，OpenAI 兼容格式
OFOXAI_API_KEY = ""  # 发布前已移除；使用前填入自己的密钥
OFOXAI_URL = "https://api.ofox.io/v1"

DEEPSEEK_API_KEY = ""  # 发布前已移除；使用前填入自己的密钥
DEEPSEEK_URL = "https://api.deepseek.com"

QWEN3_API_KEY = ""  # 发布前已移除；使用前填入自己的密钥
# QWEN3_URL = "https://llm-bq3lgthgvh81xeoa.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
QWEN3_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# =====================================================
# 模型配置（2026年4月，中国区可用）
#
# 论文对比维度：
#   1. 同系列大 vs 小：DeepSeek-V4-Pro vs V4-Flash
#   2. 新 vs 旧：DeepSeek-V4-Flash vs V3.2
#   3. 不同厂商开源 frontier：Qwen3 vs DeepSeek-V4-Pro vs Kimi-K2.5
#   4. 推理型 vs 通用型：DeepSeek-R1 vs DeepSeek-V4-Pro
#   5. 开源 OpenAI 模型：GPT-OSS-120B（OpenAI 开源的 MoE 模型）
#   6. 西方开源：Llama-4-Maverick vs Mistral-Large
# =====================================================

MODELS = {
    # --- 闭源 Frontier（通过 OfoxAI 调用）---
    # TODO: 在 https://app.ofox.io 确认确切模型名

    # 旧配置（已弃用，wildcard 中转站余额耗尽且返回 403）
    # "gpt-5": {
    #     "provider": "openai",
    #     "model": "gpt-5",
    #     "api_key": API2D_KEY,
    #     "base_url": API2D_URL,
    # },
    # "claude-sonnet-4": {
    #     "provider": "openai",
    #     "model": "claude-sonnet-4-6",
    #     "api_key": API2D_KEY,
    #     "base_url": API2D_URL + "/v1",
    # },

    "gpt-6-sol": {
        "provider": "openai",
        "model": "openai/gpt-6-sol",   # TODO: 确认 OfoxAI 上的确切模型名
        "api_key": OFOXAI_API_KEY,
        "base_url": OFOXAI_URL,
    },
    "claude-opus-5.5": {
        "provider": "anthropic",
        "model": "anthropic/claude-opus-5.5",  # TODO: 确认 OfoxAI 上的确切模型名
        "api_key": OFOXAI_API_KEY,
        "base_url": "https://api.ofox.io/anthropic",
    },

    # --- 开源 Frontier（中国厂商，通过 OpenRouter）---
    "deepseek-v4-pro": {
        "provider": "openai",
        "model": "deepseek/deepseek-v4-pro",
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
    },
    # 2026-10-22：AKG v2 双模型一致性复核用，走 DeepSeek 官方 API（与 qwen 标注分属不同厂商）。
    # 官方模型名不带 deepseek/ 前缀；标注只需短 JSON，max_tokens 取 4096。
    "deepseek-v4-pro-official": {
        "provider": "openai",
        "model": "deepseek-v4-pro",
        "api_key": DEEPSEEK_API_KEY,
        "base_url": "https://api.deepseek.com",
        "max_tokens": 4096,
    },
    "deepseek-v4-flash": {
        "provider": "openai",
        "model": "deepseek-flash",   # 官方 API 模型名（实测支持：deepseek-flash, deepseek-v4-pro）
        "api_key": DEEPSEEK_API_KEY,
        "base_url": "https://api.deepseek.com",
        # deepseek-flash 推理量大，16384 仍在部分 AR 样本上 finish_reason=length 截断（2026-10-06 三轮 patch 后剩 52 个），提高到 32768
        # 2026-10-08：32768 仍有 3 个 AR 题截断（tensorflow 1、pytorch 2）。官方 deepseek-v4-flash 最大输出 384K（api.deepseek.com），提高到 65536 补这 3 题
        # "max_tokens": 32768,
        "max_tokens": 65536,
    },
    "qwen3": {
        "provider": "openai",
        # 2026-10-21：原端点 qwen/qwen3-235b-a22b 已于 2026-10-09 下线（linux L0 全量 404），
        # 改用同一 235B 模型的 2507 快照，模型名与论文保持一致。
        # "model": "qwen/qwen3-235b-a22b",
        "model": "qwen/qwen3-235b-a22b-2507",
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
    },
    "kimi-k2.5": {
        "provider": "openai",
        "model": "moonshotai/kimi-k2.5",
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
    },
    "kimi-k3": {
        "provider": "openai",
        "model": "moonshotai/kimi-k3",
        # OfoxAI 强制 temperature=1，与实验协议 temperature=0 冲突，改用 OpenRouter
        # "api_key": OFOXAI_API_KEY,
        # "base_url": OFOXAI_URL,
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
        "reasoning_effort": "low",   # K3 默认 max 会推理过长导致超时；low 实测 3-13s 无超时
    },

    # --- 推理型模型 ---
    "deepseek-r1": {
        "provider": "openai",
        "model": "deepseek/deepseek-r1",
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
    },

    # --- 西方开源 ---
    "llama-4-maverick": {
        "provider": "openai",
        "model": "meta-llama/llama-4-maverick",
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
    },
    "gpt-oss-120b": {
        "provider": "openai",
        "model": "openai/gpt-oss-120b",
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
    },

    # --- 旧版本（消融对比）---
    "deepseek-v3.2": {
        "provider": "openai",
        "model": "deepseek/deepseek-v3.2",
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
    },

    # --- 小模型（规模效应分析）---
    # 旧配置（OpenRouter 共享池，2026-10-06/07 实测 Alibaba 上游对 qwen3-8b 频繁 429 限流，已完成 24/48 文件）
    # "qwen3-8b": {
    #     "provider": "openai",
    #     "model": "qwen/qwen3-8b",
    #     "api_key": OPENROUTER_API_KEY,
    #     "base_url": "https://openrouter.ai/api/v1",
    # },
    # 新配置（2026-10-07）：改用阿里云百炼官方 DashScope，避开 OpenRouter 共享池限流。
    # Qwen3 在 DashScope 非流式下被强制 enable_thinking=false，而 OpenRouter 默认 thinking ON，
    # 为保持口径一致，使用流式 stream=True + enable_thinking=true（两者均为 Alibaba 同源 qwen3-8b）。
    # 2026-10-23：Qwen3-8B 已于 2026-10-09 在 OpenRouter 与 DashScope 两处下线
    # （DashScope 返回 404 model_not_found，OpenRouter 明确 deprecated），linux 的 v2 DP/AR
    # 无法补跑。改用 DashScope 在线的同代 dense 小模型 qwen3.8-27b 作为 8B 小模型的替代，
    # 全量重跑 12 系统 × 4 任务。非流式下 enable_thinking=true 已验证生效（reasoning_tokens>0）。
    # 旧配置保留：
    # "qwen3-8b": {
    #     "provider": "openai",
    #     "model": "qwen3-8b",
    #     "api_key": QWEN3_API_KEY,
    #     "base_url": QWEN3_URL,
    #     "stream": True,
    #     "extra_params": {"enable_thinking": True},
    # },
    "qwen3.8-27b": {
        "provider": "openai",
        "model": "qwen3.8-27b",
        "api_key": QWEN3_API_KEY,
        "base_url": QWEN3_URL,
        "max_tokens": 8192,
        # 显式关闭 thinking：开启时实测约 41 s/题（1614 completion tokens），
        # 关闭约 1 s/题，全量 7357 题相差约 80 小时。关闭也与该模型在 DashScope 上的默认行为一致。
        "extra_params": {"enable_thinking": False},
    },
    # 2026-10-22：AKG v2 人工金标完成后，用于双模型一致性复核的国产标注模型。
    # 走 DashScope 官方兼容端点（同一 QWEN3_API_KEY），不经过 OpenRouter，避免版本漂移。
    # 原计划的 qwen3-235b-a22b-2507 在当前账号的 DashScope 上已不可用（404 model_not_found，
    # 用户模型列表里只有 qwen3.5/3.6/3.7/3.8 系列），改用同年旗舰快照 qwen3.8-max-0902，
    # 固定日期快照便于论文里标注版本。非流式调用下 Qwen3 默认关闭 thinking，无需传 enable_thinking。
    "qwen3.8-max-dashscope": {
        "provider": "openai",
        "model": "qwen3.8-max-0902",
        "api_key": QWEN3_API_KEY,
        "base_url": QWEN3_URL,
        "max_tokens": 4096,
    },
    "llama-3.1-8b": {
        "provider": "openai",
        "model": "meta-llama/llama-3.1-8b-instruct",
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
    },
}

# =====================================================
# AKG 标签标注模型（2026-10，R2-4 / R3-2 人工标注的 LLM 辅助）
#
# 仅用于 akg_ai_annotate.py 的标签可靠性标注，不参与论文评测，
# 不得并入 MODELS（否则会被实验脚本当作参评模型遍历）。
# 选取原则：与参评的 9 个模型同厂商但不同型号（claude-sonnet-5.5
# vs 参评的 claude-opus-5.5），或厂商完全未参评（gemini-4-argon），
# 降低标注误差与评测误差相关的风险。最终以人工金标为准。
# =====================================================

ANNOTATION_MODELS = {
    "gemini-4-argon": {
        "provider": "openai",
        "model": "google/gemini-4-argon",
        "api_key": OFOXAI_API_KEY,
        "base_url": OFOXAI_URL,
        "max_tokens": 4096,
    },
    "claude-sonnet-5.5": {
        "provider": "anthropic",
        "model": "anthropic/claude-sonnet-5.5",
        "api_key": OFOXAI_API_KEY,
        "base_url": "https://api.ofox.io/anthropic",
        "max_tokens": 4096,
    },
}

# =====================================================
# 项目列表
# =====================================================

PROJECTS = [
    {"name": "openfoam",      "path": "../systems/openfoam",      "system": "OpenFOAM",         "domain": "CFD simulation"},
    {"name": "nastran",       "path": "../systems/nastran",       "system": "NASTRAN-95",       "domain": "structural analysis"},
    {"name": "salome",        "path": "../systems/salome",        "system": "SALOME",           "domain": "CAE platform"},
    {"name": "paraview",      "path": "../systems/paraview",      "system": "ParaView",         "domain": "scientific visualization"},
    {"name": "kubernetes",    "path": "../systems/kubernetes",    "system": "Kubernetes",       "domain": "container orchestration"},
    {"name": "linux",         "path": "../systems/linux",         "system": "Linux Kernel",     "domain": "operating system kernel"},
    {"name": "hadoop",        "path": "../systems/hadoop",        "system": "Apache Hadoop",    "domain": "distributed computing"},
    {"name": "spring",        "path": "../systems/spring",        "system": "Spring Framework", "domain": "enterprise Java framework"},
    {"name": "codeaster",     "path": "../systems/codeaster",     "system": "Code_Aster",       "domain": "finite element analysis"},
    {"name": "openmodelica",  "path": "../systems/openmodelica",  "system": "OpenModelica",     "domain": "modeling and simulation"},
    {"name": "tensorflow",    "path": "../systems/tensorflow",    "system": "TensorFlow",       "domain": "machine learning framework"},
    {"name": "pytorch",       "path": "../systems/pytorch",       "system": "PyTorch",          "domain": "machine learning framework"},
]
