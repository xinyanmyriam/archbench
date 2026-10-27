"""
统一的 LLM 调用客户端，支持 OpenAI 兼容接口和 Anthropic

call_llm() 返回 str（向后兼容）
call_llm_with_usage() 返回 dict: {"text": str, "usage": {...}}
"""

import re
import time
import json


def call_llm(model_config, prompt, max_retries=5, temperature=0.0):
    """
    调用 LLM，返回文本响应（向后兼容）
    """
    result = call_llm_with_usage(model_config, prompt, max_retries, temperature)
    return result["text"]


def call_llm_with_usage(model_config, prompt, max_retries=5, temperature=0.0):
    """
    调用 LLM，返回文本 + token 用量

    Returns:
        dict: {
            "text": str,
            "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
            "model": str,
            "latency_ms": float,
        }
    """
    provider = model_config["provider"]

    # 429/熔断单独计数：按 "Retry after ~Ns" 等待，不消耗普通重试次数
    rate_limit_waits = 0
    max_rate_limit_waits = 30
    attempt = 0
    while True:
        try:
            start_time = time.time()

            if provider == "openai":
                text, usage = _call_openai_compatible(model_config, prompt, temperature)
            elif provider == "anthropic":
                text, usage = _call_anthropic(model_config, prompt, temperature)
            elif provider == "api2d_claude":
                text, usage = _call_api2d_claude(model_config, prompt, temperature)
            else:
                raise ValueError(f"Unknown provider: {provider}")

            latency_ms = (time.time() - start_time) * 1000

            return {
                "text": text,
                "usage": usage,
                "model": model_config.get("model", "unknown"),
                "latency_ms": round(latency_ms, 1),
            }

        except Exception as e:
            msg = str(e)
            # 400 内容安全策略拒绝：永久失败，不重试，避免消耗配额并触发熔断
            if "400" in msg and "content safety" in msg.lower():
                print(f"  Content safety rejected (no retry): {msg[:120]}")
                return {
                    "text": f"ERROR: {e}",
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "model": model_config.get("model", "unknown"),
                    "latency_ms": 0,
                }
            if ("429" in msg or "rate limit" in msg.lower()) and rate_limit_waits < max_rate_limit_waits:
                rate_limit_waits += 1
                m = re.search(r"[Rr]etry after ~?(\d+)", msg)
                wait = int(m.group(1)) + 3 if m else 30
                print(f"  Rate limited ({rate_limit_waits}/{max_rate_limit_waits}), sleep {wait}s: {msg[:120]}")
                time.sleep(wait)
                continue
            print(f"  Attempt {attempt + 1} failed: {e}")
            attempt += 1
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return {
                    "text": f"ERROR: {e}",
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "model": model_config.get("model", "unknown"),
                    "latency_ms": 0,
                }


def _call_openai_compatible(config, prompt, temperature):
    """OpenAI 兼容接口，返回 (text, usage_dict)"""
    # 流式分支：Qwen3 等模型在非流式下被强制关闭 thinking，需用流式 + enable_thinking=true
    if config.get("stream"):
        return _call_openai_streaming(config, prompt, temperature)

    from openai import OpenAI

    client = OpenAI(
        api_key=config["api_key"],
        base_url=config.get("base_url", "https://api.openai.com/v1"),
    )

    max_tokens = config.get("max_tokens", 4096)

    # 推理模型的推理强度配置（如 kimi-k3 的 reasoning_effort: low/high/max）
    create_kwargs = {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if config.get("reasoning_effort"):
        create_kwargs["reasoning_effort"] = config["reasoning_effort"]
    if config.get("extra_params"):
        # 非标准字段（如 DashScope 的 enable_thinking）必须走 extra_body，
        # 直接放进 create_kwargs 会被 openai SDK 判为非法参数
        create_kwargs["extra_body"] = config["extra_params"]

    response = client.chat.completions.create(**create_kwargs)

    choice = response.choices[0]
    finish_reason = choice.finish_reason
    content = choice.message.content

    # 空响应或被截断 → 触发重试（推理模型的 reasoning 可能耗尽 token 预算）
    if not content or not content.strip():
        raise Exception(f"Empty response (finish_reason={finish_reason})")
    if finish_reason == "length":
        raise Exception(f"Response truncated (finish_reason=length), consider increasing max_tokens")

    text = content.strip()

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens or 0,
            "completion_tokens": response.usage.completion_tokens or 0,
            "total_tokens": response.usage.total_tokens or 0,
        }

    return text, usage


def _call_openai_streaming(config, prompt, temperature):
    """OpenAI 兼容接口的流式调用，返回 (text, usage_dict)

    用于 Qwen3 等模型：DashScope 非流式强制 enable_thinking=false，
    必须用流式才能开启 thinking，以与 OpenRouter 默认（thinking ON）口径一致。
    """
    import requests

    base_url = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
    max_tokens = config.get("max_tokens", 4096)

    body = {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if config.get("reasoning_effort"):
        body["reasoning_effort"] = config["reasoning_effort"]
    if config.get("extra_params"):
        body.update(config["extra_params"])

    resp = requests.post(
        base_url + "/chat/completions",
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=300,
        stream=True,
    )
    # 连接建立即返回状态码；非 200 时读取错误体，格式对齐 openai SDK 以便上层识别 429/400
    if resp.status_code != 200:
        raise Exception(f"Error code: {resp.status_code} - {resp.text[:300]}")

    content_parts = []
    finish_reason = None
    usage = None
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except Exception:
            continue
        if chunk.get("usage"):
            u = chunk["usage"]
            usage = {
                "prompt_tokens": u.get("prompt_tokens", 0) or 0,
                "completion_tokens": u.get("completion_tokens", 0) or 0,
                "total_tokens": u.get("total_tokens", 0) or 0,
            }
        for ch in chunk.get("choices", []):
            delta = ch.get("delta", {}) or {}
            # 只累积最终答案 content；reasoning_content（思考过程）不计入 text
            if delta.get("content"):
                content_parts.append(delta["content"])
            if ch.get("finish_reason"):
                finish_reason = ch["finish_reason"]

    content = "".join(content_parts)
    if not content or not content.strip():
        raise Exception(f"Empty response (finish_reason={finish_reason})")
    if finish_reason == "length":
        raise Exception(f"Response truncated (finish_reason=length), consider increasing max_tokens")

    if usage is None:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return content.strip(), usage


def _call_anthropic(config, prompt, temperature):
    """Anthropic Claude 接口"""
    # 新版 anthropic SDK 的 messages.create() 不再接受 temperature，
    # 为保持 temperature=0 的实验协议，直接用 HTTP 调用 Messages API
    import requests

    base_url = (config.get("base_url") or "https://api.anthropic.com").rstrip("/")
    max_tokens = config.get("max_tokens", 4096)

    resp = requests.post(
        f"{base_url}/v1/messages",
        headers={
            "x-api-key": config["api_key"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": config["model"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=300,
    )
    if resp.status_code != 200:
        raise Exception(f"Anthropic HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()

    # 跳过 thinking 等非文本块
    texts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    stop_reason = data.get("stop_reason")
    if not "".join(texts).strip():
        raise Exception(f"Empty response from Anthropic (stop_reason={stop_reason})")
    if stop_reason == "max_tokens":
        raise Exception("Response truncated (stop_reason=max_tokens)")
    text = "".join(texts).strip()

    u = data.get("usage") or {}
    inp, out = u.get("input_tokens", 0) or 0, u.get("output_tokens", 0) or 0
    usage = {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out}

    return text, usage


def _call_api2d_claude(config, prompt, temperature):
    """api2d / wildcard 中转站"""
    import httpx

    url = config["base_url"].rstrip("/") + "/chat/completions"

    resp = httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        },
        json={
            "model": config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.get("max_tokens", 4096),
            "temperature": temperature,
        },
        timeout=120,
    )

    data = resp.json()

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if "usage" in data:
        usage = {
            "prompt_tokens": data["usage"].get("prompt_tokens", 0),
            "completion_tokens": data["usage"].get("completion_tokens", 0),
            "total_tokens": data["usage"].get("total_tokens", 0),
        }

    if "choices" in data:
        text = data["choices"][0]["message"]["content"].strip()
        return text, usage
    elif "content" in data:
        for block in data["content"]:
            if block.get("type") == "text":
                return block["text"].strip(), usage
        return "", usage
    elif "error" in data:
        raise Exception(f"API error: {data['error'].get('message', str(data['error']))}")
    else:
        raise Exception(f"Unexpected response: {json.dumps(data)[:200]}")
