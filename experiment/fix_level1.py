"""修复 run_level1_icse.py 中的 call_llm 调用"""
content = open('run_level1_icse.py', encoding='utf-8').read()

# 替换 CI 任务中的 call_llm
old1 = '        response = call_llm(model_config, prompt)\n        predicted = parse_ci_response(response)'
new1 = '        response_data = call_llm_with_usage(model_config, prompt)\n        response = response_data["text"]\n        predicted = parse_ci_response(response)'
content = content.replace(old1, new1)

# 替换 DP 任务中的 call_llm
old2 = '        response = call_llm(model_config, prompt)\n        predicted = parse_dp_response(response)'
new2 = '        response_data = call_llm_with_usage(model_config, prompt)\n        response = response_data["text"]\n        predicted = parse_dp_response(response)'
content = content.replace(old2, new2)

# 在所有 log_detail 中加入 usage 和 latency
old3 = '"response_raw": response[:200] if response else "",'
new3 = '"response_raw": response[:200] if response else "",\n            "usage": response_data.get("usage", {}),\n            "latency_ms": response_data.get("latency_ms", 0),'
content = content.replace(old3, new3)

open('run_level1_icse.py', 'w', encoding='utf-8').write(content)
print(f"Done. call_llm occurrences remaining: {content.count('call_llm(')}")
print(f"call_llm_with_usage occurrences: {content.count('call_llm_with_usage')}")
