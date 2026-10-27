"""
统一实验日志系统

每次实验运行都会记录：
- 开始/结束时间戳
- 模型名、项目名、任务名
- 结果摘要（准确率等）
- API 调用次数和错误次数

日志保存在 experiment/logs/ 目录下
"""

import os
import json
import time
from datetime import datetime

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, "experiment_log.jsonl")


def log_experiment(experiment_name, model, project, task, result, extra=None):
    """记录一条实验日志"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "unix_time": time.time(),
        "experiment": experiment_name,
        "model": model,
        "project": project,
        "task": task,
        "result_summary": _summarize_result(result),
        "extra": extra or {},
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


def _summarize_result(result):
    """从结果中提取关键指标"""
    if isinstance(result, dict):
        summary = {}
        for key in ["accuracy", "precision", "recall", "f1", "precision_at_5",
                    "total", "correct", "tp", "fp", "tn", "fn"]:
            if key in result:
                summary[key] = result[key]
        return summary
    return {"raw": str(result)[:200]}


def log_start(experiment_name, description=""):
    """记录实验开始"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "unix_time": time.time(),
        "event": "START",
        "experiment": experiment_name,
        "description": description,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[{entry['timestamp']}] START: {experiment_name}")


def log_end(experiment_name, summary=""):
    """记录实验结束"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "unix_time": time.time(),
        "event": "END",
        "experiment": experiment_name,
        "summary": summary,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[{entry['timestamp']}] END: {experiment_name}")


def log_error(experiment_name, model, project, task, error_msg):
    """记录错误"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "unix_time": time.time(),
        "event": "ERROR",
        "experiment": experiment_name,
        "model": model,
        "project": project,
        "task": task,
        "error": str(error_msg)[:500],
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
