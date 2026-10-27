"""
Level 0 重跑队列：按供应商分组，组内模型依次串行执行，避免同渠道并发触发限速。

用法:
    python -u run_rerun_queue.py status                 # 统计全部阵容的 缺失/不对齐/含错误/旧代码 文件
    python -u run_rerun_queue.py deepseek --smoke       # 只跑 nastran 4 个任务（结果直接作为正式结果保留）
    python -u run_rerun_queue.py deepseek               # 跑完整 12 项目 × 4 任务
    python -u run_rerun_queue.py openrouter             # OpenRouter 渠道依次跑
    python -u run_rerun_queue.py ofoxai                 # 仅手动：gpt-6-sol / opus 已基本完成

不同供应商的队列可在不同终端并行启动；同一供应商通过锁文件 logs/rerun_<group>.lock 防止重复启动。
每个模型日志写入 logs/rerun_<model>.log；单个模型失败不中断队列；每个模型跑完后自动做清单对齐 + API 错误审计。
断点续跑会跳过 results_rerun/ 中已有文件，只补缺失的。
"""
import ctypes
import json
import os
import subprocess
import sys
import time

from build_sample_manifest import MANIFEST_PATH, TASKS, expected_keys, result_keys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = "results_rerun"
LOG_DIR = os.path.join(HERE, "logs")
SMOKE_PROJECT = "nastran"

# Level 0 修订阵容（按供应商分组、按执行顺序）。
# 不在阵容中：claude-sonnet-4、gpt-5、kimi-k2.5（已弃用）；deepseek-v4-pro / r1 / v3.2（无结果）。
ROSTER = {
    "deepseek": [
        "deepseek-v4-flash",   # 官方 API（api.deepseek.com）
    ],
    "openrouter": [
        "kimi-k3",
        "gpt-oss-120b",
        "qwen3",
        "llama-4-maverick",
        "llama-3.1-8b",
        "qwen3-8b",
    ],
    "ofoxai": [
        "gpt-6-sol",
        "claude-opus-5.5",
    ],
}


def is_api_error(response):
    return (not response) or response.startswith("ERROR:")


# ---------------- 锁文件（防同一供应商重复启动） ----------------

def pid_alive(pid):
    """Windows 下 os.kill(pid, 0) 会结束进程，故用 OpenProcess/GetExitCodeProcess 判断。"""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return bool(ok) and code.value == 259  # STILL_ACTIVE


def acquire_lock(group):
    path = os.path.join(LOG_DIR, f"rerun_{group}.lock")
    if os.path.exists(path):
        with open(path) as f:
            old = f.read().strip()
        if old.isdigit() and pid_alive(int(old)):
            print(f"queue '{group}' already running (pid {old}); lock: {path}")
            sys.exit(2)
        print(f"stale lock (pid {old}) removed")
    with open(path, "w") as f:
        f.write(str(os.getpid()))
    return path


# ---------------- 审计 ----------------

def check_file(model, proj, task, manifest):
    """返回 (state, n_err)。state ∈ missing / old_code / misaligned / ok"""
    path = os.path.join(HERE, OUTPUT_DIR, f"{model}_{proj}_{task}.json")
    if not os.path.exists(path):
        return "missing", 0
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    if "n_api_errors" not in res:
        return "old_code", 0
    if result_keys(task, res) != expected_keys(task, manifest["projects"][proj][task]):
        return "misaligned", 0
    n_err = sum(1 for r in res.get("results", []) if is_api_error(r.get("raw_response", "")))
    return "ok", n_err


def audit_model(model, manifest, projects=None):
    stat = {"ok": 0, "missing": [], "old_code": [], "misaligned": [], "with_errors": []}
    for proj in projects or manifest["projects"]:
        for task in TASKS:
            state, n_err = check_file(model, proj, task, manifest)
            name = f"{proj}_{task}"
            if state == "ok":
                stat["ok"] += 1
                if n_err:
                    stat["with_errors"].append(f"{name}({n_err})")
            else:
                stat[state].append(name)
    return stat


def print_stat(model, s):
    print(f"  {model:20s} ok={s['ok']:2d} missing={len(s['missing']):2d} "
          f"misaligned={len(s['misaligned'])} old_code={len(s['old_code'])} "
          f"with_errors={len(s['with_errors'])}")
    for k in ["misaligned", "old_code", "with_errors"]:
        if s[k]:
            print(f"      {k}: {', '.join(s[k])}")


def load_manifest():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def status():
    manifest = load_manifest()
    n_total = len(manifest["projects"]) * len(TASKS)
    print(f"=== Level 0 status (results_rerun/, {n_total} files per model) ===")
    to_run = 0
    for group, models in ROSTER.items():
        print(f"[{group}]")
        for m in models:
            s = audit_model(m, manifest)
            print_stat(m, s)
            to_run += len(s["missing"]) + len(s["misaligned"]) + len(s["old_code"])
    print(f"\nfiles to (re)run: {to_run}  (misaligned/old_code 需先移入隔离目录)")


# ---------------- 队列 ----------------

def run_queue(group, smoke):
    os.makedirs(LOG_DIR, exist_ok=True)
    lock = acquire_lock(group)
    manifest = load_manifest()
    projects = [SMOKE_PROJECT] if smoke else None
    summary = []
    try:
        for model in ROSTER[group]:
            log_path = os.path.join(LOG_DIR, f"rerun_{model}.log")
            cmd = [sys.executable, "-u", "run_experiment.py",
                   "--model", model, "--task", "all", "--output", OUTPUT_DIR]
            if smoke:
                cmd += ["--project", SMOKE_PROJECT]
            print(f"\n=== [{time.strftime('%H:%M:%S')}] start {model}{' (smoke)' if smoke else ''} -> {log_path} ===",
                  flush=True)
            t0 = time.time()
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} rerun start {' '.join(cmd[2:])} =====\n")
                log.flush()
                p = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
            mins = (time.time() - t0) / 60
            s = audit_model(model, manifest, projects)
            summary.append((model, p.returncode, mins, s))
            print(f"=== [{time.strftime('%H:%M:%S')}] done {model} exit={p.returncode} ({mins:.1f} min) ===", flush=True)
            print_stat(model, s)
    finally:
        os.remove(lock)

    print("\n=== queue summary ===")
    for model, code, mins, s in summary:
        print(f"  {model:20s} exit={code}  {mins:.1f} min  ok={s['ok']} with_errors={len(s['with_errors'])}")
    print("含错误的文件用 patch_errors.py --model <m> 补跑出错样本。")
    sys.exit(1 if any(c for _, c, _, _ in summary) else 0)


def main():
    args = sys.argv[1:]
    if args == ["status"]:
        return status()
    smoke = "--smoke" in args
    groups = [a for a in args if a != "--smoke"]
    if len(groups) != 1 or groups[0] not in ROSTER:
        print(__doc__)
        sys.exit(1)
    run_queue(groups[0], smoke)


if __name__ == "__main__":
    main()
