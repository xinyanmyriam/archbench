"""
重跑前的准备：把可直接复用的结果从 results/ 复制到 results_rerun/，让断点续跑自动跳过。

复制条件（同时满足）：
  1. 模型在 Level 0 修订阵容中（run_rerun_queue.ROSTER）；
  2. 由新代码产出（结果中有 n_api_errors 字段；4 月旧代码 max_tokens=256、无 ERROR 防护，不复用）；
  3. 样本与 samples/manifest_seed42.json 逐条一致；
  4. results_rerun/ 中尚无同名文件。
含 API 错误的文件照样复制（避免整份重跑），由 status 标出后用 patch_errors.py 补。

用法:
    python prepare_rerun.py
"""
import json
import os
import shutil

from build_sample_manifest import MANIFEST_PATH, collect_effective, expected_keys, result_keys
from run_rerun_queue import ROSTER

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "results")
DST_DIR = os.path.join(HERE, "results_rerun")


def is_api_error(response):
    return (not response) or response.startswith("ERROR:")


def main():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    roster = {m for models in ROSTER.values() for m in models}
    copied = []
    for (model, proj, task), _ in sorted(collect_effective().items()):
        if model not in roster:
            continue
        src = os.path.join(SRC_DIR, f"{model}_{proj}_{task}.json")
        dst = os.path.join(DST_DIR, os.path.basename(src))
        if os.path.exists(dst) or not os.path.exists(src):
            continue
        with open(src, encoding="utf-8") as f:
            res = json.load(f)
        if "n_api_errors" not in res:
            continue
        if result_keys(task, res) != expected_keys(task, manifest["projects"][proj][task]):
            continue
        n_err = sum(1 for r in res.get("results", []) if is_api_error(r.get("raw_response", "")))
        shutil.copy2(src, dst)
        copied.append((os.path.basename(src), n_err))
    print(f"copied {len(copied)} files:")
    for name, n_err in copied:
        print(f"  {name}" + (f"  (api_errors={n_err}, 需 patch)" if n_err else ""))


if __name__ == "__main__":
    main()
