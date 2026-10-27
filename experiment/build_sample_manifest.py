"""
固定样本清单生成 + 现有结果对齐审计

1) build：复现 run_experiment.py 在 "单模型、从头一次跑完、seed=42" 下的抽样过程，
   把每个 (project, task) 抽中的行号和校验 key 写入 samples/manifest_seed42.json。
   之后所有模型 / Level 0/1/2 只读该清单，不再在运行时抽样。
2) audit：对 results/ 与 results_rerun/ 中每个模型实际生效的结果文件
   （results_rerun 优先），按清单逐条有序比对，列出需重跑的文件。

用法:
    python build_sample_manifest.py            # build + audit
    python build_sample_manifest.py --audit    # 只 audit（清单已存在）
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict

from config import PROJECTS, SAMPLE_SIZE_CI, SAMPLE_SIZE_DP, SAMPLE_SIZE_AR

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(HERE, "samples", "manifest_seed42.json")
SEED = 42
TASKS = ["ci", "ci_hard", "dp", "ar"]
TASK_FILES = {
    "ci": "component_identification.jsonl",
    "ci_hard": "component_identification_hard.jsonl",
    "dp": "dependency_prediction.jsonl",
    "ar": "architecture_reconstruction.jsonl",
}
SIZES = {"ci": SAMPLE_SIZE_CI, "ci_hard": SAMPLE_SIZE_CI, "dp": SAMPLE_SIZE_DP, "ar": SAMPLE_SIZE_AR}
# 用于校验的参考模型（已确认三者样本互相一致 = 干净一次跑完的结果）
REFERENCE_MODELS = ["gpt-6-sol", "claude-opus-5.5", "claude-sonnet-4"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def item_key(task, item):
    if task == "ci":
        return [item["file"], item["label"]]
    if task == "ci_hard":
        return [item.get("file_full") or item["file"], item["label"]]
    if task == "dp":
        return [item["module_A"], item["module_B"], item["label"]]
    raise ValueError(task)


def build():
    # 与 run_experiment.py 一致：import 时 run_experiment 与 baselines 各 seed 一次
    random.seed(SEED)
    random.seed(SEED)
    manifest = {"seed": SEED, "sizes": SIZES, "projects": {}}
    for proj in PROJECTS:
        ppath = os.path.normpath(os.path.join(HERE, proj["path"]))
        manifest["projects"][proj["name"]] = {}
        for task in TASKS:
            src = os.path.join(ppath, TASK_FILES[task])
            data = read_jsonl(src)
            n, k = len(data), SIZES[task]
            # random.sample 选出的位置只取决于 (len, k)，对 range(n) 抽样与对 data 抽样等价
            idx = random.sample(range(n), k) if n > k else list(range(n))
            entry = {
                "source_file": os.path.relpath(src, HERE).replace("\\", "/"),
                "n_pool": n,
                "indices": idx,
            }
            if task == "ar":
                gt = defaultdict(set)
                for i in idx:
                    gt[data[i]["source_module"]].add(data[i]["target_module"])
                sources = list(gt.keys())
                if len(sources) > SAMPLE_SIZE_AR:
                    sources = random.sample(sources, SAMPLE_SIZE_AR)
                entry["sources"] = sources
                entry["true_targets"] = {s: sorted(gt[s]) for s in sources}
            else:
                entry["keys"] = [item_key(task, data[i]) for i in idx]
            manifest["projects"][proj["name"]][task] = entry
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"Manifest written: {MANIFEST_PATH}")
    return manifest


def result_keys(task, res):
    rows = res.get("results", [])
    if task == "ar":
        return [r["source_module"] for r in rows]
    if task == "ci_hard":
        return [[r.get("file_full") or r["file"], r["label"]] for r in rows]
    if task == "ci":
        return [[r["file"], r["label"]] for r in rows]
    return [[r["module_A"], r["module_B"], r["label"]] for r in rows]


def expected_keys(task, entry):
    return entry["sources"] if task == "ar" else entry["keys"]


def parse_name(fname, proj_names):
    base = fname[:-5]
    for task in ["ci_hard", "ci", "dp", "ar"]:
        for p in proj_names:
            suf = f"_{p}_{task}"
            if base.endswith(suf):
                return base[: -len(suf)], p, task
    return None


def collect_effective():
    proj_names = [p["name"] for p in PROJECTS]
    eff = {}
    for d in ["results", "results_rerun"]:  # 后者覆盖前者
        for path in glob.glob(os.path.join(HERE, d, "*.json")):
            parsed = parse_name(os.path.basename(path), proj_names)
            if parsed:
                eff[parsed] = path
    return eff


def audit(manifest):
    eff = collect_effective()
    report = defaultdict(lambda: {"ok": [], "bad": []})
    for (model, proj, task), path in sorted(eff.items()):
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
        got = result_keys(task, res)
        exp = expected_keys(task, manifest["projects"][proj][task])
        tup = lambda x: tuple(x) if isinstance(x, list) else x
        overlap = len(set(map(tup, got)) & set(map(tup, exp)))
        rec = {"project": proj, "task": task, "file": os.path.relpath(path, HERE).replace("\\", "/"),
               "n": len(got), "overlap": overlap, "expected": len(exp)}
        report[model]["ok" if got == exp else "bad"].append(rec)

    print("\n=== Reference check ===")
    for m in REFERENCE_MODELS:
        if m in report:
            print(f"  {m}: ok={len(report[m]['ok'])} bad={len(report[m]['bad'])}")

    print("\n=== Per-model summary (files needing rerun) ===")
    total_bad = 0
    for model in sorted(report):
        ok, bad = report[model]["ok"], report[model]["bad"]
        total_bad += len(bad)
        by_task = defaultdict(int)
        for r in bad:
            by_task[r["task"]] += 1
        print(f"  {model:28s} ok={len(ok):2d} bad={len(bad):2d}  {dict(by_task)}")
    print(f"  TOTAL bad files: {total_bad}")

    out = os.path.join(HERE, "samples", "audit_seed42.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\nDetailed report: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true", help="only audit with existing manifest")
    args = ap.parse_args()
    if args.audit:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = build()
    audit(manifest)


if __name__ == "__main__":
    main()
