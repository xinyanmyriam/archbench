"""
按标准定义重算 AR Precision@5 = |top5 ∩ true| / 5（原实现分母为实际返回数）。
读取 results_rerun/ 下 L0/L1/L2 的 AR 结果 JSON，不修改原始文件，
输出 results_rerun/ar_p5_standard.json（逐系统 + 12 系统宏平均）。

用法（在 experiment/ 目录下运行）：python recompute_ar_p5.py
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "results_rerun")
LEVELS = {
    "L0": (ROOT, r"^(.+)_([a-z]+)_ar\.json$"),
    "L1": (os.path.join(ROOT, "level1"), r"^(.+)_([a-z]+)_ar_l1\.json$"),
    "L2": (os.path.join(ROOT, "level2"), r"^(.+)_([a-z]+)_ar_l2\.json$"),
}


def recompute(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    recs = d["results"]
    p_std = []
    for r in recs:
        hits = len(set(r["predicted"][:5]) & set(r["true_targets"]))
        p_std.append(hits / 5)
    n = len(recs)
    return {
        "n": n,
        "p5_standard": sum(p_std) / n,
        "p5_returned": d["precision_at_5"],
        "r5": d["recall_at_5"],
        "n_short_lists": sum(1 for r in recs if len(set(r["predicted"])) < 5),
    }


def main():
    out = {}
    for level, (folder, pat) in LEVELS.items():
        per = {}
        for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
            m = re.match(pat, os.path.basename(path))
            if not m:
                continue
            model, system = m.groups()
            per.setdefault(model, {})[system] = recompute(path)
        level_out = {}
        for model, systems in per.items():
            assert len(systems) == 12, f"{level}/{model}: {len(systems)} systems"
            macro = {k: sum(s[k] for s in systems.values()) / 12
                     for k in ("p5_standard", "p5_returned", "r5")}
            level_out[model] = {"macro": macro, "systems": systems}
        out[level] = level_out

    dst = os.path.join(ROOT, "ar_p5_standard.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    for level, models in out.items():
        print(f"== {level}")
        for model, v in sorted(models.items(), key=lambda x: -x[1]["macro"]["p5_standard"]):
            mm = v["macro"]
            short = sum(s["n_short_lists"] for s in v["systems"].values())
            print(f"  {model:22} P@5(std)={mm['p5_standard']:.4f}  old={mm['p5_returned']:.4f}  "
                  f"R@5={mm['r5']:.4f}  short_lists={short}")
    print("saved:", dst)


if __name__ == "__main__":
    main()
