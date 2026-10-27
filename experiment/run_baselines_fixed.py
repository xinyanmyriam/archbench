"""
非 LLM baseline 重跑（P2-4 / P2-5），不调用任何 API。

与旧 run_experiment.py --baselines 的区别：
  1. 样本：读取 samples/manifest_seed42.json 的固定样本（与所有模型评测一致），不再运行时抽样。
  2. 泄露：Graph Neighbor 使用与 Level 2 相同的屏蔽规则——
       DP：预测 (A,B) 时从图中删除 A->B；
       AR：预测 source 时删除 source -> true_targets。
     同时输出未屏蔽版本（*_unmasked），仅用于说明旧结果的泄露程度，不应写入论文主结果。
  3. AR P@5 = hits/5（与 P0-1 一致），R@5 = hits/|true|。
  4. 随机性：每个 (project, task, baseline) 使用独立的 random.Random(42)。

输出：results_rerun/baselines_fixed.json（v1）/ results_v2/baselines_fixed.json（v2）
用法：python run_baselines_fixed.py；v2：$env:SAB_MANIFEST="v2"; python run_baselines_fixed.py
v2：清单 samples/manifest_v2.json；图/频率取 baselines.load_project_data（akg_v2 模块图、
    kubernetes 模块名规范化、linux include/* 计频）；linux 模块表取 akg_v2/linux/modules_index.json。
    AR 真解为每个源模块隐藏的一半出边，屏蔽后仍有可见依赖。
"""
import json
import os
import random
from collections import Counter, defaultdict

from config import PROJECTS

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_VERSION = os.environ.get("SAB_MANIFEST", "v1")
assert MANIFEST_VERSION in ("v1", "v2")
MANIFEST_REL = "samples/manifest_seed42.json" if MANIFEST_VERSION == "v1" else "samples/manifest_v2.json"
MANIFEST = os.path.join(HERE, MANIFEST_REL)
OUT = os.path.join(HERE, "results_rerun" if MANIFEST_VERSION == "v1" else "results_v2", "baselines_fixed.json")
SEED = 42
K = 5


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_project(ppath, name):
    if MANIFEST_VERSION == "v2":
        from baselines import load_project_data
        d = load_project_data(next(p["path"] for p in PROJECTS if p["name"] == name))  # 需在 experiment/ 下运行
        mi = os.path.join(HERE, "..", "systems", name, "modules_index.json")
        if not os.path.exists(mi):
            mi = os.path.join(ppath, "modules_index.json")
        with open(mi, encoding="utf-8") as f:
            modules = json.load(f)["modules"]
        return modules, d["module_freq"], d["dep_graph"]
    with open(os.path.join(ppath, "file_index.json"), encoding="utf-8") as f:
        file_index = json.load(f)
    with open(os.path.join(ppath, "module_dependency.json"), encoding="utf-8") as f:
        deps = json.load(f)["dependencies"]
    with open(os.path.join(ppath, "modules_index.json"), encoding="utf-8") as f:
        modules = json.load(f)["modules"]
    freq = Counter(fi["module"] for fi in file_index["files"])
    graph = defaultdict(dict)
    for d in deps:
        s, t = d["source_module"], d["target_module"]
        graph[s][t] = graph[s].get(t, 0) + d.get("weight", 1)
    return modules, freq, graph


def masked_out(graph, src, hidden):
    return {t: w for t, w in graph.get(src, {}).items() if t not in hidden}


# ---------------- CI / CI-Hard ----------------

def ci_path_heuristic(file_path, candidates, rng):
    parts = set(file_path.lower().replace("\\", "/").split("/"))
    best, best_score = None, -1
    for c in candidates:
        score = len(parts & set(c.lower().replace("\\", "/").split("/")))
        if score > best_score:
            best, best_score = c, score
    return rng.choice(candidates) if best_score == 0 else best


def eval_ci(items, freq, path_key):
    out = {}
    fns = {
        "random": lambda it, r: r.choice(it["candidates"]),
        "frequency": lambda it, r: max(it["candidates"], key=lambda m: freq.get(m, 0)),
        "path_heuristic": lambda it, r: ci_path_heuristic(it[path_key], it["candidates"], r),
    }
    for name, fn in fns.items():
        rng = random.Random(SEED)
        correct = sum(fn(it, rng) == it["label"] for it in items)
        out[name] = {"accuracy": correct / len(items), "n": len(items)}
    return out


# ---------------- DP ----------------

def dp_metrics(preds, labels):
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
    n = len(labels)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (tp + tn) / n, "precision": prec, "recall": rec, "f1": f1, "n": n}


def dp_graph_neighbor(a, b, graph, mask):
    na = set(masked_out(graph, a, {b}) if mask else graph.get(a, {}))
    nb = set(graph.get(b, {}))
    if b in na:
        return 1
    return 1 if na & nb else 0


def dp_frequency(a, b, graph, mask):
    na = masked_out(graph, a, {b}) if mask else graph.get(a, {})
    return 1 if len(na) > 3 else 0


def eval_dp(items, graph):
    labels = [it["label"] for it in items]
    rng = random.Random(SEED)
    out = {"random": dp_metrics([rng.choice([0, 1]) for _ in items], labels)}
    for mask, suffix in [(True, ""), (False, "_unmasked")]:
        out["frequency" + suffix] = dp_metrics(
            [dp_frequency(it["module_A"], it["module_B"], graph, mask) for it in items], labels)
        out["graph_neighbor" + suffix] = dp_metrics(
            [dp_graph_neighbor(it["module_A"], it["module_B"], graph, mask) for it in items], labels)
    return out


# ---------------- AR ----------------

def ar_score(pred_lists, truth):
    p = r = 0.0
    for src, pred in pred_lists.items():
        hits = len(set(pred[:K]) & truth[src])
        p += hits / K
        r += hits / len(truth[src])
    n = len(pred_lists)
    return {"precision_at_5": p / n, "recall_at_5": r / n, "n_sources": n}


def ar_graph_neighbor(src, graph, hidden):
    nb = masked_out(graph, src, hidden) if hidden is not None else graph.get(src, {})
    return [t for t, _ in sorted(nb.items(), key=lambda x: (-x[1], x[0]))][:K]


def ar_two_hop(src, graph, hidden, modules):
    """屏蔽后的二跳传递：用 source 剩余已知依赖 d 的出边 d->m 权重给 m 打分；
    不足 K 个时按全图入度补齐。只用 Level 2 中可见的信息。"""
    shown = masked_out(graph, src, hidden)
    score = defaultdict(float)
    for d in shown:
        for m, w in graph.get(d, {}).items():
            if m != src and m not in shown:
                score[m] += w
    ranked = [m for m, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))]
    indeg = Counter()
    for s, tg in graph.items():
        for t in tg:
            indeg[t] += 1
    for m in sorted(modules, key=lambda m: (-indeg[m], m)):
        if len(ranked) >= K:
            break
        if m != src and m not in shown and m not in ranked:
            ranked.append(m)
    return ranked[:K]


def eval_ar(entry, modules, freq, graph):
    truth = {s: set(t) for s, t in entry["true_targets"].items()}
    sources = entry["sources"]
    rng = random.Random(SEED)
    preds = {
        "random": {s: rng.sample([m for m in modules if m != s], min(K, len(modules) - 1)) for s in sources},
        "frequency": {s: sorted([m for m in modules if m != s], key=lambda m: -freq.get(m, 0))[:K]
                      for s in sources},
        "graph_neighbor": {s: ar_graph_neighbor(s, graph, truth[s]) for s in sources},
        "graph_two_hop": {s: ar_two_hop(s, graph, truth[s], modules) for s in sources},
        "graph_neighbor_unmasked": {s: ar_graph_neighbor(s, graph, None) for s in sources},
    }
    return {name: ar_score(pl, truth) for name, pl in preds.items()}


def main():
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    results = {}
    for proj in PROJECTS:
        name = proj["name"]
        ppath = os.path.normpath(os.path.join(HERE, proj["path"]))
        modules, freq, graph = load_project(ppath, name)
        ent = manifest["projects"][name]

        def sample(task):
            data = read_jsonl(os.path.join(HERE, ent[task]["source_file"]))
            return [data[i] for i in ent[task]["indices"]]

        results[name] = {
            "ci": eval_ci(sample("ci"), freq, "file"),
            "ci_hard": eval_ci(sample("ci_hard"), freq, "file"),
            "dp": eval_dp(sample("dp"), graph),
            "ar": eval_ar(ent["ar"], modules, freq, graph),
        }
        r = results[name]
        print(f"{name:13s} CI path={r['ci']['path_heuristic']['accuracy']:.3f} freq={r['ci']['frequency']['accuracy']:.3f} | "
              f"CI-Hard path={r['ci_hard']['path_heuristic']['accuracy']:.3f} freq={r['ci_hard']['frequency']['accuracy']:.3f} | "
              f"DP F1 GN={r['dp']['graph_neighbor']['f1']:.3f} (unmasked {r['dp']['graph_neighbor_unmasked']['f1']:.3f}) "
              f"freq={r['dp']['frequency']['f1']:.3f} | "
              f"AR P/R freq={r['ar']['frequency']['precision_at_5']:.3f}/{r['ar']['frequency']['recall_at_5']:.3f} "
              f"2hop={r['ar']['graph_two_hop']['precision_at_5']:.3f}/{r['ar']['graph_two_hop']['recall_at_5']:.3f}")

    macro = {}
    for task in ["ci", "ci_hard", "dp", "ar"]:
        for bl in results[PROJECTS[0]["name"]][task]:
            for metric in results[PROJECTS[0]["name"]][task][bl]:
                if metric.startswith("n"):
                    continue
                vals = [results[p["name"]][task][bl][metric] for p in PROJECTS]
                macro.setdefault(task, {}).setdefault(bl, {})[metric] = sum(vals) / len(vals)
    print("\n=== Macro-average over 12 systems ===")
    print(json.dumps(macro, indent=1))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "manifest": MANIFEST_REL,
                   "note": "*_unmasked uses the queried edge (leaky); report masked variants only.",
                   "per_system": results, "macro": macro}, f, indent=1)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
