"""校验：all_targets 与 v2 图（load_project_data）出边完全一致，确保 L2 可见依赖不泄漏隐藏边。"""
import os
os.environ["SAB_MANIFEST"] = "v2"
import json
from config import PROJECTS
from baselines import load_project_data

m = json.load(open(os.path.join("samples", "manifest_v2.json"), encoding="utf-8"))
bad = 0
for p in PROJECTS:
    g = load_project_data(p["path"])["dep_graph"]
    ar = m["projects"][p["name"]]["ar"]
    for s in ar["sources"]:
        full = set(ar["all_targets"][s])
        hid = set(ar["true_targets"][s])
        out = set(g.get(s, {})) - {s}
        if full != out or not hid <= full:
            bad += 1
            print(p["name"], s, len(full), len(out), sorted(full ^ out)[:5])
print("mismatch:", bad)
