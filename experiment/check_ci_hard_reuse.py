"""核对 L2 CI-Hard 能否复用 v1 结果：分别在 SAB_MANIFEST=v1/v2 下构建 prompt，逐题比对；
并核对 v1 结果文件逐题 file/label/letter_map 与 v2 题目一致、api_errors=0。
用法：python check_ci_hard_reuse.py dump v1|v2  ;  python check_ci_hard_reuse.py compare"""
import hashlib, json, os, subprocess, sys

if len(sys.argv) > 1 and sys.argv[1] == "dump":
    os.environ["SAB_MANIFEST"] = sys.argv[2]
    sys.path.insert(0, ".")
    from config import PROJECTS
    import run_level2 as R
    out = {}
    for p in PROJECTS:
        ctx = R.load_project_context(p["path"])
        out[p["name"]] = [{"h": hashlib.sha1(it["prompt"].encode()).hexdigest(), "gold": it["gold"],
                           "lm": it["letter_map"], "file": it["meta"]["file"], "label": it["meta"]["label"]}
                          for it in R.build_items("ci_hard", p, ctx)]
    json.dump(out, open(f"_ci_hard_items_{sys.argv[2]}.json", "w", encoding="utf-8"))
    sys.exit(0)

for v in ("v1", "v2"):
    subprocess.run([sys.executable, __file__, "dump", v], check=True)
a = json.load(open("_ci_hard_items_v1.json", encoding="utf-8"))
b = json.load(open("_ci_hard_items_v2.json", encoding="utf-8"))
ok = True
for proj in b:
    same = sum(x["h"] == y["h"] for x, y in zip(a[proj], b[proj]))
    line = f"{proj:13s} prompt identical {same}/{len(b[proj])}"
    for model in ("gpt-6-sol", "claude-opus-5.5"):
        d = json.load(open(f"results_rerun/level2/{model}_{proj}_ci_hard_l2.json", encoding="utf-8"))
        r = d["results"]
        match = sum(x["file"] == y["file"] and x["label"] == y["label"]
                    and (x.get("predicted_letter") is None or y["lm"].get(x["predicted_letter"]) == x.get("predicted_module"))
                    for x, y in zip(r, b[proj]))
        line += f" | {model} n={len(r)} err={d['n_api_errors']} match={match} acc={d['accuracy']:.3f}"
        ok &= d["n_api_errors"] == 0 and match == len(b[proj]) == len(r)
    ok &= same == len(b[proj])
    print(line)
print("ALL OK" if ok else "MISMATCH")
for v in ("v1", "v2"):
    os.remove(f"_ci_hard_items_{v}.json")
