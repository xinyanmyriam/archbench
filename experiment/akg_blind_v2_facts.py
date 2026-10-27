"""为 akg_ai_blind_v2.csv 的每条边生成“事实层”证据包（不含判定、不含推荐）。
用法：
    python akg_blind_v2_facts.py            # 生成 akg_ai_blind_v2_facts.json / .txt
    python akg_blind_v2_facts.py --show 5 --from 1
事实内容：源文件与语句、写明的引用串、解析目标、目标是否在磁盘上（文件/目录/缺失）、
解析目标文件的模块归属与 target_module 的父子关系、候选路径及其磁盘存在性、语言类型。
"""
import csv, json, os, re, sys
from collections import OrderedDict

sys.path.insert(0, ".")
from config import PROJECTS

SRC = "akg_ai_blind_v2.csv"
OUT_JSON = "akg_ai_blind_v2_facts.json"
OUT_TXT = "akg_ai_blind_v2_facts.txt"

PROJ = {p["name"]: p for p in PROJECTS}
PROJ_RESOLVE = {"name": None}


def load_index(name):
    path = os.path.join(PROJ[name]["path"], "file_index.json")
    fi = json.load(open(path, encoding="utf-8"))["files"]
    path2mod = {x["path"]: x["module"] for x in fi}
    mods = json.load(open(os.path.join(PROJ[name]["path"], "modules_index.json"),
                          encoding="utf-8"))["modules"]
    v2mods = os.path.join("..", "systems", name, "modules_index.json")
    if os.path.exists(v2mods) and name == "linux":
        mods = json.load(open(v2mods, encoding="utf-8"))["modules"]
    return path2mod, set(mods)


def module_of(path, path2mod):
    if path in path2mod:
        return path2mod[path]
    parts = path.split("/")
    if parts[0] == "include" and len(parts) > 2:
        return "include/" + parts[1]
    return None


def relation(target, model):
    """target_module 与解析文件所属模块的层级关系（事实）"""
    if model is None:
        return "unindexed"
    if target == model:
        return "same"
    if model.startswith(target + "/"):
        return "target_is_ancestor_of_file_module"
    if target.startswith(model + "/"):
        return "target_is_descendant_of_file_module"
    return "unrelated"


def lang_of(f):
    f = f.lower()
    for ext, lang in ((".cc", "C++"), (".cpp", "C++"), (".cxx", "C++"), (".c", "C"),
                      (".h", "C/C++ header"), (".hxx", "C++ header"), (".hpp", "C++ header"),
                      (".py", "Python"), (".go", "Go"), (".java", "Java"),
                      (".f", "Fortran"), (".f90", "Fortran"), (".f77", "Fortran")):
        if f.endswith(ext):
            return lang
    return "other"


def check_c_include(stmt, source_file, resolved):
    """C/C++：#include 引用串与解析路径的机械比对（只给等式结果）"""
    m = re.search(r'#\s*include\s*[<"]([^>"]+)[>"]', stmt)
    if not m:
        return {}
    ref = m.group(1)
    src_dir = os.path.dirname(source_file)
    facts = {"ref": ref}
    if ref.startswith("../") or ref.startswith("./"):
        facts["ref_relative_resolved"] = os.path.normpath(os.path.join(src_dir, ref))
    else:
        facts["ref_suffix_of_resolved"] = resolved.endswith("/" + ref) or resolved == ref
        facts["resolved_suffix"] = "/".join(resolved.split("/")[-len(ref.split("/")):]) if ref else ""
    return facts


def check_py_import(stmt, resolved):
    m = re.search(r'^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))', stmt)
    if not m:
        return {}
    mod = (m.group(1) or m.group(2)).split(" as ")[0].strip()
    base = mod.replace(".", "/")
    facts = {"ref": mod}
    facts["ref_module_path"] = base
    facts["resolved_is_module_init"] = resolved == base + "/__init__.py"
    facts["resolved_is_module_file"] = resolved == base + ".py"
    facts["resolved_under_ref"] = resolved.startswith(base + "/")
    return facts


def check_go_import(stmt, resolved):
    m = re.search(r'"([\w\.\-/]+)"', stmt)
    if not m:
        return {}
    ref = m.group(1)
    return {"ref": ref, "resolved_suffix": "/".join(resolved.split("/")[-len(ref.split("/")):]),
            "ref_suffix_of_resolved": resolved.endswith("/" + ref)}


def check_java_import(stmt, resolved):
    m = re.search(r'import\s+(?:static\s+)?([\w\.]+)', stmt)
    if not m:
        return {}
    fqn = m.group(1)
    return {"ref": fqn, "ref_path": fqn.replace(".", "/"),
            "resolved_is_ref_path": resolved == fqn.replace(".", "/") + ".java" or resolved.endswith("/" + fqn.replace(".", "/") + ".java")}


rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
edges = OrderedDict()
for r in rows:
    edges.setdefault(r["pilot_id"], []).append(r)

cache = {}
out = []
for pid, rs in edges.items():
    e = rs[0]
    name = e["system"]
    if name not in cache:
        cache[name] = load_index(name)
    path2mod, mods = cache[name]
    PROJ_RESOLVE["name"] = name
    examples = []
    for r in rs:
        lang = lang_of(r["source_file"])
        resolved = r["resolved_target"]
        f = {"file": r["source_file"], "line": r["line"], "statement": r["statement"],
             "resolved": resolved, "resolved_on_disk": r["resolved_on_disk"],
             "resolved_module": module_of(resolved, path2mod),
             "lang": lang}
        if lang.startswith("C"):
            f.update(check_c_include(r["statement"], r["source_file"], resolved))
        elif lang == "Python":
            f.update(check_py_import(r["statement"], resolved))
        elif lang == "Go":
            f.update(check_go_import(r["statement"], resolved))
        elif lang == "Java":
            f.update(check_java_import(r["statement"], resolved))
        examples.append(f)
    rel = relation(e["target_module"], module_of(rs[0]["resolved_target"], path2mod))
    out.append({
        "pilot_id": pid, "system": name,
        "source_module": e["source_module"], "target_module": e["target_module"],
        "weight": int(e["weight"]), "n_candidates": int(e["n_candidates"]),
        "target_in_modules_index": e["target_module"] in mods,
        "resolved_file_module": module_of(rs[0]["resolved_target"], path2mod),
        "relation": rel,
        "candidates": [c.strip() for c in e["candidates_top15"].split(";") if c.strip()],
        "examples": examples,
    })

json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

with open(OUT_TXT, "w", encoding="utf-8") as fh:
    for o in out:
        fh.write(f"\n[{o['pilot_id']}] {o['system']} {o['source_module']} -> {o['target_module']} "
                 f"weight={o['weight']} candidates={o['n_candidates']}\n")
        fh.write(f"  target_in_modules_index={o['target_in_modules_index']} "
                 f"resolved_file_module={o['resolved_file_module']} relation={o['relation']}\n")
        for x in o["examples"]:
            fh.write(f"  {x['file']}:{x['line']} ({x['lang']}) {x['statement']}\n")
            fh.write(f"     resolved={x['resolved']} disk={x['resolved_on_disk']} mod={x['resolved_module']}\n")
            skip = {"file", "line", "statement", "resolved", "resolved_on_disk", "resolved_module", "lang"}
            extra = {k: v for k, v in x.items() if k not in skip}
            if extra:
                fh.write(f"     checks={json.dumps(extra, ensure_ascii=False)}\n")
        if o["n_candidates"] > 1:
            fh.write("  candidates: " + " | ".join(o["candidates"]) + "\n")
print(f"written {OUT_JSON}, {OUT_TXT}; edges={len(out)}")

if "--show" in sys.argv:
    n = int(sys.argv[sys.argv.index("--show") + 1])
    start = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 1
    for o in out[start - 1:start - 1 + n]:
        print(f"\n[{o['pilot_id']}] {o['system']} {o['source_module']} -> {o['target_module']} "
              f"(weight={o['weight']}, candidates={o['n_candidates']})")
        print(f"  target_in_index={o['target_in_modules_index']} resolved_file_module={o['resolved_file_module']} "
              f"relation={o['relation']}")
        for x in o["examples"]:
            print(f"  {x['file']}:{x['line']} [{x['lang']}] {x['statement']}")
            print(f"     -> {x['resolved']} ({x['resolved_on_disk']}, mod={x['resolved_module']})")
            skip = {"file", "line", "statement", "resolved", "resolved_on_disk", "resolved_module", "lang"}
            extra = {k: v for k, v in x.items() if k not in skip}
            if extra:
                print("     " + json.dumps(extra, ensure_ascii=False))
