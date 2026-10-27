#!/usr/bin/env python3
"""
AKG v2 rebuild driver: strict re-resolution of raw dependency statements and
relabeling of the existing DP / AR task items (no resampling).

Run:  python -u akg_v2/rebuild_generic.py [system ...]
Out:  akg_v2/<sys>/{file_dependency.json, module_dependency.json,
      dependency_prediction.jsonl, architecture_reconstruction.jsonl,
      architecture_reconstruction_filtered.jsonl, stats.json}
      akg_v2/summary_generic.json
"""

import json
import os
import posixpath
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from resolvers import (FileIndex, build_disk_index, CppResolver, LinuxCppResolver,  # noqa: E402
                       JavaResolver, PythonResolver, module_edges, relabel_dp,
                       relabel_ar, jaccard)

SYSTEMS = ["linux", "hadoop", "spring", "paraview", "pytorch", "tensorflow",
           "codeaster", "openmodelica"]

# system -> audit key (experiment/akg_impact_audit.py SYSTEMS)
CPP_EXT = (".c", ".cpp", ".cxx", ".cc", ".h", ".hpp", ".hxx")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def dump(obj, p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def dump_jsonl(items, p):
    with open(p, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def linux_module_map(f2m, disk_paths, modules):
    """Add include files: top-level include/<sub>/... -> new module 'include/<sub>'
    (module_depth=2 naming); files under any other pruned 'include' dir
    (arch/<X>/include/..., drivers/gpu/drm/amd/include/...) -> the existing
    depth-2 module of their path, if that module exists."""
    mods = set(modules)
    new_mods = set()
    added = Counter()
    for p in disk_paths:
        if p in f2m:
            continue
        parts = p.split("/")
        if parts[0] == "include":
            if len(parts) >= 3:
                m = "include/" + parts[1]
            else:
                m = "include"
            f2m[p] = m
            new_mods.add(m)
            added[m] += 1
        elif "include" in parts[:-1]:
            m = "/".join(parts[:2])
            if m in mods:
                f2m[p] = m
                added["(existing) " + ("arch/*" if parts[0] == "arch" else m)] += 1
    return new_mods, added


def rebuild(system):
    sd = os.path.join(ROOT, system)
    od = os.path.join(HERE, system)
    os.makedirs(od, exist_ok=True)
    cfg = load(os.path.join(sd, "config.json"))
    repo = os.path.join(sd, cfg["repo_path"])
    assert os.path.isdir(repo), repo
    lang = cfg["language"]
    fi = load(os.path.join(sd, "file_index.json"))["files"]
    f2m = {f["path"]: f["module"] for f in fi}
    modules = load(os.path.join(sd, "modules_index.json"))["modules"]
    raws = load(os.path.join(sd, "dependency_raw.json"))["dependencies"]
    old_fd = load(os.path.join(sd, "file_dependency.json"))["dependencies"]
    old_md = {(d["source_module"], d["target_module"]): d["weight"]
              for d in load(os.path.join(sd, "module_dependency.json"))["dependencies"]}

    unignore = ("include",) if system == "linux" else ()
    disk = build_disk_index(repo, cfg.get("source_extensions"), cfg.get("ignore_dirs"), unignore)
    idx = FileIndex(disk)
    print(f"[{system}] disk index {len(idx.paths)} files (file_index {len(f2m)}), raw {len(raws)}", flush=True)
    missing_from_disk = sum(1 for p in f2m if p not in idx.paths)

    new_mods, added = set(), Counter()
    if system == "linux":
        new_mods, added = linux_module_map(f2m, disk, modules)

    cpp = py = java = None
    if lang == "java":
        java = JavaResolver(idx)
    else:
        cpp = (LinuxCppResolver if system == "linux" else CppResolver)(idx)
        cpp.prepare({r["target"].strip() for r in raws if r["type"] == "include"})
        if lang == "mixed_cpp_python":
            py = PythonResolver(idx)

    links, recs = [], []
    reasons, drops = Counter(), Counter()
    for r in raws:
        src, tgt, typ = r["source"], r["target"], r["type"]
        if typ == "include":
            res, why = cpp.resolve(src, tgt)
        elif typ == "import" and lang == "java":
            res, why = java.resolve(src, tgt)
        elif typ == "import" and src.endswith(".py") and py is not None:
            res, why = py.resolve(src, tgt)
        else:
            res, why = None, "unsupported_type"
        if res is not None:
            if res == src:
                res, why = None, "self_link"
            elif res not in f2m:
                res, why = None, "target_unmapped"
        if res is None:
            drops[why] += 1
        else:
            reasons[why] += 1
            links.append({"source": src, "target": res, "type": typ})
        recs.append((src, tgt, res, why))

    new_e = module_edges(((l["source"], l["target"]) for l in links), f2m)
    dump({"system": cfg["system"], "total": len(links), "dependencies": links},
         os.path.join(od, "file_dependency.json"))
    dump({"system": cfg["system"], "total": len(new_e),
          "dependencies": [{"source_module": a, "target_module": b, "weight": w}
                           for (a, b), w in new_e.items()]},
         os.path.join(od, "module_dependency.json"))

    # ---- tasks
    dp = jsonl(os.path.join(sd, "dependency_prediction.jsonl"))
    ar = jsonl(os.path.join(sd, "architecture_reconstruction.jsonl"))
    edges = set(new_e)
    dp_new = relabel_dp(dp, edges)
    ar_new = relabel_ar(ar, edges)
    dump_jsonl(dp_new, os.path.join(od, "dependency_prediction.jsonl"))
    dump_jsonl(ar_new, os.path.join(od, "architecture_reconstruction.jsonl"))
    dump_jsonl([{k: v for k, v in it.items() if k in ("source_module", "target_module")}
                for it in ar_new if it["label"] == 1],
               os.path.join(od, "architecture_reconstruction_filtered.jsonl"))

    # verify task items: same count, same order, non-label fields verbatim
    assert len(dp_new) == len(dp) and len(ar_new) == len(ar)
    for a, b in zip(dp, dp_new):
        assert {k: v for k, v in a.items() if k != "label"} == \
               {k: v for k, v in b.items() if k != "label"}
    for a, b in zip(ar, ar_new):
        assert all(b[k] == v for k, v in a.items())

    flips = Counter(f"{a['label']}->{b['label']}" for a, b in zip(dp, dp_new))
    old_set = set(old_md)
    out_old, out_new, gt_old, gt_new = (defaultdict(set) for _ in range(4))
    for a, b in old_set:
        out_old[a].add(b)
    for a, b in edges:
        out_new[a].add(b)
    for it in ar_new:
        gt_old[it["source_module"]].add(it["target_module"])
        if it["label"]:
            gt_new[it["source_module"]].add(it["target_module"])
    ar_srcs = list(dict.fromkeys(it["source_module"] for it in ar))
    jac_graph = [jaccard(out_old[s], out_new[s]) for s in ar_srcs]
    jac_task = [jaccard(gt_old[s], gt_new[s]) for s in ar_srcs]

    mods_set = set(modules)
    edges_orig_only = {e for e in edges if e[0] in mods_set and e[1] in mods_set}
    stats = {
        "system": system,
        "raw_statements": len(raws),
        "raw_reused_from": f"{system}/dependency_raw.json (full target text kept)",
        "disk_index_files": len(idx.paths),
        "file_index_files": len(fi),
        "file_index_files_missing_on_disk": missing_from_disk,
        "linux_new_include_modules": sorted(new_mods),
        "linux_added_file_mappings": dict(added),
        "links_old": len(old_fd),
        "links_new": len(links),
        "resolve_reasons": dict(reasons.most_common()),
        "drop_reasons": dict(drops.most_common()),
        "module_edges_old": len(old_set),
        "module_edges_new": len(edges),
        "module_edges_new_between_original_modules": len(edges_orig_only),
        "module_edges_new_touching_new_include_modules": len(edges) - len(edges_orig_only),
        "module_edges_kept": len(old_set & edges),
        "module_edges_added": len(edges - old_set),
        "module_edges_removed": len(old_set - edges),
        "weight_old": sum(old_md.values()),
        "weight_new": sum(new_e.values()),
        "new_edges_lt_30pct_of_old": len(edges_orig_only) < 0.3 * len(old_set),
        "dp_items": len(dp),
        "dp_positive_old": sum(t["label"] for t in dp),
        "dp_positive_new": sum(t["label"] for t in dp_new),
        "dp_flips": dict(flips),
        "dp_flip_1_to_0": flips.get("1->0", 0),
        "dp_flip_0_to_1": flips.get("0->1", 0),
        "ar_items": len(ar),
        "ar_items_edge_still_present": sum(t["label"] for t in ar_new),
        "ar_items_empty_new_target_set": sum(1 for t in ar_new if not t["new_target_set"]),
        "ar_source_modules": len(ar_srcs),
        "ar_source_modules_empty_new_target_set": sum(1 for s in ar_srcs if not out_new[s]),
        "ar_jaccard_graph_mean": round(sum(jac_graph) / len(jac_graph), 4) if jac_graph else None,
        "ar_jaccard_task_mean": round(sum(jac_task) / len(jac_task), 4) if jac_task else None,
        "top_removed": [f"{a} -> {b} ({w})" for (a, b), w in
                        sorted(((e, old_md[e]) for e in old_set - edges), key=lambda x: -x[1])[:10]],
        "top_added": [f"{a} -> {b} ({w})" for (a, b), w in
                      sorted(((e, new_e[e]) for e in edges - old_set), key=lambda x: -x[1])[:10]],
    }
    stats["audit"] = audit_new_links(system, recs, f2m)
    stats["examples"] = check_examples(system, recs)
    dump(stats, os.path.join(od, "stats.json"))
    print(f"  links {stats['links_old']} -> {stats['links_new']}, edges {stats['module_edges_old']} -> "
          f"{stats['module_edges_new']} (orig-only {len(edges_orig_only)}), DP 1->0 {stats['dp_flip_1_to_0']} "
          f"0->1 {stats['dp_flip_0_to_1']}, AR empty {stats['ar_items_empty_new_target_set']}, "
          f"Jg {stats['ar_jaccard_graph_mean']} Jt {stats['ar_jaccard_task_mean']}, "
          f"clear_error {stats['audit'].get('clear_error_pct')}%", flush=True)
    return stats


# ------------------------------------------------------------------ audit

AUDIT_KEY = {"linux": "linux", "hadoop": "hadoop", "spring": "spring", "paraview": "paraview",
             "pytorch": "pytorch", "tensorflow": "tensorflow", "codeaster": "codeaster",
             "openmodelica": "openmodelica"}


def audit_new_links(system, recs, f2m):
    """Classify every new link with experiment/akg_impact_audit.py classifiers
    (imported, not modified).  The audit Index is pointed at the new
    file->module map so linux include files are not mis-flagged as config_gap."""
    try:
        sys.path.insert(0, os.path.join(ROOT, "experiment"))
        import akg_impact_audit as A
        from akg_ai_evidence import CLEAR_ERROR
    except Exception as e:  # pragma: no cover
        return {"error": repr(e)}
    data_dir, repo_rel, lang = A.SYSTEMS[AUDIT_KEY[system]]
    ix = A.Index(os.path.join(ROOT, data_dir), repo_rel)
    ix.f2m = dict(f2m)
    ix.all_paths = set(ix.f2m)
    ix.by_base = defaultdict(list)
    for p in ix.f2m:
        ix.by_base[os.path.basename(p)].append(p)
    ix.pkg2path = {}
    for p in ix.f2m:
        q = p[:-5] if p.endswith(".java") else p
        ix.pkg2path[q.replace("/", ".")] = p
    ix.suffix = defaultdict(list)
    for pkg, p in ix.pkg2path.items():
        parts = pkg.split(".")
        for k in range(len(parts), 0, -1):
            ix.suffix[".".join(parts[-k:])].append(p)
    ix.build_suffix_map({t for s, t, res, w in recs if res and "/" in t})
    v = Counter()
    ex = defaultdict(list)
    for src, raw, res, why in recs:
        if res is None:
            continue
        if src.endswith(".py") and lang == "mixed":
            c = A.classify_py(ix, src, raw, res)
        elif lang == "java":
            c = A.classify_java(ix, src, raw, res)
        else:
            c = A.classify_cpp(ix, src, raw, res)
        v[c] += 1
        if c in CLEAR_ERROR and len(ex[c]) < 3:
            ex[c].append(f"{src} | {raw} -> {res} [{why}]")
    n = sum(v.values())
    ce = sum(c for k, c in v.items() if k in CLEAR_ERROR)
    return {"verdicts": dict(v.most_common()), "clear_error": ce,
            "clear_error_pct": round(100 * ce / n, 3) if n else None,
            "clear_error_examples": dict(ex)}


# ------------------------------------------------------------------ known bad examples

def check_examples(system, recs):
    out = {}

    def res_of(pred):
        return Counter((res or f"<dropped:{why}>") for s, t, res, why in recs if pred(s, t))

    if system == "linux":
        r = res_of(lambda s, t: s == "arch/alpha/boot/bootp.c" and t == "linux/kernel.h")
        out["bootp.c <linux/kernel.h>"] = {"result": dict(r), "ok": dict(r) == {"include/linux/kernel.h": 1}}
        r = res_of(lambda s, t: s == "arch/alpha/boot/bootp.c" and t == "asm/io.h")
        out["bootp.c <asm/io.h>"] = {"result": dict(r), "ok": all(k.startswith("arch/alpha/") or k.startswith("include/asm-generic") or k.startswith("<dropped") for k in r)}
    if system == "spring":
        r = res_of(lambda s, t: t == "org.jspecify.annotations.Nullable")
        out["org.jspecify.annotations.Nullable"] = {"result": dict(r), "ok": all(k.startswith("<dropped") for k in r)}
    if system == "hadoop":
        r = res_of(lambda s, t: t == "java.util.Queue")
        out["java.util.Queue"] = {"result": dict(r), "ok": all(k.startswith("<dropped") for k in r)}
        r = res_of(lambda s, t: t == "com.qcloud.cos.utils.Base64")
        out["com.qcloud.cos.utils.Base64"] = {"result": dict(r), "ok": all(k.startswith("<dropped") for k in r)}
    if system == "tensorflow":
        r = res_of(lambda s, t: t == "absl/status/statusor.h")
        out["absl/status/statusor.h"] = {"result": dict(r), "ok": all(k.startswith("<dropped") for k in r)}
        r = res_of(lambda s, t: t == "tensorflow/core/platform/status.h")
        out["tensorflow/core/platform/status.h"] = {"result": dict(r), "ok": all(k in ("tensorflow/core/platform/status.h",) or k.startswith("<dropped") for k in r)}
    if system == "pytorch":
        r = res_of(lambda s, t: t == "os" and s.endswith(".py"))
        out["python os"] = {"result": dict(r), "ok": all(k.startswith("<dropped") for k in r)}
        r = res_of(lambda s, t: t == "ATen/core/ScalarType.h")
        out["ATen/core/ScalarType.h"] = {"result": dict(r), "ok": "aten/src/ATen/ScalarType.h" not in r}
        r = res_of(lambda s, t: t == "ATen/Functions.h")
        out["ATen/Functions.h"] = {"result": dict(r), "ok": "aten/src/ATen/templates/Functions.h" not in r}
    if system == "codeaster":
        r = res_of(lambda s, t: t == "scipy.optimize")
        out["scipy.optimize"] = {"result": dict(r), "ok": all(k.startswith("<dropped") for k in r)}
    return out


def main():
    systems = sys.argv[1:] or SYSTEMS
    summary = {}
    sp = os.path.join(HERE, "summary_generic.json")
    if os.path.exists(sp):
        summary = load(sp)
    for s in systems:
        st = rebuild(s)
        summary[s] = {k: st[k] for k in (
            "links_old", "links_new", "module_edges_old", "module_edges_new",
            "module_edges_new_between_original_modules", "module_edges_kept",
            "module_edges_added", "module_edges_removed", "new_edges_lt_30pct_of_old",
            "dp_positive_old", "dp_positive_new", "dp_flip_1_to_0", "dp_flip_0_to_1",
            "ar_items", "ar_items_edge_still_present", "ar_items_empty_new_target_set",
            "ar_jaccard_graph_mean", "ar_jaccard_task_mean")}
        summary[s]["clear_error_pct"] = st["audit"].get("clear_error_pct")
        summary[s]["examples_ok"] = all(v["ok"] for v in st["examples"].values())
        dump(summary, sp)


if __name__ == "__main__":
    main()
