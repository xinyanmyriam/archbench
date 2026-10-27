#!/usr/bin/env python3
"""
AKG v2 rebuild driver for the special-cased systems:
kubernetes (Go package dirs), openfoam (re-extracted includes), salome
(per-repo + cross-repo, unified graph), nastran (copied, consistency check).

Same output format / bookkeeping / assertions as rebuild_generic.py.

Run:  python -u akg_v2/rebuild_special.py [system ...]
Out:  akg_v2/<sys>/{file_dependency.json, module_dependency.json,
      dependency_prediction.jsonl, architecture_reconstruction.jsonl,
      architecture_reconstruction_filtered.jsonl, stats.json}
      akg_v2/salome/cross_file_dependency.json
      akg_v2/summary_special.json
"""

import os
import posixpath
import re
import shutil
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import rebuild_generic as G  # noqa: E402
from rebuild_generic import load, jsonl, dump, dump_jsonl, CPP_EXT  # noqa: E402
from resolvers import (FileIndex, build_disk_index, CppResolver,  # noqa: E402
                       module_edges, relabel_dp, relabel_ar, jaccard)

SYSTEMS = ["kubernetes", "openfoam", "salome", "nastran"]
IGNORE_DIRS = {".git", "build", "bin", "dist", "node_modules", "target", ".idea", ".vscode"}
SALOME_REPOS = ["kernel", "geom", "smesh", "medcoupling", "yacs", "gui"]

G.AUDIT_KEY.setdefault("openfoam", "openfoam")
G.AUDIT_KEY.setdefault("salome", "salome")


class StrictUniqueCpp(CppResolver):
    """CppResolver without the same-top-level tie-break: several suffix
    candidates are only accepted when one of them is the path relative to the
    including file (C quote-include semantics); otherwise dropped."""

    def resolve(self, src, inc):
        res, why = super().resolve(src, inc)
        if why == "path_same_top_preferred":
            return None, "path_ambiguous_same_top"
        return res, why


def _audit_import():
    sys.path.insert(0, os.path.join(ROOT, "experiment"))
    import akg_impact_audit as A
    from akg_ai_evidence import CLEAR_ERROR
    return A, CLEAR_ERROR


# ------------------------------------------------------------------ shared finalize

def finalize(system, od, task_dir, old_links, links_new, new_e, old_md, reasons, drops,
             audit, extra=None, modules=None):
    """new_e / old_md: Counter/dict (src_module, tgt_module) -> weight in the
    namespace used by the DP/AR task items."""
    dp = jsonl(os.path.join(task_dir, "dependency_prediction.jsonl"))
    ar = jsonl(os.path.join(task_dir, "architecture_reconstruction.jsonl"))
    edges = set(new_e)
    dp_new = relabel_dp(dp, edges)
    ar_new = relabel_ar(ar, edges)
    dump_jsonl(dp_new, os.path.join(od, "dependency_prediction.jsonl"))
    dump_jsonl(ar_new, os.path.join(od, "architecture_reconstruction.jsonl"))
    dump_jsonl([{k: v for k, v in it.items() if k in ("source_module", "target_module")}
                for it in ar_new if it["label"] == 1],
               os.path.join(od, "architecture_reconstruction_filtered.jsonl"))

    assert len(dp_new) == len(dp) and len(ar_new) == len(ar)
    for a, b in zip(dp, dp_new):
        assert {k: v for k, v in a.items() if k != "label"} == \
               {k: v for k, v in b.items() if k != "label"}
    for a, b in zip(ar, ar_new):
        assert all(b[k] == v for k, v in a.items())
        assert set(b) - set(a) <= {"label", "new_target_set"}
    # re-read from disk and re-check
    dp_disk = jsonl(os.path.join(od, "dependency_prediction.jsonl"))
    ar_disk = jsonl(os.path.join(od, "architecture_reconstruction.jsonl"))
    assert dp_disk == dp_new and ar_disk == ar_new

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

    stats = {"system": system}
    stats.update(extra or {})
    stats.update({
        "links_old": old_links,
        "links_new": links_new,
        "resolve_reasons": dict(reasons.most_common()),
        "drop_reasons": dict(drops.most_common()),
        "module_edges_old": len(old_set),
        "module_edges_new": len(edges),
        "module_edges_kept": len(old_set & edges),
        "module_edges_added": len(edges - old_set),
        "module_edges_removed": len(old_set - edges),
        "module_edges_new_between_original_modules":
            len({e for e in edges if e[0] in modules and e[1] in modules}) if modules else len(edges),
        "weight_old": sum(old_md.values()),
        "weight_new": sum(new_e.values()),
        "new_edges_lt_30pct_of_old": len(edges) < 0.3 * len(old_set),
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
        "task_items_verbatim_check": "passed (count, order, non-label fields)",
        "top_removed": [f"{a} -> {b} ({w})" for (a, b), w in
                        sorted(((e, old_md[e]) for e in old_set - edges), key=lambda x: -x[1])[:10]],
        "top_added": [f"{a} -> {b} ({w})" for (a, b), w in
                      sorted(((e, new_e[e]) for e in edges - old_set), key=lambda x: -x[1])[:10]],
        "audit": audit,
    })
    dump(stats, os.path.join(od, "stats.json"))
    print(f"  links {stats['links_old']} -> {stats['links_new']}, edges {stats['module_edges_old']} -> "
          f"{stats['module_edges_new']}, DP 1->0 {stats['dp_flip_1_to_0']} 0->1 {stats['dp_flip_0_to_1']}, "
          f"AR empty-src {stats['ar_source_modules_empty_new_target_set']}, "
          f"Jg {stats['ar_jaccard_graph_mean']} Jt {stats['ar_jaccard_task_mean']}, "
          f"clear_error {audit.get('clear_error_pct')}%", flush=True)
    return stats


def write_graph(od, system_name, links, edges, fname="file_dependency.json"):
    dump({"system": system_name, "total": len(links), "dependencies": links},
         os.path.join(od, fname))
    if edges is not None:
        dump({"system": system_name, "total": len(edges),
              "dependencies": [{"source_module": a, "target_module": b, "weight": w}
                               for (a, b), w in edges.items()]},
             os.path.join(od, "module_dependency.json"))


# ------------------------------------------------------------------ kubernetes

def k8s_normalize(m):
    parts = m.split("/")
    if m.startswith("staging/src/k8s.io/"):
        return "staging/src/k8s.io/" + parts[3] if len(parts) > 3 else m
    elif len(parts) > 1:
        return parts[0] + "/" + parts[1]
    return m


def norm_edges(c):
    out = Counter()
    for (a, b), w in c.items():
        na, nb = k8s_normalize(a), k8s_normalize(b)
        if na != nb:
            out[(na, nb)] += w
    return out


def rebuild_kubernetes():
    system = "kubernetes"
    sd = os.path.join(ROOT, system)
    od = os.path.join(HERE, system)
    os.makedirs(od, exist_ok=True)
    repo = os.path.join(sd, "kubernetes")
    assert os.path.isdir(repo), repo
    fi = load(os.path.join(sd, "file_index.json"))["files"]
    f2m = {f["path"]: f["module"] for f in fi}
    raws = load(os.path.join(sd, "dependency_raw.json"))["dependencies"]
    old_fd = load(os.path.join(sd, "file_dependency.json"))["dependencies"]
    old_md_raw = {(d["source_module"], d["target_module"]): d["weight"]
                  for d in load(os.path.join(sd, "module_dependency.json"))["dependencies"]}
    staging_repos = {d for d in os.listdir(os.path.join(repo, "staging/src/k8s.io"))
                     if os.path.isdir(os.path.join(repo, "staging/src/k8s.io", d))}
    # package dirs that are in the module space: contain >=1 indexed non-test .go file
    godirs = {posixpath.dirname(p) for p in f2m if p.endswith(".go") and not p.endswith("_test.go")}
    print(f"[{system}] file_index {len(f2m)}, go package dirs {len(godirs)}, raw {len(raws)}, "
          f"staging repos {len(staging_repos)}", flush=True)

    isdir_cache = {}

    def isdir(rel):
        if rel not in isdir_cache:
            p = os.path.join(repo, rel)
            isdir_cache[rel] = os.path.isdir(p) or os.path.islink(p)
        return isdir_cache[rel]

    links, recs = [], []
    reasons, drops = Counter(), Counter()
    invalid_examples, ext_k8s = [], Counter()
    for r in raws:
        src, imp, typ = r["source"], r["target"].strip(), r["type"]
        t, why = None, None
        if typ != "import":
            why = "unsupported_type"
        elif imp.startswith("k8s.io/kubernetes/"):
            t, why = imp[len("k8s.io/kubernetes/"):], "k8s_main_repo"
        elif imp == "k8s.io/kubernetes":
            why = "k8s_main_repo_root"
        elif imp.startswith("k8s.io/"):
            rp = imp.split("/")[1]
            if rp in staging_repos:
                t, why = "staging/src/" + imp, "k8s_staging_repo"
            else:
                why = "external_k8s_repo"
                ext_k8s["k8s.io/" + rp] += 1
        elif imp.startswith("pkg/") or imp.startswith("cmd/"):
            why = "invalid_non_module_import"
            invalid_examples.append(f"{src} | {imp}")
        elif "." in imp.split("/")[0]:
            why = "external_module"
        else:
            why = "stdlib"
        if t is not None:
            if not isdir(t):
                t, why = None, "target_dir_missing"
            elif t not in godirs:
                t, why = None, "target_outside_file_index"
        if t is None:
            drops[why] += 1
        else:
            reasons[why] += 1
            links.append({"source": src, "target": t, "type": typ})
        recs.append((src, imp, t, why))

    # module edges, original granularity: target module = package dir
    raw_e = Counter()
    for l in links:
        a, b = f2m.get(l["source"]), l["target"]
        if a and a != b:
            raw_e[(a, b)] += 1
    write_graph(od, system, links, raw_e)
    new_n = norm_edges(raw_e)
    old_n = norm_edges(Counter(old_md_raw))

    # audit
    A, CLEAR_ERROR = _audit_import()
    ix = A.Index(sd, "kubernetes/kubernetes")
    v, ex = Counter(), defaultdict(list)
    for src, imp, t, why in recs:
        if t is None:
            continue
        c = A.classify_go(ix, imp, t)
        v[c] += 1
        if c in CLEAR_ERROR and len(ex[c]) < 3:
            ex[c].append(f"{src} | {imp} -> {t}")
    n = sum(v.values())
    ce = sum(c for k, c in v.items() if k in CLEAR_ERROR)
    audit = {"verdicts": dict(v), "clear_error": ce,
             "clear_error_pct": round(100 * ce / n, 3) if n else None,
             "clear_error_examples": dict(ex)}
    # old-link audit for reference (target dir existence)
    old_v = Counter(A.classify_go(ix, None, d["target"]) for d in old_fd)

    extra = {
        "raw_statements": len(raws),
        "raw_reused_from": "kubernetes/dependency_raw.json",
        "file_index_files": len(fi),
        "go_package_dirs_in_file_index": len(godirs),
        "granularity": "file -> package directory (as original); target module = package dir; "
                       "task labels use normalize_module()",
        "module_edges_old_raw_names": len(old_md_raw),
        "module_edges_new_raw_names": len(raw_e),
        "old_links_target_dir_audit": dict(old_v),
        "old_links_to_staging_k8s_io_kubernetes": sum(
            1 for d in old_fd if d["target"].startswith("staging/src/k8s.io/kubernetes")),
        "external_k8s_repos_dropped": dict(ext_k8s.most_common()),
        "invalid_non_module_imports": invalid_examples[:20],
    }
    return finalize(system, od, sd, len(old_fd), len(links), new_n, old_n, reasons, drops,
                    audit, extra)


# ------------------------------------------------------------------ openfoam

INC_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*([<"])([^">\n]+)[">]', re.M)
OF_SRC_EXT = {".C", ".H", ".cxx", ".txx", ".L", ".lyy-m4", ".cc", ".rl", ".cpp", ".Cver",
              ".hxx", ".h", ".c", ".hpp", ".y", ".l", ".yy", ".ll", ".lyy"}


def rebuild_openfoam():
    system = "openfoam"
    sd = os.path.join(ROOT, system)
    od = os.path.join(HERE, system)
    os.makedirs(od, exist_ok=True)
    repo = os.path.join(sd, "openfoam")
    assert os.path.isdir(repo), repo
    fi = load(os.path.join(sd, "file_index.json"))["files"]
    f2m = {f["path"]: f["module"] for f in fi}       # same as original: last entry wins
    modules = set(load(os.path.join(sd, "modules_index.json"))["modules"])
    old_raws = load(os.path.join(sd, "dependency_raw.json"))["dependencies"]
    old_fd = load(os.path.join(sd, "file_dependency.json"))["dependencies"]
    old_md = {(d["source_module"], d["target_module"]): d["weight"]
              for d in load(os.path.join(sd, "module_dependency.json"))["dependencies"]}

    disk = build_disk_index(repo, None, IGNORE_DIRS)      # every file -> stricter uniqueness
    idx = FileIndex(disk)
    srcs = sorted(p for p in f2m if posixpath.splitext(p)[1] in OF_SRC_EXT and p in idx)

    raws = []
    for p in srcs:
        with open(os.path.join(repo, p), encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        for m in INC_RE.finditer(txt):
            raws.append((p, m.group(1), m.group(2).strip()))
    print(f"[{system}] disk {len(idx.paths)} files, file_index {len(fi)} entries / {len(f2m)} unique, "
          f"sources scanned {len(srcs)}, includes {len(raws)} (old raw {len(old_raws)})", flush=True)

    cpp = StrictUniqueCpp(idx)
    cpp.prepare({inc for _, _, inc in raws})
    links, recs = [], []
    reasons, drops = Counter(), Counter()
    angle_would = Counter()
    for src, delim, inc in raws:
        if delim == "<":
            res, why = cpp.resolve(src, inc)
            angle_would["resolvable_to_project_file" if res is not None and res in f2m
                        else "not_resolvable"] += 1
            res, why = None, "angle_system_include"
        else:
            res, why = cpp.resolve(src, inc)
        if res is not None:
            if res == src:
                res, why = None, "self_link"
            elif res not in f2m:
                res, why = None, "target_unmapped"
        if res is None:
            drops[why] += 1
        else:
            reasons[why] += 1
            links.append({"source": src, "target": res, "type": "include"})
        recs.append((src, inc, res, why))

    new_e = module_edges(((l["source"], l["target"]) for l in links), f2m)
    write_graph(od, system, links, new_e)
    audit = G.audit_new_links(system, recs, f2m)
    dummy = Counter(l["target"].split("/")[2] for l in links
                    if l["target"].startswith("src/dummyThirdParty/"))
    extra = {
        "raw_statements": len(raws),
        "raw_old_statements": len(old_raws),
        "raw_reextracted": "line-anchored #include, quote + angle, full path kept; "
                           "unique file_index paths with C/C++-like extensions",
        "raw_old_with_path": sum(1 for r in old_raws if "/" in r["target"]),
        "raw_new_with_path": sum(1 for _, d, i in raws if "/" in i and d == '"'),
        "raw_new_angle": sum(1 for _, d, _ in raws if d == "<"),
        "angle_includes_if_resolved_generically": dict(angle_would),
        "sources_scanned": len(srcs),
        "disk_index_files": len(idx.paths),
        "file_index_entries": len(fi),
        "file_index_unique_paths": len(f2m),
        "file_index_files_missing_on_disk": sum(1 for p in f2m if p not in idx),
        "links_to_dummyThirdParty_stubs": dict(dummy),
    }
    return finalize(system, od, sd, len(old_fd), len(links), new_e, old_md, reasons, drops,
                    audit, extra, modules)


# ------------------------------------------------------------------ salome

SALOME_EXT = CPP_EXT + (".hh", ".txx", ".ixx", ".tcc", ".inl")


def rebuild_salome():
    system = "salome"
    base = os.path.join(ROOT, "salome")
    ud = os.path.join(base, "unified")
    od = os.path.join(HERE, system)
    os.makedirs(od, exist_ok=True)
    src_root = os.path.join(base, "salome")
    uf2m = {f["path"]: f["module"] for f in load(os.path.join(ud, "file_index.json"))["files"]}
    modules = set(load(os.path.join(ud, "modules_index.json"))["modules"])
    old_md = {(d["source_module"], d["target_module"]): d["weight"]
              for d in load(os.path.join(ud, "module_dependency.json"))["dependencies"]}
    old_intra = load(os.path.join(ud, "file_dependency.json"))["dependencies"]
    old_cross = load(os.path.join(base, "crossrepo", "cross_file_dependency.json"))["dependencies"]

    per = {}
    gpaths = []
    for repo in SALOME_REPOS:
        rd = os.path.join(base, "reposanalysis", repo)
        rpath = os.path.join(src_root, repo)
        assert os.path.isdir(rpath), rpath
        rf2m = {f["path"]: f["module"] for f in load(os.path.join(rd, "file_index.json"))["files"]}
        raws = load(os.path.join(rd, "dependency_raw.json"))["dependencies"]
        disk = build_disk_index(rpath, SALOME_EXT, IGNORE_DIRS)
        per[repo] = (rf2m, raws, FileIndex(disk))
        gpaths += [repo + "/" + p for p in disk]
    gidx = FileIndex(gpaths)
    gidx.build_suffix({r["target"].strip() for _, raws, _ in per.values() for r in raws
                       if "/" in r["target"] and ".." not in r["target"].split("/")})

    intra, cross, recs = [], [], []
    reasons, drops = Counter(), Counter()
    per_repo = {}
    for repo in SALOME_REPOS:
        rf2m, raws, idx = per[repo]
        cpp = StrictUniqueCpp(idx)
        cpp.prepare({r["target"].strip() for r in raws if r["type"] == "include"})
        n_in = n_x = 0
        for r in raws:
            src, inc = r["source"], r["target"].strip()
            usrc = repo + "/" + src
            if r["type"] != "include":
                res, why = None, "unsupported_type"
            else:
                res, why = cpp.resolve(src, inc)
            ures, is_cross = None, False
            if res is not None:
                if res == src:
                    why = "self_link"
                elif res not in rf2m:
                    why = "target_unmapped"
                else:
                    ures = repo + "/" + res
            elif why in ("path_not_found", "bare_not_found"):
                # cross-repo: only a globally unique basename / path suffix
                if "/" in inc:
                    if ".." in inc.split("/") or inc.startswith("./"):
                        cands = []
                    else:
                        cands = gidx.suffix(inc)
                    tag = "cross_path"
                else:
                    cands = gidx.by_base.get(inc, [])
                    tag = "cross_bare"
                cands = [c for c in cands if not c.startswith(repo + "/")]
                if len(cands) == 1:
                    if cands[0] in uf2m:
                        ures, why, is_cross = cands[0], tag + "_unique", True
                    else:
                        why = tag + "_target_unmapped"
                elif len(cands) > 1:
                    why = tag + "_ambiguous"
                else:
                    why = why + "_global"
            if ures is None:
                drops[why] += 1
            else:
                reasons[why] += 1
                (cross if is_cross else intra).append(
                    {"source": usrc, "target": ures, "type": "include"})
                n_x += is_cross
                n_in += not is_cross
            recs.append((usrc, inc, ures, why))
        per_repo[repo] = {"raw": len(raws), "intra_links": n_in, "cross_links_from": n_x,
                          "disk_index_files": len(idx.paths), "file_index_files": len(rf2m)}
        print(f"[{system}] {repo}: raw {len(raws)} intra {n_in} cross {n_x}", flush=True)

    # unified merge as in crossInfoGeneration.py: per-repo module weights + cross counts
    intra_e = module_edges(((l["source"], l["target"]) for l in intra), uf2m)
    cross_e = module_edges(((l["source"], l["target"]) for l in cross), uf2m)
    new_e = Counter(intra_e)
    new_e.update(cross_e)
    write_graph(od, system, intra, new_e)
    dump({"dependencies": cross}, os.path.join(od, "cross_file_dependency.json"))

    audit = G.audit_new_links(system, recs, uf2m)
    # audit cross links separately
    A, CLEAR_ERROR = _audit_import()
    extra = {
        "raw_statements": sum(v["raw"] for v in per_repo.values()),
        "raw_reused_from": "salome/reposanalysis/<repo>/dependency_raw.json",
        "per_repo": per_repo,
        "links_old_intra": len(old_intra),
        "links_old_cross": len(old_cross),
        "links_new_intra": len(intra),
        "links_new_cross": len(cross),
        "module_edges_new_intra_only": len(intra_e),
        "module_edges_new_cross_only": len(cross_e),
        "global_disk_index_files": len(gidx.paths),
        "unified_file_index_files": len(uf2m),
        "file_dependency_json_contains": "intra-repo links (as original unified); "
                                         "cross-repo links in cross_file_dependency.json",
    }
    return finalize(system, od, ud, len(old_intra) + len(old_cross), len(intra) + len(cross),
                    new_e, old_md, reasons, drops, audit, extra, modules)


# ------------------------------------------------------------------ nastran

def rebuild_nastran():
    system = "nastran"
    sd = os.path.join(ROOT, system)
    od = os.path.join(HERE, system)
    os.makedirs(od, exist_ok=True)
    f2m = {f["path"]: f["module"] for f in load(os.path.join(sd, "file_index.json"))["files"]}
    modules = set(load(os.path.join(sd, "modules_index.json"))["modules"]) \
        if os.path.exists(os.path.join(sd, "modules_index.json")) else None
    shutil.copyfile(os.path.join(sd, "file_dependency.json"), os.path.join(od, "file_dependency.json"))
    shutil.copyfile(os.path.join(sd, "module_dependency.json"), os.path.join(od, "module_dependency.json"))
    fd = load(os.path.join(od, "file_dependency.json"))["dependencies"]
    old_md = {(d["source_module"], d["target_module"]): d["weight"]
              for d in load(os.path.join(od, "module_dependency.json"))["dependencies"]}
    new_e = module_edges(((d["source"], d["target"]) for d in fd), f2m)
    consistent = dict(new_e) == old_md
    print(f"[{system}] file links {len(fd)}, recomputed module edges == original: {consistent}", flush=True)

    A, CLEAR_ERROR = _audit_import()
    recs, _, a_edges, _ = A.build_links("nastran")
    v = Counter(r["verdict"] for r in recs)
    replay = Counter((r["src"], r["resolved"]) for r in recs)
    shipped = Counter((d["source"], d["target"]) for d in fd)
    n = sum(v.values())
    ce = sum(c for k, c in v.items() if k in CLEAR_ERROR)
    audit = {"verdicts": dict(v), "clear_error": ce,
             "clear_error_pct": round(100 * ce / n, 3) if n else None,
             "replayed_links_equal_file_dependency": replay == shipped,
             "replayed_links_only": sum((replay - shipped).values()),
             "file_dependency_only": sum((shipped - replay).values()),
             "replayed_module_edges_equal": dict(a_edges) == old_md}
    extra = {"mode": "copied original file_dependency / module_dependency (consistency check)",
             "recomputed_module_edges_equal_original": consistent}
    return finalize(system, od, sd, len(fd), len(fd), new_e, old_md, Counter({"copied": len(fd)}),
                    Counter(), audit, extra, modules)


# ------------------------------------------------------------------ main

RUN = {"kubernetes": rebuild_kubernetes, "openfoam": rebuild_openfoam,
       "salome": rebuild_salome, "nastran": rebuild_nastran}


def main():
    systems = sys.argv[1:] or SYSTEMS
    summary = {}
    sp = os.path.join(HERE, "summary_special.json")
    if os.path.exists(sp):
        summary = load(sp)
    for s in systems:
        st = RUN[s]()
        summary[s] = {k: st[k] for k in (
            "links_old", "links_new", "module_edges_old", "module_edges_new",
            "module_edges_new_between_original_modules", "module_edges_kept",
            "module_edges_added", "module_edges_removed", "new_edges_lt_30pct_of_old",
            "dp_positive_old", "dp_positive_new", "dp_flip_1_to_0", "dp_flip_0_to_1",
            "ar_items", "ar_items_edge_still_present", "ar_items_empty_new_target_set",
            "ar_source_modules", "ar_source_modules_empty_new_target_set",
            "ar_jaccard_graph_mean", "ar_jaccard_task_mean")}
        summary[s]["clear_error_pct"] = st["audit"].get("clear_error_pct")
        for k in ("module_edges_old_raw_names", "module_edges_new_raw_names",
                  "links_old_intra", "links_old_cross", "links_new_intra", "links_new_cross",
                  "raw_statements", "raw_old_statements", "recomputed_module_edges_equal_original"):
            if k in st:
                summary[s][k] = st[k]
        dump(summary, sp)


if __name__ == "__main__":
    main()
