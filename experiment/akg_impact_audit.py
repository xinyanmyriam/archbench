#!/usr/bin/env python3
"""
Audit the AKG dependency labels of all 12 systems (R2-4 / R3-2).

For every raw dependency statement we mirror the resolver that the original
pipeline used for that system, and then classify the resulting file-level link
as one of:

  verified_path            C/C++ include with a path that matches the resolved file
  verified_same_dir        bare include resolved to the source's own directory
  verified_unique_basename bare include, only one file with that name in the repo
  ambiguous_basename       bare include, several same-named files (prefix heuristic)
  fanout_ambiguous         salome: every same-named file is kept (one statement -> N links)
  config_gap               the real file exists on disk but is missing from file_index
  external_misresolved     the dependency comes from outside the project (not on disk),
                           yet it was forced onto a project file by name collision
  wrong_simple_name        Java: matched by simple class name while the correct FQN
                           resolves to a different file
  verified_fqn             Java: exact fully-qualified name
  verified_module          Python: exact dotted module path
  external_misresolved_py  Python: standard library / third-party module forced onto
                           a project file by name collision
  verified_dir             Go: import maps to an existing package directory
  target_dir_missing       Go: import maps to a directory that does not exist
  verified_symbol          Fortran: symbol found in subroutine_index

A link is "trusted" if its verdict starts with "verified"; otherwise it is
"suspect" (except plain ambiguity, which is reported separately).

From the labels we derive, per system:
  * share of file-level links and of graph weight that rests on trusted links
  * module edges with no trusted support at all
  * DP positive labels that lose their support, AR task edges that disappear

The auditor also asserts that replaying the original resolver reproduces
module_dependency.json exactly, so the numbers are tied to the shipped data.

Run:  python -u experiment/akg_impact_audit.py
Out:  experiment/akg_impact_audit.json
"""

import json
import os
import re
from collections import Counter, defaultdict

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP)
OUT = os.path.join(EXP, "akg_impact_audit.json")

SYSTEMS = {
    # system: (AKG data dir, source tree, language)
    "nastran":     ("nastran",               "nastran/NASTRAN-95",      "fortran"),
    "openfoam":    ("openfoam",              "openfoam/openfoam",       "include_basename"),
    "salome":      ("salome/unified",        "salome/salome",           "salome"),
    "paraview":    ("paraview",              "paraview/ParaView",       "cpp"),
    "kubernetes":  ("kubernetes",            "kubernetes/kubernetes",   "go"),
    "linux":       ("linux",                 "linux/linux",             "cpp"),
    "hadoop":      ("hadoop",                "hadoop/hadoop",           "java"),
    "spring":      ("spring",                "spring/spring-framework", "java"),
    "codeaster":   ("codeaster",             "codeaster/codeaster",     "mixed"),
    "openmodelica": ("openmodelica",         "openmodelica/OpenModelica", "mixed"),
    "tensorflow":  ("tensorflow",            "tensorflow/tensorflow",   "mixed"),
    "pytorch":     ("pytorch",               "pytorch/pytorch",         "mixed"),
}

SALOME_REPOS = ["kernel", "geom", "smesh", "medcoupling", "yacs", "gui"]


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cpl(a, b):
    """common path prefix length, in directory levels"""
    n = 0
    for x, y in zip(a.split("/"), b.split("/")):
        if x != y:
            break
        n += 1
    return n


class Index:
    """Mirrors file_index.json plus the on-disk repo tree."""

    def __init__(self, project_dir, repo_rel):
        data = load(os.path.join(project_dir, "file_index.json"))
        self.files = data["files"]
        self.f2m = {f["path"]: f["module"] for f in self.files}
        self.all_paths = set(self.f2m)
        self.by_base = defaultdict(list)
        for p in self.f2m:
            self.by_base[os.path.basename(p)].append(p)
        self.repo = os.path.join(ROOT, repo_rel)
        self.pkg2path = {}
        for p in self.f2m:
            q = p[:-5] if p.endswith(".java") else p
            self.pkg2path[q.replace("/", ".")] = p
        # every package-path suffix -> indexed files (Maven/Gradle paths carry the
        # source root, so the import FQN is only a suffix of the recorded path)
        self.suffix = defaultdict(list)
        for pkg, p in self.pkg2path.items():
            parts = pkg.split(".")
            for k in range(len(parts), 0, -1):
                self.suffix[".".join(parts[-k:])].append(p)

    def on_disk(self, rel):
        return os.path.exists(os.path.join(self.repo, rel))

    def dir_on_disk(self, rel):
        p = os.path.join(self.repo, rel)
        # a git symlink checked out on Windows is a plain file; treat both as a hit
        return os.path.isdir(p) or os.path.islink(p)

    def build_suffix_map(self, wanted):
        """path suffix (up to 8 segments) -> indexed paths, only for the wanted keys."""
        self.path_suffix = {k: [] for k in wanted}
        for p in self.f2m:
            parts = p.split("/")
            for k in range(1, min(len(parts), 8) + 1):
                s = "/".join(parts[-k:])
                if s in self.path_suffix:
                    self.path_suffix[s].append(p)


def java_fqn_from_path(path):
    p = path[:-5] if path.endswith(".java") else path
    return p.replace("/", ".")


def resolve_cpp(index, src, inc):
    """pipeline/generate_all.py include branch."""
    cands = index.by_base.get(os.path.basename(inc), [])
    if len(cands) == 1:
        return cands[0]
    if cands:
        return max(cands, key=lambda c: cpl(src, c))
    if inc in index.all_paths:
        return inc
    return None


_DISK_CACHE = {}


def on_disk_repo(index, rel):
    key = (index.repo, rel)
    if key not in _DISK_CACHE:
        _DISK_CACHE[key] = os.path.exists(os.path.join(index.repo, rel))
    return _DISK_CACHE[key]


def classify_cpp(index, src, inc, resolved):
    if inc.startswith("./") or inc.startswith("../") or "/../" in inc:
        # relative include: resolves against the source file's directory
        exp = os.path.normpath(os.path.join(os.path.dirname(src), inc)).replace("\\", "/")
        if resolved == exp:
            return "verified_path"
    if "/" in inc:
        # A C/C++ include with a path resolves either to a path ending in that
        # string (project source root / include root) or relative to the source file.
        cands = set(index.path_suffix.get(inc, ()))
        if resolved in cands or resolved == inc:
            return "verified_path"
        if cands:
            return "wrong_path"          # the correct file is indexed, another was chosen
        for probe in (inc, "include/" + inc):
            if on_disk_repo(index, probe):
                return "config_gap"      # real file exists but is missing from file_index
        return "external_misresolved"
    if resolved == os.path.join(os.path.dirname(src), inc).replace("\\", "/"):
        return "verified_same_dir"
    cands = index.by_base.get(os.path.basename(inc), [])
    if len(cands) == 1:
        return "verified_unique_basename"
    return "ambiguous_basename"


def resolve_py(index, target):
    t = target.replace(".", "/")
    for c in (t + ".py", t + "/__init__.py"):
        if c in index.all_paths:
            return c
    parts = target.split(".")
    cands = index.by_base.get(parts[-1] + ".py", [])
    if cands:
        return max(cands, key=lambda c: cpl(t, c))
    return None


def classify_py(index, src, mod, resolved):
    """A Python import resolves against one of the package roots above the source file
    (the project may add a root prefix such as 'catalo/' that is not part of the
    import string), so every prefix of the source path is a candidate root.
    Relative imports ('.mod', '..pkg.mod') resolve against the source's package."""
    if mod.startswith("."):
        lvl = len(mod) - len(mod.lstrip("."))
        pkg = src.split("/")[:-1]
        base = pkg[:len(pkg) - (lvl - 1)] if lvl - 1 <= len(pkg) else []
        rest = mod.lstrip(".").replace(".", "/")
        exp = "/".join(base + ([rest] if rest else []))
        if resolved in (exp + ".py", exp + "/__init__.py"):
            return "verified_module"
        return "wrong_module"            # relative import is internal by definition
    t = mod.replace(".", "/")
    parts = src.split("/")
    cands = set()
    for k in range(len(parts), 0, -1):
        root = "/".join(parts[:k - 1])
        for form in (t + ".py", t + "/__init__.py"):
            cands.add((root + "/" + form).lstrip("/"))
    if resolved in cands:
        return "verified_module"
    if cands & index.all_paths:
        return "wrong_module"
    if on_disk_repo(index, t + ".py") or on_disk_repo(index, t):
        return "config_gap"
    return "external_misresolved_py"


def resolve_java(index, target):
    """pipeline/generate_all.py java branch: exact FQN, else simple class name."""
    if target in index.pkg2path:
        return index.pkg2path[target]
    parts = target.split(".")
    cands = index.by_base.get(parts[-1] + ".java", [])
    if len(cands) == 1:
        return cands[0]
    if cands:
        pkg_path = "/".join(parts[:-1])
        return max(cands, key=lambda c: cpl(pkg_path, c))
    return None


def java_target(index, fqn):
    """Indexed files the FQN really points to; trailing segments are stripped so that
    nested classes and static-member imports resolve to their enclosing class."""
    parts = fqn.split(".")
    for k in range(len(parts), 0, -1):
        hit = index.suffix.get(".".join(parts[:k]))
        if hit:
            return hit
    return []


def classify_java(index, src, fqn, resolved):
    tgt = java_target(index, fqn)
    if resolved in tgt:
        return "verified_fqn"
    if not tgt:
        if on_disk_repo(index, fqn.replace(".", "/") + ".java"):
            return "config_gap"
        return "external_misresolved"
    cands = index.by_base.get(fqn.split(".")[-1] + ".java", [])
    if len(cands) == 1:
        return "verified_unique_class"
    return "wrong_simple_name"


def classify_go(index, imp, resolved):
    if index.dir_on_disk(resolved):
        return "verified_dir"
    return "target_dir_missing"


def salome_indexes():
    """repo -> (per-repo Index built from reposanalysis output, repo dir name)"""
    out = {}
    for repo in SALOME_REPOS:
        d = os.path.join(ROOT, "salome/reposanalysis", repo)
        if not os.path.isfile(os.path.join(d, "file_index.json")):
            continue
        out[repo] = Index(d, os.path.join("salome/salome", repo))
    return out


def build_links(system):
    """Replay the original resolver. Returns (records, index, rendered_graph)."""
    data_dir, repo_rel, lang = SYSTEMS[system]
    d = os.path.join(ROOT, data_dir)
    recs = []          # dict(src=..., raw=..., resolved=..., verdict=...)
    edges = Counter()  # (src_module, tgt_module) -> weight, replayed
    extra_notes = {}

    if system == "salome":
        idx = Index(d, repo_rel)                     # unified index for the module map
        f2m = idx.f2m
        repos = salome_indexes()
        for repo, ridx in repos.items():
            raws = load(os.path.join(ROOT, "salome/reposanalysis", repo, "dependency_raw.json"))["dependencies"]
            for r in raws:
                src, inc = repo + "/" + r["source"], r["target"]
                base = os.path.basename(inc)
                cands = ridx.by_base.get(base, [])
                if not cands:
                    continue
                for c in cands:
                    tgt = repo + "/" + c
                    if tgt == src:
                        continue
                    if len(cands) == 1:
                        v = "verified_unique_basename"
                    elif inc == c or tgt.endswith("/" + inc):
                        v = "verified_path"
                    else:
                        v = "fanout_ambiguous"
                    recs.append({"src": src, "raw": inc, "resolved": tgt, "verdict": v})
                    a, b = f2m.get(src), f2m.get(tgt)
                    if a and b and a != b:
                        edges[(a, b)] += 1
        # cross-repo links are produced by a separate script; keep them, flagged
        cross = load(os.path.join(ROOT, "salome/crossrepo/cross_file_dependency.json"))["dependencies"]
        for c in cross:
            recs.append({"src": c["source"], "raw": "(cross-repo)", "resolved": c["target"],
                         "verdict": "crossrepo_unverified"})
            a, b = f2m.get(c["source"]), f2m.get(c["target"])
            if a and b and a != b:
                edges[(a, b)] += 1
        extra_notes["crossrepo_links"] = len(cross)
        return recs, idx, edges, extra_notes

    idx = Index(d, repo_rel)
    raws = load(os.path.join(d, "dependency_raw.json"))["dependencies"]
    if lang in ("cpp", "mixed"):
        idx.build_suffix_map({r["target"] for r in raws
                              if r["type"] == "include" and "/" in r["target"]})

    if system == "nastran":
        sub = {k.upper(): v for k, v in load(os.path.join(d, "subroutine_index.json")).items()}
        for r in raws:
            t = sub.get(r["target"].upper())
            if t is None or t == r["source"] or t not in idx.all_paths:
                continue
            recs.append({"src": r["source"], "raw": r["target"], "resolved": t, "verdict": "verified_symbol"})
            a, b = idx.f2m.get(r["source"]), idx.f2m.get(t)
            if a and b and a != b:
                edges[(a, b)] += 1
        return recs, idx, edges, extra_notes

    if system == "kubernetes":
        for r in raws:
            imp = r["target"]
            if imp.startswith("k8s.io/"):
                t = "staging/src/" + imp
            elif imp.startswith("pkg/") or imp.startswith("cmd/"):
                t = imp
            else:
                continue
            recs.append({"src": r["source"], "raw": imp, "resolved": t,
                         "verdict": classify_go(idx, imp, t)})
            a, b = idx.f2m.get(r["source"]), t        # target module is the package dir
            if a and a != b:
                edges[(a, b)] += 1
        return recs, idx, edges, extra_notes

    if system == "openfoam":
        # basename only; any include carrying a path is silently dropped
        dropped = 0
        for r in raws:
            inc = r["target"]
            cands = idx.by_base.get(inc, [])
            if not cands:
                dropped += 1
                continue
            t = cands[0] if len(cands) == 1 else max(cands, key=lambda c: cpl(r["source"], c))
            recs.append({"src": r["source"], "raw": inc, "resolved": t,
                         "verdict": "verified_unique_basename" if len(cands) == 1 else "ambiguous_basename"})
            a, b = idx.f2m.get(r["source"]), idx.f2m.get(t)
            if a and b and a != b:
                edges[(a, b)] += 1
        extra_notes["raw_dropped"] = dropped
        return recs, idx, edges, extra_notes

    for r in raws:
        src, target, typ = r["source"], r["target"], r["type"]
        if typ == "call":
            continue
        is_py = lang == "mixed" and src.endswith(".py") and typ == "import"
        if is_py:
            t = resolve_py(idx, target)
            if t and t != src:
                recs.append({"src": src, "raw": target, "resolved": t,
                             "verdict": classify_py(idx, src, target, t)})
                a, b = idx.f2m.get(src), idx.f2m.get(t)
                if a and b and a != b:
                    edges[(a, b)] += 1
        elif typ == "import" and lang == "java":
            t = resolve_java(idx, target)
            if t and t != src:
                recs.append({"src": src, "raw": target, "resolved": t,
                             "verdict": classify_java(idx, src, target, t)})
                a, b = idx.f2m.get(src), idx.f2m.get(t)
                if a and b and a != b:
                    edges[(a, b)] += 1
        elif typ == "include":
            t = resolve_cpp(idx, src, target)
            if t and t != src:
                recs.append({"src": src, "raw": target, "resolved": t,
                             "verdict": classify_cpp(idx, src, target, t)})
                a, b = idx.f2m.get(src), idx.f2m.get(t)
                if a and b and a != b:
                    edges[(a, b)] += 1
    return recs, idx, edges, extra_notes


TRUSTED = ("verified", "fanout_ambiguous_not_used")
# verdicts that confirm the target from the statement itself (path, FQN, dotted module,
# package directory, or symbol); 'verified_unique_basename' is only name-level evidence
PATH_VERIFIED = ("verified_path", "verified_fqn", "verified_module", "verified_dir",
                 "verified_symbol", "verified_same_dir")


def _kg_norm(m):
    """kubernetes/generate_benchmark_tasks.py:normalize_module, applied both to the
    module list and to the endpoints of module_dependency.json. Package paths are
    collapsed to a top-level package ('staging/src/k8s.io/api/core/v1' ->
    'staging/src/k8s.io/api', 'pkg/volume/util' -> 'pkg/volume')."""
    parts = m.split("/")
    if m.startswith("staging/src/k8s.io/"):
        return "staging/src/k8s.io/" + parts[3] if len(parts) > 3 else m
    if len(parts) > 1:
        return parts[0] + "/" + parts[1]
    return m


def _raw_count(system, data_dir):
    if system == "salome":
        n = 0
        for repo in SALOME_REPOS:
            p = os.path.join(ROOT, "salome/reposanalysis", repo, "dependency_raw.json")
            if os.path.isfile(p):
                n += len(load(p)["dependencies"])
        p = os.path.join(ROOT, "salome/crossrepo/cross_dependency_raw.json")
        if os.path.isfile(p):
            n += len(load(p)["dependencies"])
        return n
    p = os.path.join(ROOT, data_dir, "dependency_raw.json")
    return len(load(p)["dependencies"]) if os.path.isfile(p) else 0


def audit_system(system):
    recs, idx, edges, notes = build_links(system)
    data_dir = SYSTEMS[system][0]
    shipped = {(d["source_module"], d["target_module"]): d["weight"]
               for d in load(os.path.join(ROOT, data_dir, "module_dependency.json"))["dependencies"]}
    replay_ok = dict(edges) == shipped

    verdicts = Counter(r["verdict"] for r in recs)
    trusted_link = lambda r: r["verdict"].startswith("verified")
    n_trusted = sum(1 for r in recs if trusted_link(r))

    # per-edge support
    def tmod(resolved):
        # Kubernetes imports resolve to package directories, and the shipped graph
        # uses that directory as the target module
        return resolved if system == "kubernetes" else idx.f2m.get(resolved)

    sup_all, sup_trusted, sup_path = Counter(), Counter(), Counter()
    for r in recs:
        a, b = idx.f2m.get(r["src"]), tmod(r["resolved"])
        if a and b and a != b:
            sup_all[(a, b)] += 1
            if trusted_link(r):
                sup_trusted[(a, b)] += 1
            if r["verdict"] in PATH_VERIFIED:
                sup_path[(a, b)] += 1

    edges_total = len(shipped)
    edges_no_trusted = sum(1 for e in shipped if sup_trusted.get(e, 0) == 0)
    w_total = sum(shipped.values())
    w_no_trusted = sum(w for e, w in shipped.items() if sup_trusted.get(e, 0) == 0)
    w_trusted = sum(min(w, sup_trusted.get(e, 0)) for e, w in shipped.items())

    tasks = os.path.join(ROOT, data_dir)
    dp = jsonl(os.path.join(tasks, "dependency_prediction.jsonl"))
    ar = jsonl(os.path.join(tasks, "architecture_reconstruction.jsonl"))
    pos = [t for t in dp if t["label"] == 1]
    neg = [t for t in dp if t["label"] == 0]

    if system == "kubernetes":
        # the benchmark tasks are defined on collapsed package names, so the
        # support and graph statistics are recomputed in that naming space
        nz = _kg_norm
        sup_trusted = Counter({(nz(a), nz(b)): c for (a, b), c in sup_trusted.items() if nz(a) != nz(b)})
        sup_path = Counter({(nz(a), nz(b)): c for (a, b), c in sup_path.items() if nz(a) != nz(b)})
        shipped = {(nz(a), nz(b)): w for (a, b), w in shipped.items() if nz(a) != nz(b)}
        edges_total = len(shipped)
        edges_no_trusted = sum(1 for e in shipped if sup_trusted.get(e, 0) == 0)
        w_total = sum(shipped.values())
        w_no_trusted = sum(w for e, w in shipped.items() if sup_trusted.get(e, 0) == 0)
        w_trusted = sum(min(w, sup_trusted.get(e, 0)) for e, w in shipped.items())
        pair = lambda t: (nz(t["module_A"]), nz(t["module_B"]))
        apair = lambda t: (nz(t["source_module"]), nz(t["target_module"]))
    else:
        pair = lambda t: (t["module_A"], t["module_B"])
        apair = lambda t: (t["source_module"], t["target_module"])

    pos_lost = sum(1 for t in pos if sup_trusted.get(pair(t), 0) == 0)
    ar_lost = sum(1 for t in ar if sup_trusted.get(apair(t), 0) == 0)
    src_mods = {apair(t)[0] for t in ar}
    src_mods_emptied = sum(1 for s in src_mods
                           if not any(e[0] == s and sup_trusted.get(e, 0) for e in shipped))

    # same counters restricted to path/FQN/module/symbol-verified support
    edges_no_path = sum(1 for e in shipped if sup_path.get(e, 0) == 0)
    w_no_path = sum(w for e, w in shipped.items() if sup_path.get(e, 0) == 0)
    w_path = sum(min(w, sup_path.get(e, 0)) for e, w in shipped.items())
    pos_lost_path = sum(1 for t in pos if sup_path.get(pair(t), 0) == 0)
    ar_lost_path = sum(1 for t in ar if sup_path.get(apair(t), 0) == 0)

    # a couple of representative wrong links per failure mode
    examples = defaultdict(list)
    for r in recs:
        v = r["verdict"]
        if not v.startswith("verified") and len(examples[v]) < 3:
            examples[v].append(f'{r["src"]} | {r["raw"]} -> {r["resolved"]}')

    return {
        "system": system,
        "language": SYSTEMS[system][2],
        "replay_reproduces_module_dependency": replay_ok,
        "raw_statements": _raw_count(system, data_dir),
        "links": len(recs),
        "links_trusted": n_trusted,
        "links_trusted_pct": round(100 * n_trusted / len(recs), 1) if recs else None,
        "verdicts": dict(verdicts),
        "notes": notes,
        "module_edges": edges_total,
        "module_weight": w_total,
        "edges_without_trusted_support": edges_no_trusted,
        "edges_without_trusted_support_pct": round(100 * edges_no_trusted / edges_total, 1) if edges_total else None,
        "weight_without_trusted_support": w_no_trusted,
        "weight_without_trusted_support_pct": round(100 * w_no_trusted / w_total, 1) if w_total else None,
        "weight_trusted": w_trusted,
        "weight_trusted_pct": round(100 * w_trusted / w_total, 1) if w_total else None,
        "dp_positive": len(pos), "dp_negative": len(neg),
        "dp_positive_pairs_in_graph": sum(1 for t in pos if pair(t) in shipped),
        "dp_positive_losing_support": pos_lost,
        "dp_positive_losing_support_pct": round(100 * pos_lost / len(pos), 1) if pos else None,
        "ar_tasks": len(ar), "ar_edges_lost": ar_lost,
        "ar_pairs_in_graph": sum(1 for t in ar if apair(t) in shipped),
        "ar_edges_lost_pct": round(100 * ar_lost / len(ar), 1) if ar else None,
        # strict (path/FQN/module/symbol-verified) view
        "edges_without_path_support": edges_no_path,
        "edges_without_path_support_pct": round(100 * edges_no_path / edges_total, 1) if edges_total else None,
        "weight_without_path_support_pct": round(100 * w_no_path / w_total, 1) if w_total else None,
        "weight_path_support_pct": round(100 * w_path / w_total, 1) if w_total else None,
        "dp_positive_losing_path_support": pos_lost_path,
        "dp_positive_losing_path_support_pct": round(100 * pos_lost_path / len(pos), 1) if pos else None,
        "ar_edges_losing_path_support_pct": round(100 * ar_lost_path / len(ar), 1) if ar else None,
        "ar_source_modules": len(src_mods),
        "ar_source_modules_without_trusted_targets": src_mods_emptied,
        "examples": {k: v for k, v in examples.items()},
    }


def main():
    out = {}
    for system in SYSTEMS:
        print(f"[{system}] auditing ...", flush=True)
        out[system] = audit_system(system)
        r = out[system]
        print(f"  replay={r['replay_reproduces_module_dependency']} "
              f"links={r['links']} trusted={r['links_trusted_pct']}% "
              f"weight_trusted={r['weight_trusted_pct']}% "
              f"dp_pos_lost={r['dp_positive_losing_support_pct']}%", flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
