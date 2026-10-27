#!/usr/bin/env python3
"""
Fill in file-level evidence for the 200-edge AKG validation sample (R3-2 / R2-4, P0-5).

The earlier sampler (akg_sampling_validation.py) tried to re-resolve raw
import strings and found target-verified evidence for only 23/200 edges.
This script keeps the SAME 200 sampled module edges and instead takes the
evidence from the already-resolved file-level graph (file_dependency.json),
which is exactly what module_dependency.json aggregates. For every edge it
reports how many file-level links support it, samples up to three of them,
and locates the corresponding statement (line number + text) in the source
tree so that a human annotator can judge each link without re-running the
pipeline.

It also draws a recall sample: source files for which the annotator lists
the internal dependencies the AKG missed.

Outputs (experiment/):
  akg_validation_evidence.csv   precision sheet, one row per sampled edge
  akg_recall_sample.csv         recall sheet, one row per sampled file
  akg_validation_evidence_summary.json
"""

import csv
import json
import os
import random
import re
from collections import Counter, defaultdict

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP)
SEED = 42
MAX_EXAMPLES = 3
RECALL_PER_SYSTEM = {"default": 4, "linux": 5, "kubernetes": 5}  # 10*4 + 2*5 = 50

SAMPLE_CSV = os.path.join(EXP, "akg_validation_sample.csv")
OUT_EVIDENCE = os.path.join(EXP, "akg_validation_evidence.csv")
OUT_RECALL = os.path.join(EXP, "akg_recall_sample.csv")
OUT_SUMMARY = os.path.join(EXP, "akg_validation_evidence_summary.json")

# system -> (AKG data dir, source tree) relative to ROOT
SYSTEMS = {
    "nastran": ("nastran", "nastran/NASTRAN-95"),
    "openfoam": ("openfoam", "openfoam/openfoam"),
    "salome": ("salome/unified", "salome/salome"),
    "paraview": ("paraview", "paraview/ParaView"),
    "kubernetes": ("kubernetes", "kubernetes/kubernetes"),
    "linux": ("linux", "linux/linux"),
    "hadoop": ("hadoop", "hadoop/hadoop"),
    "spring": ("spring", "spring/spring-framework"),
    "codeaster": ("codeaster", "codeaster/codeaster"),
    "openmodelica": ("openmodelica", "openmodelica/OpenModelica"),
    "tensorflow": ("tensorflow", "tensorflow/tensorflow"),
    "pytorch": ("pytorch", "pytorch/pytorch"),
}

SOURCE_EXT = {
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hxx", ".hh", ".C", ".H",
    ".py", ".java", ".go", ".f", ".F", ".f90", ".F90", ".for", ".mo",
}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_system(system):
    data_dir, _ = SYSTEMS[system]
    d = os.path.join(ROOT, data_dir)
    file_to_mod = {f["path"]: f["module"] for f in load(os.path.join(d, "file_index.json"))["files"]}
    fdeps = list(load(os.path.join(d, "file_dependency.json"))["dependencies"])
    raws = []
    if system == "salome":
        cross = os.path.join(ROOT, "salome/crossrepo/cross_file_dependency.json")
        fdeps += load(cross)["dependencies"]
        repo_dir = os.path.join(ROOT, "salome/reposanalysis")
        for repo in sorted(os.listdir(repo_dir)):
            p = os.path.join(repo_dir, repo, "dependency_raw.json")
            if os.path.isfile(p):
                for r in load(p)["dependencies"]:
                    raws.append({**r, "source": repo + "/" + r["source"]})
    else:
        raws = load(os.path.join(d, "dependency_raw.json"))["dependencies"]
    return file_to_mod, fdeps, raws


def target_module(system, target, file_to_mod):
    # Kubernetes resolves Go imports to package directories, and the module
    # graph uses that directory directly as the target module.
    if system == "kubernetes":
        return target
    return file_to_mod.get(target)


def read_lines(path):
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                return f.read().splitlines()
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')
GO_STR_RE = re.compile(r'"([^"]+)"')


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def looks_commented(line):
    s = line.lstrip()
    if s.startswith(("//", "/*", "*", "--")):
        return True
    if line[:1] in ("C", "c", "*", "!") and not line[:1].isspace() and len(line) > 1 and line[1] == " ":
        return True  # fixed-form Fortran comment
    return False


PY_FROM_RE = re.compile(r"^\s*from\s+(\.*[\w.]*)\s+import\s+(.+)")
PY_IMPORT_RE = re.compile(r"^\s*import\s+([\w.]+(?:\s*,\s*[\w.]+)*)")
JAVA_IMPORT_RE = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+(?:\.\*)?)\s*;")
JAVA_PKG_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;")


def java_fqn(path):
    """org/x/Y.java under .../java/ -> org.x.Y (best effort)."""
    p = os.path.splitext(path)[0]
    for marker in ("/src/main/java/", "/src/test/java/", "/java/", "/src/"):
        if marker in p:
            p = p.split(marker, 1)[1]
            break
    return p.replace("/", ".")


def py_dotted(path):
    p = os.path.splitext(path)[0].replace("/", ".")
    return p[: -len(".__init__")] if p.endswith(".__init__") else p


def ends_with_dotted(full, part):
    return full == part or full.endswith("." + part)


def find_statement(system, dep, lines, src_root):
    """Locate the statement that yields dep and check that it resolves to dep's target.

    Returns (line_no, text, match_kind, auto_check). auto_check is
    'consistent', 'ambiguous_basename', or 'suspect:<reason>'.
    """
    tgt, typ = dep["target"], dep["type"]
    base, st = os.path.basename(tgt), stem(tgt)
    cands = []  # (line_no, text, kind, check)
    src_pkg = None
    for i, line in enumerate(lines, 1):
        if typ == "include":
            m = INCLUDE_RE.match(line)
            if m:
                inc = m.group(1)
                if tgt == inc or tgt.endswith("/" + inc):
                    cands.append((i, line, "include_path_suffix", "consistent"))
                elif os.path.basename(inc) == base:
                    check = ("suspect:include_path_mismatch" if "/" in inc else "ambiguous_basename")
                    cands.append((i, line, "include_basename", check))
            elif typ == "include" and "import" in line and re.search(r"\b%s\b" % re.escape(st), line):
                cands.append((i, line, "import_name", "unchecked"))
        elif typ == "call":
            if re.search(r"\bCALL\s+%s\b" % re.escape(st), line, re.I):
                cands.append((i, line, "fortran_call", "consistent"))
        elif typ == "import" and system == "kubernetes":
            for s in GO_STR_RE.findall(line):
                if "/" in s and (tgt == s or tgt.endswith("/" + s)):
                    exists = os.path.isdir(os.path.join(src_root, tgt))
                    cands.append((i, line, "go_import_path",
                                  "consistent" if exists else "suspect:target_dir_missing"))
        elif typ == "import" and tgt.endswith(".java"):
            pm = JAVA_PKG_RE.match(line)
            if pm:
                src_pkg = pm.group(1)
            m = JAVA_IMPORT_RE.match(line)
            if m:
                name, fqn = m.group(1), java_fqn(tgt)
                if name == fqn:
                    cands.append((i, line, "java_import_class", "consistent"))
                elif name.endswith(".*") and fqn.rsplit(".", 1)[0] == name[:-2]:
                    cands.append((i, line, "java_import_wildcard", "consistent"))
                elif name.rsplit(".", 1)[-1] == st:
                    cands.append((i, line, "java_import_simple_name", "suspect:fqn_mismatch"))
        elif typ == "import":
            tdot = py_dotted(tgt)
            m = PY_FROM_RE.match(line)
            mods = []
            if m:
                mod, names = m.group(1), m.group(2)
                names = [n.strip().split(" as ")[0].strip("() ") for n in names.split(",")]
                if mod.startswith("."):
                    if any(n == st for n in names) or mod.lstrip(".").split(".")[-1:] == [st] or st in mod:
                        cands.append((i, line, "python_relative_import", "consistent"))
                    continue
                mods = [mod] + [mod + "." + n for n in names if n]
            else:
                m = PY_IMPORT_RE.match(line)
                if m:
                    mods = [x.strip() for x in m.group(1).split(",")]
            if not mods:
                continue
            if any(ends_with_dotted(tdot, x) and (x == tdot or "." in x) for x in mods):
                cands.append((i, line, "python_import_module", "consistent"))
            elif any(x.split(".")[-1] == st or x == st for x in mods):
                cands.append((i, line, "python_import_name", "suspect:module_path_mismatch"))
    if typ == "import" and tgt.endswith(".java") and not cands and src_pkg:
        tpkg = java_fqn(tgt).rsplit(".", 1)[0]
        if tpkg == src_pkg:
            for i, line in enumerate(lines, 1):
                if re.search(r"\b%s\b" % re.escape(st), line):
                    return i, line.strip()[:200], "java_same_package_use", "consistent"
        return "", "", "not_found", "suspect:no_matching_import"
    if not cands:
        return "", "", "not_found", "suspect:no_statement"
    order = {"consistent": 0, "ambiguous_basename": 1, "unchecked": 2}
    cands.sort(key=lambda c: (looks_commented(c[1]), order.get(c[3], 3)))
    i, line, kind, check = cands[0]
    if looks_commented(line):
        kind += "+maybe_comment"
    return i, line.strip()[:200], kind, check


def granularity_flag(src, tgt):
    if src.startswith(tgt + "/") or tgt.startswith(src + "/"):
        return "nested_modules"
    return ""


def main():
    rng = random.Random(SEED)
    with open(SAMPLE_CSV, encoding="utf-8") as f:
        sample = list(csv.DictReader(f))
    by_sys = defaultdict(list)
    for r in sample:
        by_sys[r["system"]].append(r)

    evidence_rows, recall_rows = [], []
    stats = {"edges": 0, "weight_match": 0, "status": Counter(), "match_kind": Counter(),
             "granularity": Counter(), "auto_check": Counter(), "edges_with_suspect_link": 0,
             "edges_all_suspect": 0, "ambiguous_basename_links": 0, "examples": 0, "per_system": {}}

    for system in SYSTEMS:
        print(f"[{system}] loading ...", flush=True)
        file_to_mod, fdeps, raws = load_system(system)
        src_root = os.path.join(ROOT, SYSTEMS[system][1])
        module_files = set(file_to_mod.values())
        base_count = Counter(os.path.basename(p) for p in file_to_mod)

        support = defaultdict(list)
        out_links = defaultdict(list)
        for d in fdeps:
            a = file_to_mod.get(d["source"])
            b = target_module(system, d["target"], file_to_mod)
            out_links[d["source"]].append(d["target"])
            if a and b and a != b:
                support[(a, b)].append(d)
        raw_by_src = defaultdict(list)
        for r in raws:
            raw_by_src[r["source"]].append(r["target"])

        line_cache = {}
        sys_stat = Counter()
        for row in by_sys.get(system, []):
            src, tgt, w = row["source_module"], row["target_module"], int(row["weight"])
            links = support.get((src, tgt), [])
            stats["edges"] += 1
            stats["weight_match"] += int(len(links) == w)
            picked = rng.sample(links, min(MAX_EXAMPLES, len(links)))
            out = {
                "sample_id": row["sample_id"], "system": system,
                "source_module": src, "target_module": tgt, "weight": w,
                "support_links": len(links), "weight_consistent": int(len(links) == w),
            }
            found = suspect = 0
            for k in range(MAX_EXAMPLES):
                p = f"ex{k + 1}_"
                if k >= len(picked):
                    out.update({p + "source": "", p + "line": "", p + "statement": "",
                                p + "resolved_target": "", p + "match": "", p + "auto_check": "",
                                p + "same_basename_files": ""})
                    continue
                d = picked[k]
                path = d["source"]
                if path not in line_cache:
                    line_cache[path] = read_lines(os.path.join(src_root, path))
                lines = line_cache[path]
                if lines is None:
                    ln, text, kind, check = "", "", "source_unreadable", "suspect:source_unreadable"
                else:
                    ln, text, kind, check = find_statement(system, d, lines, src_root)
                found += kind not in ("not_found", "source_unreadable")
                suspect += check.startswith("suspect")
                stats["auto_check"][check] += 1
                amb = base_count.get(os.path.basename(d["target"]), 0) if d["type"] == "include" else ""
                if d["type"] == "include" and amb and amb > 1:
                    stats["ambiguous_basename_links"] += 1
                stats["match_kind"][kind] += 1
                stats["examples"] += 1
                out.update({p + "source": path, p + "line": ln, p + "statement": text,
                            p + "resolved_target": d["target"], p + "match": kind,
                            p + "auto_check": check, p + "same_basename_files": amb})
            if not picked:
                status = "no_support"
            elif found == len(picked):
                status = "all_located"
            elif found:
                status = "partly_located"
            else:
                status = "none_located"
            flag = granularity_flag(src, tgt)
            if system == "kubernetes" and tgt not in module_files:
                flag = (flag + ";" if flag else "") + "target_is_package_not_module"
            stats["edges_with_suspect_link"] += int(suspect > 0)
            stats["edges_all_suspect"] += int(bool(picked) and suspect == len(picked))
            out.update({"evidence_status": status, "n_suspect_links": suspect, "auto_flags": flag,
                        "label_annotator1": "", "label_annotator2": "", "label_final": "", "comment": ""})
            stats["status"][status] += 1
            for f_ in filter(None, flag.split(";")):
                stats["granularity"][f_] += 1
            sys_stat[status] += 1
            evidence_rows.append(out)
        stats["per_system"][system] = dict(sys_stat)

        # recall sheet: random source files that the AKG indexed
        n = RECALL_PER_SYSTEM.get(system, RECALL_PER_SYSTEM["default"])
        pool = sorted(p for p in file_to_mod if os.path.splitext(p)[1] in SOURCE_EXT)
        rng.shuffle(pool)
        chosen = []
        for p in pool:
            if os.path.isfile(os.path.join(src_root, p)):
                chosen.append(p)
                if len(chosen) == n:
                    break
        for path in chosen:
            resolved = sorted(set(out_links.get(path, [])))
            raw = sorted(set(raw_by_src.get(path, [])))
            internal_mods = sorted({target_module(system, t, file_to_mod) or "?" for t in resolved}
                                   - {file_to_mod[path]})
            recall_rows.append({
                "recall_id": len(recall_rows) + 1, "system": system, "file": path,
                "module": file_to_mod[path],
                "akg_raw_statements": len(raw), "akg_resolved_links": len(resolved),
                "akg_target_modules": " | ".join(internal_mods),
                "akg_resolved_targets": " | ".join(resolved),
                "akg_unresolved_raw": " | ".join(t for t in raw if not any(r.endswith(t.replace('.', '/')) or r.endswith(t) for r in resolved))[:2000],
                "missed_internal_deps": "", "spurious_links": "", "annotator": "", "comment": "",
            })

    evidence_rows.sort(key=lambda r: int(r["sample_id"]))
    for path, rows in ((OUT_EVIDENCE, evidence_rows), (OUT_RECALL, recall_rows)):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    stats["status"] = dict(stats["status"])
    stats["match_kind"] = dict(stats["match_kind"])
    stats["granularity"] = dict(stats["granularity"])
    stats["auto_check"] = dict(stats["auto_check"])
    stats["recall_files"] = len(recall_rows)
    stats["seed"] = SEED
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
