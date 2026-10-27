#!/usr/bin/env python3
"""
Evidence packs for LLM-assisted AKG edge annotation (pilot: 20 edges).

Step 1  edge-level audit category for all 200 edges of akg_validation_evidence.csv,
        from the link verdicts of akg_impact_audit.build_links (cached in
        akg_ai_edge_category.json so the remaining 180 edges reuse it).
Step 2  stratified sample (category x system, seed 42) of PILOT_N edges.
Step 3  per example link, repository facts computed on the source-tree disk
        (not file_index): statement context, whether the resolved target exists,
        and the candidate files/dirs the reference string can denote.

The packs contain raw facts only; audit verdicts and the auto_* / match columns
of the evidence CSV are deliberately left out.

Outputs (experiment/):
  akg_ai_pilot_evidence.jsonl  model input, one edge per line
  akg_ai_pilot_blind.csv       human blind-annotation sheet
  akg_ai_pilot_strata.csv      pilot_id -> sample_id, system, audit category (do not show)
"""

import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import akg_impact_audit as A

EXP = A.EXP
ROOT = A.ROOT
SEED = 42
PILOT_N = 20
MAX_CANDS = 15
EVIDENCE_CSV = os.path.join(EXP, "akg_validation_evidence.csv")
CAT_CACHE = os.path.join(EXP, "akg_ai_edge_category.json")
OUT_JSONL = os.path.join(EXP, "akg_ai_pilot_evidence.jsonl")
OUT_BLIND = os.path.join(EXP, "akg_ai_pilot_blind.csv")
OUT_STRATA = os.path.join(EXP, "akg_ai_pilot_strata.csv")
OUT_CLUES = os.path.join(EXP, "akg_ai_pilot_clues.csv")
OUT_GRAN = os.path.join(EXP, "akg_ai_pilot_granularity.csv")

CLEAR_ERROR = {"config_gap", "external_misresolved", "external_misresolved_py", "wrong_path",
               "wrong_simple_name", "wrong_module", "target_dir_missing"}
CATEGORIES = ("path_verified", "name_unique", "clear_error", "undeterminable")


def read_edges():
    with open(EVIDENCE_CSV, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- step 1
def edge_category(verdicts):
    if any(v in A.PATH_VERIFIED for v in verdicts):
        return "path_verified"
    if any(v.startswith("verified") for v in verdicts):
        return "name_unique"
    if verdicts and all(v in CLEAR_ERROR for v in verdicts):
        return "clear_error"
    return "undeterminable"


def compute_categories(rows):
    if os.path.isfile(CAT_CACHE):
        return A.load(CAT_CACHE)
    by_sys = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)
    out = {}
    for system, rs in by_sys.items():
        print(f"[audit] {system} ...", flush=True)
        recs, idx, _, _ = A.build_links(system)
        want = {(r["source_module"], r["target_module"]) for r in rs}
        fold = system == "kubernetes"
        if fold:
            want |= {(A._kg_norm(a), A._kg_norm(b)) for a, b in want}
        verd = defaultdict(Counter)
        links = defaultdict(list)      # (src_file, resolved) -> raw strings, for example lookup
        for rec in recs:
            a = idx.f2m.get(rec["src"])
            b = rec["resolved"] if fold else idx.f2m.get(rec["resolved"])
            if not a or not b or a == b:
                continue
            keys = [(a, b)] + ([(A._kg_norm(a), A._kg_norm(b))] if fold else [])
            for k in keys:
                if k in want:
                    verd[k][rec["verdict"]] += 1
            links[(rec["src"], rec["resolved"])].append(rec["raw"])
        for r in rs:
            k = (r["source_module"], r["target_module"])
            matched = "exact"
            if k not in verd and fold:
                k, matched = (A._kg_norm(k[0]), A._kg_norm(k[1])), "folded"
            v = verd.get(k, Counter())
            raws = {}
            for i in (1, 2, 3):
                s, t = r[f"ex{i}_source"], r[f"ex{i}_resolved_target"]
                if s:
                    raws[str(i)] = sorted(set(links.get((s, t), [])))
            out[r["sample_id"]] = {
                "system": system,
                "category": edge_category(set(v)) if v else "undeterminable",
                "edge_match": matched if v else "no_replayed_links",
                "verdicts": dict(v),
                "example_raw": raws,
            }
        del recs, idx, links
    with open(CAT_CACHE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    return out


# ---------------------------------------------------------------- step 2
def stratified_sample(rows, cats, n):
    rng = random.Random(SEED)
    cells = defaultdict(lambda: defaultdict(list))   # category -> system -> sample_ids
    for r in rows:
        c = cats[r["sample_id"]]
        cells[c["category"]][c["system"]].append(r["sample_id"])
    present = [c for c in CATEGORIES if cells[c]]
    quota = {c: n // len(present) for c in present}
    for c in present[: n - sum(quota.values())]:
        quota[c] += 1
    # a category smaller than its quota hands the rest to the others
    sizes = {c: sum(len(v) for v in cells[c].values()) for c in present}
    spare = sum(max(0, quota[c] - sizes[c]) for c in present)
    quota = {c: min(quota[c], sizes[c]) for c in present}
    while spare:
        for c in present:
            if spare and quota[c] < sizes[c]:
                quota[c] += 1
                spare -= 1
    picked = []
    for c in present:
        pools = {s: rng.sample(ids, len(ids)) for s, ids in sorted(cells[c].items())}
        # larger cells first (ties in random order) so linux / kubernetes, which hold
        # most of the 200 edges, are not crowded out by one-edge systems
        systems = sorted(rng.sample(sorted(pools), len(pools)), key=lambda s: -len(pools[s]))
        got = []
        while len(got) < quota[c]:        # round-robin over systems, random inside each
            for s in systems:
                if pools[s] and len(got) < quota[c]:
                    got.append(pools[s].pop())
        picked += got
    return picked


# ---------------------------------------------------------------- step 3
class Disk:
    """On-disk file list of a source tree."""

    def __init__(self, repo):
        self.repo = repo
        self.by_base = defaultdict(list)
        self.by_stem = defaultdict(list)
        for d, dirs, files in os.walk(repo):
            dirs[:] = [x for x in dirs if x != ".git"]
            rel = os.path.relpath(d, repo).replace("\\", "/")
            rel = "" if rel == "." else rel + "/"
            for fn in files:
                p = rel + fn
                self.by_base[fn].append(p)
                self.by_stem[os.path.splitext(fn)[0].upper()].append(p)

    def kind(self, rel):
        p = os.path.join(self.repo, rel)
        if os.path.isdir(p):
            return "dir"
        if os.path.isfile(p):
            return "file"
        return "missing"

    def ends_with(self, suffix):
        return sorted(p for p in self.by_base.get(suffix.split("/")[-1], [])
                      if p == suffix or p.endswith("/" + suffix))


INC_RE = re.compile(r'#\s*include\s*([<"])([^>"]+)[>"]')
GO_RE = re.compile(r'"([^"]+)"')
JAVA_RE = re.compile(r'^\s*import\s+(?:static\s+)?([\w.]+)')
PY_FROM = re.compile(r'^\s*from\s+(\.*[\w.]*)\s+import')
PY_IMP = re.compile(r'^\s*import\s+([\w.]+)')


def raw_from_statement(stmt, ext):
    for rx in (INC_RE,):
        m = rx.search(stmt)
        if m:
            return m.group(2)
    if ext == ".go":
        m = GO_RE.search(stmt)
        return m.group(1) if m else None
    if ext == ".java":
        m = JAVA_RE.match(stmt)
        return m.group(1) if m else None
    if ext == ".py":
        m = PY_FROM.match(stmt) or PY_IMP.match(stmt)
        return m.group(1) if m else None
    return None


def context(repo, src, line, k=2):
    try:
        with open(os.path.join(repo, src), encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    lo, hi = max(1, line - k), min(len(lines), line + k)
    return [f"{i}: {lines[i - 1]}" for i in range(lo, hi + 1)]


def go_module(repo):
    p = os.path.join(repo, "go.mod")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            for l in f:
                if l.startswith("module "):
                    return l.split()[1]
    return None


def candidates(system, disk, src, raw, ext):
    """(rule description, list of (path, kind)) for the reference string."""
    srcdir = os.path.dirname(src)
    if ext == ".go":
        mod = go_module(disk.repo) or ""
        probes = []
        if mod and (raw == mod or raw.startswith(mod + "/")):
            probes.append(raw[len(mod) + 1:])
        probes += ["staging/src/" + raw, "vendor/" + raw]
        hits = [(p, disk.kind(p)) for p in probes]
        return (f"Go import path; repo go.mod module = {mod!r}; probed main-module path, "
                "staging/src/<import>, vendor/<import>", hits)
    if ext == ".java":
        parts = raw.split(".")
        for k in range(len(parts), 1, -1):
            hits = disk.ends_with("/".join(parts[:k]) + ".java")
            if hits:
                return (f"Java FQN -> file path suffix '{'/'.join(parts[:k])}.java'"
                        + (f" (last {len(parts) - k} segment(s) treated as member/nested)" if k < len(parts) else ""),
                        [(p, "file") for p in hits])
        simple = disk.by_base.get(parts[-1] + ".java", [])
        return ("Java FQN -> no file with that path suffix; listed files with the same simple class name",
                [(p, "file") for p in sorted(simple)])
    if ext == ".py":
        if raw.startswith("."):
            lvl = len(raw) - len(raw.lstrip("."))
            pkg = src.split("/")[:-1]
            base = pkg[:len(pkg) - (lvl - 1)]
            rest = raw.lstrip(".").replace(".", "/")
            exp = "/".join(base + ([rest] if rest else []))
            probes = [exp + ".py", exp + "/__init__.py"]
            return ("Python relative import resolved against the source package",
                    [(p, disk.kind(p)) for p in probes])
        t = raw.replace(".", "/")
        hits = disk.ends_with(t + ".py") + disk.ends_with(t + "/__init__.py")
        return (f"Python module '{raw}' -> files ending with '{t}.py' or '{t}/__init__.py'",
                [(p, "file") for p in sorted(set(hits))])
    if system == "nastran" or ext.lower() in (".f", ".for", ".f90"):
        hits = disk.by_stem.get(raw.upper(), [])
        return (f"Fortran routine '{raw}' -> files whose name stem is '{raw.upper()}'",
                [(p, "file") for p in sorted(hits)])
    # C / C++ include
    out = []
    if raw.startswith("./") or raw.startswith("../") or "/../" in raw:
        exp = os.path.normpath(os.path.join(srcdir, raw)).replace("\\", "/")
        out.append((exp, disk.kind(exp)))
    same = (srcdir + "/" + raw).lstrip("/")
    if disk.kind(same) == "file" and same not in [p for p, _ in out]:
        out.append((same, "file"))
    if "/" in raw:
        rule = (f"C/C++ include with path -> source-dir-relative file, plus every file whose path "
                f"ends with '{raw}'")
        hits = disk.ends_with(raw.lstrip("./") if raw.startswith("./") else raw)
    else:
        rule = f"C/C++ include without path -> source-dir file, plus every file named '{raw}'"
        hits = sorted(disk.by_base.get(raw, []))
    out += [(p, "file") for p in hits if p not in [q for q, _ in out]]
    return rule, out


def build_pack(row, cat, idx, disk, lang_note):
    system = row["system"]
    exs = []
    for i in (1, 2, 3):
        src = row[f"ex{i}_source"]
        if not src:
            continue
        ext = os.path.splitext(src)[1]
        line = int(row[f"ex{i}_line"]) if row[f"ex{i}_line"].isdigit() else None
        stmt = row[f"ex{i}_statement"]
        resolved = row[f"ex{i}_resolved_target"]
        raws = [r for r in cat["example_raw"].get(str(i), []) if r != "(cross-repo)"]
        raw = next((r for r in raws if r in stmt), raws[0] if raws else None) or raw_from_statement(stmt, ext)
        rule, cands = candidates(system, disk, src, raw, ext) if raw else ("reference string not recoverable", [])
        tmod = resolved if system == "kubernetes" else idx.f2m.get(resolved)
        exs.append({
            "source_file": src,
            "source_module": idx.f2m.get(src),
            "line": line,
            "statement": stmt,
            "context": context(disk.repo, src, line) if line else [],
            "reference_string": raw,
            "resolved_target": resolved,
            "resolved_target_on_disk": disk.kind(resolved),
            "resolved_target_module": tmod,
            "candidate_rule": rule,
            "n_candidates": len(cands),
            "candidates": [{"path": p, "on_disk": k,
                            "module": (p if system == "kubernetes"
                                       else idx.f2m.get(p, "(not in AKG file index)"))}
                           for p, k in cands[:MAX_CANDS]],
        })
    return {
        "sample_id": row["sample_id"],
        "system": system,
        "edge": {"source_module": row["source_module"], "target_module": row["target_module"],
                 "weight": int(row["weight"]), "supporting_file_links": int(row["support_links"] or 0)},
        "module_notes": lang_note,
        "examples": exs,
    }


MODULE_NOTES = {
    "kubernetes": "Source modules are AKG modules from file_index; target modules are the "
                  "Go package directory the import was resolved to (benchmark tasks later "
                  "collapse both to two path levels, e.g. pkg/volume, staging/src/k8s.io/api).",
    "salome": "Paths are relative to the SALOME super-tree (kernel/, geom/, smesh/, ...).",
}


# ---------------------------------------------------------------- clue cards
def expansion(system, src, raw, ext):
    """Mechanical expansion of the reference string: which path forms the rule maps it
    to. Facts only, no selection among them."""
    srcdir = os.path.dirname(src)
    if ext == ".go":
        return [f"导入路径按 go.mod 的 module 名推导：<import 去掉 module 前缀>、staging/src/{raw}、vendor/{raw}"]
    if ext == ".java":
        return [f"import 全限定名：{raw}", f"全限定名换路径：{raw.replace('.', '/')}.java"]
    if ext == ".py":
        if raw.startswith("."):
            return [f"相对导入按源文件所在包解析，源文件目录为 {srcdir}"]
        t = raw.replace(".", "/")
        return [f"点路径换斜杠：{t}", f"对应文件：{t}.py 或 {t}/__init__.py"]
    if system == "nastran" or ext.lower() in (".f", ".for", ".f90"):
        return [f"子程序名对应文件名词干：{raw.upper()}"]
    out = []
    if raw.startswith("./") or raw.startswith("../") or "/../" in raw:
        out.append("相对 include 按源文件所在目录解析："
                   + os.path.normpath(os.path.join(srcdir, raw)).replace("\\", "/"))
    out.append(f"同目录候选：{(srcdir + '/' + raw).lstrip('/')}")
    if "/" in raw:
        out.append(f"include 搜索路径候选：<include 根>/{raw.lstrip('./')}")
    else:
        out.append(f"include 搜索路径候选：<include 根>/{raw}，以及仓库中所有名为 {raw} 的文件")
    return out


def write_clues(pack_path, out_path):
    with open(pack_path, encoding="utf-8") as f:
        packs = [json.loads(l) for l in f if l.strip()]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["pilot_id", "system", "ex", "判定规则", "引用串", "规则展开（非磁盘事实）",
                    "解析目标", "解析目标是否在磁盘", "磁盘候选路径", "磁盘候选类型",
                    "候选全限定名（末 N 段）", "候选所属模块"])
        for p in packs:
            for j, x in enumerate(p["examples"], 1):
                ext = os.path.splitext(x["source_file"])[1]
                raw = x["reference_string"] or ""
                exp = " ；".join(expansion(p["system"], x["source_file"], raw, ext)) if raw else ""
                nseg = len(raw.split("."))          # Java: the import's own segment count
                cs = x["candidates"] or [{}]
                for c in cs:
                    path = c.get("path", "")
                    fqn = ""
                    if ext == ".java" and path:
                        parts = path[:-5].split("/")
                        fqn = ".".join(parts[-min(nseg, len(parts)):])
                    w.writerow([p["pilot_id"], p["system"], j, x["candidate_rule"], raw, exp,
                                x["resolved_target"], x["resolved_target_on_disk"],
                                c.get("path", ""), c.get("on_disk", ""), fqn, c.get("module", "")])
    print(f"wrote {out_path}", flush=True)


def system_modules(system):
    """module -> (n files, n files sitting in a subdirectory below the module), from file_index."""
    project_dir = os.path.join(ROOT, A.SYSTEMS[system][0])
    files = A.load(os.path.join(project_dir, "file_index.json"))["files"]
    st = defaultdict(lambda: [0, 0])
    for f in files:
        m, p = f.get("module") or "", f.get("path") or ""
        st[m][0] += 1
        if p.startswith(m + "/") and "/" in p[len(m) + 1:]:
            st[m][1] += 1
    return st


def ancestor_of(a, b):
    """True if a's path segments are a proper prefix of b's (a is an ancestor module of b)."""
    sa, sb = a.split("/"), b.split("/")
    return len(sa) < len(sb) and sb[:len(sa)] == sa


def write_granularity(pack_path, out_path):
    with open(pack_path, encoding="utf-8") as f:
        packs = [json.loads(l) for l in f if l.strip()]
    mods, depth = {}, {}
    print("system module scale (段数分布: file_index 里模块路径的段数 -> 模块个数):")
    for system in sorted({p["system"] for p in packs}):
        mods[system] = system_modules(system)
        depth[system] = sorted(Counter(len(m.split("/")) for m in mods[system]).items())
        print(f"  {system}: 模块 {len(mods[system])} 个, {depth[system]}", flush=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["pilot_id", "sample_id", "system", "source_module", "target_module",
                    "源模块段数", "目标模块段数", "嵌套关系", "源模块文件数", "源模块子目录文件数",
                    "目标模块文件数", "目标模块子目录文件数", "备注"])
        for p in packs:
            st = mods[p["system"]]
            s, t = p["edge"]["source_module"], p["edge"]["target_module"]
            ss, tt = st.get(s, [0, 0]), st.get(t, [0, 0])
            rel = ("source 是 target 的上级" if ancestor_of(s, t) else
                   "target 是 source 的上级" if ancestor_of(t, s) else "无嵌套")
            note = "; ".join(f"{n} 不在 file_index 里" for n, v in (("source_module", ss), ("target_module", tt))
                             if v[0] == 0)
            w.writerow([p["pilot_id"], p["sample_id"], p["system"], s, t,
                        len(s.split("/")), len(t.split("/")), rel,
                        ss[0], ss[1], tt[0], tt[1], note])
    print(f"wrote {out_path}", flush=True)


def main():
    if "--clues" in sys.argv:
        write_clues(OUT_JSONL, OUT_CLUES)
        return 0
    if "--granularity" in sys.argv:
        write_granularity(OUT_JSONL, OUT_GRAN)
        return 0
    rows = read_edges()
    cats = compute_categories(rows)
    print("category x system:", Counter((v["category"], v["system"]) for v in cats.values()))
    pick = stratified_sample(rows, cats, PILOT_N)
    byid = {r["sample_id"]: r for r in rows}
    rng = random.Random(SEED + 1)
    order = rng.sample(pick, len(pick))          # presentation order hides the strata

    packs, ctx = [], {}
    for sid in sorted(pick, key=lambda s: byid[s]["system"]):
        system = byid[sid]["system"]
        if system not in ctx:
            print(f"[disk] {system} ...", flush=True)
            data_dir, repo_rel, _ = A.SYSTEMS[system]
            ctx = {system: (A.Index(os.path.join(ROOT, data_dir), repo_rel),
                            Disk(os.path.join(ROOT, repo_rel)))}
        idx, disk = ctx[system]
        packs.append(build_pack(byid[sid], cats[sid], idx, disk, MODULE_NOTES.get(system, "")))
    pmap = {p["sample_id"]: p for p in packs}

    with open(OUT_JSONL, "w", encoding="utf-8") as fj, \
         open(OUT_BLIND, "w", encoding="utf-8-sig", newline="") as fb, \
         open(OUT_STRATA, "w", encoding="utf-8-sig", newline="") as fs:
        wb = csv.writer(fb)
        wb.writerow(["pilot_id", "sample_id", "system", "source_module", "target_module", "weight",
                     "ex", "source_file", "line", "statement", "reference_string", "resolved_target",
                     "resolved_on_disk", "resolved_target_module", "n_candidates", "candidates_top15",
                     "label_blind", "confidence_1to3", "comment"])
        ws = csv.writer(fs)
        ws.writerow(["pilot_id", "sample_id", "system", "audit_category", "edge_match", "link_verdicts"])
        for k, sid in enumerate(order, 1):
            pid = f"P{k:02d}"
            p = dict(pilot_id=pid, **pmap[sid])
            fj.write(json.dumps(p, ensure_ascii=False) + "\n")
            e = p["edge"]
            for j, x in enumerate(p["examples"], 1):
                wb.writerow([pid, sid, p["system"], e["source_module"], e["target_module"], e["weight"],
                             j, x["source_file"], x["line"], x["statement"], x["reference_string"],
                             x["resolved_target"], x["resolved_target_on_disk"], x["resolved_target_module"],
                             x["n_candidates"],
                             " ; ".join(f'{c["path"]} [{c["on_disk"]}, module={c["module"]}]'
                                        for c in x["candidates"]),
                             *(["", "", ""] if j == 1 else ["^", "^", "^"])])
            c = cats[sid]
            ws.writerow([pid, sid, c["system"], c["category"], c["edge_match"],
                         json.dumps(c["verdicts"], ensure_ascii=False)])
    print("pilot strata:", Counter(cats[s]["category"] for s in pick),
          Counter(cats[s]["system"] for s in pick))


if __name__ == "__main__":
    sys.exit(main())
