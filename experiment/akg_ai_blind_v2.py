#!/usr/bin/env python3
"""
Blind annotation sheet for the rebuilt (akg_v2) AKG validation sample.

Same column layout as akg_ai_pilot_blind.csv, so the reviewer can annotate the
200 v2 edges exactly as in the pilot: one block of rows per edge, the first row
of a block takes label/confidence/comment, the following rows carry "^".

Facts are read from the source tree on disk (statement, line, whether the
resolved target exists, candidate paths). No audit verdict and no model output
is written to this file.

Output: experiment/akg_ai_blind_v2.csv
"""

import csv
import json
import os
import random
import re
import sys
from collections import defaultdict

import akg_impact_audit as A
import akg_ai_evidence as E

EXP = A.EXP
ROOT = A.ROOT
AKG2 = os.path.join(ROOT, "systems")
SAMPLE = os.path.join(EXP, "akg_validation_sample_v2.csv")
OUT = os.path.join(EXP, "akg_ai_blind_v2.csv")
MAX_CANDS = 15
SEED = 42

INC_RE = re.compile(r'#\s*include\s*([<"])([^>"]+)[>"]')
GO_RE = re.compile(r'"([^"]+)"')
JAVA_RE = re.compile(r'^\s*import\s+(?:static\s+)?([\w.]+)\s*;')
PY_FROM = re.compile(r'^\s*from\s+(\.*[\w.]*)\s+import')
PY_IMP = re.compile(r'^\s*import\s+([\w.]+)')
CALL_RE = re.compile(r'\bCALL\s+([A-Za-z_]\w*)', re.I)

LINK_RE = re.compile(r'^(?P<src>.+?) -> (?P<tgt>.+?) \((?P<typ>[^)]*)\)$')


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def dataset_dir(system):
    base = os.path.join(ROOT, system)
    if system == "salome":
        base = os.path.join(base, "unified")
    return base


def k8s_norm(m):
    parts = m.split("/")
    if len(parts) >= 4 and parts[:3] == ["staging", "src", "k8s.io"]:
        return "/".join(parts[:4])
    return "/".join(parts[:2])


class ModMap:
    """file / import path -> module, following the akg_v2 mapping rules."""

    def __init__(self, system):
        self.system = system
        self.f2m = {f["path"]: f["module"]
                    for f in load(os.path.join(dataset_dir(system), "file_index.json"))["files"]}
        self.modules = set(load(os.path.join(dataset_dir(system), "modules_index.json"))["modules"])

    def __call__(self, path):
        if self.system == "kubernetes":
            return k8s_norm(path)
        if path in self.f2m:
            return self.f2m[path]
        if self.system != "linux":
            return None
        parts = path.split("/")
        if parts[0] == "include":
            return "include/" + parts[1] if len(parts) >= 3 else "include"
        if "include" in parts[:-1]:
            m = "/".join(parts[:2])
            if m in self.modules:
                return m
        return None


class Lines:
    """Cached file contents for statement lookup."""

    def __init__(self, repo):
        self.repo = repo
        self.cache = {}

    def get(self, path):
        if path not in self.cache:
            try:
                with open(os.path.join(self.repo, path), encoding="utf-8", errors="replace") as f:
                    self.cache[path] = f.read().splitlines()
            except OSError:
                self.cache[path] = []
        return self.cache[path]


def dotted(target):
    t = target
    if t.endswith("/__init__.py"):
        t = t[:-12]
    elif t.endswith(".py"):
        t = t[:-3]
    return t.replace("/", ".")


def find_statement(system, lines, src, target, typ):
    """(line_no, statement, reference_string) of the statement that produced the link."""
    body = lines.get(src)
    if not body:
        return None, "", ""
    ext = os.path.splitext(src)[1].lower()
    base = os.path.basename(target)

    if typ == "include" or ext in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx", ".hh"):
        best = None
        for i, l in enumerate(body, 1):
            m = INC_RE.search(l)
            if not m:
                continue
            raw = m.group(2)
            if target == raw:
                score = 3
            elif target.endswith("/" + raw):
                score = 2
            elif os.path.basename(raw) == base:
                score = 1
            else:
                continue
            if best is None or score > best[0]:
                best = (score, i, l.strip(), raw)
                if score == 3:
                    break
        return (best[1], best[2], best[3]) if best else (None, "", "")

    if typ == "import" and (ext == ".go" or system == "kubernetes"):
        best = None
        for i, l in enumerate(body, 1):
            s = l.strip()
            if s.startswith("//"):
                continue
            if not (s.startswith('"') or "import" in s):
                continue
            for raw in GO_RE.findall(l):
                if "/" not in raw:
                    continue
                if target.endswith(raw) or target == raw:
                    score = 3
                elif raw.endswith(target) or raw.endswith("/" + target):
                    score = 2          # main module import, e.g. k8s.io/kubernetes/pkg/fieldpath
                else:
                    continue
                if best is None or score > best[0]:
                    best = (score, i, l.strip(), raw)
        return (best[1], best[2], best[3]) if best else (None, "", "")

    if typ == "import" and ext == ".java":
        pkg = os.path.dirname(target).replace("/", ".")
        stem = os.path.splitext(base)[0]
        for i, l in enumerate(body, 1):
            m = JAVA_RE.match(l)
            if not m:
                continue
            raw = m.group(1)
            if raw.split(".")[-1] != stem:
                continue
            if dotted(target).endswith(raw):
                return i, l.strip(), raw
            if pkg.endswith(".".join(raw.split(".")[:-1])):
                return i, l.strip(), raw
        return None, "", ""

    if typ == "import" and ext == ".py":
        for i, l in enumerate(body, 1):
            m = PY_FROM.match(l) or PY_IMP.match(l)
            if not m:
                continue
            raw = m.group(1)
            if not raw or raw == ".":
                continue
            if dotted(target).endswith(dotted(raw).lstrip(".")):
                return i, l.strip(), raw
        return None, "", ""

    if typ == "call":
        stem = os.path.splitext(base)[0]
        for i, l in enumerate(body, 1):
            m = CALL_RE.search(l)
            if m and m.group(1).lower() == stem.lower():
                return i, l.strip(), m.group(1)
        return None, "", ""

    # fallback: any quoted reference that resolves to the target
    for i, l in enumerate(body, 1):
        for raw in re.findall(r'["<]([^">]+)[">]', l):
            if target.endswith("/" + raw) or target == raw:
                return i, l.strip(), raw
    return None, "", ""


def main():
    with open(SAMPLE, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    by_sys = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)

    out_rows = []
    stats = defaultdict(lambda: defaultdict(int))

    for system, srows in by_sys.items():
        data_dir, repo_rel, _ = A.SYSTEMS[system]
        repo = os.path.join(ROOT, repo_rel)
        print(f"[{system}] {len(srows)} edges, reading disk ...", flush=True)
        disk = E.Disk(repo)
        lines = Lines(repo)
        mm = ModMap(system)

        for r in srows:
            exs = []
            for s in (r["file_level_examples"] or "").split(" | "):
                m = LINK_RE.match(s)
                if not m:
                    continue
                src, tgt, typ = m.group("src"), m.group("tgt"), m.group("typ")
                ln, stmt, raw = find_statement(system, lines, src, tgt, typ)
                stats[system]["links"] += 1
                if ln is None:
                    stats[system]["no_statement"] += 1
                ext = os.path.splitext(src)[1]
                rule, cands = (E.candidates(system, disk, src, raw, ext) if raw
                               else ("reference string not recoverable", []))
                exs.append({
                    "source_file": src,
                    "line": ln,
                    "statement": stmt,
                    "reference_string": raw,
                    "resolved_target": tgt,
                    "resolved_on_disk": disk.kind(tgt),
                    "resolved_target_module": mm(tgt),
                    "n_candidates": len(cands),
                    "candidates": [{"path": p, "on_disk": k, "module": mm(p) or "(not in AKG file index)"}
                                   for p, k in cands[:MAX_CANDS]],
                })
                if len(exs) >= 3:
                    break

            out_rows.append((system, r, exs))

        del disk, lines

    rng = random.Random(SEED + 1)
    rng.shuffle(out_rows)

    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pilot_id", "sample_id", "system", "source_module", "target_module", "weight",
                    "ex", "source_file", "line", "statement", "reference_string", "resolved_target",
                    "resolved_on_disk", "resolved_target_module", "n_candidates", "candidates_top15",
                    "label_blind", "confidence_1to3", "comment"])
        for k, (system, r, exs) in enumerate(out_rows, 1):
            pid = f"V{k:03d}"
            for j, x in enumerate(exs, 1):
                w.writerow([pid, r["sample_id"], system, r["source_module"], r["target_module"],
                            r["weight"], j, x["source_file"], x["line"] or "", x["statement"],
                            x["reference_string"], x["resolved_target"], x["resolved_on_disk"],
                            x["resolved_target_module"] or "", x["n_candidates"],
                            " ; ".join(f'{c["path"]} [{c["on_disk"]}, module={c["module"]}]'
                                       for c in x["candidates"]),
                            *(["", "", ""] if j == 1 else ["^", "^", "^"])])
            if not exs:
                w.writerow([pid, r["sample_id"], system, r["source_module"], r["target_module"],
                            r["weight"], 0, "", "", "", "", "", "", "", 0, "", "", "", ""])

    print(f"\nwrote {OUT}: {len(out_rows)} edges")
    for system, st in stats.items():
        print(f"  {system:14s} links={st['links']:4d} no_statement={st['no_statement']}")
    blank = [r for r in out_rows if not r[2]]
    print(f"edges without any example: {len(blank)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
