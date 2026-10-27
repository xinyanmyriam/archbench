#!/usr/bin/env python3
"""
Freeze the originally released dataset (pre-AKG-fix) as dataset_v1/.

Copies every data file (*.json / *.jsonl) of the 12 systems, keeping the
relative path, plus experiment/samples/*.json. Writes SHA256 checksums to
dataset_v1/CHECKSUMS.sha256 and marks the copies read-only. Source code
repositories under each system dir are not copied.

Re-running with --verify checks the snapshot against both the checksums and
the live originals (to detect later accidental edits).
"""
import glob
import hashlib
import os
import shutil
import stat
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "dataset_v1")
SYSTEMS = ["linux", "hadoop", "spring", "paraview", "pytorch", "tensorflow",
           "codeaster", "openmodelica", "kubernetes", "openfoam", "salome", "nastran"]


def files():
    out = set()
    for s in SYSTEMS:
        for p in ("*.json", "*.jsonl", "unified/*.json*", "reposanalysis/*/*.json*"):
            out.update(glob.glob(os.path.join(ROOT, s, p)))
    out.update(glob.glob(os.path.join(ROOT, "experiment", "samples", "*.json")))
    return sorted(os.path.relpath(f, ROOT) for f in out)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def snapshot():
    if os.path.exists(DST):
        sys.exit(f"{DST} already exists; refusing to overwrite. Use --verify.")
    lines = []
    for rel in files():
        src, dst = os.path.join(ROOT, rel), os.path.join(DST, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        h = sha(dst)
        assert h == sha(src), rel
        lines.append(f"{h}  {rel.replace(os.sep, '/')}")
        os.chmod(dst, stat.S_IREAD)
    with open(os.path.join(DST, "CHECKSUMS.sha256"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(DST, "VERSION.txt"), "w", encoding="utf-8") as f:
        f.write("SoftArchBench dataset v1: as originally submitted (before the AKG\n"
                "dependency-label fix). Frozen 2026-10-20. Corrected labels: akg_v2/.\n")
    print(f"snapshot: {len(lines)} files -> {DST}")


def verify():
    bad = 0
    with open(os.path.join(DST, "CHECKSUMS.sha256"), encoding="utf-8") as f:
        rows = [l.split("  ", 1) for l in f.read().splitlines() if l]
    for h, rel in rows:
        for base, tag in ((DST, "snapshot"), (ROOT, "original")):
            p = os.path.join(base, rel)
            if not os.path.exists(p) or sha(p) != h:
                bad += 1
                print(f"MISMATCH [{tag}] {rel}")
    print(f"verified {len(rows)} files, mismatches: {bad}")
    return bad


if __name__ == "__main__":
    sys.exit(1 if verify() else 0) if "--verify" in sys.argv else snapshot()
