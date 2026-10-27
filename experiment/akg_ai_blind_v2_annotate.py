#!/usr/bin/env python3
"""
Build the v2 evidence pack from akg_ai_blind_v2_facts.json and run annotation models on it.

The human blind pass is already finished (akg_ai_blind_v2.csv, 200/200 Correct), so this
script only produces independent model labels for the agreement analysis in R2-5 / R3-2.

The pack keeps repository facts only: the edge, the source statements, the resolved
target, whether it exists on disk, and the full candidate list. Derived audit fields
(relation, ref_suffix_of_resolved, resolved_file_module) are dropped so the models do
not see any hint of the tool's or the human's verdict.

Outputs (experiment/):
  akg_ai_blind_v2_evidence.jsonl      one line per edge (facts only)
  akg_ai_blind_v2_model_labels.csv    pilot_id, model, label, confidence, reason, parse_ok
  akg_ai_blind_v2_model_raw.jsonl     raw response per call, for auditing
"""

import argparse
import csv
import json
import os
import sys

import akg_ai_annotate as A
import config
import llm_client

EXP = os.path.dirname(os.path.abspath(__file__))
FACTS = os.path.join(EXP, "akg_ai_blind_v2_facts.json")
PACK = os.path.join(EXP, "akg_ai_blind_v2_evidence.jsonl")


def parse_candidate(s):
    """'path [file, module=m]' -> structured dict."""
    path, _, rest = s.partition(" [")
    rest = rest.rstrip("]")
    on_disk, _, mod = rest.partition(", module=")
    return {"path": path, "on_disk": on_disk, "module": mod or None}


def to_pack(rec):
    return {
        "pilot_id": rec["pilot_id"],
        "system": rec["system"],
        "edge": {
            "source_module": rec["source_module"],
            "target_module": rec["target_module"],
            "weight": rec["weight"],
        },
        "target_module_in_index": rec["target_in_modules_index"],
        "examples": [
            {
                "source_file": x["file"],
                "line": x["line"],
                "statement": x["statement"],
                "reference_string": x.get("ref", ""),
                "resolved_target": x["resolved"],
                "resolved_target_on_disk": x["resolved_on_disk"],
                "resolved_target_module": x["resolved_module"],
            }
            for x in rec["examples"]
        ],
        "n_candidates": rec["n_candidates"],
        "candidates": [parse_candidate(c) for c in rec["candidates"]],
    }


def build_pack():
    facts = json.load(open(FACTS, encoding="utf-8"))
    packs = [to_pack(r) for r in facts]
    with open(PACK, "w", encoding="utf-8") as f:
        for p in packs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"pack: {PACK} ({len(packs)} edges)", flush=True)
    return packs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3-235b-dashscope",
                    help="comma-separated keys in config.MODELS")
    ap.add_argument("--out-prefix", default="akg_ai_blind_v2")
    args = ap.parse_args()

    packs = build_pack()
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    cfg = {m: config.MODELS[m] for m in names}
    out_labels = os.path.join(EXP, f"{args.out_prefix}_model_labels.csv")
    out_raw = os.path.join(EXP, f"{args.out_prefix}_model_raw.jsonl")

    rows, raw = [], []
    for name in names:
        ok = fail = 0
        for rec in packs:
            pid = rec["pilot_id"]
            try:
                res = llm_client.call_llm_with_usage(cfg[name], A.build_prompt(rec), max_retries=3)
                text = res.get("text", "")
            except Exception as e:                     # API error: keep, never parse as label
                text, res = "", {"error": f"{type(e).__name__}: {e}"}
            lab, conf, reason, parsed = A.parse_label(text)
            ok, fail = (ok + 1, fail) if parsed else (ok, fail + 1)
            rows.append([pid, name, lab, conf or "", reason, int(parsed)])
            raw.append({"pilot_id": pid, "model": name, "raw": text,
                        "usage": res.get("usage"), "error": res.get("error"),
                        "parsed": parsed})
            print(f"  {name} {pid} {'ok' if parsed else 'FAIL'} {lab}", flush=True)
        print(f"{name}: ok={ok} parse_failed={fail}", flush=True)

    with open(out_labels, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pilot_id", "model", "label", "confidence", "reason", "parse_ok"])
        w.writerows(rows)
    with open(out_raw, "w", encoding="utf-8") as f:
        for r in raw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out_labels} ({len(rows)} rows), {out_raw}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
