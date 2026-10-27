# SoftArchBench: A Multi-Domain Benchmark for Evaluating LLM Understanding of Industrial Software Architecture

SoftArchBench (formerly ArchBench) is a benchmark for evaluating how well Large Language Models understand
the software architecture of real industrial systems. It covers **12 industrial open-source systems** across
5 application domains and 8 programming languages, with **4 architecture-understanding tasks** evaluated at
**3 context levels** (zero-context, code-context, dependency-graph context).

The benchmark data totals **1,404 modules**, **7,988 module-level dependencies** (validated on a stratified
manual sample of 200 edges; see the paper, Section 3.10), and **603,457 file-level dependency links**.

## Repository layout

| Path | Content |
|------|---------|
| `config.json` | Benchmark metadata and per-system module counts |
| `systems/<name>/` | Benchmark data for one of the 12 systems |
| `pipeline/` | Architecture Knowledge Graph extraction code (rebuild) |
| `experiment/` | Evaluation harness, prompts, baselines, manifest, results, validation artifacts |
| `requirements.txt` | Python dependencies (`requests`, `numpy`) |

### `systems/<name>/`

| File | Content |
|------|---------|
| `modules_index.json` | Module list |
| `module_dependency.json` | Module-level dependency graph (rebuilt, manually validated) |
| `file_dependency.json` | File-level dependency links with provenance |
| `file_index.json` | File-to-module index |
| `component_identification.jsonl` | CI task pool (full path → module) |
| `component_identification_hard.jsonl` | CI-Hard task pool (filename only) |
| `component_identification_content.jsonl` | CI-Content task pool |
| `component_identification_obfuscated.jsonl` | CI-Obfuscated task pool |
| `dependency_prediction.jsonl` | DP task pool (module pair → dependency label) |
| `architecture_reconstruction.jsonl` | AR task pool (source module → outgoing targets) |

The files `architecture_kg.jsonl`, `architectural_smells.jsonl`, `change_impact.jsonl`,
`layer_inference.jsonl`, and `transitive_dependency.jsonl` are retained for archival purposes only;
they are **not** part of the current benchmark tasks.

### `experiment/`

- `prompts.py`, `prompts_level1.py`, `prompts_level2.py` — prompt construction for the three context levels.
- `run_experiment.py` — Level-0 (zero-context) evaluation; `run_level1_icse.py` — Level-1 (code-context);
  `run_level2.py` — Level-2 (dependency-graph context).
- `run_baselines_fixed.py` — the four non-LLM baselines (path heuristic, frequency heuristic,
  seeded random predictor, two-hop graph traversal).
- `config.py` — model endpoints and API keys. **API keys were removed before publication;
  fill in your own keys before running.**
- `samples/manifest_v2.json` — the fixed sampling manifest (seed 42) answered by every model and baseline;
  `samples/manifest_seed42.json` and `samples/manifest_v2_fullout.json` are provenance snapshots.
- `results_v2/` — per-model result files and aggregate summaries.
- Validation artifacts for the Architecture Knowledge Graph (paper, Section 3.10):
  `akg_validation_sample_v2.csv` (the 200-edge stratified sample with file-level provenance),
  `akg_ai_blind_v2_facts.txt` (fact-only evidence packs), `akg_ai_blind_v2.csv` and
  `akg_ai_blind_v2_deepseek_model_labels.csv` (independent model label sheets),
  `akg_annotation_guide.md` (annotation protocol), and `akg_ai_pilot_*` (the pre-rebuild pilot
  that recovered the error classes the rebuild was designed to remove).

## Reproducing the results

```bash
pip install -r requirements.txt
# 1. put your own API keys into experiment/config.py
# 2. run from the experiment/ directory:
cd experiment
SAB_MANIFEST=v2 python run_experiment.py --task all --model all --project all   # Level 0
SAB_MANIFEST=v2 python run_level2.py --model <model> --project all --task all    # Level 2 (graph context)
python run_baselines_fixed.py                                                    # non-LLM baselines
```

Level-1 (code-context) evaluation additionally requires local checkouts of the 12 subject systems;
set the checkout locations in the `REPO_PATHS` dictionary of `run_level1_icse.py` before running it.

The extraction pipeline in `pipeline/` documents how the dependency graphs were rebuilt from source
text (C/C++ `#include`, Java/Go/Python `import`, Fortran `CALL`) with conservative resolution rules;
it expects the original working-tree layout and is provided for auditability rather than as a
self-contained re-run script.

## License and citation

Citation information will be added upon publication.
