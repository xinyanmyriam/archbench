# ArchBench: A Multi-Domain Benchmark for Evaluating LLM Understanding of Industrial Software Architecture

## Overview

ArchBench is a benchmark for evaluating Large Language Model (LLM) understanding of industrial software architecture. It covers **12 industrial open-source systems** across 5 application domains and 8 programming languages, with **4 architecture understanding tasks**.

## Subject Systems

| System | Domain | Modules | Language | Source |
|--------|--------|---------|----------|--------|
| OpenFOAM | CFD Simulation | 69 | C++ | [OpenFOAM](https://github.com/OpenFOAM/OpenFOAM-dev) |
| NASTRAN-95 | Structural FEA | 6 | Fortran | [NASTRAN-95](https://github.com/nasa/NASTRAN-95) |
| SALOME | CAE Platform | 197 | C++ | [SALOME](https://git.salome-platform.org/) |
| ParaView | Visualization | 143 | C++ | [ParaView](https://github.com/Kitware/ParaView) |
| Kubernetes | Container Orchestration | 90 | Go | [Kubernetes](https://github.com/kubernetes/kubernetes) |
| Linux Kernel | OS Kernel | 411 | C | [Linux](https://github.com/torvalds/linux) |
| Hadoop | Distributed Computing | 31 | Java | [Hadoop](https://github.com/apache/hadoop) |
| Spring Framework | Enterprise Java | 22 | Java | [Spring](https://github.com/spring-projects/spring-framework) |
| Code_Aster | FEA | 131 | Mixed | [Code_Aster](https://gitlab.com/codeaster/src) |
| OpenModelica | Modeling & Simulation | 21 | Mixed | [OpenModelica](https://github.com/OpenModelica/OpenModelica) |
| TensorFlow | ML Framework | 133 | Mixed | [TensorFlow](https://github.com/tensorflow/tensorflow) |
| PyTorch | ML Framework | 119 | Mixed | [PyTorch](https://github.com/pytorch/pytorch) |

## Tasks

1. **Component Identification (CI)**: Given a file path, determine which module it belongs to (4-choice MCQ)
2. **Component Identification - Hard (CI-Hard)**: Same as CI but with only the filename (no directory path)
3. **Dependency Prediction (DP)**: Given two modules, determine if a dependency exists between them
4. **Architecture Reconstruction (AR)**: Given a module, predict its dependencies

## Repository Structure

```
archbench/
├── README.md
├── systems/
│   ├── openfoam/
│   │   ├── modules_index.json          # Module list
│   │   ├── module_dependency.json      # Ground-truth dependencies
│   │   ├── file_index.json             # File-to-module mapping
│   │   ├── file_dependency.json        # File-level dependencies
│   │   ├── component_identification.jsonl      # CI task data
│   │   ├── component_identification_hard.jsonl # CI-Hard task data
│   │   ├── dependency_prediction.jsonl         # DP task data
│   │   └── architecture_reconstruction.jsonl   # AR task data
│   ├── kubernetes/
│   │   └── ...
│   └── ... (12 systems total)
└── config.json                         # Benchmark metadata
```

## Data Format

### modules_index.json
```json
{
  "modules": ["src/mesh", "src/parallel", "applications/solvers", ...]
}
```

### module_dependency.json
```json
{
  "dependencies": [
    {"source_module": "src/mesh", "target_module": "src/OpenFOAM", "weight": 15},
    ...
  ]
}
```

### Task JSONL (e.g., component_identification_hard.jsonl)
Each line is a JSON object:
```json
{"file": "solver.cpp", "file_full": "src/mesh/solver.cpp", "label": "src/mesh", "options": ["src/mesh", "src/parallel", "applications/solvers", "src/OpenFOAM"]}
```

## Usage

```python
import json

# Load system architecture
with open("systems/kubernetes/modules_index.json") as f:
    modules = json.load(f)["modules"]

with open("systems/kubernetes/module_dependency.json") as f:
    deps = json.load(f)["dependencies"]

# Load task data
tasks = []
with open("systems/kubernetes/component_identification_hard.jsonl") as f:
    for line in f:
        tasks.append(json.loads(line))
```

## Citation

```bibtex
@misc{archbench2026,
  author = {[Anonymous]},
  title = {ArchBench: A Multi-Domain Benchmark for Evaluating LLM Understanding of Industrial Software Architecture},
  year = {2026}
}
```

## License

The benchmark data (ground-truth architectures, task instances) is released under CC-BY-4.0. Source code of the subject systems retains their original licenses.
