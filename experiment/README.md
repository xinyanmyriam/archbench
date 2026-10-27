# Architecture Understanding Benchmark - Experiment

## Experiment Design

### Tasks
1. **Component Identification (CI)**: Given a file path, identify which module it belongs to (4-choice)
2. **Dependency Prediction (DP)**: Given two modules, predict if a dependency exists (binary)
3. **Architecture Reconstruction (AR)**: Given a source module, predict its dependency target (open-ended)

### Models
- GPT-4o
- Claude 3.5 Sonnet
- DeepSeek-V3
- Llama 3.1 70B
- Qwen2.5 72B

### Baselines
- Random
- Frequency-based
- Path Heuristic (for CI)
- Graph Neighbor (for DP/AR)

### Metrics
- CI: Accuracy
- DP: Accuracy, Precision, Recall, F1
- AR: Precision@K, Recall@K
