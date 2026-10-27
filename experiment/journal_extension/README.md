# Journal Extension: CCF-A (TSE / TOSEM)

## 在 ICSE 基础上扩展 30%+ 新内容

### 新增实验模块

1. **Multi-hop Reasoning Tasks** (`multihop_tasks.py`)
   - Transitive dependency prediction (A→B→C)
   - Change impact analysis
   - Architectural smell detection (cyclic deps, god modules)
   - Architectural layer inference

2. **Level 2/3 Evaluation** (`run_level2_3.py`)
   - Level 2: 提供部分依赖图作为上下文
   - Level 3: 提供架构文档/设计决策描述

3. **Error Analysis** (`error_analysis.py`)
   - 错误分类 taxonomy
   - 按错误类型统计
   - 与系统特征的相关性分析

4. **Training Data Correlation** (`familiarity_analysis.py`)
   - GitHub stars / forks / issues 数量
   - StackOverflow 讨论量
   - 与模型表现的相关性

5. **Formalization** (`architecture_knowledge_level.py`)
   - Architecture Knowledge Level (AKL) 形式化定义
   - 量化指标计算

### 使用方式

```bash
# 生成 multi-hop 任务数据
python generate_multihop_tasks.py --project all

# 运行 Level 2/3 实验
python run_level2_3.py --model deepseek-v4-flash --level 2 --project all

# 错误分析
python error_analysis.py --results-dir ../results/

# 熟悉度相关性分析
python familiarity_analysis.py
```
