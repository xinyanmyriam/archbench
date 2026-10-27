# AKG 标签验证试跑：人工金标 vs 双模型对比与统计

- 试跑边数：20；进入金标精确率/分层统计：20；进入「模型 vs 人工」一致性统计：19（一致性统计排除 P04）
- 模型：gemini-4-argon, claude-sonnet-5.5；解析失败：0
- 缺失置信度：无
- 排除理由：方案A：该边的结论已由助手在对话中逐步演示给出，不构成独立人工标注，计算模型-人工一致率时排除；在论文方法说明中如实记录 1/20 为示例。

## 1. 人工金标分布与边精确率

| 标签 | 数 |
|---|---|
| Correct | 14 |
| Wrong-resolution | 3 |
| Spurious | 2 |
| Granularity | 1 |
| Unclear | 0 |

边精确率 = Correct / (Correct + Wrong-resolution + Spurious) = 14/19 = 73.7% (Wilson 95% CI [51.2%, 88.2%])
Granularity / Unclear 单列：1 / 0（不进分母）
对照（剔除一致性排除样本 P04 后）：14/18 = 77.8% (Wilson 95% CI [54.8%, 91.0%])

## 2. 按系统

| 系统 | n | Correct | Wrong-res | Spurious | Granularity | 边精确率 |
|---|---|---|---|---|---|---|
| codeaster | 2 | 2 | 0 | 0 | 0 | 100.0% |
| hadoop | 1 | 0 | 0 | 1 | 0 | 0.0% |
| kubernetes | 2 | 1 | 1 | 0 | 0 | 50.0% |
| linux | 1 | 0 | 1 | 0 | 0 | 0.0% |
| openfoam | 1 | 0 | 0 | 0 | 1 | n/a |
| openmodelica | 1 | 1 | 0 | 0 | 0 | 100.0% |
| paraview | 1 | 1 | 0 | 0 | 0 | 100.0% |
| pytorch | 2 | 1 | 0 | 1 | 0 | 50.0% |
| salome | 6 | 6 | 0 | 0 | 0 | 100.0% |
| spring | 1 | 1 | 0 | 0 | 0 | 100.0% |
| tensorflow | 2 | 1 | 1 | 0 | 0 | 50.0% |

## 3. 按审计类别（后分层加权）

| 审计类别 | 试跑 n | 池占比 | Correct | 边精确率 |
|---|---|---|---|---|
| clear_error | 5 | 61.5% | 0/5 | 0.0% |
| name_unique | 5 | 9.5% | 4/4 | 100.0% |
| path_verified | 5 | 26.5% | 5/5 | 100.0% |
| undeterminable | 5 | 2.5% | 5/5 | 100.0% |

后分层加权精确率（按审计类别、权重取自 200 边候选池）= **38.5%** (bootstrap 95% CI [38.5%, 38.5%])
> 注：本次试跑各审计类别层内边精确率在重采样下无变异（clear_error 层全为非 Correct，其余三层全为 Correct，Granularity 不计入分母），bootstrap CI 退化，只能报点估计；该加权值的区间需待全量样本才能给出。
> 试跑是等量分层抽样（每类 5 条），与候选池的真实类别占比差异很大，故池化值需用后分层加权。

## 4. 模型 vs 人工金标

| 模型 | n | 一致 | 一致率 (Wilson 95%) | Cohen's κ |
|---|---|---|---|---|
| gemini-4-argon | 19 | 19 | 100.0% [83.2, 100.0] | 1.000 |
| claude-sonnet-5.5 | 19 | 18 | 94.7% [75.4, 99.1] | 0.875 |

### gemini-4-argon 混淆矩阵（人工 → 模型）

| | Correct | Wrong-resolution | Spurious | Granularity | Unclear |
|---|---|---|---|---|---|
| Correct | 14 | 0 | 0 | 0 | 0 |
| Wrong-resolution | 0 | 3 | 0 | 0 | 0 |
| Spurious | 0 | 0 | 1 | 0 | 0 |
| Granularity | 0 | 0 | 0 | 1 | 0 |
| Unclear | 0 | 0 | 0 | 0 | 0 |

### claude-sonnet-5.5 混淆矩阵（人工 → 模型）

| | Correct | Wrong-resolution | Spurious | Granularity | Unclear |
|---|---|---|---|---|---|
| Correct | 14 | 0 | 0 | 0 | 0 |
| Wrong-resolution | 0 | 3 | 0 | 0 | 0 |
| Spurious | 0 | 1 | 0 | 0 | 0 |
| Granularity | 0 | 0 | 0 | 1 | 0 |
| Unclear | 0 | 0 | 0 | 0 | 0 |

## 5. 双模型一致性

- gemini-4-argon vs claude-sonnet-5.5：18/19 = 94.7% (Wilson 95% CI [75.4%, 99.1%])，κ = 0.875

## 6. 不一致明细（仅一致性统计所覆盖的边）

| 边 | 系统 | 审计类别 | 人工 | gemini-4-argon | claude-sonnet-5.5 |
|---|---|---|---|---|---|
| P19 | pytorch | clear_error | Spurious | Spurious | Wrong-resolution |

## 7. 数据质量备注

- 盲审表 `weight` 列为该边在审计分组内的链接条数（如 P06=209、P10=1104），非抽样权重；本统计为边级、不加权。
- 排除样本（P04）仅从一致性计算中剔除，其金标仍计入精确率与分层统计；若论文采用更保守口径（也剔除该样本），边精确率取第 1 节的对照值。
- 结论：以上金标统计基于 20 条边、一致性统计基于 19 条边；样本量小，按系统的 CI 仅作方向参考。

