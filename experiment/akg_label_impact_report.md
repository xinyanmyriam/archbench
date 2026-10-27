# AKG 依赖标签受影响程度审计报告

- 审计对象：12 个系统的 `module_dependency.json` 及其派生的 DP（依赖预测）/ AR（架构重建）任务标签
- 结论指向的审稿意见：R2（ground truth 可靠性、可复现性）、R3-2
- 审计脚本：`experiment/akg_impact_audit.py`（本次新增）
- 原始输出：`experiment/akg_impact_audit.json`
- Linux 专项试验：`experiment/akg_fix_trial_linux.py`、`experiment/akg_fix_trial_linux.json`
- 抽样证据：`experiment/akg_validation_evidence.csv`、`experiment/akg_annotation_guide.md`
- 审计日期：2026-10-19

> 说明：本报告全部为**自动判定**结果，未经人工确认。判定规则镜像了论文原始流水线（`pipeline/generate_all.py`、各系统 `generate_file_dependency.py`、`salome/crossinfoGeneration.py`、`kubernetes/generate_benchmark_tasks.py`）的解析逻辑，并断言"重放解析器能逐边复现 `module_dependency.json`"（12/12 系统通过）。数值可复现，但精度以人工标注为准。
>
> 修订记录：初版把相对路径的 C/C++ include（`../x.h`）和 Python 相对导入（`..pkg.mod`）一律判为错误；现已改为相对源文件目录/所在包解析后再比对（codeaster 9.1%→1.4%、openmodelica 16.0%→8.0%、linux 98.0%→97.2%，其余系统边级结果不变）。下文均为修订后数字。

---

## 0. 结论摘要

判定标准（两个口径，均按**边数**衡量，因为 DP/AR 是逐边标签）：

- **失去诚实支撑**：该边的全部支撑链接都落在"明确错误"或"无法判定"两类，即没有任何一条属于"路径可验证"或"名称唯一"。这是判定受损程度的**主口径**。
- **未经路径/符号确证**：该边没有任何路径、FQN、模块路径或符号级的确认，只有名称级的证据。该口径只在"原始数据保留了路径"的系统上有判别力（见 §2.3）。

| 系统 | 语言 | DP 正例失去诚实支撑 | DP 正例未经路径确证 | AR 任务边失去诚实支撑 | 模块边 | 判定 |
|---|---|---|---|---|---|---|
| linux | C/C++ | **97.2%**（1944/2000） | 97.7%（1953/2000） | 1941/2000 | 5,947 | 严重 |
| hadoop | Java | **43.7%**（59/135） | 54.1%（73/135） | 59/135 | 135 | 严重 |
| salome | C/C++ | **41.3%**（405/980） | 97.9%（959/980） | 405/980 | 980 | 严重（41.3% 为上限，见 §3.4） |
| kubernetes | Go | **37.0%**（147/397） | 37.0%（147/397） | 147/397 | 397（归一化后） | 严重 |
| pytorch | Python/C++ | **36.4%**（277/762） | 36.6%（279/762） | 277/762 | 762 | 严重 |
| tensorflow | Python/C++ | **32.0%**（472/1476） | 32.3%（477/1476） | 472/1476 | 1,476 | 严重 |
| spring | Java | **31.1%**（37/119） | 31.1%（37/119） | 37/119 | 119 | 严重 |
| openfoam | C++ | 13.6%（27/199） | 100.0%（199/199） | 27/199 | 199 | 中等 |
| openmodelica | Modelica/C++ | 8.0%（2/25） | 48.0%（12/25） | 2/25 | 25 | 轻微 |
| codeaster | Python/C++ | 1.4%（5/352） | 19.3%（68/352） | 5/352 | 352 | 基本无影响 |
| paraview | C/C++ | 0.3%（1/311） | 100.0%（311/311） | 1/311 | 311 | 基本无影响 |
| nastran | Fortran | 0.0%（0/4） | 0.0%（0/4） | 0/4 | 4 | 无影响 |

要点：

1. **7/12 个系统的 DP/AR 标签有明显问题**，失去诚实支撑的边从 31% 到 97% 不等。
2. **CI / CI-Hard / CI-Obfuscated 三个任务族不受影响**：它们的标签来自 `file_index`（文件归属哪个模块），与依赖解析无关。
3. 全库共 861,433 条原始依赖语句、501,246 条文件级链接，其中自动判定为**明确错误**的约 150,942 条（30.1%）。
4. linux 的影响不是"部分错误"，而是**几乎整张模块图失效**：97.2% 的模块边无诚实支撑，路径确证的权重仅占 0.6%。
5. "未经路径确证"一列不能单独解读：paraview、openfoam 的这一列是 100%，但两者实际的明确错误率分别只有 0.2% 和 0.0%（它们的原始语句只保存文件名，见 §2.3）。
6. 偏差方向：除 salome 外，"失去诚实支撑"是**下界**——"名称唯一"被计为有支撑，且本审计只检查假边、不检查漏边（见 §8）。

---

## 1. 审计方法

对每条原始依赖语句，镜像该系统的原始解析器，得到 `(源文件, 原始语句, 解析目标)`，再按证据强度判为四类之一：

| 类别 | 含义 | 判定依据 |
|---|---|---|
| **路径可验证** | 语句本身携带足够信息，能与解析目标对上 | C/C++ 带路径 include 命中索引路径后缀、源文件同目录，或相对路径（`../x.h`）按源文件目录解析后命中；Java FQN 精确命中；Python 点路径命中包根，或相对导入按所在包解析后命中；Go 命中存在的包目录；Fortran 符号命中子程序索引 |
| **名称唯一（弱证据）** | 语句只有文件名，仓库内同名文件唯一 | 解析结果合理，但不受路径约束，**不能算作错误，也不能算作确证** |
| **明确错误** | 语句与解析目标不一致 | 索引覆盖缺口、外部依赖被强行映射、同名冲突选错、Go 目标目录不存在 |
| **无法判定** | 语句同名候选多个或跨仓 fan-out | 需要人工判断 |

由此得到系统级指标：

- 文件级链接构成（`path` / `name` / `bad` / `unk` 占比）
- 模块边可信支撑：**失去诚实支撑**的边数（无"路径可验证"且无"名称唯一"证据）与**未经路径确证**的边数
- 任务级影响：DP 正例、AR 任务边在两口径下失去支撑的比例。除 linux 外，DP 正例与 AR 都覆盖了全部模块边，因此三者比例相同；linux 的 DP 正例与 AR 是两套独立的 2000 条抽样（重叠 688 条），比例接近但不相同

**一致性核对**：审计脚本断言重放解析器能逐边复现已发布的 `module_dependency.json`（12/12 通过），DP/AR 的模块名与已发布图完全对应（12/12，正例全部落在图中；kubernetes 需按 `generate_benchmark_tasks.py:normalize_module` 折叠包路径后对应）。因此下表的损失比例不是命名口径造成的假象。

---

## 2. 逐系统结果

### 2.1 任务层面影响

| 系统 | DP 正例/总数 | 失去诚实支撑 | 未经路径确证 | AR 任务边失去诚实支撑 | 无诚实支撑的模块边 |
|---|---|---|---|---|---|
| linux | 2000/4000 | 1944 | 1953 | 1941/2000 | 5,778/5,947（97.2%） |
| hadoop | 135/270 | 59 | 73 | 59/135 | 59/135（43.7%） |
| salome | 980/1960 | 405 | 959 | 405/980 | 405/980（41.3%） |
| kubernetes | 397/794 | 147 | 147 | 147/397 | 147/397（37.0%） |
| pytorch | 762/1524 | 277 | 279 | 277/762 | 277/762（36.4%） |
| tensorflow | 1476/2952 | 472 | 477 | 472/1476 | 472/1476（32.0%） |
| spring | 119/238 | 37 | 37 | 37/119 | 37/119（31.1%） |
| openfoam | 199/398 | 27 | 199 | 27/199 | 27/199（13.6%） |
| openmodelica | 25/50 | 2 | 12 | 2/25 | 2/25（8.0%） |
| codeaster | 352/704 | 5 | 68 | 5/352 | 5/352（1.4%） |
| paraview | 311/622 | 1 | 311 | 1/311 | 1/311（0.3%） |
| nastran | 4/8 | 0 | 0 | 0/4 | 0/4（0.0%） |

注意三点：

- hadoop、spring 的**权重**损失很小（无支撑权重分别为 4.3%、4.3%），但**边**损失很大（43.7%、31.1%）：失真的边普遍是低权重边（权重 1–2）。DP/AR 按边抽样，因此对指标的伤害由边数决定，不由权重决定。其他系统的权重指标见 `akg_impact_audit.json`。
- kubernetes 的边数是**任务实际使用的归一化粒度**（`staging/src/k8s.io/api/core/v1` 折叠为 `staging/src/k8s.io/api`），原始图有 3,270 条边；两口径数值相同，因为 kubernetes 的链接没有"名称唯一"这一类。
- paraview 的"未经路径确证"为 100%，但它只有 1/311 条边失去诚实支撑——该列要在 §2.3 的口径下解读。

### 2.2 解析层面构成（文件级链接）

| 系统 | 原始语句 | 链接 | 路径可验证 | 名称唯一 | 明确错误 | 无法判定 |
|---|---|---|---|---|---|---|
| linux | 341,042 | 179,846 | 38.0% | 4.7% | **56.8%** | 0.5% |
| tensorflow | 137,017 | 79,605 | 72.4% | 0.0% | **27.6%** | 0.0% |
| openfoam | 54,731 | 54,355 | 0.0% | 91.7% | 0.0% | 8.3% |
| pytorch | 50,368 | 33,682 | 70.1% | 0.3% | **29.6%** | 0.1% |
| hadoop | 88,634 | 46,999 | 91.0% | 0.3% | **8.7%** | 0.0% |
| kubernetes | 42,164 | 21,119 | 68.4% | 0.0% | **31.6%** | 0.0% |
| spring | 39,443 | 22,251 | 79.3% | 0.0% | **20.6%** | 0.0% |
| salome | 36,576 | 18,439 | 0.3% | 85.4% | 0.0% | **14.4%** |
| paraview | 26,697 | 11,075 | 50.0% | 49.7% | 0.2% | 0.1% |
| openmodelica | 23,968 | 16,552 | 93.0% | 2.8% | 3.8% | 0.4% |
| codeaster | 13,207 | 10,403 | 68.7% | 23.6% | 7.7% | 0.0% |
| nastran | 7,586 | 6,920 | 100.0% | 0.0% | 0.0% | 0.0% |

- openfoam、salome 的"明确错误"为 0 **不代表解析正确**，而是原始语句只保存了文件名（openfoam 54,731 条中仅 142 条保留路径），路径信息在解析阶段就已丢失，错误**无法被这种方式检出**。这两个系统的风险体现在"名称唯一/无法判定"两列。
- 明确错误合计约 150,942 条链接，占全部链接的 30.1%。

### 2.3 两个口径怎么读

「未经路径确证」高、但「明确错误」低，说明该系统的链接主要靠**名称唯一**成立，而不是解析器出错。典型：

| 系统 | 原始数据是否保留 include 路径 | path% | name% | bad% | 解读 |
|---|---|---|---|---|---|
| linux | 是（`<linux/kernel.h>` 保留目录） | 38.0 | 4.7 | 56.8 | 路径可对不上，只能退化为同名匹配 → 真实错误 |
| tensorflow / pytorch / spring / hadoop / kubernetes / openmodelica / codeaster | 是 | 68–93 | ≤24 | 4–32 | 路径级解析为主，错误集中在外部依赖误映射与同名冲突 |
| paraview | 部分 | 50.0 | 49.7 | **0.2** | 名称唯一匹配，错误率极低 |
| openfoam | **否**（54,731 条中仅 142 条带路径） | 0.0 | 91.7 | **0.0** | 解析器已丢弃路径，无法用路径判错，8.3% 为前缀启发式猜测 |
| salome | **否** | 0.3 | 85.4 | **0.0** | 同上，且跨仓链接为同名 fan-out |

结论：**「失去诚实支撑」是判定受损系统的主口径**；「未经路径确证」在 paraview 这类仓库会造成高估，只作辅助参考。

---

## 3. 失败模式与代表例

### 3.1 索引覆盖缺口（linux 为主）

`file_index` 未收录 `include/` 与 `arch/*/include/` 两类目录（均为 0 个文件），而 341,042 条 include 中有 264,453 条（77.5%）是 `<linux/…>`、`<asm/…>` 这类带路径的头文件。解析器于是退化为按文件名匹配，挂到了同名文件上：

- `arch/alpha/boot/bootp.c | #include <linux/kernel.h>` → `arch/sparc/kernel/kernel.h`
- `arch/alpha/boot/bootp.c | #include <linux/mm.h>` → `arch/arm/mm/mm.h`
- `arch/alpha/boot/bootp.c | #include <asm/io.h>` → `arch/powerpc/boot/io.h`

这类错误 93,883 条（`config_gap`），另有 7,969 条（`external_misresolved`）指向仓库中同样不存在的路径。两者合计占 linux 全部链接的 56.6%。

### 3.2 外部依赖被强行映射（Java/Python 为主）

项目外的库（标准库、第三方依赖）被按同名文件挂进项目内，凭空造出模块边：

- spring：`org.jspecify.annotations.Nullable` → `spring-core/.../org/springframework/lang/Nullable.java`（3,392 条）
- hadoop：`com.qcloud.cos.utils.Base64` → `hadoop-azure/.../azurebfs/utils/Base64.java`；`java.util.Queue` → mapreduce 内的 `Queue.java`（1,713 条）
- tensorflow：`absl/status/statusor.h` → `tensorflow/core/platform/statusor.h`；`absl/types/span.h` → `tensorflow/core/tfrt/mlrt/bytecode/span.h`（7,082 条）
- pytorch：`ATen/Functions.h` → `aten/src/ATen/templates/Functions.h`；`os` → `torch/_dynamo/polyfills/os.py`（4,687 条）
- codeaster：`scipy.optimize` → `code_aster/MacroCommands/Utils/optimize.py`（4 条）

### 3.3 同名冲突（正确目标其实已在索引内，却选了另一个）

- pytorch：`ATen/core/ScalarType.h` → `aten/src/ATen/ScalarType.h`（正确目标 `c10/core/ScalarType.h` 在索引内）
- tensorflow：`tensorflow/core/platform/status.h` → `tensorflow/cc/experimental/base/public/status.h`
- spring：`org.springframework.util.CollectionU…` 的简单类名匹配落到同名的 webflux 类
- hadoop：`com.qcloud.cos.model.ListObjectsRequest` → tosfs 下的同名类
- pytorch：相对导入 `..util.utils`（应为 `package/util/utils.py`）→ `package/oss/utils.py`；codeaster：`..Language.DataStructure` → `code_aster/Cata/DataStructure.py`

计数：`wrong_path` 合计 9,634 条（tensorflow 6,427、pytorch 2,346、openmodelica 497、linux 347、paraview 17）；`wrong_simple_name` 合计 3,588 条（hadoop 2,387、spring 1,201）；`wrong_module`（Python）合计 1,557 条（codeaster 799、pytorch 753、tensorflow 5）。

### 3.4 粒度不一致与 fan-out

- **kubernetes**：`file_dependency` 的 target 是 Go 包目录，而 source 侧是模块；生成任务时 `normalize_module` 再把包路径折叠成顶层包（`pkg/volume/util` → `pkg/volume`），于是原始 3,270 条边折叠为 397 条。同时 6,683 条导入被映射到不存在的目录（如 `k8s.io/klog/v2` → `staging/src/k8s.io/klog/v2`）。
- **openfoam（1 条）、kubernetes（7 条）**：边端点是父子模块，例如 openfoam 的 `src/thermophysicalModels → src`。
- **salome**：跨仓解析（`crossinfoGeneration.py`）对每个同名文件都产出一条链接，不做唯一性约束，共 2,649 条（占权重 30.2%）。这 405 条无支撑的边**可能是真的**，也可能是同名误配，因此 salome 的 41.3% 应视为**上限**，需要单独复核。

---

## 4. Linux 专项：严格重建试验

用严格规则（带路径的 include 必须路径一致，否则不解析；同名歧义直接丢弃；裸 include 优先同目录）重建 Linux 模块图，不改动任何已发布数据：

| 指标 | 现版图 | 严格规则 |
|---|---|---|
| 文件级链接 | 179,846 | 75,184 |
| 模块边 | 5,947 | 165（保留 164，新增 1） |
| 总权重 | 88,597 | 1,153 |
| 200 条抽检样本中的边 | 86 条 | 0 条保留 |
| DP 标签（2000 正例） | — | 1,947 条翻转为 0 |
| AR 任务边（2000 条） | — | 仅 50 条仍存在；源模块目标集 Jaccard 均值 0.02 |

现版图中权重最大的边多为误配产物：

- `sound/soc → arch/arm`（1,371）
- `drivers/media → drivers/net`（1,079）
- `drivers/net → arch/sparc`（830）

**严格重建后的图也不可用**：165 条边过于稀疏，且 Linux 模块间的真实依赖大量经由 `include/` 中转（`drivers/net` 引用 `include/linux/skbuff.h`，实现却在 `net/core`），只要 `include/` 不算模块，静态 include 就无法反映这些依赖。

---

## 5. 对论文结果的影响面

| 任务 / 章节 | 标签来源 | 是否受影响 |
|---|---|---|
| CI、CI-Hard、CI-Obfuscated | `file_index` | 不受影响 |
| Level 0 DP | `module_dependency` | 受影响，可**离线重新打分**（模型输出已保存，无需重新调用 API） |
| Level 0 AR | 同上 | 同上 |
| Level 1 DP/AR | 提示词给的是原始 import 语句，标签仍来自 `module_dependency` | 主要需重新打分；`compute_level1_leakage.py` 的同源诊断需用新标签重算 |
| Level 2 DP/AR | 提示词内嵌依赖图，图内含错误边 | 受影响系统的提示词本身有噪声，需重新调用模型 |
| 基线（Frequency、Graph Neighbor） | 同一张图 | 与模型同源，需用新图重算；Graph Neighbor 尤其依赖该图 |

结论层面：错误标签会**系统性压低**模型分数（把真依赖判为错、把假依赖当成正确答案），因此"LLM 架构理解能力弱"的结论有一部分来自标签噪声；跨系统比较也被污染（7 个系统受不同程度影响）。

---

## 6. 后续选项

| 方案 | 内容 | 影响面 | 代价 |
|---|---|---|---|
| **A. 剔除严重受影响系统的 DP/AR** | linux（97.2%）、hadoop（43.7%）、salome（41.3%，待复核）、kubernetes（37.0%）、pytorch（36.4%）、tensorflow（32.0%）、spring（31.1%）的 DP/AR 结果不作为主要证据，或单独报告并标注标签噪声 | 摘要、结论、DP/AR 表格、跨系统均值 | 中：需重算均值，无需重跑 API |
| **B. 修复解析器后重建标签** | 补 `include/` 等索引缺口、禁止外部依赖映射进项目内、同名冲突改为不解析、salome 跨仓改为唯一性约束 | 全部 DP/AR 与 Level 2 | 高：所有受影响系统需重新生成标签并重跑 Level 2 |
| **C. 先人工标注，再据实报告** | 用 `akg_annotation_guide.md` 的 5 类标签在 200 条边 + 50 文件样本上双人标注，报告精确率（含 95% CI）、错误分布、Cohen's κ、召回率 | §3 QA、R2-4/R3-2 回复 | 低 |
| **D. 只修 linux 的索引缺口后重建** | 试验显示重建后图不可用（165 边），不建议 | — | — |

建议顺序：先做 C 拿到**经过人工确认的数字**，再决定 A 或 B。C 的数字决定了 A 的剔除范围是否站得住，也决定 B 的优先级。

---

## 7. 复现方式

```
python -u experiment/akg_impact_audit.py        # 12 系统审计，约 4 分钟
python -u experiment/akg_fix_trial_linux.py     # Linux 严格重建试验
python -u experiment/akg_fill_evidence.py       # 生成 200 条边证据与 50 文件召回样本
```

输出文件：

- `experiment/akg_impact_audit.json`：逐系统指标、判定分布、代表例
- `experiment/akg_fix_trial_linux.json`：Linux 严格重建对比
- `experiment/akg_validation_evidence.csv`、`experiment/akg_recall_sample.csv`：人工标注输入
- `experiment/akg_annotation_guide.md`：标注规则与统计口径

## 8. 已知局限

1. **全部为自动判定**，未人工确认。偏差方向是单向的：判定规则在三处偏宽松——"名称唯一"计为有支撑（裸 include 的系统头文件若恰好与项目内唯一文件同名，会被当作支撑）、C/C++ 带路径 include 按路径后缀匹配、Java 嵌套类按外层类判 `verified_unique_class`——因此"失去诚实支撑"是**下界**。唯一例外是 salome 的跨仓 fan-out（见第 3 条），为上界。
2. **linux 的 `config_gap` 与 `external_misresolved` 划分存在不确定**：`<asm/io.h>` 类头文件位于 `arch/*/include/`，本方法按常规 include 根判定，两者之间可能有少量串档，但两类同为"路径无法对上"，不影响结论。
3. **salome 跨仓链接未验证**：审计只确认其构成方式（同名 fan-out），未逐条复核，故其 41.3% 为上限。
4. **openfoam、salome 的原始数据在解析阶段丢弃了 include 路径**，本方法无法判定其正确性，需回到源码重新解析才能量化。
5. 审计覆盖**已发布快照**，重跑实验（`results_rerun/`）若使用了不同的标签版本，需另行核对。
6. **只检查了假边（DP 正例 / AR 目标中的错误边），未检查漏边**。真实依赖若没进图，会被当作 DP 负例或从 AR 目标集中缺失：linux 中所有经由 `include/`、`arch/*/include/` 的依赖都不在图内；kubernetes 中 4,156 条 `k8s.io/kubernetes/...` 自引用（占 raw 42,164 条的 9.9%）被映射到不存在的 `staging/src/k8s.io/kubernetes` 目录；openfoam 丢弃了带路径的 include。因此标签的实际错误率高于本报告数字。
