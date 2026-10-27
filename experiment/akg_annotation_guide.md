# AKG 依赖标注说明（R2-4 / R3-2）

输入由 `akg_fill_evidence.py` 生成（seed=42，沿用 `akg_validation_sample.csv` 的 200 条模块边）：

- `akg_validation_evidence.csv`：精确率表，每行一条模块边
- `akg_recall_sample.csv`：召回表，每行一个源文件（50 个）
- `akg_validation_evidence_summary.json`：自动统计

两名标注者独立填写 `label_annotator1` / `label_annotator2`，完成后再讨论分歧，填 `label_final`。

## 1. 精确率表（模块边）

### 1.1 每行要看什么

- `source_module → target_module`，以及 `weight`。`weight` 等于支撑这条边的文件级链接数，`weight_consistent=1` 表示已核对相等。
- `ex{1..3}_*`：最多 3 条随机抽取的文件级链接，每条包括：
  - `source`、`line`、`statement`：源文件、行号、原始语句；
  - `resolved_target`：AKG 把这条语句解析到的文件或目录；
  - `match`：语句的匹配方式；
  - `auto_check`：自动一致性检查结果，只作参考，不能代替人工判断；
  - `same_basename_files`：仓库中同名文件的个数，大于 1 表示解析存在歧义。
- `auto_flags`：粒度提示，取值为 `nested_modules` 或 `target_is_package_not_module`。

### 1.2 auto_check 含义

| 值 | 含义 | 标注时 |
|---|---|---|
| consistent | 语句与 resolved_target 的路径、FQN 或包路径一致 | 核实后通常判 Correct |
| ambiguous_basename | `#include "x.h"` 不带路径，且有同名文件 | 按 include 搜索路径人工判断 |
| suspect:include_path_mismatch | `#include <a/b.h>` 被解析到路径不符的同名文件 | 多为 Wrong-resolution |
| suspect:fqn_mismatch | Java 简单类名相同，但包名不同 | 多为 Wrong-resolution |
| suspect:module_path_mismatch | Python 末段名字相同，但模块路径不同，常见于 `import torch/logging/triton` 被解析到项目内同名文件 | 多为 Wrong-resolution 或 Spurious |
| suspect:target_dir_missing | Go 导入被映射到一个不存在的目录 | 多为 Spurious 或 Wrong-resolution |
| suspect:no_statement / no_matching_import | 源文件中找不到对应语句 | 多为 Spurious |
| unchecked | 无法自动判断 | 人工判断 |

### 1.3 标签（每条边一个）

先逐条判断示例链接，再按下面的规则合成边标签。

| 标签 | 定义 |
|---|---|
| **Correct** | 至少一条示例链接确实把 source_module 中的文件静态依赖到 target_module 中的文件，且解析目标正确 |
| **Wrong-resolution** | 依赖语句真实存在，但解析到了错误的文件或模块，例如同名头文件、同名类 |
| **Spurious** | 找不到真实的静态依赖语句，或者目标在项目外（标准库、第三方库、vendor），却被映射到了项目内 |
| **Granularity** | 依赖本身正确，但边端点的粒度不一致，例如父子模块（nested_modules），或 target 是包目录而 source 是模块（target_is_package_not_module） |
| **Unclear** | 信息不足、生成代码、条件编译等，无法确定 |

合成规则：

1. 只要有一条示例链接是正确的，且 target_module 层级合理，就判 Correct。
2. 示例全部错误时，按多数错误类型判 Wrong-resolution 或 Spurious。
3. 依赖本身正确，但端点粒度有问题，判 Granularity。在报告中，Granularity 作为单独一类统计，不计入错误。
4. 有疑问时，在 `comment` 中写明理由。

判定要点：

- C/C++：看 `#include <dir/x.h>` 的路径前缀和 resolved_target 是否一致。只有 basename 相同不算正确。Linux 的 `<linux/*.h>` 应对应 `include/linux/*.h`。注意，当前 file_index 没有收录 `include/` 目录。
- Java：比较 import 的 FQN 和 resolved_target 推导出的 FQN。
- Python：`import torch`、`import logging` 这类指向顶层包或标准库的导入，被解析到 `torch/_dynamo/variables/torch.py` 这类文件时，判 Spurious。
- Go：`k8s.io/utils/...` 这类 vendor 或外部包被映射到 `staging/src/...` 下不存在的目录时，判 Spurious；目录路径映射错位时，判 Wrong-resolution。
- Fortran：`CALL X` 对应 `X.f` 中的子程序，判 Correct。

## 2. 召回表（文件）

每行给出一个 AKG 已索引的源文件，以及 AKG 为它解析出的链接，包括 `akg_resolved_targets` 和 `akg_target_modules`。标注者需要：

1. 打开源文件，列出它对**项目内其他模块**的全部静态依赖，即 include、import、call。不包括外部库和标准库。
2. `missed_internal_deps`：真实存在但 AKG 没有给出的依赖，填目标文件或模块，多个之间用 ` | ` 分隔。
3. `spurious_links`：AKG 给出了，但实际不成立的链接。
4. 填写 `annotator`，需要时填写 `comment`。

## 3. 统计与报告

- **边精确率** = Correct /（Correct + Wrong-resolution + Spurious）。Granularity 和 Unclear 单列，不计入分母。按 Wilson 方法给出 95% CI，同时报告按系统的分布。
- **错误分布**：Wrong-resolution、Spurious、Granularity 各占多少，按系统和语言分开报告。
- **一致性**：用两名标注者的 `label_annotator1/2` 计算 Cohen's κ，在 5 类标签上计算。
- **文件级召回率** = AKG 正确链接数 /（AKG 正确链接数 + missed_internal_deps 数），先在 50 个文件上汇总，再按系统报告。
- 样本是按系统规模分层抽取的，Linux 占 86 条，Kubernetes 占 48 条。报告总体值时，要同时给出按系统的值，避免大系统主导结论。
