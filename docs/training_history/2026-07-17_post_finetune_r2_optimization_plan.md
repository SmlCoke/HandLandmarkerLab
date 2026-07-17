# Finetune r2 后两天优化计划

日期：2026-07-17

可用时间：2 天

适用基线：`v2-pretrain-r3`、`v2-finetune-r1`、`v2-finetune-r2 Gold 50%`

本文是当前两天冲刺计划，不是通用操作手册。当前数据和指标见[当前训练与数据状态](../training_system/current_training_status.md)。

## 1. 两天内的决策

r2 只把 Gold 从每 batch 34.375% 提高到 50%，结果：

```text
r1 mean pixel  23.1402
r2 mean pixel  23.1415
r1/r2 infer    都是 192 张输入、140 张有检测、217 个检测
spread <0.10   97 → 99
```

因此两天内不再做：

- 继续试 70%/80% Gold；
- 扩大模型参数量；
- 无目的增加 MediaPipe pseudo；
- 没有自动排序和去重的全量人工复核；
- 固定关节方向、绝对骨长或“把点强行拉开”的后处理。

两天内只做三件事：

1. 程序自动找出最值得标的塌缩/高误差 ROI；
2. 团队并行精标 600～800 个高价值新 ROI，硬上限 800；
3. 在同一新数据快照上训练两个必做候选；只有第二个候选已经改善且时间充足时，才运行第三个可选候选。

人工总工作应控制在：

| 人工任务 | 目标工作量 |
|---|---:|
| 查看自动错误 overlay | 不超过 40 张，约 15～20 分钟 |
| 录制困难手势 | 20～30 分钟 |
| CVAT 精标 | 团队合计 600～800 ROI，硬上限 800；按约 100 张/job 分工 |
| 查看候选 infer/上板 | 每个候选固定少量样例 |

人均正式标注量约为“冻结预算 ÷ 标注人数”：800 张由 4/6/8 人分工时，人均约为 200/134/100 张。若无法把人均量控制在约 200 张以内，就选 600 预算。

不要求人工写 CSV、JSONL、候选 ID、排除表、SHA 或数据聚合配置。开始导出前先按实际参加人数冻结总预算：人手充足选 800，人手不足选 600；不要先生成 800 张任务、再留下未处理图片。

## 2. 计划前提：先修改仓库，再开始人工操作

下文人工流程建立在“修改后的 HLMF/HLML”上。以下拟新增命令在代码落地、测试通过并同步服务器之前都不能运行，也不能要求人工用旧仓库手工补足其功能。

顺序必须是：

```text
先完成本地 HLMF/HLML 代码修改
  → 单元测试与小样例通过
  → Git 提交/推送
  → 服务器两个仓库 git pull
  → 服务器 compile/test/preflight
  → 程序自动生成不重复任务
  → 人工才开始录制、看 overlay 和 CVAT
```

若程序修改在第 1 天前半天仍不能通过 gate，停止新增人工任务，不能让人工先标一批无法稳定导入的数据。

## 3. 程序需要先完成的最小修改

只实现两天关键路径需要的功能，不扩展为大型通用平台。

### 3.1 HLML：自动错误分析与限额选样

拟新增：

```bash
make analyze-finetune-errors \
  BASELINE_FINETUNE_ID=v2-finetune-r1 \
  CANDIDATE_FINETUNE_ID=v2-finetune-r2
```

程序读取既有 Val/infer，不修改数据，自动产生：

```text
hand_landmarker_runs/v2-finetune-r2/analysis/error_audit/
├── summary.json
├── per_roi_metrics.jsonl
├── paired_comparison.json
└── overlays/              # 总数最多 40 张
```

每个 Val ROI 至少计算：pixel error、spread、预测/Gold spread ratio、20 条骨边误差、dataset、crop ID 和 SHA。固定 infer 至少计算：Palm ROI、hand flag、spread、r1/r2 逐点差异。

只输出四类代表图：

```text
student_collapse
high_landmark_error
palm_or_roi_failure
improved_vs_baseline
```

每类 top-K 加少量随机样本，总量不超过 40。

### 3.2 HLML：第二轮 selection 和累计排除

拟新增：

```bash
make prepare-finetune-round \
  FINETUNE_ROUND_ID=r02 \
  FINETUNE_GOLD_BUDGET=<600-or-800> \
  NEW_RECORDED_SOURCE_ID=new_recorded_gold_r01
```

程序必须自动读取已有 Gold aggregate、历史 selection、Val/Test 和 source registry，并使用以下身份排除重复：

```text
parent_global_crop_id
global_crop_id
source image identity
ROI SHA-256
归一化像素 SHA-256
```

本轮只选择：

- 600 预算优先分为：`disagreement_gold_r02` 400 ROI + `new_recorded_gold_r01` 最多 200 ROI；
- 800 预算优先分为：`disagreement_gold_r02` 500 ROI + `new_recorded_gold_r01` 最多 300 ROI；
- 不再制作第二轮 `negative_removed_gold`，现有 260 条可训练记录先保留使用；
- HLMF 先确定新录制 task 的实际数量；HLML 再把 `总预算 - new-recorded 实际数` 作为 disagreement 数量。如果新录制有效 ROI 不足，程序因此自动用更多不重复 disagreement 补足，但总 CVAT 数硬封顶 800。

Disagreement 选择应优先：student collapse、高 pixel/NME、高 bone error，同时按 dataset/session/姿态分散，不能被连续相似帧占满。

与 replay 重合不是错误：新 Gold 可以晋升旧 replay，finetune curate 会自动保留 Gold、移除重复 replay。已经进入历史 Gold/CVAT 或 Val/Test 的 ROI 必须排除。

### 3.3 HLMF：从旧 Gold 自动建立新 finetune 工作区

拟新增：

```bash
make seed_finetune_gold \
  BASE_FINETUNE_ID=v2-finetune-r1 \
  HAND_FINETUNE_ID=v2-finetune-r3
```

程序自动把 r1 已认证的三个 Gold source 迁入新的 r3 工作区，保持图片字节不变，并重新生成/认证 descriptor、路径和 aggregate 输入。人工不复制 source、不改 descriptor。

要求：

- r1 目录只读；
- r3 目标已存在时 fail-closed；
- 每个文件 SHA 与原 source 一致；
- 输出 seed report，记录 base/new ID、文件数和 aggregate SHA；
- Dragon、旧 disagreement、旧 negative-removed Gold 全部保留。

### 3.4 HLMF：支持第二轮 source ID 和新录制限额

HLMF 必须支持：

```text
disagreement_gold_r02       source_kind=disagreement_gold
new_recorded_gold_r01       source_kind=new_recorded_gold
```

`finalize_train_finetune` 必须能聚合同一 `source_kind` 的多个 source ID，并跨轮去重/冲突检查。

`native_existing` 新增程序控制的最大任务量，由冻结总预算自动解析为 200 或 300 ROI；按时间间隔、session、Palm/ROI 和图像差异做确定性抽样。人工不从原图目录手选这些图片。

每个导出包还要生成 `qc/cvat_job_plan.json`，建议 CVAT `segment size=100`，列出每个 job 的图片范围、数量和 SHA。人工只在 CVAT 中把互不重叠的 job 分配给不同成员，不手工拆图片目录。

### 3.5 HLML：结构 loss 和实验 profile

只新增两个标签驱动、可关闭的结构项：

1. `bone_vector_loss`：预测与 Gold 的 20 条骨向量 Huber；
2. `spread_ratio_loss`：预测与 Gold 的整体 spread log-ratio Huber。

两项在两天候选中只对 valid Gold positive 生效；pseudo replay 的结构项默认关闭，避免大量 teacher 伪标签再次主导新约束。negative、landmark mask 无效和 ignored 均不得误用。不得加入固定手势模板、固定骨长或固定关节朝向。

为减少配置文件，建议只增加一个 profile 入口：

```text
data_only
structure
structure_roi_aug
```

每个 run 自动保存 resolved config。`HAND_FINETUNE_ID` 仍表示 frozen data workspace；新增一个可选 `FINETUNE_EXPERIMENT_ID` 只路由 run/eval/infer/export，避免复制图片或混淆导出 provenance。

### 3.6 自动比较

拟新增：

```bash
make compare-finetune-runs \
  BASELINE_FINETUNE_ID=v2-finetune-r1 \
  CANDIDATE_FINETUNE_ID=<candidate-id>
```

自动报告 overall/per-dataset/per-landmark、配对差值、PCK、presence、handedness、infer 检测数、spread 和 top 改善/退化 overlay。

### 3.7 程序修改验收

人工开始前必须全部满足：

- HLMF/HLML `make compile` 或等价语法检查通过；
- 两仓库单元测试通过；
- 用临时小目录验证 seed 不改旧 source；
- 第二轮 selector 在重复执行时结果确定，且不会选中已有 Gold/Val/Test；
- 任务上限严格不超过配置值；
- 600/800 两档预算能解析为正确的 disagreement/new-recorded 配额，且每个 CVAT job 不超过 100 张；
- 新 loss 的 mask、梯度和 batch reduction 测试通过；
- experiment ID、data ID、manifest 和 export provenance 能正确绑定；
- 仓库文档中的新命令与 Makefile 一致。

## 4. 修改完成后的两天人工与训练流程

以下命令以修改后的仓库为准。

## 5. Day 1：自动选样 + 团队并行精标 600～800 ROI

### 5.1 同步并验证新代码

```bash
cd /root/HandLandmarksFab
git pull
conda activate anfab
make test

cd /root/HandLandmarkerLab
git pull
conda activate hand-landmarker-tf29
make paths
make compile
make test-unit
```

任何测试失败都停止，不开始 CVAT。

### 5.2 自动建立 r3 Gold 工作区

```bash
cd /root/HandLandmarksFab
conda activate anfab

make seed_finetune_gold \
  HAND_DATA_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0 \
  BASE_FINETUNE_ID=v2-finetune-r1 \
  HAND_FINETUNE_ID=v2-finetune-r3
```

人工只查看 seed report 是否 `status=ok`、三个旧 Gold source 是否都存在、SHA 是否一致。

### 5.3 自动错误分析

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29

make analyze-finetune-errors \
  BASELINE_FINETUNE_ID=v2-finetune-r1 \
  CANDIDATE_FINETUNE_ID=v2-finetune-r2
```

人工查看不超过 40 张 overlay，用 15～20 分钟确认：主要失败确实包括握拳、侧向张掌、数字“1”或完整 ROI 上的 student collapse。无需做逐图表格。

### 5.4 录制来源 e

只录以下五类：

```text
握拳
数字“1”
侧向张掌
遮挡/手指重叠
近/远距离和轻微 ROI 偏移
```

要求：

- 新的 train-only session；
- 左右手都包含；
- 每类短序列，不大量录制连续重复帧；
- 原文件不重编码；
- 总录制时间 20～30 分钟；若多人参与，优先让不同人员各录一个独立 session、使用不同背景/距离，而不是由一人延长同一段视频。

HLMF 00～03：

```bash
cd /root/HandLandmarksFab
conda activate anfab
export HAND_DATA_ROOT=/path/to/new-recorded-r01

make validate_images_train
make palm_detection_train
make build_roi_train
make run_mediapipe_train
```

### 5.5 程序生成两个 CVAT task

先按已确认的标注人数冻结 `600` 或 `800`。下面以 800 为例，先让 HLMF 从新录制 source 自动选择最多 300 ROI；若冻结的是 600 预算，把该值改为 200：

```bash
cd /root/HandLandmarksFab
conda activate anfab

make export_finetune_gold \
  HAND_DATA_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0 \
  HAND_FINETUNE_ID=v2-finetune-r3 \
  FINETUNE_SOURCE_ID=new_recorded_gold_r01 \
  FINETUNE_SOURCE_MODE=native_existing \
  FINETUNE_RAW_SOURCE_ROOT=/path/to/new-recorded-r01 \
  FINETUNE_MAX_ITEMS=300
```

随后 HLML 读取新录制 task descriptor 和 export report，把剩余预算全部分配给不重复 disagreement：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29

make prepare-finetune-round \
  HAND_FINETUNE_ID=v2-finetune-r3 \
  FINETUNE_ROUND_ID=r02 \
  FINETUNE_GOLD_BUDGET=800 \
  NEW_RECORDED_SOURCE_ID=new_recorded_gold_r01
```

最后由 HLMF 导出 disagreement：

```bash
cd /root/HandLandmarksFab
conda activate anfab

make export_finetune_gold \
  HAND_DATA_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0 \
  HAND_FINETUNE_ID=v2-finetune-r3 \
  FINETUNE_SOURCE_ID=disagreement_gold_r02 \
  FINETUNE_SOURCE_MODE=selection_subset
```

程序必须打印两个 task 的总数、每个 source 的分层分布和 CVAT job 规划。正常情况下两者合计应恰好等于冻结预算；候选不足时可以少于预算，但绝不允许超过 600/800 上限。

### 5.6 人工 CVAT

两个 source 分别创建一个 CVAT image task，创建时按 `cvat_job_plan.json` 设置约 100 张一个 job，再把互不重叠的 job 分配给团队成员。团队合计处理冻结的 600 或 800 ROI，硬上限 800。每张图选择：

- 完整 21 点 + Left/Right/unknown；或
- `no_hand`；或
- `ignore_for_training`。

多人开始正式标注前，先共同处理同一组 10 张校准图，由一名负责人统一确认 21 点编号、手腕/指尖位置、Left/Right 视角和 `ignore_for_training` 标准。正式任务不重复标注；每位成员完成后，由负责人抽查其约 5% 的图片。无法可靠标注时直接 `ignore_for_training`，不要耗时猜点。全部 job 完成后，分别从两个完整 task 导出 `CVAT for images 1.1` 为 `reviewed.xml`。

### 5.7 自动导入、聚合和 curate

```bash
cd /root/HandLandmarksFab
conda activate anfab

make import_finetune_gold \
  HAND_DATA_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0 \
  HAND_FINETUNE_ID=v2-finetune-r3

make finalize_train_finetune \
  HAND_DATA_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0 \
  HAND_FINETUNE_ID=v2-finetune-r3
```

然后：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29

make prepare-finetune-sources HAND_FINETUNE_ID=v2-finetune-r3
make finetune-curate \
  HAND_FINETUNE_ID=v2-finetune-r3 \
  FINETUNE_PROFILE=data_only
make check-finetune-data HAND_FINETUNE_ID=v2-finetune-r3
make inspect-finetune HAND_FINETUNE_ID=v2-finetune-r3
```

`data_only` profile 在 modified repo 中同时解析第 5.8 节的 Gold role 权重，并把它们冻结进 r3 curation report。所有 Gold、replay、去重、泄漏、实际 draw 和 SHA 由程序处理。人工只查看 report 是否 `ok/pass`。

### 5.8 当晚训练候选 A

候选 A 只验证新 Gold 与 Gold 内部来源重平衡，不改 loss、增强或 Gold/pseudo 总比例。r2 已经否定“只提高 Gold 总比例”；本候选保持 `gold_fraction=0.35`，但让新困难数据获得足够抽样权重：

```yaml
profile: data_only
gold_fraction: 0.35
target_gold_weight:
  dragon_gold: 0.40
  negative_removed_gold: 0.15
  disagreement_gold: 0.30
  new_recorded_gold: 0.15
```

同一 role 下的旧、新多轮 source 先合并去重，再在 role 内按来源规模开平方分配；稀有 cell 仍受重复率上限保护。程序把目标权重、重归一化结果和实际 draw 写入 resolved config/report，人工不手算。

```bash
make finetune-smoke \
  HAND_FINETUNE_ID=v2-finetune-r3 \
  FINETUNE_EXPERIMENT_ID=v2-finetune-r3a \
  FINETUNE_PROFILE=data_only

make finetune-train \
  HAND_FINETUNE_ID=v2-finetune-r3 \
  FINETUNE_EXPERIMENT_ID=v2-finetune-r3a \
  FINETUNE_PROFILE=data_only

make eval-val-finetune FINETUNE_EXPERIMENT_ID=v2-finetune-r3a
make infer-finetune FINETUNE_EXPERIMENT_ID=v2-finetune-r3a
```

## 6. Day 2：结构 loss、可选增强、导出与上板

### 6.1 先自动比较候选 A

```bash
make compare-finetune-runs \
  BASELINE_FINETUNE_ID=v2-finetune-r1 \
  CANDIDATE_FINETUNE_ID=v2-finetune-r3a
```

人工只查看 summary 和自动生成的改善/退化 overlay。

### 6.2 候选 B：结构 loss

同一 frozen r3 数据，保持候选 A 的 Gold 35%、Gold 内部权重和原增强，只对 Gold positive 增加结构 loss；pseudo replay 继续参与原 landmarks/presence/handedness 目标，但不参与新增结构项：

```bash
make finetune-smoke \
  HAND_FINETUNE_ID=v2-finetune-r3 \
  FINETUNE_EXPERIMENT_ID=v2-finetune-r3b \
  FINETUNE_PROFILE=structure

make finetune-train \
  HAND_FINETUNE_ID=v2-finetune-r3 \
  FINETUNE_EXPERIMENT_ID=v2-finetune-r3b \
  FINETUNE_PROFILE=structure

make eval-val-finetune FINETUNE_EXPERIMENT_ID=v2-finetune-r3b
make infer-finetune FINETUNE_EXPERIMENT_ID=v2-finetune-r3b
make compare-finetune-runs \
  BASELINE_FINETUNE_ID=v2-finetune-r3a \
  CANDIDATE_FINETUNE_ID=v2-finetune-r3b
```

### 6.3 候选 C：仅在需要时运行

只有 B 的塌缩减少且 Val 没有明显退化，但仍表现出 ROI 偏移敏感时，才运行 `structure_roi_aug`。建议 moderate 参数：rotation `±10°`、scale `0.90～1.10`、translation `0.05`。

若 B 没有任何收益，不运行 C，避免在坏方向上继续组合。

### 6.4 两天内的晋级标准

最低要求：

1. Val mean pixel 必须低于当前最好 geometry 的 `22.2212 px`；目标为 `≤21.55 px`；
2. P90 不高于 `39.57 px`，优选 `≤38.5 px`；
3. PCK@0.10 优选 `≥32%`；
4. presence positive recall 不低于 `94%`；
5. handedness accuracy 不低于 `74%`；
6. 固定 infer 有检测图片不低于 135/140；
7. spread `<0.10` 至少比 r1 的 97 减少 15%，并由人工确认不是错误拉伸；
8. Peak/Soar 不能只改善一个来源、显著破坏另一个；
9. ONNX 和厂商工具链必须通过。

若没有候选超过 geometry，不为了“用了 Gold”而强行选 finetune；保留 geometry、multitask、r1 和 r2 作为 fallback，由固定 Val、固定 infer 和上板目标共同选择。

### 6.5 导出与上板

对两天内最优候选运行：

```bash
make export-finetune FINETUNE_EXPERIMENT_ID=<best-candidate-id>
```

程序检查 ONNX parity、A1 算子、大小、training provenance 和转换数据。人工负责厂商工具链转换和固定场景上板：正面张掌、握拳、侧向张掌、数字“1”、左右手。

方案、checkpoint、threshold、Palm/ROI、ONNX 和后处理全部冻结后，才运行一次：

```bash
make eval-test-finetune FINETUNE_EXPERIMENT_ID=<best-candidate-id>
```

## 7. 超时与降级方案

- 程序修改超过第 1 天前半天仍未通过测试：停止新增人工标注，优先交付现有最佳候选。
- 标注人数或可用时间不足：必须在导出 CVAT 包之前选择 600 预算，不生成无法完成的 800 张任务。
- 已选择 800 后个别成员临时退出：优先重新分配未开始的 job；仍无法完成时，不得把空白图冒充 negative，应停止该 source 的 strict import，并回到程序重新生成较小的新 source ID。
- 新录制有效 ROI 不足：程序用不重复 disagreement 补足，但总数仍不超过冻结的 600/800 预算。
- 候选 A 无改善：不再增加 Gold 比例，直接测试一次结构 loss。
- 候选 B 无改善：停止结构/增强路线，不运行候选 C。
- 所有新候选均未超过 geometry：保持现有 fallback，不在最后数小时修改模型结构。

## 8. 最短人工清单

- [ ] 等待两仓库修改、测试、提交并同步完成；
- [ ] 运行新命令，查看最多 40 张错误 overlay；
- [ ] 录制 20～30 分钟困难手势；
- [ ] 按实际人手在导出前冻结 600 或 800 预算；
- [ ] 共同标 10 张校准图，再按约 100 张/job 分工，团队总量绝对不超过 800；
- [ ] 负责人抽查每位成员约 5% 的标注；
- [ ] 放回两个 `reviewed.xml`；
- [ ] 查看自动 gate/compare，不手工整理数据；
- [ ] 对最优候选做厂商转换和固定手势上板；
- [ ] 冻结后运行一次 Test。
