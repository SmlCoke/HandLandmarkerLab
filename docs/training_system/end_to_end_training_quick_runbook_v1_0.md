# HLMF + HLML 完整训练简易操作手册

版本：1.0

日期：2026-07-16

用途：在已经理解[完整训练流程 v1.0](end_to_end_training_workflow_v1_0.md)后，按本手册直接执行日常操作。

完整流程文档负责解释原理、术语、数据契约和异常处理；本手册只保留执行顺序、命令、关键输出与停止条件。两者冲突时，以完整流程文档为准。

## 0. 开始前先看这八条

1. 当前 `v2-pretrain-r3` 的 HLMF pretrain、HLML curate、geometry smoke、geometry 正式训练、Val 和独立推理已经完成，不要重跑。
2. `make pretrain-curate-reviewed` 只读取 `negative_reviewed`；上传后必须保持目录层级和 PNG SHA 不变，任何额外归档、符号链接或改写都会被拒绝。
3. 正式 multitask 前必须通过 `make check-multitask-data`；程序会按真实负例数量解析 epoch size，并限制 cell 平均重复与单行最大期望重复。
4. 本文列出的接口已经实现；首次部署或更新后仍必须先通过 `make compile && make test-unit`，再处理真实数据。
5. 调参只看固定 Val 和固定推理样例；Test 只在最终 checkpoint、阈值和方案全部冻结后运行一次。
6. 不手写或修改候选 ID、CSV、JSONL、SHA、review decision 或最终训练标签；不靠删行绕过 gate。
7. HLMF 与 HLML 不同时写同一输出。共享物理根相同，但 HLMF 使用 `HAND_DATA_ROOT`，HLML 使用 `HAND_TRAIN_ROOT`。
8. 当前 Val/Test 没有 negative，只能公平比较 landmarks，不能用其 `hand_flag_accuracy` 证明背景拒绝能力。

状态标记：

| 标记 | 含义 |
|---|---|
| `[现有]` | 当前仓库已支持 |
| `[已实现]` | 代码、配置和测试已落地，满足数据前置条件后可以运行 |
| `[r3 已完成]` | 当前实验已经做完，直接跳过 |
| `[可选]` | 时间不足时可以不做 |

## 1. 固定路径和实验 ID

当前 B 阶段已经存在的固定值：

```bash
export HAND_TRAIN_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0
export HAND_DATA_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0
export HAND_PRETRAIN_ID=v2-pretrain-r3
```

finetune ID 默认为 `v2-finetune-r1`。HLML Makefile 已独立定义并导出 `HAND_FINETUNE_ID`；`make paths` 会同时打印 pretrain 与 finetune 两个 ID。

仓库：

```text
HLMF  /root/HandLandmarksFab       conda: anfab
HLML  /root/HandLandmarkerLab      conda: hand-landmarker-tf29
```

确认 HLML Makefile 顶部同时存在：

```make
HAND_PRETRAIN_ID := v2-pretrain-r3
HAND_FINETUNE_ID ?= v2-finetune-r1
```

然后运行：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
make paths
make compile
make test-unit
```

`make paths` 打印的 PRETRAIN_ID、FINETUNE_ID 或数据根不正确时立即停止。它还会打印 pretrain curated/run 与 finetune workspace/run 根目录。

## 2. 阶段 A：从新数据制作到 geometry `[r3 已完成]`

当前 r3 直接跳到第 3 节。以后新建 pretrain ID 时才执行本节。

### A1. HLMF 制作一个普通 pseudo Train source `[现有]`

原始图片必须与 Val/Test/inference 按完整采集 session 隔离。

```bash
cd /root/HandLandmarksFab
conda activate anfab
export HAND_DATA_ROOT=/path/to/one_source

make validate_images_train
make palm_detection_train
make build_roi_train
make run_mediapipe_train
```

确认 source 中存在：

```text
02_roi_crops/images/
02_roi_crops/hand_roi_crops_manifest.jsonl
02_roi_crops/hand_landmarks_autolabel_draft.jsonl
qc/
```

把这些训练产物按 source 目录复制到：

```text
/root/autodl-tmp/TrainFab/HLML-2.0/train_sources/<dataset>/
```

不要重新保存 PNG，不要手工改 JSONL 中的旧绝对路径。

### A2. HLMF 聚合 pretrain `[现有]`

在 HLMF `configs/finalize_train.yaml` 为每个 source 登记唯一 `dataset_id`，然后：

```bash
cd /root/HandLandmarksFab
conda activate anfab
export HAND_DATA_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0
make finalize_train_pretrain
```

只在以下条件全部满足时继续：

```text
train_pretrain_merged/qc/finalize_train_pretrain_report.json
status = ok
fatal_errors = []
missing images = 0
每个预期 source 的 included > 0
```

### A3. HLML curate 与 geometry `[现有]`

先为新实验设置新的 `HAND_PRETRAIN_ID`，不能覆盖旧实验。

```bash
cd /root/HandLandmarkerLab
git pull
conda activate hand-landmarker-tf29

make paths
make compile
make test-unit
make pretrain-curate
make test
make doctor
make inspect-geometry
make pretrain-geometry-smoke
make pretrain-geometry
make eval-val-geometry
```

先停下来检查 Val，并冻结本次候选 checkpoint/方案；Val 报告不完整或指标明显异常时不要继续。确认后再运行：

```bash
make infer-geometry
make export-geometry
```

继续条件：

```text
hand_landmarker_runs/<PRETRAIN_ID>/geometry/training_report.json       status=complete
hand_landmarker_runs/<PRETRAIN_ID>/geometry/experiment_metadata.json  status=complete
checkpoints/best.weights.h5 存在
固定推理样例没有明显关键点爆炸或全塌缩
```

不要在此时运行 Test；保留 geometry best 和 ONNX 作为 fallback。

## 3. 阶段 B：导入人工负例并训练 multitask

这是当前 `v2-pretrain-r3` 的实际起点。

### B1. 打包并上传 `negative_reviewed`

本地只保留人工确认“没有手”的 1,049 张 PNG。不要包含两个 ZIP。

```powershell
Set-Location "D:\CICIEC\MediaPipe\Trainfab\HLML-2.0\negative_candidates\negative_candidates"
7z a -t7z ..\v2-pretrain-r3-negative-reviewed.7z .\* -xr!*.zip
7z t ..\v2-pretrain-r3-negative-reviewed.7z
```

可经夸克网盘中转。把 7z 上传到服务器后：

```bash
REVIEW_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0/hand_landmarker_reviews/v2-pretrain-r3
ARCHIVE=/root/autodl-tmp/v2-pretrain-r3-negative-reviewed.7z

7z t "$ARCHIVE"
test ! -e "$REVIEW_ROOT/negative_reviewed"
mkdir -p "$REVIEW_ROOT/negative_reviewed"
7z x "$ARCHIVE" -o"$REVIEW_ROOT/negative_reviewed"
find "$REVIEW_ROOT/negative_reviewed" -type f -iname '*.png' | wc -l
find "$REVIEW_ROOT/negative_reviewed" -type f ! -iname '*.png' -print
```

第一个 `find` 必须输出 `1049`，第二个不得输出任何路径。否则停止，不运行 curate。

不要人工删除服务器原 `negative_candidates`，不要手工创建 `negative_removed`、`negative_quarantine` 或 decision 文件。

### B2. 导入、分区和 gate `[已实现]`

先确认服务器已拉取包含事务复核和自动 epoch-size gate 的最新代码，并通过 `make compile && make test-unit`，再运行：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29

make paths
make pretrain-curate-reviewed
make check-multitask-data
make inspect-multitask
```

报告必须得到：

```text
原候选                 48,643
人工上传                 1,049
进入 multitask           1,022
negative_quarantine         27
negative_removed         47,594
resolved epoch size      约 6,400
Runtime negative 平均重复 约 3.90625 次/epoch
```

重点文件：

```text
hand_landmarker_reviews/v2-pretrain-r3/review_report.json
hand_landmarker_reviews/v2-pretrain-r3/negative_removed_manifest.jsonl
train_pretrain_curated/v2-pretrain-r3/qc/curation_report.json
hand_landmarker_runs/v2-pretrain-r3/multitask_data_gate.json
```

数量不守恒、SHA 不一致、gate 非 pass 或 epoch size 仍接近 60,974 时停止。

### B3. Multitask 训练 `[已实现]`

```bash
make check-multitask-data
make inspect-multitask
make pretrain-multitask
make eval-val-multitask
make infer-multitask
make export-multitask
```

接受 multitask 的最低条件：

- `training_report.json` 和 `experiment_metadata.json` 均为 `complete`；
- 从 r3 geometry best 初始化；
- Val landmark mean/P90 相比 geometry 的相对退化不超过约 3%；
- 固定推理样例的 landmarks 没有明显恶化；
- `hand_flag` 对假 ROI 的拒绝有改善。

若不满足，保留 geometry best，不继续把差的 multitask 当 finetune 起点。

## 4. 阶段 C：并行准备 finetune 数据 `[已实现]`

Multitask 训练时即可并行做本节。

### C1. 先设置人工预算

通常只调整以下两个配置文件中的少量字段：

```text
configs/prepare_finetune_sources.yaml:
  selection.negative_removed.enabled/max_items/per_dataset_max
  selection.teacher_student.enabled/max_items/per_dataset_max
  selection.pretrain_replay.max_records

configs/curate_finetune.yaml:
  sources.<gold_role>.target_gold_weight
```

默认预算：

```text
b negative_removed       最多 300 ROI
c teacher-student 分歧   最多 300 ROI
d pretrain replay        最多 10,000 条

Gold role 目标权重：
Dragon 60% / b 20% / c 15% / e 5%
```

不要手写候选 ID、TXT、CSV 或 JSONL。

### C2. Dragon Gold `[已实现]`

```bash
cd /root/HandLandmarksFab
conda activate anfab

make prepare_dragon_gold \
  DRAGON_RAW_ROOT=/path/to/HandViolenceEnhanced0716/dragon \
  HAND_DATA_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0 \
  HAND_FINETUNE_ID=v2-finetune-r1
```

只在报告得到以下固定统计时接受：

```text
qc/gold_source_report.json: counts.matched_rois = 5,191
qc/gold_source_report.json: counts.included     = 5,189
qc/gold_source_report.json: counts.ignored      = 2
finetune_source.json: handedness_policy = unavailable
训练行：handedness mask/loss weight = 0
```

同时必须确认：4,093 张未被 Hand 标注引用的图片没有进入 Gold；850 张 `p=0` 只进入 reject audit，没有被制造成 negative。

人工只查看程序生成的 64 张 overlay；若方向、Palm→Hand 对应或投影有系统错误，停止。

默认结果目录：

```text
finetune/v2-finetune-r1/sources/gold/dragon_gold_0716_v1/finetune_source.json
finetune/v2-finetune-r1/sources/gold/dragon_gold_0716_v1/qc/gold_source_report.json
finetune/v2-finetune-r1/sources/gold/dragon_gold_0716_v1/qc/overlays/
```

### C3. 自动生成 b/c/d `[已实现]`

```bash
# 冻结 r3 只需发布一次父源索引；不会改写 pretrain labels
cd /root/HandLandmarksFab
conda activate anfab
make build_pretrain_source_registry \
  HAND_DATA_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0 \
  HAND_PRETRAIN_ID=v2-pretrain-r3

cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
make paths
make prepare-finetune-sources
```

确认 `make paths` 打印 `HAND_PRETRAIN_ID=v2-pretrain-r3` 与 `HAND_FINETUNE_ID=v2-finetune-r1` 后再继续。

程序应自动产生：

```text
finetune/v2-finetune-r1/mining/                 b/c selection request
finetune/v2-finetune-r1/sources/replay/         d replay
```

然后让 HLMF 生成 b/c CVAT task：

b/c 已有原 ROI、manifest 和 teacher draft，不要重新运行 HLMF 03。

```bash
cd /root/HandLandmarksFab
conda activate anfab

make export_finetune_gold \
  HAND_DATA_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0 \
  HAND_FINETUNE_ID=v2-finetune-r1 \
  FINETUNE_SOURCE_ID=negative_removed_gold \
  FINETUNE_SOURCE_MODE=selection_subset

make export_finetune_gold \
  HAND_DATA_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0 \
  HAND_FINETUNE_ID=v2-finetune-r1 \
  FINETUNE_SOURCE_ID=disagreement_gold \
  FINETUNE_SOURCE_MODE=selection_subset
```

固定身份和类型如下；不要另传不存在的 `FINETUNE_SOURCE_KIND`：

| 候选 | `FINETUNE_SOURCE_ID` / 请求目录 | 请求内嵌 `source_kind` |
|---|---|---|
| b | `negative_removed_gold` | `reviewed_hard_gold` |
| c | `disagreement_gold` | `disagreement_gold` |

检查以下文件后再进入 CVAT：

```text
train_pretrain_merged/qc/pretrain_source_registry.jsonl
train_pretrain_merged/qc/pretrain_source_registry_report.json
finetune/v2-finetune-r1/mining/negative_removed_gold/selection_request.jsonl
finetune/v2-finetune-r1/mining/disagreement_gold/selection_request.jsonl
finetune/v2-finetune-r1/cvat/negative_removed_gold/task_descriptor.json
finetune/v2-finetune-r1/cvat/disagreement_gold/task_descriptor.json
```

### C4. 人工完成 b/c CVAT

每张图片必须明确选择一种状态：

1. 21 点 skeleton + Left；
2. 21 点 skeleton + Right；
3. 21 点 skeleton + unknown handedness；
4. 显式 `no_hand`；
5. `ignore_for_training`。

一个 positive 只能选择 Left/Right/unknown 之一。不能把“什么都没标”当 negative。

把每个 task 的 CVAT for images 1.1 XML 放回 descriptor 指定的 `reviewed.xml` 后：

```bash
cd /root/HandLandmarksFab
conda activate anfab
export HAND_DATA_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0
export HAND_FINETUNE_ID=v2-finetune-r1

make import_finetune_gold
```

只有 strict import 全部成功后，b/c 才会发布到：

```text
finetune/v2-finetune-r1/sources/gold/
```

此时还没有最终 Gold aggregate；统一在 C6 生成。

### C5. 新录制 e `[可选；已实现]`

时间不足可以跳过。若执行，先用独立 raw root 跑 HLMF 00～03：

```bash
cd /root/HandLandmarksFab
conda activate anfab
E_SOURCE_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0/finetune/v2-finetune-r1/raw/new_recorded_gold_v1
export HAND_DATA_ROOT="$E_SOURCE_ROOT"

make validate_images_train
make palm_detection_train
make build_roi_train
make run_mediapipe_train
```

再走 finetune strict CVAT，而不是普通 Train import：

```bash
export HAND_DATA_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0
export HAND_FINETUNE_ID=v2-finetune-r1

make export_finetune_gold \
  FINETUNE_SOURCE_MODE=native_existing \
  FINETUNE_RAW_SOURCE_ROOT="$E_SOURCE_ROOT" \
  FINETUNE_SOURCE_ID=new_recorded_gold_v1

# 人工完成 CVAT 并放回 reviewed.xml 后：
make import_finetune_gold FINETUNE_SOURCE_ID=new_recorded_gold_v1
```

### C6. 聚合全部 Gold `[已实现；始终执行]`

完成本次实际启用的 a/b/c/e 后，无论其中哪些可选来源缺失，都统一运行一次：

```bash
cd /root/HandLandmarksFab
conda activate anfab
export HAND_DATA_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0
export HAND_FINETUNE_ID=v2-finetune-r1
make finalize_train_finetune
```

确认输出存在：

```text
finetune/v2-finetune-r1/hmlf_gold_merged/hmlf_gold_aggregate.json
finetune/v2-finetune-r1/hmlf_gold_merged/05_labels/
finetune/v2-finetune-r1/hmlf_gold_merged/qc/finalize_train_finetune_report.json
```

只有实际存在的 Gold source 全部通过 strict gate 后才能进入阶段 D。

## 5. 阶段 D：聚合 finetune 训练集 `[已实现]`

允许 a～e 中部分可选 Gold 来源缺失；但已经存在的 source 必须完整通过严格检查。d replay 必须存在。

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29

make paths
make finetune-curate
make check-finetune-data
make inspect-finetune
```

`make finetune-curate` 会先自动执行来源门禁。必须检查：

```text
finetune/v2-finetune-r1/qc/finetune_sources_gate.json         status=ok
hand_landmarker_runs/v2-finetune-r1/finetune_data_gate.json  status=ok
```

继续条件：

- 至少 1 个有效 Gold source；
- 至少 256 个 Gold positive；
- d replay 存在并同时保留受控 pseudo positive；其中所有 negative 都必须是带完整人工 review decision 的 confirmed negative；
- source descriptor、标签、图片和 SHA 一致；
- Gold 与 replay 重合时只保留 Gold；
- 与固定 Val/Test 无 ID、source group、图片 SHA 或归一化像素 SHA 泄漏；
- Gold `no_hand` 与 pseudo confirmed negative 分别具有正确的人工证据；
- epoch 级 sampling plan 可行。

关键输出：

```text
train_finetune_merged/v2-finetune-r1/
├── 05_labels/hand_training_labels_finetune.jsonl
├── 05_labels/hand_training_labels_finetune_smoke.jsonl
├── audit/finetune_smoke_selection.jsonl
└── qc/sha256_manifest.json
```

任何 gate 失败都停止，不手工删 JSONL 行绕过。

## 6. 阶段 E：Finetune smoke、正式训练与交付 `[已实现]`

### E1. Smoke

```bash
make finetune-smoke
make check-finetune-smoke
```

Smoke 必须：

- 使用程序持久化并认证的 256 行；
- 同时认证正式 full config、初始 multitask checkpoint 和 curation manifest；
- landmarks、hand flag、handedness 三类有效监督通过固定数值门禁；
- required 或 effective quota>0 的 cell 都实际被抽到；
- 可选且缺失的 Gold negative 只记录 `not_applicable/redistributed`。

固定硬门槛：

```text
landmark mean MAE ≤ 0.02
landmark P90 MAE  ≤ 0.04
landmark max MAE  ≤ 0.10
hand flag BCE     ≤ 0.08，accuracy ≥ 0.98
handedness BCE    ≤ 0.15，accuracy ≥ 0.95（只统计有效 mask）
```

Smoke 失败时不能启动 full train。

### E2. 正式训练、Val、推理与导出

```bash
make finetune-train
make eval-val-finetune
make infer-finetune
make export-finetune
make conversion-data-finetune
```

不要手工设置通用配置内部使用的路由变量。上述 `*-finetune` Make 目标会自动使用 `HAND_FINETUNE_ID`、`run phase=finetune`、`checkpoint stage=finetune` 和 `configs/train_finetune.yaml`；geometry/multitask 目标则自动使用 `HAND_PRETRAIN_ID` 与各自 phase。`make finetune-train` 在 full trainer 前会复核数据、Train/Val/Test inspection 和现有 smoke gate，不会隐式重训 smoke。

输出：

```text
hand_landmarker_runs/v2-finetune-r1/finetune/
hand_landmarker_runs/v2-finetune-r1/eval/finetune/val/
hand_landmarker_inference/v2-finetune-r1/finetune/
hand_landmarker_runs/v2-finetune-r1/export/finetune/
```

使用 Val 与固定推理样例比较 geometry、multitask、finetune。最终交付使用 Val 选出的 best checkpoint，不默认使用 last。

### E3. 冻结后只运行一次 Test

先固定：

```text
FINETUNE_ID
checkpoint
hand_flag threshold
所有推理后处理
ONNX/export 配置
```

再执行：

```bash
make eval-test-finetune
```

不要根据 Test 结果返回修改训练参数；若需要再实验，创建新的 finetune ID，并仍用 Val 选方案。

## 7. 每阶段只看这些文件

| 阶段 | 首要报告/输出 |
|---|---|
| HLMF pretrain finalize | `train_pretrain_merged/qc/finalize_train_pretrain_report.json` |
| HLML pretrain curate | `train_pretrain_curated/<PRETRAIN_ID>/qc/curation_report.json` |
| 负例导入 | `hand_landmarker_reviews/<PRETRAIN_ID>/review_report.json` |
| Multitask gate | `hand_landmarker_runs/<PRETRAIN_ID>/multitask_data_gate.json` |
| Geometry/Multitask | 对应 run 的 `training_report.json`、`history.json`、`checkpoints/best.weights.h5` |
| Dragon | source 的 adapter/QC report 与 64 张 overlay |
| Pretrain source registry | `train_pretrain_merged/qc/pretrain_source_registry_report.json` |
| b/c 请求 | `finetune/<FINETUNE_ID>/mining/<source_id>/selection_report.json` |
| Gold aggregate | `finetune/<FINETUNE_ID>/hmlf_gold_merged/hmlf_gold_aggregate.json` |
| Finetune source gate | `finetune/<FINETUNE_ID>/qc/finetune_sources_gate.json` |
| Finetune curate | `train_finetune_merged/<FINETUNE_ID>/qc/curation_report.json`、`sha256_manifest.json` |
| Finetune data gate | `hand_landmarker_runs/<FINETUNE_ID>/finetune_data_gate.json` |
| Finetune smoke | smoke run 的 `smoke_gate_report.json` |
| Finetune full | `hand_landmarker_runs/<FINETUNE_ID>/finetune/training_report.json` |

## 8. 任何一项出现就立即停止

- ID 或数据根与预期不同；
- 目标 run/source 目录已经存在且命令可能覆盖；
- 输入、manifest、descriptor 或图片 SHA 不一致；
- 数量不守恒；
- 非可选 source 缺失，或已存在的可选 source 内部损坏；
- `fatal_errors` 非空、gate 非 pass、训练状态不是 complete；
- Train 与 Val/Test 发生任何身份或图片泄漏；
- smoke 与当前 full config/checkpoint/manifest 不一致；
- 输出出现 NaN、关键点重新“漫天飞舞”或大面积塌缩；
- 想通过手工改 JSONL、复制 ID 或删除错误行绕过程序门禁。

遇到停止条件时，回到[完整训练流程 v1.0](end_to_end_training_workflow_v1_0.md)查看对应章节，不自行猜测修复。

## 9. 当前最短执行清单

- [x] `negative_reviewed/removed/quarantine` 事务导入与测试已完成；
- [x] multitask 重复率保护与测试已完成；
- [ ] 上传 1,049 张 PNG；
- [ ] 运行 review finalize、gate、inspect；
- [ ] 训练并评估 multitask；
- [ ] Multitask 训练期间并行准备 Dragon 与 b/c/d；
- [ ] 人工只处理 b/c CVAT；
- [ ] HLMF strict import 与 Gold aggregate；
- [ ] HLML finetune-curate、gate、inspect；
- [ ] finetune smoke；
- [ ] finetune full、Val、固定推理和导出；
- [ ] 冻结后运行一次 Test。
