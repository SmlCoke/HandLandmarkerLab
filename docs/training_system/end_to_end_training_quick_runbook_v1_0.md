# HLMF + HLML 端到端训练快速操作手册

版本：1.0

用途：作为 HLMF 数据制作、HLML pretrain/finetune、评估和导出的最高级操作入口。

本文只保留执行顺序、命令、文件摆放、结果目录和停止条件。术语、契约和异常处理见[完整训练流程](end_to_end_training_workflow_v1_0.md)；当前服务器、数据集和实验进度见[当前训练与数据状态](current_training_status.md)。实际依赖以两个仓库各自的 Makefile 为准。

## 1. 流程与分工

```text
原图
  → HLMF 制作 pseudo source
  → HLMF 聚合 pretrain
  → HLML curate + geometry
  → 人工删除式复核 negative
  → HLML multitask
  → HLML 自动选择 Gold 候选和 replay
  → HLMF CVAT 导出/导入 + Gold 聚合
  → HLML finetune curate + smoke + full
  → Val + 固定 infer + ONNX
  → 冻结方案后运行一次 Test
```

| 参与者 | 工作 |
|---|---|
| HLMF | 原图、Palm、Hand ROI、MediaPipe 草稿、CVAT、Gold 聚合 |
| HLML | curate、训练、评估、推理、困难样本选择、ONNX |
| 人工 | 录制/搬运、负例删除复核、CVAT 精标、查看报告和上板 |

人工不写候选 CSV/JSONL、review decision、SHA 或最终训练标签。

## 2. 路径与实验 ID

本文使用：

```text
<DATA_ROOT>    中央数据根
<PRETRAIN_ID>  一次 geometry + multitask 的唯一 ID
<FINETUNE_ID>  一次 finetune 数据与训练的唯一 ID
<SOURCE_ID>    一个数据来源的唯一 ID
```

服务器仓库：

```text
HLMF  /root/HandLandmarksFab    conda: anfab
HLML  /root/HandLandmarkerLab   conda: hand-landmarker-tf29
```

每次正式实验前，在 HLML Makefile 顶部设置并提交：

```make
HAND_TRAIN_ROOT := <DATA_ROOT>
HAND_PRETRAIN_ID := <PRETRAIN_ID>
HAND_FINETUNE_ID ?= <FINETUNE_ID>
```

新实验必须使用新 ID，不覆盖旧目录。然后：

```bash
cd /root/HandLandmarkerLab
git pull
conda activate hand-landmarker-tf29
make paths
make compile
make test-unit
make doctor
```

`make paths` 不正确时停止。

中央数据布局：

```text
<DATA_ROOT>/
├── train_sources/                 # HLMF pseudo sources
├── eval_sources/                  # 固定 Val/Test sources
├── train_pretrain_merged/         # HLMF pretrain 聚合
├── train_pretrain_curated/        # HLML pretrain 快照
├── hand_landmarker_reviews/       # negative 人工复核
├── finetune/                       # Gold/CVAT/mining/replay
├── train_finetune_merged/         # HLML finetune 快照
├── val_merged/                     # 固定 Val
├── test_merged/                    # 锁定 Test
├── hand_landmarker_runs/          # 训练/评估/导出
└── hand_landmarker_inference/     # 固定 infer
```

## 3. HLMF 制作 pretrain 数据

Train、Val、Test、固定 infer 必须按完整录制 session 隔离。

### 3.1 制作一个 pseudo Train source

```bash
cd /root/HandLandmarksFab
conda activate anfab
export HAND_DATA_ROOT=/path/to/raw-source

make validate_images_train
make palm_detection_train
make build_roi_train
make run_mediapipe_train
```

确认存在：

```text
02_roi_crops/images/
02_roi_crops/hand_roi_crops_manifest.jsonl
02_roi_crops/hand_landmarks_autolabel_draft.jsonl
qc/
```

把完整 source 放到：

```text
<DATA_ROOT>/train_sources/<SOURCE_ID>/
```

可以压缩搬运，但不能重保存、改名或手改 JSONL/SHA。

### 3.2 聚合 pretrain

在 HLMF `configs/finalize_train.yaml` 为每个 source 注册唯一 `dataset_id`，然后：

```bash
cd /root/HandLandmarksFab
conda activate anfab
make finalize_train_pretrain \
  HAND_DATA_ROOT=<DATA_ROOT> \
  HAND_PRETRAIN_ID=<PRETRAIN_ID>
```

检查：

```text
train_pretrain_merged/05_labels/hand_training_labels_pretrain.jsonl
train_pretrain_merged/qc/finalize_train_pretrain_report.json
```

报告必须 `status=ok`，无 fatal/missing，预期 source 均有 included 记录。

## 4. HLML curate 与 geometry

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29

make paths
make pretrain-curate
make test
make inspect-geometry
make pretrain-geometry-smoke
make pretrain-geometry
make eval-val-geometry
```

关键结果：

```text
train_pretrain_curated/<PRETRAIN_ID>/
hand_landmarker_reviews/<PRETRAIN_ID>/negative_candidates/
hand_landmarker_runs/<PRETRAIN_ID>/geometry/training_report.json
hand_landmarker_runs/<PRETRAIN_ID>/geometry/checkpoints/best.weights.h5
hand_landmarker_runs/<PRETRAIN_ID>/eval/geometry/val/metrics.json
```

Smoke、full、Val 和固定样例均正常后：

```bash
make infer-geometry
make export-geometry
```

调参期间不运行 Test。

## 5. 人工负例复核与 multitask

### 5.1 删除式复核

把 `negative_candidates/` 压缩并下载到本地，解压后：

- 看到手、手指、手腕或疑似手部就删除；
- 模糊、遮挡、过暗、过曝、截断或无法判断也删除；
- 只保留明确没有手的 ROI；
- 不修改、重保存或重命名保留图片。

保持相对目录不变，把保留结果重新压缩、上传并解压到：

```text
<DATA_ROOT>/hand_landmarker_reviews/<PRETRAIN_ID>/negative_reviewed/
```

归档文件不能留在该目录；服务器原 `negative_candidates/` 在程序事务提交前不得人工清理。

```bash
REVIEW_ROOT=<DATA_ROOT>/hand_landmarker_reviews/<PRETRAIN_ID>
find "$REVIEW_ROOT/negative_reviewed" -type f ! -iname '*.png' -print
find "$REVIEW_ROOT/negative_reviewed" -type f -iname '*.png' | wc -l
```

第一条不得输出文件，第二条应与本地保留数一致。

### 5.2 导入复核并训练 multitask

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29

make pretrain-curate-reviewed
make check-multitask-data
make inspect-multitask
make pretrain-multitask
make eval-val-multitask
make infer-multitask
make export-multitask
```

检查：

```text
hand_landmarker_reviews/<PRETRAIN_ID>/review_report.json
train_pretrain_curated/<PRETRAIN_ID>/qc/curation_report.json
hand_landmarker_runs/<PRETRAIN_ID>/multitask_data_gate.json
hand_landmarker_runs/<PRETRAIN_ID>/multitask/training_report.json
hand_landmarker_runs/<PRETRAIN_ID>/eval/multitask/val/metrics.json
```

复核事务未提交、数量不守恒、SHA 不一致或 gate 非 `pass` 时停止。

## 6. 准备 finetune 数据

Finetune 来源：

| 角色 | 来源 | 人工工作 |
|---|---|---|
| a | Dragon Gold | 查看 overlay |
| b | `negative_removed` 困难样本 | CVAT |
| c | teacher–student 分歧 | CVAT |
| d | pretrain replay | 无，程序自动 |
| e | 新录制数据 | 录制 + CVAT，可选 |

至少一个 Gold source 和 replay 必须存在。

### 6.1 发布 source registry

```bash
cd /root/HandLandmarksFab
conda activate anfab
make build_pretrain_source_registry \
  HAND_DATA_ROOT=<DATA_ROOT> \
  HAND_PRETRAIN_ID=<PRETRAIN_ID>
```

结果：

```text
train_pretrain_merged/qc/pretrain_source_registry.jsonl
train_pretrain_merged/qc/pretrain_source_registry_report.json
```

### 6.2 可选导入 Dragon

Dragon 根目录直接包含 `images/`、`annotations_hand.txt` 和 `annotations_palm.txt`：

```bash
make prepare_dragon_gold \
  DRAGON_RAW_ROOT=/path/to/dragon \
  HAND_DATA_ROOT=<DATA_ROOT> \
  HAND_FINETUNE_ID=<FINETUNE_ID>
```

查看 `finetune/<FINETUNE_ID>/sources/gold/<dragon-id>/qc/overlays/`；方向、Palm→Hand、ROI 或投影有系统错误时停止。

### 6.3 HLML 自动选择 b/c/d

在 `configs/prepare_finetune_sources.yaml` 设置 b/c 人工上限和 replay 上限，然后：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
make paths
make prepare-finetune-sources
```

程序生成：

```text
finetune/<FINETUNE_ID>/mining/negative_removed_gold/
finetune/<FINETUNE_ID>/mining/disagreement_gold/
finetune/<FINETUNE_ID>/sources/replay/pretrain_replay/
```

### 6.4 HLMF 导出 b/c CVAT

```bash
cd /root/HandLandmarksFab
conda activate anfab

make export_finetune_gold \
  HAND_DATA_ROOT=<DATA_ROOT> \
  HAND_FINETUNE_ID=<FINETUNE_ID> \
  FINETUNE_SOURCE_ID=negative_removed_gold \
  FINETUNE_SOURCE_MODE=selection_subset

make export_finetune_gold \
  HAND_DATA_ROOT=<DATA_ROOT> \
  HAND_FINETUNE_ID=<FINETUNE_ID> \
  FINETUNE_SOURCE_ID=disagreement_gold \
  FINETUNE_SOURCE_MODE=selection_subset
```

Task：

```text
finetune/<FINETUNE_ID>/cvat/<source_id>/
├── 02_roi_crops/images/
├── cvat_autolabel.xml
├── task_descriptor.json
└── qc/
```

### 6.5 人工 CVAT 与导入

每张图片明确选择一种：

1. 完整 21 点 + `Left`；
2. 完整 21 点 + `Right`；
3. 完整 21 点 + `unknown_handedness`；
4. `no_hand`；
5. `ignore_for_training`。

不能把空白标注当 negative。无法可靠标注时用 `ignore_for_training`，不要为了门控把真实越界点拉回 ROI。

从 CVAT 导出 `CVAT for images 1.1`，保存为：

```text
finetune/<FINETUNE_ID>/cvat/<source_id>/reviewed.xml
```

导入：

```bash
cd /root/HandLandmarksFab
conda activate anfab
make import_finetune_gold \
  HAND_DATA_ROOT=<DATA_ROOT> \
  HAND_FINETUNE_ID=<FINETUNE_ID>
```

### 6.6 可选新录制 e

新 train-only session 先在独立 raw root 跑 HLMF 00～03，再导出 strict CVAT：

```bash
export HAND_DATA_ROOT=/path/to/new-recorded-raw
make validate_images_train
make palm_detection_train
make build_roi_train
make run_mediapipe_train

make export_finetune_gold \
  HAND_DATA_ROOT=<DATA_ROOT> \
  HAND_FINETUNE_ID=<FINETUNE_ID> \
  FINETUNE_SOURCE_ID=<new-source-id> \
  FINETUNE_SOURCE_MODE=native_existing \
  FINETUNE_RAW_SOURCE_ROOT=/path/to/new-recorded-raw
```

完成人工 CVAT 后，使用同一 source ID 运行 `make import_finetune_gold`。

### 6.7 聚合 Gold

```bash
cd /root/HandLandmarksFab
conda activate anfab
make finalize_train_finetune \
  HAND_DATA_ROOT=<DATA_ROOT> \
  HAND_FINETUNE_ID=<FINETUNE_ID>
```

检查：

```text
finetune/<FINETUNE_ID>/hmlf_gold_merged/hmlf_gold_aggregate.json
finetune/<FINETUNE_ID>/hmlf_gold_merged/05_labels/
finetune/<FINETUNE_ID>/hmlf_gold_merged/qc/finalize_train_finetune_report.json
```

实际存在的任一 source 未通过 strict gate 时停止。

## 7. HLML finetune

### 7.1 聚合快照

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29

make paths
make check-finetune-sources
make finetune-curate
make check-finetune-data
make inspect-finetune
```

结果：

```text
train_finetune_merged/<FINETUNE_ID>/
├── 05_labels/hand_training_labels_finetune.jsonl
├── 05_labels/hand_training_labels_finetune_smoke.jsonl
├── audit/finetune_smoke_selection.jsonl
└── qc/sha256_manifest.json
```

关键 gate：

```text
finetune/<FINETUNE_ID>/qc/finetune_sources_gate.json
hand_landmarker_runs/<FINETUNE_ID>/finetune_data_gate.json
```

### 7.2 Smoke、full、Val、infer、export

```bash
make finetune-smoke
make check-finetune-smoke
make finetune-train
make eval-val-finetune
make infer-finetune
make export-finetune
```

只需独立重建转换数据时：

```bash
make conversion-data-finetune
```

结果：

```text
hand_landmarker_runs/<FINETUNE_ID>/finetune_smoke/
hand_landmarker_runs/<FINETUNE_ID>/finetune/
hand_landmarker_runs/<FINETUNE_ID>/eval/finetune/val/
hand_landmarker_inference/<FINETUNE_ID>/finetune/
hand_landmarker_runs/<FINETUNE_ID>/export/finetune/
```

最终候选使用 Val 选出的 `best.weights.h5`，不默认使用 last。

## 8. 冻结后 Test

先冻结 finetune ID、best checkpoint、hand flag threshold、Palm/ROI/后处理和 ONNX 配置，再运行一次：

```bash
make eval-test-finetune
```

Test 不用于继续调参；需要改变方案时建立新实验并仍用 Val 选择。

## 9. 首要报告

| 阶段 | 报告/产物 |
|---|---|
| Pretrain finalize | `train_pretrain_merged/qc/finalize_train_pretrain_report.json` |
| Pretrain curate | `train_pretrain_curated/<PRETRAIN_ID>/qc/curation_report.json` |
| Negative review | `hand_landmarker_reviews/<PRETRAIN_ID>/review_report.json` |
| Multitask gate | `hand_landmarker_runs/<PRETRAIN_ID>/multitask_data_gate.json` |
| Train | 对应 phase 的 `training_report.json`、`history.json`、`best.weights.h5` |
| Finetune mining | `finetune/<FINETUNE_ID>/mining/<source_id>/selection_report.json` |
| Gold aggregate | `finetune/<FINETUNE_ID>/hmlf_gold_merged/hmlf_gold_aggregate.json` |
| Finetune curate | `train_finetune_merged/<FINETUNE_ID>/qc/curation_report.json`、`sha256_manifest.json` |
| Finetune smoke | `finetune_smoke/smoke_gate_report.json` |
| Val | `eval/<phase>/val/metrics.json`、`predictions.jsonl` |
| Export | `hand_landmarker_v2.onnx`、`hand_landmarker_v2.contract.json` |

## 10. 立即停止条件

- ID、数据根或目标目录不正确；
- 目标 source/run 已存在且可能覆盖；
- manifest、descriptor、图片或 SHA 不一致；
- 数量不守恒，或 fatal/conflict/gate 非预期；
- Train 与 Val/Test 泄漏；
- smoke 与 full config/checkpoint/manifest 不一致；
- 训练不是 `complete`，或输出出现 NaN、爆炸、严重塌缩；
- 必须手工改 JSONL、ID、decision 或 SHA 才能继续。

遇到停止条件时查看[完整训练流程](end_to_end_training_workflow_v1_0.md)，修复根因，不绕过门禁。
