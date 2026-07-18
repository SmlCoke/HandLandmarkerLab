# HLML 最终 Finetune 执行计划

更新时间：2026-07-19。本计划从“geometry/multitask 已完成、两批新录制 Gold 已标完但尚未 import”开始，写到最终评估、推理、导出。通用原理见 [HLML 完整训练流程](HLML_training_workflow.md)。以下命令由人工按顺序执行；程序负责认证、去重、聚合、冻结、训练和报告生成。

本轮采用低人工工作量路线：不再制作新的 disagreement/negative-removed CVAT task，直接使用 600 个已经标完的新录制 ROI、有效 Dragon 0718、两批历史人工 Gold和 mandatory replay。

## 1. 固定环境和 ID

每次新登录先执行：

```bash
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_GOLD_ROOT=/root/autodl-tmp/DatesetFab/GoldSource
export HAND_TRAIN_ROOT=/root/autodl-tmp/TrainFab/HLML-3.0
export HAND_PRETRAIN_ID=v3-pretrain-r1
export HAND_FINETUNE_ID=v3-finetune-final-r1
export FINETUNE_EXPERIMENT_ID=v3-finetune-final-dataonly-r1
export FINETUNE_PROFILE=data_only
```

固定来源决定：

| source ID | 本轮决定 | 原因 |
|---|---|---|
| `disagreement_gold_hlml2.0` | 启用 | 复用历史人工困难样本 Gold |
| `negative_removed_gold_hlml2.0` | 启用 | 复用历史人工困难样本 Gold |
| `new_recorded_gold_0718_r01` | 启用 | 新录制同域，300 ROI |
| `new_recorded_gold_0718_r02` | 启用 | 新录制同域，300 ROI |
| `dragon_gold_0718_v1` | 启用 | 已确认符合部署/评测域 |
| `dragon_gold_0716_v1` | 禁用 | 有损视频/JPEG 域不一致 |

不要复用其他已有 finetune 数据目录或实验目录。如果这组来源要改变，创建新的 `HAND_FINETUNE_ID`；如果只改变训练 profile/超参数，保持数据 ID 不变、换新的 `FINETUNE_EXPERIMENT_ID`。

## 2. 更新代码并运行快速门控

HLMF：

```bash
cd /root/HandLandmarksFab
git pull --ff-only
conda activate anfab
make compile
make test
```

HLML：

```bash
cd /root/HandLandmarkerLab
git pull --ff-only
conda activate hand-landmarker-tf29
make paths
make compile
make test-unit
make doctor
```

任一步失败都先停止，不要在代码/环境门控失败时发布 Gold 或创建训练快照。

## 3. 严格导入两批已标完的新录制 Gold

先确认输入确实存在：

```bash
test -f "$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_0718_r01/task/reviewed.xml"
test -f "$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_0718_r02/task/reviewed.xml"
```

逐批导入，便于报错时准确定位：

```bash
cd /root/HandLandmarksFab
conda activate anfab

make import_finetune_gold \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_SOURCE_ID=new_recorded_gold_0718_r01

make import_finetune_gold \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_SOURCE_ID=new_recorded_gold_0718_r02
```

程序负责：验证 CVAT 完整性、21 点/presence/handedness、ROI 边界和 SHA；生成 `published/`；把 XML/任务描述符移入 `published/audit/`；删除已完成的 `task/`。人工不移动 JSONL，不复制 ROI，不手改 descriptor。

导入后检查：

```bash
test -f "$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_0718_r01/published/finetune_source.json"
test -f "$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_0718_r02/published/finetune_source.json"
test ! -d "$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_0718_r01/task"
test ! -d "$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_0718_r02/task"
```

若 import 报 blocking error，查看对应 `task/qc/` 报告并只修 CVAT XML；不要放宽几何门控或删除整批任务。

## 4. 自动建立 replay 和 disagreement 分数池

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29

make prepare-finetune-sources \
  HAND_PRETRAIN_ID=v3-pretrain-r1 \
  HAND_FINETUNE_ID=v3-finetune-final-r1
```

程序会使用认证后的 pretrain registry 和 curated multitask 标签：保留全部人工确认负样本，再在默认 10000 条总上限内按 `POS_RUNTIME:POS_LOW_PALM = 0.75:0.25` 确定性补充 positives；同时用 geometry student 与 MediaPipe teacher 生成 disagreement 分数池，并排除 GoldSource、pending task、Val/Test 的已占用身份。

本轮不运行 `prepare-finetune-round`，也不把分数池导出到 CVAT。它只是保留给未来分析，不会自行进入训练。

检查：

```bash
test -f "$HAND_TRAIN_ROOT/finetune/v3-finetune-final-r1/sources/replay/pretrain_replay/finetune_source.json"
test -f "$HAND_TRAIN_ROOT/finetune/v3-finetune-final-r1/mining/teacher_student/disagreement_scores.jsonl"
test -f "$HAND_TRAIN_ROOT/finetune/v3-finetune-final-r1/mining/prepare_finetune_sources_report.json"
```

重点查看报告中的 registry/authentication 状态、replay 实际数量、confirmed negative 数量、各 positive 类型/来源分布和 `historical_gold_exclusion`。任何 SHA、缺图或 registry 失败都先停止。

## 5. HLMF 聚合全部 published Gold

```bash
cd /root/HandLandmarksFab
conda activate anfab

make finalize_train_finetune \
  HAND_FINETUNE_ID=v3-finetune-final-r1
```

程序扫描 `GoldSource/*/*/published/finetune_source.json`，认证包括禁用候选在内的全部批次，跨来源去重并拒绝标签冲突。输出：

```text
$HAND_TRAIN_ROOT/finetune/v3-finetune-final-r1/hmlf_gold_merged/
├── 05_labels/hand_train_catalog_finetune.jsonl
├── 05_labels/hand_training_labels_finetune.jsonl
├── hmlf_gold_aggregate.json
└── qc/finalize_train_finetune_report.json
```

这里出现 `dragon_gold_0716_v1` 是正常的：HLMF 聚合负责认证全仓数据，还没有做本次训练选择。

## 6. 冻结每个 Gold 批次是否参与本轮训练

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29

make prepare-finetune-gold-selection \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  GOLD_ENABLE_SOURCE_IDS=disagreement_gold_hlml2.0,negative_removed_gold_hlml2.0,new_recorded_gold_0718_r01,new_recorded_gold_0718_r02,dragon_gold_0718_v1
```

程序把命令列出的五批写成 `enabled: true`，把仓库中其他所有 published 批次明确写成 `false`。人工打开检查：

```text
$HAND_TRAIN_ROOT/finetune/v3-finetune-final-r1/gold_selection.yaml
```

必须确认：五个目标 source 为 true；`dragon_gold_0716_v1` 为 false；每个 published source 恰好一项；descriptor 路径和 SHA 已冻结。生成后不要再发布新 Gold；若来源组合错误，放弃该数据 ID，用新 ID重建，不能手改选择文件。

## 7. 聚合 enabled Gold + mandatory replay，运行数据门控

```bash
make check-finetune-sources \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_PROFILE=data_only

make finetune-curate \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_PROFILE=data_only

make check-finetune-data \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_PROFILE=data_only

make inspect-finetune \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_PROFILE=data_only
```

查看：

```text
$HAND_TRAIN_ROOT/train_finetune_merged/v3-finetune-final-r1/05_labels/hand_training_labels_finetune.jsonl
$HAND_TRAIN_ROOT/train_finetune_merged/v3-finetune-final-r1/qc/curation_report.json
$HAND_TRAIN_ROOT/train_finetune_merged/v3-finetune-final-r1/qc/
```

人工只需检查报告和少量可视化，确认：replay 恰好一份且 mandatory；五批目标 Gold enabled；0716 Dragon disabled；Gold/replay 去重正常；Val/Test leakage 为 0；图片/SHA 全通过；训练角色数量和权重合理。

## 8. Smoke 门控

```bash
make finetune-smoke \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-final-dataonly-r1 \
  FINETUNE_PROFILE=data_only
```

`finetune-smoke` 已自动包含数据检查、inspect、256 ROI 小集训练和 smoke 结果门控。完成后可再次显式复查：

```bash
make check-finetune-smoke \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-final-dataonly-r1 \
  FINETUNE_PROFILE=data_only
```

查看：

```text
$HAND_TRAIN_ROOT/hand_landmarker_runs/v3-finetune-final-dataonly-r1/finetune_smoke/
```

确认 loss 有效下降、checkpoint 已生成、日志无 NaN/Inf，Gold/replay 都被抽到且门控为 pass。smoke 只证明训练链路可用，不代表最终精度。

## 9. 正式 finetune

Smoke 通过后由人工启动：

```bash
make finetune-train \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-final-dataonly-r1 \
  FINETUNE_PROFILE=data_only
```

训练目录：

```text
$HAND_TRAIN_ROOT/hand_landmarker_runs/v3-finetune-final-dataonly-r1/finetune/
```

查看 `logs/history.csv`、resolved config、data report 和 `checkpoints/best.weights.h5`。本轮先只验证数据变化，不同时修改网络、结构 loss 和 ROI augmentation。

## 10. Val 与独立 infer

```bash
make eval-val-finetune \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-final-dataonly-r1

make infer-finetune \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-final-dataonly-r1
```

查看：

```text
$HAND_TRAIN_ROOT/hand_landmarker_runs/v3-finetune-final-dataonly-r1/eval/finetune/val/metrics.json
$HAND_TRAIN_ROOT/hand_landmarker_inference/v3-finetune-final-dataonly-r1/finetune/
```

先比较 geometry/multitask 和现有 finetune baseline 的 mean/median/P90 pixel error、NME、PCK、handedness、presence；再快速检查固定代表图中的拳头、数字 1、侧掌、旋转、遮挡、边缘和左右手，记录漏检、塌缩、整体平移及局部手指偏差。

若已有可配对 baseline，再执行：

```bash
make analyze-finetune-errors \
  BASELINE_FINETUNE_ID=<baseline-experiment-id> \
  CANDIDATE_FINETUNE_ID=v3-finetune-final-dataonly-r1

make compare-finetune-runs \
  BASELINE_FINETUNE_ID=<baseline-experiment-id> \
  CANDIDATE_FINETUNE_ID=v3-finetune-final-dataonly-r1
```

只有 Val、独立 infer 和代表 overlay 的总体证据都支持改善，才把本实验锁为候选 winner。

## 11. Locked Test、ONNX 和转换数据

锁定 winner 后才运行一次 Test：

```bash
make eval-test-finetune \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-final-dataonly-r1

make export-finetune \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-final-dataonly-r1

make conversion-data-finetune \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-final-dataonly-r1
```

结果：

```text
$HAND_TRAIN_ROOT/hand_landmarker_runs/v3-finetune-final-dataonly-r1/eval/finetune/test/
$HAND_TRAIN_ROOT/hand_landmarker_exports/v3-finetune-final-dataonly-r1/finetune/
```

确认 ONNX 数值/算子/体积门控通过后，再走厂商工具链转换和上板。最终用固定的原始无损 TIFF 手势集合回归，不根据 Test 结果反复改数据或超参数。

## 12. 本轮不做什么

- 不执行 `prepare-finetune-round`，不新增 disagreement CVAT 工作量；
- 不启用新的 negative-removed 选样；
- 不删除历史 Gold 来避免重复，程序会按多种认证身份排重；
- 不复制 DatesetFab 图像到 HLML 工作区；
- 不启用 `dragon_gold_0716_v1`；
- 不在第一个正式实验中同时改结构 loss、模型规模或 replay 配比；
- 不覆盖已有数据 ID、实验 ID、selection 或训练目录。
