# HLML 最终 Finetune 冲刺执行计划

更新时间：2026-07-18。本文件从当前 geometry 训练开始，写到最终模型导出和上板。通用原理见 [HLML 完整训练流程](HLML_training_workflow.md)。

最终目标：完成 `v3-pretrain-r1` geometry/multitask，制作最多 800 个新的 TIFF 同域 Gold ROI，复用两批 HLML-2.0 人工 Gold，禁用 Dragon，建立新的 `v3-finetune-final-r1` 数据快照并完成最终 finetune。

## 1. 固定环境和 ID

每次登录：

```bash
cd /root/HandLandmarkerLab
git pull --ff-only
conda activate hand-landmarker-tf29

export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_TRAIN_ROOT=/root/autodl-tmp/TrainFab/HLML-3.0
export HAND_PRETRAIN_ID=v3-pretrain-r1
export HAND_FINETUNE_ID=v3-finetune-final-r1

make compile
make test
```

本计划使用：

```text
finetune 数据快照 ID: v3-finetune-final-r1
首个正式实验 ID: v3-finetune-final-dataonly-r1
Gold round ID: final_r01
```

不要复用已经作废的 `new_recorded_gold_r01`，不要覆盖其他 finetune 目录。

## 2. 等待并验收当前 geometry

只查看，不重新启动：

```bash
ps -eo pid,lstart,cmd | grep -E '[t]rain.py|[m]ake pretrain-geometry'
tail -n 10 "$HAND_TRAIN_ROOT/hand_landmarker_runs/$HAND_PRETRAIN_ID/geometry/logs/history.csv"
```

进程自然结束后确认：

```bash
test -f "$HAND_TRAIN_ROOT/hand_landmarker_runs/$HAND_PRETRAIN_ID/geometry/checkpoints/best.weights.h5"
make eval-val-geometry
make infer-geometry
```

查看：

```text
$HAND_TRAIN_ROOT/hand_landmarker_runs/$HAND_PRETRAIN_ID/eval/geometry/val/metrics.json
$HAND_TRAIN_ROOT/hand_landmarker_inference/$HAND_PRETRAIN_ID/geometry/
```

人工重点看握拳、侧向张掌、数字 1 是否仍明显塌缩；这里用于记录基线，不因个别失败就跳过 multitask。

## 3. 完成最终 multitask

```bash
make check-multitask-data
make inspect-multitask
make pretrain-multitask
```

训练完成后：

```bash
make eval-val-multitask
make infer-multitask
make export-multitask
```

检查：

```text
$HAND_TRAIN_ROOT/hand_landmarker_runs/$HAND_PRETRAIN_ID/multitask/checkpoints/best.weights.h5
$HAND_TRAIN_ROOT/hand_landmarker_runs/$HAND_PRETRAIN_ID/eval/multitask/val/metrics.json
$HAND_TRAIN_ROOT/hand_landmarker_inference/$HAND_PRETRAIN_ID/multitask/
$HAND_TRAIN_ROOT/hand_landmarker_exports/$HAND_PRETRAIN_ID/multitask/
```

程序负责 checkpoint、指标和导出门控；人工只需快速看固定困难姿态和少量代表性 overlay。HLMF 的 r02/r03 录制、自动标注和 CVAT 可以与 multitask 并行进行。

## 4. 制作新的 TIFF 同域 Gold

按 HLMF 下一步计划制作：

```text
GoldSource/new_recorded_gold/new_recorded_gold_r02
GoldSource/new_recorded_gold/new_recorded_gold_r03
```

推荐两个 task 各最多 400，人工总预算最多 800。r01 因双手 ROI 作废，不计入预算，也不参与排重或训练。

进入下一节前至少确认 task descriptor 已存在：

```bash
test -f "$HAND_DATASET_ROOT/GoldSource/new_recorded_gold/new_recorded_gold_r02/task/task_descriptor.json"
test -f "$HAND_DATASET_ROOT/GoldSource/new_recorded_gold/new_recorded_gold_r03/task/task_descriptor.json"
```

如果最终只制作一批，就从后续命令的 ID 列表删除另一批，不得写不存在的示例 ID。

## 5. 建立 replay 和困难样本分数池

必须在 multitask best checkpoint 存在后运行：

```bash
make prepare-finetune-sources \
  HAND_PRETRAIN_ID=v3-pretrain-r1 \
  HAND_FINETUNE_ID=v3-finetune-final-r1
```

程序自动完成：

- 从 pretrain curated 数据构建强制参与的 replay；
- 用 multitask student 预测可选正样本；
- 生成 student/MediaPipe teacher disagreement 分数；
- 扫描全部历史 published Gold、当前 pending task 和固定 Val/Test，排除重复身份。

检查：

```text
$HAND_TRAIN_ROOT/finetune/v3-finetune-final-r1/sources/replay/pretrain_replay/finetune_source.json
$HAND_TRAIN_ROOT/finetune/v3-finetune-final-r1/mining/teacher_student/disagreement_scores.jsonl
$HAND_TRAIN_ROOT/finetune/v3-finetune-final-r1/mining/prepare_finetune_sources_report.json
```

报告中的 `historical_gold_exclusion` 应包含两个 `_hlml2.0` published 来源和 r02/r03 pending task。若缺少任何一个，停止，不要继续抽样。

## 6. 用 800 总预算冻结最终 Gold round

r02、r03 task 已建立后执行：

```bash
make prepare-finetune-round \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_ROUND_ID=final_r01 \
  FINETUNE_GOLD_BUDGET=800 \
  NEW_RECORDED_SOURCE_IDS=new_recorded_gold_r02,new_recorded_gold_r03
```

程序读取两个 task 的实际数量。disagreement 数量自动等于 `800 - r02实际数 - r03实际数`，候选不足时可以更少，但绝不会超过预算或抽到历史/待标重复 ROI。

查看：

```text
$HAND_TRAIN_ROOT/finetune/v3-finetune-final-r1/mining/rounds/final_r01/disagreement_gold_final_r01/selection_report.json
$HAND_TRAIN_ROOT/finetune/v3-finetune-final-r1/mining/rounds/final_r01/disagreement_gold_final_r01/selection_request.jsonl
```

如果 `selection_report.json` 的最终数量为 0，不建立空 CVAT task。若大于 0，返回 HLMF 按其下一步计划第 7 节导出、标注和 import。

时间紧张时不再新增一批 negative-removed 人工任务：最终训练直接复用 `negative_removed_gold_hlml2.0`。程序生成的新的 negative candidate request 可以保留供以后使用，但不应挤占本轮 800 人工预算。

## 7. 等待 HLMF 发布并生成最终聚合

在 HLMF 完成 r02/r03 和可选 disagreement import 后，确认：

```bash
test -f "$HAND_DATASET_ROOT/GoldSource/new_recorded_gold/new_recorded_gold_r02/published/finetune_source.json"
test -f "$HAND_DATASET_ROOT/GoldSource/new_recorded_gold/new_recorded_gold_r03/published/finetune_source.json"
test -f "$HAND_TRAIN_ROOT/finetune/v3-finetune-final-r1/hmlf_gold_merged/hmlf_gold_aggregate.json"
```

成功发布的批次不再有 task；Dragon 是唯一按设计长期保留 source/published 的当前来源。

## 8. 冻结逐批 Gold 选择

当前推荐启用：

- `disagreement_gold_hlml2.0`；
- `negative_removed_gold_hlml2.0`；
- `new_recorded_gold_r02`；
- `new_recorded_gold_r03`；
- `disagreement_gold_final_r01`（仅当实际制作并 published）。

不启用 `dragon_gold_0716_v1`。如果没有 final disagreement，命令为：

```bash
make prepare-finetune-gold-selection \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  GOLD_ENABLE_SOURCE_IDS=disagreement_gold_hlml2.0,negative_removed_gold_hlml2.0,new_recorded_gold_r02,new_recorded_gold_r03
```

若它已 published，则在列表末尾追加 `,disagreement_gold_final_r01`。

人工逐项查看：

```text
$HAND_TRAIN_ROOT/finetune/v3-finetune-final-r1/gold_selection.yaml
```

必须确认仓库中每个 published 子批次都有一项决定，Dragon 为 `enabled: false`，两个 `_hlml2.0` 和本轮 TIFF Gold 为 `true`。清单冻结后不要再发布新 Gold；如需改变来源组合，使用新的 `HAND_FINETUNE_ID`。

## 9. Curate 和数据门控

先使用风险最低的 `data_only`：

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
$HAND_TRAIN_ROOT/train_finetune_merged/v3-finetune-final-r1/qc/curation_report.json
$HAND_TRAIN_ROOT/train_finetune_merged/v3-finetune-final-r1/qc/
```

确认：replay 恰好一份且 mandatory；Dragon disabled；两个历史和所有选定新批次 enabled；Val/Test leakage 为 0；source selection 与 `gold_selection.yaml` 完全一致。

## 10. Smoke 与正式 finetune

```bash
make finetune-smoke \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-final-dataonly-r1 \
  FINETUNE_PROFILE=data_only

make check-finetune-smoke \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-final-dataonly-r1 \
  FINETUNE_PROFILE=data_only
```

Smoke 通过后由人工启动正式训练：

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

首轮不要同时改网络、损失和数据权重。先用新 Gold 来源回答“TIFF 同域人工标签是否改善塌缩/漏检”。只有 data_only 门控和训练正常且仍有时间，才用新 experiment ID 尝试 structure profile。

## 11. Val、独立 infer 和错误分析

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

人工只重点检查：拳头、数字 1、侧掌、旋转/遮挡、边缘和左右手。记录漏检、塌缩、整体平移和局部手指误差，不用遍历全部图片。

如已有可比较 baseline：

```bash
make analyze-finetune-errors \
  BASELINE_FINETUNE_ID=<baseline-experiment-id> \
  CANDIDATE_FINETUNE_ID=v3-finetune-final-dataonly-r1

make compare-finetune-runs \
  BASELINE_FINETUNE_ID=<baseline-experiment-id> \
  CANDIDATE_FINETUNE_ID=v3-finetune-final-dataonly-r1
```

## 12. 锁定最终模型、Test、导出和上板

只有 Val、独立 infer 和代表性 overlay 均优于 baseline 后才锁定 winner：

```bash
make eval-test-finetune \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-final-dataonly-r1

make export-finetune \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-final-dataonly-r1
```

检查 ONNX 后走厂商工具链并上板，使用板端原始无损 TIFF 固定手势集合做最终验收。若 Test 或上板未改善，不回头改 Test 集；保留当前 winner 和所有指标，按 Val/infer 证据选择提交版本。

## 13. 最终交付清单

- geometry、multitask、finetune 三阶段 best checkpoint；
- Val/Test `metrics.json`；
- 独立 infer 代表图和错误分析报告；
- 最终 `gold_selection.yaml`、curation report 和 replay descriptor；
- HLMF `hmlf_gold_aggregate.json`；
- ONNX、厂商转换产物和上板记录；
- 明确记录 Dragon disabled、r01 invalidated，以及实际启用的每个 Gold source ID。
