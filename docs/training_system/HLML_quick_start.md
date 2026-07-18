# HLML Quick Start

本文只列通用操作。当前使用的 ID、预算和实验安排见 [当前下一步计划](HLML_next_step_plan.md)，原理与排错见 [完整训练流程](HLML_training_workflow.md)。

## 1. 初始化

```bash
cd /root/HandLandmarksFab
git pull --ff-only
conda activate anfab
make compile
make test

cd /root/HandLandmarkerLab
git pull --ff-only
conda activate hand-landmarker-tf29
make paths
make compile
make test-unit
make doctor
```

## 2. HLMF 聚合基础数据

```bash
cd /root/HandLandmarksFab
conda activate anfab
make finalize_train_pretrain
make build_pretrain_source_registry
make finalize_val
make finalize_test
```

## 3. Pretrain

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29

make pretrain-curate
# 人工复制 negative_candidates 为 negative_reviewed，删除含手/模糊图片
make pretrain-curate-reviewed

make inspect-geometry
make inspect-geometry-smoke
make pretrain-geometry-smoke
make check-geometry-smoke
make pretrain-geometry
make eval-val-geometry
make infer-geometry

make check-multitask-data
make inspect-multitask
make pretrain-multitask
make eval-val-multitask
make infer-multitask
make export-multitask
```

## 4. 建立 Finetune Gold 和 replay

```bash
cd /root/HandLandmarksFab
conda activate anfab

# 每一批 Dragon 使用不同批次 ID
make prepare_dragon_gold \
  HAND_FINETUNE_ID=<finetune-data-id> \
  DRAGON_SOURCE_ROOT=$HAND_DATASET_ROOT/GoldSource/dragon/<dragon-batch-id>/source \
  DRAGON_BATCH_ID=<dragon-batch-id>

make finalize_train_finetune HAND_FINETUNE_ID=<finetune-data-id>

cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
make prepare-finetune-sources HAND_FINETUNE_ID=<finetune-data-id>
```

## 5. 一轮新录制 + disagreement Gold

新录制来源先在 HLMF 跑完 00～03，再导出：

```bash
cd /root/HandLandmarksFab
make export_finetune_gold \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_SOURCE_ID=new_recorded_gold_<round-id> \
  FINETUNE_SOURCE_MODE=native_existing \
  FINETUNE_RAW_SOURCE_ROOT=$HAND_DATASET_ROOT/GoldSource/new_recorded_gold/new_recorded_gold_<round-id>/source \
  FINETUNE_MAX_ITEMS=<new-recorded-limit>
```

HLML 冻结剩余 disagreement：

```bash
cd /root/HandLandmarkerLab
make prepare-finetune-round \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_ROUND_ID=<round-id> \
  FINETUNE_GOLD_BUDGET=<round-budget> \
  NEW_RECORDED_SOURCE_IDS=new_recorded_gold_<round-id>[,new_recorded_gold_<other-id>]
```

HLMF 导出、人工 CVAT、导入并聚合：

```bash
cd /root/HandLandmarksFab
make export_finetune_gold \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_SOURCE_ID=disagreement_gold_<round-id> \
  FINETUNE_SOURCE_MODE=selection_subset

# reviewed.xml 放入 GoldSource/<domain>/<source-id>/task/ 后
make import_finetune_gold HAND_FINETUNE_ID=<finetune-data-id>
make finalize_train_finetune HAND_FINETUNE_ID=<finetune-data-id>
```

## 6. Finetune 候选

先为 GoldSource 中每个 published 子批次冻结显式决定。列出的 ID 启用，其余逐项写为禁用；replay 没有开关且必须存在：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29

make prepare-finetune-gold-selection \
  HAND_FINETUNE_ID=<finetune-data-id> \
  GOLD_ENABLE_SOURCE_IDS=<id-a>,<id-b>,<id-c>

make finetune-curate HAND_FINETUNE_ID=<finetune-data-id> FINETUNE_PROFILE=<profile>
make check-finetune-data HAND_FINETUNE_ID=<finetune-data-id>
make inspect-finetune HAND_FINETUNE_ID=<finetune-data-id>

make finetune-smoke \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_EXPERIMENT_ID=<experiment-id> \
  FINETUNE_PROFILE=<profile>
make check-finetune-smoke \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_EXPERIMENT_ID=<experiment-id> \
  FINETUNE_PROFILE=<profile>
make finetune-train \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_EXPERIMENT_ID=<experiment-id> \
  FINETUNE_PROFILE=<profile>
make eval-val-finetune FINETUNE_EXPERIMENT_ID=<experiment-id>
make infer-finetune FINETUNE_EXPERIMENT_ID=<experiment-id>
```

同一数据快照比较多个 profile 时，保持 `HAND_FINETUNE_ID` 不变，每个候选使用新的 `FINETUNE_EXPERIMENT_ID`。

## 7. 比较、Test 和导出

```bash
make compare-finetune-runs \
  BASELINE_FINETUNE_ID=<baseline-experiment-id> \
  CANDIDATE_FINETUNE_ID=<candidate-experiment-id>

# winner 冻结后才运行 locked Test
make eval-test-finetune FINETUNE_EXPERIMENT_ID=<winner>
make export-finetune FINETUNE_EXPERIMENT_ID=<winner>
make conversion-data-finetune FINETUNE_EXPERIMENT_ID=<winner>
```
