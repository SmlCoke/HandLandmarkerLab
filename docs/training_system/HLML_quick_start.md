# HLML 3.0 Quick Start

本文只列操作。首次运行和排错请阅读 [完整训练流程](HLML_training_workflow.md)。

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

默认根目录：

```text
HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
HAND_TRAIN_ROOT=/root/autodl-tmp/TrainFab/HLML-3.0
HAND_PRETRAIN_ID=v3-pretrain-r1
HAND_FINETUNE_ID=v3-finetune-r1
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
# 人工复制候选树到 negative_reviewed，删除含手/模糊图片
make pretrain-curate-reviewed

make inspect-geometry-smoke
make pretrain-geometry-smoke
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

## 4. 建立 Finetune Gold

```bash
cd /root/HandLandmarksFab
conda activate anfab
make prepare_dragon_gold HAND_FINETUNE_ID=v3-finetune-r1

cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
make prepare-finetune-sources HAND_FINETUNE_ID=v3-finetune-r1
```

可选新录制来源先在 HLMF 跑完 00～03，再限额导出：

```bash
cd /root/HandLandmarksFab
make export_finetune_gold \
  HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_SOURCE_ID=new_recorded_gold_r01 \
  FINETUNE_SOURCE_MODE=native_existing \
  FINETUNE_RAW_SOURCE_ROOT=/root/autodl-tmp/DatesetFab/<source> \
  FINETUNE_MAX_ITEMS=300
```

HLML 自动用 disagreement 补足到冻结的 600/800 预算：

```bash
cd /root/HandLandmarkerLab
make prepare-finetune-round \
  HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_ROUND_ID=r01 \
  FINETUNE_GOLD_BUDGET=800 \
  NEW_RECORDED_SOURCE_ID=new_recorded_gold_r01

cd /root/HandLandmarksFab
make export_finetune_gold \
  HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_SOURCE_ID=disagreement_gold_r01 \
  FINETUNE_SOURCE_MODE=selection_subset
```

人工按 `qc/cvat_job_plan.json` 分工；完成后把各完整 task 的 XML 放成 `reviewed.xml`：

```bash
make import_finetune_gold HAND_FINETUNE_ID=v3-finetune-r1
make finalize_train_finetune HAND_FINETUNE_ID=v3-finetune-r1
```

## 5. Finetune 候选

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29

make finetune-curate HAND_FINETUNE_ID=v3-finetune-r1 FINETUNE_PROFILE=data_only
make check-finetune-data HAND_FINETUNE_ID=v3-finetune-r1
make inspect-finetune HAND_FINETUNE_ID=v3-finetune-r1

make finetune-smoke HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-r1a FINETUNE_PROFILE=data_only
make finetune-train HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-r1a FINETUNE_PROFILE=data_only
make eval-val-finetune FINETUNE_EXPERIMENT_ID=v3-finetune-r1a
make infer-finetune FINETUNE_EXPERIMENT_ID=v3-finetune-r1a

make finetune-smoke HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-r1b FINETUNE_PROFILE=structure
make finetune-train HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-r1b FINETUNE_PROFILE=structure
make eval-val-finetune FINETUNE_EXPERIMENT_ID=v3-finetune-r1b
make infer-finetune FINETUNE_EXPERIMENT_ID=v3-finetune-r1b
```

## 6. 比较、导出

```bash
make compare-finetune-runs \
  BASELINE_FINETUNE_ID=v3-finetune-r1a \
  CANDIDATE_FINETUNE_ID=v3-finetune-r1b

make export-finetune FINETUNE_EXPERIMENT_ID=<winner>
make conversion-data-finetune FINETUNE_EXPERIMENT_ID=<winner>
```

最后只对冻结候选运行 locked Test 和上板验证。
