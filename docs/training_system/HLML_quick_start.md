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

import 成功后 task 自动退休，最终批次应看到 `published/finetune_source.json`，不应继续保留同批 task。Dragon 保留 `source + published`；其他目录按 source/task/published 的真实语义判断，不手工合并或复制。

## 6. Finetune 候选

先为 GoldSource 中每个 published 子批次冻结显式决定。列出的 ID 启用，其余逐项写为禁用；replay 没有开关且必须存在：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29

make prepare-finetune-gold-selection \
  HAND_FINETUNE_ID=<finetune-data-id> \
  GOLD_ENABLE_SOURCE_IDS=<id-a>,<id-b>,<id-c>

make finetune-curate HAND_FINETUNE_ID=<finetune-data-id> FINETUNE_PROFILE=<profile>
make check-finetune-data \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_EXPERIMENT_ID=<experiment-id> \
  FINETUNE_PROFILE=<profile> \
  FINETUNE_GOLD_LOSS_WEIGHT=1.0 \
  FINETUNE_PSEUDO_LOSS_WEIGHT=1.0
make inspect-finetune HAND_FINETUNE_ID=<finetune-data-id>

make finetune-smoke \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_EXPERIMENT_ID=<experiment-id> \
  FINETUNE_PROFILE=<profile> \
  FINETUNE_GOLD_LOSS_WEIGHT=1.0 \
  FINETUNE_PSEUDO_LOSS_WEIGHT=1.0
make check-finetune-smoke \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_EXPERIMENT_ID=<experiment-id> \
  FINETUNE_PROFILE=<profile> \
  FINETUNE_GOLD_LOSS_WEIGHT=1.0 \
  FINETUNE_PSEUDO_LOSS_WEIGHT=1.0
make finetune-train \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_EXPERIMENT_ID=<experiment-id> \
  FINETUNE_PROFILE=<profile> \
  FINETUNE_GOLD_LOSS_WEIGHT=1.0 \
  FINETUNE_PSEUDO_LOSS_WEIGHT=1.0
make eval-val-finetune FINETUNE_EXPERIMENT_ID=<experiment-id>
make infer-finetune FINETUNE_EXPERIMENT_ID=<experiment-id>
```

同一数据快照比较多个 profile 时，保持 `HAND_FINETUNE_ID` 不变，每个候选使用新的 `FINETUNE_EXPERIMENT_ID`。

`training.gold_fraction` 是每批 Gold 的**抽样占比**；上面两个 `*_LOSS_WEIGHT` 才是进入 Loss 的 Gold/pseudo 倍率，二者不要混淆。默认 `1.0:1.0`。改变倍率时换新 `FINETUNE_EXPERIMENT_ID`，并在 data gate、smoke、smoke gate、正式训练四条命令中始终传同一组值。查看：

```bash
python -m json.tool \
  /root/autodl-tmp/TrainFab/HLML-3.0/hand_landmarker_runs/<experiment-id>/finetune_data_gate.json
```

重点读取 `sampling.loss_weighting.configured_supervision_tier_weights` 和 `epoch0_effective_head_weight_mass_fraction`。倍率必须大于 0；不能把 mandatory replay 设为 0。若要让 Gold 的逐样本 Loss 倍率为 pseudo 的两倍，把四条命令中的值统一改成 `FINETUNE_GOLD_LOSS_WEIGHT=2.0 FINETUNE_PSEUDO_LOSS_WEIGHT=1.0`；无需重做数据快照。

补充：finetune smoke 内部固定使用正/负均衡 overfit 抽样和 smoke-only 较快学习率，以便 256 ROI 同时检验三个输出 head；正式训练的抽样与学习率仍以 `train_finetune.yaml` 和 data gate 为准，不要手工覆盖 smoke 的内部探针配置。

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
