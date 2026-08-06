# HLML 常见问题与解答

## Q1：如何修改 `configs/`，确保训练使用正确的数据集？

HLML 4.0 不在训练配置中手写 ROI/JSONL 路径，也不把数据复制到某个训练 run。数据成员只由 `configs/datasets.yaml` 中的 HLMF 发布 ID 和 proposal variant 决定；实际根目录由 `HAND_DATASET_ROOT` 指定。

当前服务器示例：

```bash
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_TRAIN_ROOT=/root/autodl-tmp/TrainFab/HLML-4.0
export HLML_PRETRAIN_DATASET_ID=FullEnhance0801
export HLML_EVAL_DATASET_ID=FullEnhanceVal0801
export HLML_PROPOSAL_VARIANT=eos-1.0
```

这些环境变量分别填入 geometry/multitask/multi-finetune 的 `datasets[].dataset_id`、Val/Test 的 `evaluation.*[].dataset_id`，以及所有成员的 `proposal_variant`。

开始 multitask 前还必须设置一个已经由 HLMF 发布的真负样本 ID；开始 multi-finetune 前还必须设置已发布的困难正样本 selection：

```bash
export HLML_NEGATIVE_DATASET_ID=<published_negative_dataset_id>
export HLML_SELECTION_ID=<published_selection_id>
```

如果需要长期固定，也可以直接把相同值写入 `configs/datasets.yaml`；环境变量适合服务器单次实验，不会把服务器数据状态写进通用工作流文档。

修改后依次验证：

```bash
make config-check
make data-audit HLML_STAGE=geometry
# 后续阶段分别改为 multitask / multi_finetune
```

`config-check` 只检查配置可解析；`data-audit` 才会实际解析 HLMF manifest、核对 Registry、split、variant、文件路径并解码全部所选 `256×256` 灰度 ROI。只有审计报告 `membership.errors=[]` 才可训练。`performer_cross_split` 是否阻塞由 `configs/datasets.yaml` 的 `policies.performer_cross_split` 决定：`warn` 只警告，要求人员严格隔离时改为 `fail`。

每次正式实验还应使用新的、可追溯的运行身份：

```bash
export HLML_SNAPSHOT_ID=<new_snapshot_id>
export HLML_EXPERIMENT_ID=<new_experiment_id>
export HLML_RELEASE_ID=<new_release_id>
```

这些 ID 只标识 HLML 索引、训练和发布产物，不得写入 HLMF 标注数据身份或 `HAND_DATASET_ROOT` 的数据集目录。
