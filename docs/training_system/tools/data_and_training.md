# HLML 4.0 数据与训练契约

## 数据选择

HLML 只读取 HLMF 3.0 已发布 manifest。`datasets` 由 `dataset_id + proposal_variant` 选择 positive；`negative_datasets`、`hard_datasets` 分别由长期可复用的发布 ID 选择；`gold_datasets` 可选读取新录制且经 CVAT 复核的 Gold。所有成员与权重只在 `configs/datasets.yaml` 声明，不手工拼接 JSONL 或复制 ROI。

当前 v3 positive 使用四个 Eos-2.1 + HaMeR r4 Train dataset；multitask 使用完整 `neg-eos_2.0-hcf0813-hp0.5`。正样本发布域和已发布负样本域分别执行 proposal variant 唯一性门禁，因此可保留各自 Eos 版本；同一域内一个 capture source 选择多个 variant 仍失败。

`HAND_DATASET_ROOT` 是图片与标签的长期仓库；`HAND_TRAIN_ROOT` 只保存零拷贝 snapshot 索引、训练/评估/导出产物，不复制图片。

## 门控

- capture source 与 raw image 不跨 Train/Val/Test。
- 同一发布域内，一个 capture source 只能有一个 proposal variant。
- `roi_id`、Registry 和相对路径一致且不重复。
- crop 必须位于 `HAND_DATASET_ROOT` 且解码为单通道 `256×256`。
- 人员跨 split 默认 warning；Test 不反馈给训练、阈值、mining 或 winner 选择。
- 不对每一步反复计算图片 SHA-256。

## 三阶段

- geometry：positive only，配置任何 negative dataset 都失败。
- multitask：同一模型版本的 geometry winner + published true negatives；当前完整负样本集为 `neg-eos_2.0-hcf0813-hp0.5`。
- multi-finetune：multitask winner + CVAT-reviewed published hard dataset + 可选 reviewed Gold + published true negative + mandatory pretrain replay。没有已发布 hard dataset 时必须失败，不接受占位 ID；默认 hard/gold 55%、replay 45%，replay 必须大于零。
