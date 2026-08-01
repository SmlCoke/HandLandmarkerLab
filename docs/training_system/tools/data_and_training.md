# HLML 4.0 数据与训练契约

## 数据选择

HLML 只读取 HLMF 3.0 发布 manifest。positive 由 `dataset_id + proposal_variant` 选择，true negative 由 `negative_dataset_id` 选择，困难 positive 由 `selection_id` 选择。成员关系禁止依赖手工拼接 JSONL 或复制 ROI。

`HAND_DATASET_ROOT` 是图片与标签的长期仓库；`HAND_TRAIN_ROOT` 只能保存：

```text
snapshots/<snapshot_id>/<stage>/{train,val,test}.jsonl
snapshots/<snapshot_id>/<stage>/snapshot.json
runs/<experiment_id>/...
mining/<snapshot_id>/...
releases/<release_id>/...
```

snapshot 的 `crop_path` 指向 `HAND_DATASET_ROOT` 内原文件，另保存 `warehouse_crop_relpath`。图片不会复制到训练目录。

## 门控

- capture source 与 raw image 不跨 Train/Val/Test。
- 同一运行的一个 capture source 只能有一个 proposal variant。
- `roi_id`、registry 和相对路径一致。
- crop 必须位于 `HAND_DATASET_ROOT` 且解码为单通道 `256×256`。
- 人员跨 split 默认 warning，可配置为 error。
- 不反复计算图片 SHA-256。

## 三阶段

- geometry：positive only，任何 negative dataset 配置都失败。
- multitask：geometry winner + published true negatives，按 negative dataset 权重采样。
- multi-finetune：multitask winner + selection hard positive + 可选 `new_datasets` 新录制 positive + 按 ID/权重选择的 published true negative + mandatory pretrain replay；默认 55/45，replay 必须大于零，hard/new ROI 从 replay 中排除以避免重复。
