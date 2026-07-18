# HLML 当前下一步计划

更新时间：2026-07-18。本文只记录当前冲刺安排；通用原理见 [完整训练流程](HLML_training_workflow.md)。

## 1. 当前结论

1. 正式 geometry 正常训练，等待 `geometry/checkpoints/best.weights.h5`。
2. `v3-pretrain-r1` 的负样本人工删除复核已经完成。
3. `new_recorded_gold_r01` 已导出 300 个 TIFF 同域 ROI，人工标注进行中。
4. HLML-2.0 的 `disagreement_gold` 和 `negative_removed_gold` 都是有效人工 Gold。此前未在 3.0 使用，是因为它们绑在旧训练工作区，而不是质量失效；归档到 GoldSource 后应重新纳入候选。
5. Dragon Gold 保留但本次禁用：H.264/I420/JPEG 抽帧域与板端无损 TIFF 评测域不一致。
6. 本轮可以制作多个 `new_recorded_gold_r*`，人工任务合计最多 800 个 ROI。

## 2. 程序已经自动化的工作

- HLMF 把 Gold 固定发布为 `DatesetFab/GoldSource/<domain>/<source-id>/{task,published}`。
- HLMF 聚合会认证 GoldSource 中所有 published 批次并跨来源去重。
- HLML 生成 replay 和 disagreement score pool 时，会扫描所有历史 published Gold 和 pending task，排除已经标注/待标 ROI。
- `prepare-finetune-round` 可同时接收多个 new-recorded source ID，先求其任务总数，再用总预算的剩余额度选择 disagreement。
- HLML 为每次 finetune 生成 `gold_selection.yaml`，要求 GoldSource 中每个 published 子批次都有显式 true/false，并锁定 descriptor SHA256。
- replay 不在 Gold 开关中，始终强制参与。

## 3. 人工当前要做什么

### 3.1 完成 r01

继续完成 `new_recorded_gold_r01` 的完整 21 点、presence 和 handedness 标注。确定无手使用 `no_hand`；无法可靠标注使用 `ignore_for_training`。

返回 XML 放到：

```text
/root/autodl-tmp/DatesetFab/GoldSource/new_recorded_gold/new_recorded_gold_r01/task/reviewed.xml
```

由 HLMF 单批导入：

```bash
make import_finetune_gold FINETUNE_SOURCE_ID=new_recorded_gold_r01
```

### 3.2 决定是否追加 r02/r03

如果 r01 完成后仍有人工余量，优先录制以下薄弱姿态的新无损 TIFF 图片流：握拳及开合、数字 1、侧向张掌、指间遮挡、手腕旋转、画面边缘、不同距离和左右手。每批使用新的 source ID，并放在：

```text
DatesetFab/GoldSource/new_recorded_gold/<source-id>/source/images/
```

每批单独 autolabel、导出、标注和 import。多个批次的任务数之和不得超过团队剩余人工上限；不要为了凑 800 重复相似连续帧。

## 4. Geometry 完成后

只读取 best checkpoint，不改动旧 run 布局：

```bash
make eval-val-geometry
make infer-geometry
make check-multitask-data
make inspect-multitask
make pretrain-multitask
make eval-val-multitask
make infer-multitask
make export-multitask
```

人工负责观察握拳、侧向掌、数字 1 的 infer；程序负责指标、checkpoint 和导出门控。

## 5. 建立 replay 和 disagreement 候选池

Multitask 完成后：

```bash
make prepare-finetune-sources \
  HAND_FINETUNE_ID=v3-finetune-r1 \
  HAND_PRETRAIN_ID=v3-pretrain-r1
```

查看：

```text
HLML-3.0/finetune/v3-finetune-r1/sources/replay/pretrain_replay/finetune_source.json
HLML-3.0/finetune/v3-finetune-r1/mining/teacher_student/disagreement_scores.jsonl
HLML-3.0/finetune/v3-finetune-r1/mining/prepare_finetune_sources_report.json
```

报告中的 `historical_gold_exclusion` 应列出 GoldSource 的 published 标签和 pending task manifest。

## 6. 用多个 new-recorded task 冻结 disagreement

假设当前已导出 `new_recorded_gold_r01,r02`，总人工预算为 800：

```bash
make prepare-finetune-round \
  HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_ROUND_ID=r01 \
  FINETUNE_GOLD_BUDGET=800 \
  NEW_RECORDED_SOURCE_IDS=new_recorded_gold_r01,new_recorded_gold_r02
```

程序读取每个 task descriptor，计算 new-recorded 合计数，再从 score pool 选择不重复 disagreement 补足剩余额度。若 new-recorded 已达到 800，disagreement 数量为 0；不会超预算。

输出：

```text
HLML-3.0/finetune/v3-finetune-r1/mining/rounds/r01/disagreement_gold_r01/
```

若选中数大于 0，回 HLMF 导出 `disagreement_gold_r01`、人工标注并 import。

## 7. HLMF 最终聚合

确认所有本轮 task 均已 published，然后：

```bash
cd /root/HandLandmarksFab
make finalize_train_finetune HAND_FINETUNE_ID=v3-finetune-r1
```

不得在尚有计划任务未导入时提前冻结最终聚合；输出目录已存在时不要直接覆盖，应先确认它是不是旧契约产物。

## 8. 逐批选择参与本次 finetune 的 Gold

回到 HLML。当前建议启用：

- `disagreement_gold`（历史人工 Gold）；
- `negative_removed_gold`（历史困难正样本人工 Gold）；
- 所有本轮完成的 `new_recorded_gold_r*`；
- 本轮完成的 `disagreement_gold_r01`（如果存在）。

不要启用 `dragon_gold_0716_v1`。命令示例：

```bash
make prepare-finetune-gold-selection \
  HAND_FINETUNE_ID=v3-finetune-r1 \
  GOLD_ENABLE_SOURCE_IDS=disagreement_gold,negative_removed_gold,new_recorded_gold_r01,new_recorded_gold_r02,disagreement_gold_r01
```

如果某示例批次实际上不存在，就从列表删除。命令会把仓库中其余每个 published 批次明确写成 `enabled: false`。

人工查看：

```text
HLML-3.0/finetune/v3-finetune-r1/gold_selection.yaml
```

逐项确认领域、source ID 和 `enabled`。清单存在即不可重新生成；如需另一组合，使用新的 `HAND_FINETUNE_ID`。

## 9. Curate、smoke 和正式 finetune

```bash
make check-finetune-sources HAND_FINETUNE_ID=v3-finetune-r1
make finetune-curate HAND_FINETUNE_ID=v3-finetune-r1
make check-finetune-data HAND_FINETUNE_ID=v3-finetune-r1
make inspect-finetune HAND_FINETUNE_ID=v3-finetune-r1
make finetune-smoke HAND_FINETUNE_ID=v3-finetune-r1 FINETUNE_EXPERIMENT_ID=<experiment-id>
make finetune-train HAND_FINETUNE_ID=v3-finetune-r1 FINETUNE_EXPERIMENT_ID=<experiment-id>
```

重点检查 `curation_report.json`：

- 每个 published 批次都出现在 `source_selection`；
- Dragon 为 disabled；
- 历史 disagreement、negative-removed 与选定 new-recorded 为 enabled；
- replay 恰好一个且 mandatory；
- `disabled_source_rows`、Gold/replay 数量和 Val/Test leakage 均符合预期。

## 10. 评估决策

```bash
make eval-val-finetune FINETUNE_EXPERIMENT_ID=<experiment-id>
make infer-finetune FINETUNE_EXPERIMENT_ID=<experiment-id>
make analyze-finetune-errors BASELINE_FINETUNE_ID=<baseline> CANDIDATE_FINETUNE_ID=<experiment-id>
```

人工只需查看程序选出的代表性 overlays 和固定困难手势。以 Val 像素误差/PCK、infer 漏检与塌缩数共同决策；winner 冻结后才运行 Test、ONNX 导出和板端转换。
