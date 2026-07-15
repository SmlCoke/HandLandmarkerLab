# 训练来源负例及 ROI 域审计

审计日期：2026-07-15。本文记录服务器现有 `v2-pretrain-r1` 数据的只读检查结果，不修改服务器文件。

## 1. 审计对象

主要文件：

```text
/root/autodl-tmp/TrainFab/HLML-2.0/train_pretrain_merged/05_labels/hand_training_labels_pretrain.jsonl
/root/autodl-tmp/TrainFab/HLML-2.0/train_pretrain_curated/v2-pretrain-r1/audit/negative_review_queue.jsonl
/root/autodl-tmp/TrainFab/HLML-2.0/train_pretrain_curated/v2-pretrain-r1/qc/curation_report.json
/root/autodl-tmp/TrainFab/HLML-2.0/hand_landmarker_reviews/v2-pretrain-r1/negative_candidates/
```

merged canonical 共 35,454 行。curation 报告记录 25,162 个可进入 geometry 的 positive、9,106 个待人工复核 negative candidate，以及 3,916 个与已确认手部证据重叠的候选；当前状态为等待人工复核。

## 2. 新来源实际产生了负例

新增的 5 个来源共有 9,623 行，其中 8,627 个 positive、996 个 negative。996 个 negative 全部是 `NEG_RUNTIME_CANDIDATE`，并且全部存在于 `negative_review_queue.jsonl`：

| dataset_id | Positive | `NEG_RUNTIME_CANDIDATE` |
|---|---:|---:|
| `dragon_train_0714` | 4,408 | 324 |
| `peak_train_0714_bright` | 456 | 95 |
| `peak_train_0714_dark` | 3 | 199 |
| `soar_train_0714_bright` | 3,637 | 145 |
| `soar_train_0714_dark` | 123 | 233 |
| 合计 | 8,627 | 996 |

旧来源共有 25,831 行，其中 17,721 个 positive、8,110 个 negative。旧来源可同时产生 `NEG_LOW_PALM_CANDIDATE` 与 `NEG_RUNTIME_CANDIDATE`。

因此，“新增来源没有在 pretrain-curate 中产生负例”并不成立。最可能的观察误区是只查看了：

```text
negative_candidates/NEG_LOW_PALM_CANDIDATE/
```

新增来源的 996 张图实际位于：

```text
negative_candidates/NEG_RUNTIME_CANDIDATE/<dataset_id>/
```

服务器审查工作区一共有 9,106 个候选文件，与 review queue 完全一致。

## 3. 为什么 official 后端没有 low-Palm 候选

`HandLandmarkerFab/hand_autolabel/palm_mediapipe.py` 的 MediaPipe official full/tiled 后端只有在官方 Hand Landmarker 成功返回 landmarks 时才生成 proposal。它使用 landmark ID `[0,1,5,9,13,17]` 的外接框，再按 `compatible_bbox_expand=0.25` 扩框；proposal score 来自 handedness score（缺失时为 1.0），不是 MediaPipe 内部 Palm Detector 的置信度。

这带来两个结果：

1. official 模型在原图或 tile 上完全未检出手时，不会暴露内部低分 Palm proposal，也就无法产生 `NEG_LOW_PALM_CANDIDATE`；这类情况是“无 proposal”，不能自动解释为“无手”。
2. 已成功构造的 ROI 会在标注第 03 阶段再次运行 official Hand Landmarker。如果这次 ROI 推理失败，canonical 记录表现为 `palm_valid=true`、`hand_presence=false`，最终成为 `NEG_RUNTIME_CANDIDATE`。

各新增来源的 Palm QC 都显示 backend=`mediapipe_official`、mode=`tasks_tiled`、Palm 阶段 `negative_candidates=0`；这只说明没有低分 Palm proposal，并不表示后续 curation 没有 negative。第 03 阶段在原始 ROI 上的失败量很大，之后 canonical finalize/downsample 最终保留了上述 996 个 runtime candidate。

不能把 official 后端的“没有 proposal”原图直接标成负例。需要背景 hard negative 时，应继续使用能够暴露低分 proposal 的 AetherSign ONNX Palm，或在数据制作端增加独立的背景 proposal 策略，再进行人工复核。

## 4. 两种 ROI 不是同一个输入域

以下统计使用 positive 的 21 点归一化外接框，衡量“手在 256×256 ROI 中占多大比例”。数值越大表示裁剪越紧，`min margin` 越小表示手越靠近边界。

| 来源/集合 | 行数 | bbox width mean / median | bbox height mean / median | min margin mean / median |
|---|---:|---:|---:|---:|
| 新 MediaPipe-derived，可进入 geometry | 7,588 | 0.4480 / 0.4291 | 0.5489 / 0.5440 | 0.1207 / 0.1220 |
| 旧 AetherSign Palm，可进入 geometry | 17,574 | 0.2750 / 0.2656 | 0.3146 / 0.3070 | 0.2416 / 0.2491 |
| Val Gold | 1,226 | 0.2844 | 0.3295 | 0.2500 |
| Test Gold | 985 | 0.2832 | 0.3423 | 0.2523 |

新来源中的手平均约比旧来源宽 1.63 倍、高 1.74 倍，边缘余量约减半。这不是可以忽略的微小误差，而是明显的 crop-scale domain shift。

原始 positive 的越界率也支持这一结论：

- 新 MediaPipe-derived：1,039 / 8,627 = 12.04%；
- 旧 AetherSign Palm：147 / 17,721 = 0.83%。

这些越界 positive 已被 curation 隔离，不会进入 geometry，但高隔离率说明 `compatible_bbox_expand=0.25` 对当前任务偏紧。当前 geometry 数据中仍有 7,588 个新来源 positive，占 25,162 个样本的 30.16%；训练增强 `scale_range=[0.95,1.05]` 不足以消除 1.6～1.7 倍的尺度差。

## 5. 对训练、评估和上板的影响

### 训练

少量、质量合格的尺度变化能提升鲁棒性，但当前差异过大。模型会同时学习“手占 ROI 约 27%×31%”和“约 45%×55%”两个域；如果紧裁剪来源继续增加，可能牺牲主要部署域的关键点精度，尤其是靠近 ROI 边缘的手腕和指尖。12.04% 的原始越界率还会浪费大量新数据。

### Val/Test

现有 Val/Test 的手占比与旧 AetherSign Palm 来源高度一致，因此它们更接近当前板端输入域。这有利于选出部署相关 checkpoint，但不会充分度量 MediaPipe 紧裁剪域的效果。不要为了让新增来源的分数好看而改变锁定 Test；如需监控该域，应另建不参与主 checkpoint 选择的 diagnostic split。

### 实际上板

板端仍使用 AetherSign Palm，并按 `scale=1.8`、`shift_y=-0.1` 构造 ROI。新增来源不会改变板端预处理代码，但会通过训练分布影响模型。当前 Val/Test 和旧来源与板端更一致；因此主要风险不是“板端突然得到更小 ROI”，而是模型被过多紧裁剪样本拉离板端域。

## 6. 修正建议

优先级从高到低：

1. **首选同域生成。** 正式 Train/Val/Test 尽量统一使用实际板端的 AetherSign ONNX Palm 和同一套 ROI `scale/shift/rotation`。MediaPipe official 可用于发现手和生成伪 landmarks，但不应让它单独定义最终训练 crop 几何。
2. **校准 official 兼容框。** 如果必须由 core landmarks 构框，调整 HandLandmarkerFab 的 `compatible_bbox_expand`，使落盘 positive 的 hand occupancy 接近 Val/Test，而不是凭肉眼判断“只小一点”。由当前比例粗估，合理中心约为 `0.7～0.8`，但必须在数据制作端对 `0.50/0.65/0.75/0.85` 做小样本网格审计后再冻结。
3. **量化验收目标。** 建议 positive 越界率 `<1%`，median bbox width `0.27～0.30`、height `0.31～0.35`、median min margin `0.22～0.27`；同时逐图可视化确认旋转、目标手归属和腕部边界。
4. **重新落盘而非训练时修补。** crop 改变会同时改变图片、landmark 坐标、哈希和 candidate 类型，应从 HLMF 相应 ROI/label/finalize 阶段重新生成 `train_pretrain_merged`，再使用新的 `HAND_PRETRAIN_ID` 执行 curation；不要在 loader 内临时缩放。
5. **保留负例语义。** official 未检出是 abstention，不是 true negative；996 个 runtime candidate 仍须按删除式流程人工复核，只有明确背景才能进入 multitask。

本轮不修改 HandLandmarkerLab 的训练增强或既有 Val/Test，因为首先应在数据制作边界统一 ROI 域。完成一批校准数据后，再根据落盘统计决定是否把训练 scale augmentation 从当前的窄范围适度放宽。
