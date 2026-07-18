# HLML 当前训练状态

更新时间：2026-07-19。本文件只记录当前服务器事实；通用原理和命令见 [HLML 完整训练流程](HLML_training_workflow.md)，接下来应实际执行的顺序见 [HLML 下一步计划](HLML_next_step_plan.md)。

## 1. 代码、数据和运行根目录

```text
HLMF 仓库: /root/HandLandmarksFab
HLML 仓库: /root/HandLandmarkerLab
长期数据仓库: /root/autodl-tmp/DatesetFab
HLML 3.0 工作区: /root/autodl-tmp/TrainFab/HLML-3.0
pretrain ID: v3-pretrain-r1
计划 finetune 数据 ID: v3-finetune-final-r1
```

Pretrain 原始来源的规范入口已经统一为：

```text
/root/autodl-tmp/DatesetFab/PretrainSource/HandViolence0708/
/root/autodl-tmp/DatesetFab/PretrainSource/HandViolenceEnhanced0714/
```

HLMF 的 source registry 也应只记录这些规范路径。旧顶层别名 `HandViolence0708/`、`HandViolenceEnhanced0714/` 和旧 Dragon 原始目录 `HandViolenceEnhanced0716/` 已完成真实性核对后清理；训练清单仍通过 registry 恢复到 DatesetFab 真源，不复制图片。

## 2. Geometry 与 multitask 已完成

运行根目录：

```text
/root/autodl-tmp/TrainFab/HLML-3.0/hand_landmarker_runs/v3-pretrain-r1/
```

已完成：

- geometry 正式训练与 `eval-val-geometry`；
- multitask 正式训练与 `eval-val-multitask`；
- multitask ONNX 导出。

关键产物：

```text
geometry/checkpoints/best.weights.h5
multitask/checkpoints/best.weights.h5
eval/geometry/val/metrics.json
eval/multitask/val/metrics.json
export/multitask/hand_landmarker_v2.onnx
```

Val 的定位指标如下：

| 阶段 | 平均像素误差 | 中位像素误差 | P90 像素误差 | mean NME | PCK@0.10 | handedness accuracy |
|---|---:|---:|---:|---:|---:|---:|
| geometry | 20.73 px | 18.21 px | 37.32 px | 0.1973 | 0.3342 | 0.4560 |
| multitask | 22.01 px | 18.38 px | 39.18 px | 0.2079 | 0.3019 | 0.6843 |

multitask 明显提高了 handedness，但 Val 关键点定位没有超过 geometry。当前 Val 全是 positive，因此 presence accuracy 为 1 不能证明对真实背景/漏检的泛化已经解决。现有结果足以作为 finetune 初始化和对照基线，但不能把 pretrain 指标当作最终模型质量结论。

## 3. GoldSource 当前批次

### 3.1 已发布，可被本轮显式选择

```text
GoldSource/disagreement_gold/disagreement_gold_hlml2.0/published/
GoldSource/negative_removed_gold/negative_removed_gold_hlml2.0/published/
GoldSource/dragon/dragon_gold_0718_v1/published/
```

- `disagreement_gold_hlml2.0`：300 条人工审核记录，其中 237 条可训练、63 条 ignored；
- `negative_removed_gold_hlml2.0`：300 条人工审核记录，其中 260 条可训练、40 条 ignored；
- `dragon_gold_0718_v1`：406 条，全部可训练；已发布，且已确认数据域符合本次板端无损图像目标。

### 3.2 已完成 reviewed.xml，尚待 import/publish

```text
GoldSource/new_recorded_gold/new_recorded_gold_0718_r01/task/reviewed.xml
GoldSource/new_recorded_gold/new_recorded_gold_0718_r02/task/reviewed.xml
```

两批各 300 个 Hand ROI，共 600 个，人工标注已经完成。当前它们仍是 task，不会进入 HLMF 聚合或 HLML 训练；下一步必须分别严格 import，成功后 task 自动退休并生成同批次 `published/`。

### 3.3 保留但本轮禁用

```text
GoldSource/dragon/dragon_gold_0716_v1/{source,published}/
```

该批来自 H.264/I420/JPEG 链路，与最终板端原始无损 TIFF 域不一致。数据保留用于审计和其他实验，但本次 `gold_selection.yaml` 应为 `enabled: false`。

## 4. 最终 finetune 尚未开始

截至本次检查，`v3-finetune-final-r1` 还没有建立以下产物：

- mandatory pretrain replay；
- 本轮 disagreement 分数池；
- `hmlf_gold_merged` 全仓 Gold 聚合；
- `gold_selection.yaml`；
- `train_finetune_merged/v3-finetune-final-r1` 冻结训练集；
- finetune smoke、正式 run、eval、infer 或 export。

时间策略已经确定：不再新增本轮 disagreement/negative-removed 人工标注；复用两批 HLML-2.0 Gold，加入两批已经标完的新录制 Gold 和有效的 `dragon_gold_0718_v1`，同时保留强制 replay。具体命令见下一步计划。
