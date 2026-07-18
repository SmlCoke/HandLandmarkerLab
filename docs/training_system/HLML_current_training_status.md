# HLML 当前训练状态

更新时间：2026-07-18。本文只记录当前服务器事实；通用操作见 [完整训练流程](HLML_training_workflow.md)，冲刺执行顺序见 [当前下一步计划](HLML_next_step_plan.md)。

## 1. 代码与根目录

```text
HLMF: /root/HandLandmarksFab
HLML: /root/HandLandmarkerLab
数据仓库: /root/autodl-tmp/DatesetFab
训练工作区: /root/autodl-tmp/TrainFab/HLML-3.0
pretrain ID: v3-pretrain-r1
```

规范数据入口：

```text
DatesetFab/
├── PretrainSource/
├── GoldSource/
│   ├── disagreement_gold/disagreement_gold_hlml2.0/published/
│   ├── negative_removed_gold/negative_removed_gold_hlml2.0/published/
│   └── dragon/dragon_gold_0716_v1/{source,published}/
└── eval_sources/
```

## 2. Pretrain 状态

- `v3-pretrain-r1` 的人工负样本删除式复核已经完成。
- 正式 `pretrain-geometry` 仍在运行；现有 `HLML-3.0` 输入目录和 run 布局不做迁移。
- geometry 完成后使用其 `geometry/checkpoints/best.weights.h5` 进入 multitask。

## 3. Finetune Gold 状态

- `disagreement_gold_hlml2.0`：HLML-2.0 人工 Gold，300 ROI，可在最终 finetune 显式启用。
- `negative_removed_gold_hlml2.0`：HLML-2.0 困难样本人工 Gold，300 ROI，可在最终 finetune 显式启用。
- `dragon_gold_0716_v1`：保留，但 H.264/I420/JPEG 来源域与最终板端无损 TIFF 评测域不一致，最终选择应为 disabled。
- `new_recorded_gold_r01`：因大量双手同时进入单个 Hand ROI 已作废并清理，没有进入 published，也不得复用该 ID。
- 当前没有 pending Gold task；下一轮从 `new_recorded_gold_r02` 开始，可制作多个独立新录制批次，人工总量最多 800 ROI。

## 4. Gold 生命周期

- `source`：原始录制或 Dragon 原始数据；
- `task`：等待 CVAT 人工/等待 import 的冻结任务；
- `published`：认证后的最终训练来源。

HLMF import 成功后 task 自动退休，必要 XML 和任务描述符进入 `published/audit/`。Dragon 因 source 与 published 数据性质不同长期保留两者；其余来源不长期并存 task/published。

## 5. 尚未开始的最终 finetune

最终 finetune 建议使用新的数据快照 ID `v3-finetune-final-r1`，避免与此前试验性目录混用。它尚未建立 replay、disagreement round、Gold 聚合、`gold_selection.yaml` 或训练 run。完整命令见下一步计划。
