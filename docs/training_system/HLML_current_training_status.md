# HLML 当前训练与数据状态

更新时间：2026-07-18。本文只记录会变化的项目事实。通用原理见 [完整训练流程](HLML_training_workflow.md)，命令清单见 [Quick Start](HLML_quick_start.md)。

## 1. 服务器和代码

```text
SSH: root@connect.nmb2.seetacloud.com:41877
HLML: /root/HandLandmarkerLab
HLMF: /root/HandLandmarksFab
数据仓库: /root/autodl-tmp/DatesetFab
当前派生工作区: /root/autodl-tmp/TrainFab/HLML-3.0
```

两个系统使用新的长期数据源契约，不再从旧 finetune 训练目录 seed Gold。Gold 的启停粒度是每个 `source-id` 子批次，不是整个领域。

## 2. DatesetFab 当前结构

```text
DatesetFab/
├── GoldSource/
│   ├── new_recorded_gold/
│   ├── disagreement_gold/
│   ├── negative_removed_gold/
│   └── dragon/
├── PretrainSource/
├── eval_sources/
├── HandFinetune0713/
└── HandFinetune0715/
```

`HandFinetune0713/0715` 尚未正式 autolabel/训练，暂留原位。当前 geometry 仍可使用它启动时的旧 Pretrain 路径；规范 PretrainSource 不改变该 run 的数据或权重。

## 3. 已有 Gold

| 领域/批次 | 当前状态 |
|---|---|
| `disagreement_gold/disagreement_gold` | HLML-2.0 人工精标 300 ROI，已归档 published |
| `negative_removed_gold/negative_removed_gold` | HLML-2.0 人工精标 300 ROI，已归档 published |
| `dragon/dragon_gold_0716_v1` | 5,191 ROI，5,189 trainable、2 ignored；保留但当前禁用 |
| `new_recorded_gold/new_recorded_gold_r01` | TIFF 同域来源，任务 300 ROI；人工标注进行中 |

历史 `negative_removed_gold` 不是无效数据。它此前没有进入 HLML-3.0，只是因为 Gold 被绑在 HLML-2.0 工作区；迁入 GoldSource 后可以在新 finetune 选择清单中重新启用。

## 4. 当前 Pretrain 进度

- geometry smoke 已通过。
- 正式 `pretrain-geometry` 正在运行；本轮仓库更新不修改其 HLML-3.0 输入目录和 run。
- `v3-pretrain-r1` 的负样本人工删除复核已经完成。
- geometry 完成后继续 multitask；只需要从当前 run 获取 best checkpoint，不要求旧工作区采用新数据仓库布局。

## 5. 当前 Finetune 准备

- `new_recorded_gold_r01` 的原始输入是无损 TIFF 图片流，autolabel 使用本批覆盖阈值 `negative_candidate_threshold=0.3`。
- 已导出 300 个 ROI，人工标注正在进行。
- 允许继续制作多个 `new_recorded_gold_r*`；每批独立 task/published，HLML 抽样会跨所有历史与 pending 批次排重。
- 所有计划 Gold 发布后，HLMF 生成全仓聚合；随后 HLML 通过 `make prepare-finetune-gold-selection` 为每个 published 子批次显式 true/false，并锁定 descriptor SHA256。
- replay 始终强制参与，不能通过 Gold 清单关闭。

完成重要阶段后只更新本文，不把当前状态写回通用 Workflow 或 Quick Start。
