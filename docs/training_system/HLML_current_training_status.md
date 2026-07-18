# HLML 当前训练与数据状态

更新时间：2026-07-18

本文只记录会变化的项目事实。通用原理和命令见 [完整训练流程](HLML_training_workflow.md)，最短命令清单见 [Quick Start](HLML_quick_start.md)。

## 1. 代码版本

- HLML 当前契约：3.0；`HLML-3.0` tag 是初始发布点，后续修复继续位于同一契约，不再为本次更新打 tag。
- HLMF 当前契约：2.0；`HLMF-2.0` tag 是初始发布点，本次 Dragon 多批次接口更新不改变数据契约主版本。
- 两套系统均不兼容 HLML-2.0/HLMF-1.x 的旧配置和工作区。
- 当前服务器可见 NVIDIA GeForce RTX 3090（24 GB）；TensorFlow 已成功使用 GPU 完成 geometry smoke。

## 2. 服务器与根目录

```text
SSH: root@connect.nmb2.seetacloud.com:41877
HLML repo: /root/HandLandmarkerLab
HLMF repo: /root/HandLandmarksFab

只读可再生数据仓库: /root/autodl-tmp/DatesetFab
生成工作区:         /root/autodl-tmp/TrainFab/HLML-3.0
```

pretrain、Val 和 Test 的 `crop_path` 直接指向 DatesetFab，没有在 TrainFab 建立图片副本。TrainFab 只保存标签、QC、训练产物以及人工标注或 ROI 变换确实需要的派生图片。

## 3. HLML-3.0 已生成的数据

HLMF 2.0 已完成真实数据聚合：

| 产物 | 当前结果 |
|---|---:|
| pretrain source registry | 362,470 条，7 个 source 全部 `status=ok` |
| pretrain included | 108,926 条 |
| pretrain excluded/catalog audit | 253,544 / 362,470 条 |
| Val included/ignored | 1,226 / 8 条 |
| Test included/ignored | 985 / 24 条 |
| Dragon Gold manifest | 5,191 ROI |
| Dragon Gold 标签本身 trainable/ignored | 5,189 / 2 ROI |

主要报告：

```text
HLML-3.0/train_pretrain_merged/qc/finalize_train_pretrain_report.json
HLML-3.0/train_pretrain_merged/qc/pretrain_source_registry_report.json
HLML-3.0/val_merged/qc/finalize_val_report.json
HLML-3.0/test_merged/qc/finalize_test_report.json
HLML-3.0/finetune/v3-finetune-r1/sources/gold/dragon_gold_0716_v1/qc/gold_source_report.json
```

## 4. 历史基线

HLML-2.0 已完成 `v2-pretrain-r3` multitask 和 `v2-finetune-r1/r2`。历史观察是：正面张掌较准，握拳、侧向张掌和数字 1 等困难姿态较差；Val 平均关键点误差约 20 px 以上；infer 中仍有大量关键点塌缩；单独把 Gold batch 比例提高到 50% 没有改善。这些结果只用于解释 3.0 的 Gold 重平衡、自动错误分析和结构 loss 设计，不复制进新工作区。

## 5. 当前训练进度

- `inspect-geometry`、`inspect-geometry-smoke` 和 `pretrain-geometry-smoke` 已运行；smoke 训练完成 300 epoch 并保存 best checkpoint。
- 此前 `pretrain-geometry` 在前置 `check-geometry-smoke` 退出，报错为 `KeyError: 0`。根因是 gate 仍以旧列表下标读取 3.0 已改为语义 mapping 的 targets/sample weights；正式 geometry 训练没有启动，也没有生成 `geometry/` run 目录。
- 本次代码更新修复 gate。由于 smoke provenance 严格绑定 Git commit，服务器部署时将旧 smoke 目录保留为备份，并在新 commit 下重新生成标准 `smoke/`、通过 gate；正式 geometry 仍由人工启动。此前执行的 `eval-val-geometry` 不能视为有效评估，因为正式 geometry checkpoint 尚不存在。
- 人工 true-negative 复核和 multitask 正式训练尚未完成。
- `v3-finetune-r1` 当前已保存并认证 Dragon Gold，但 `configs/curate_finetune.yaml` 已按 source ID 禁用它：视频 H.264/I420 及 JPEG 抽帧域与板端无损 TIFF 评测域不一致。它不会进入后续 finetune；新录制和 disagreement 的 600/800 总预算任务尚未冻结和标注。
- `data_only`、`structure`、可选 `structure_roi_aug` 候选尚未正式训练。
- 新模型的 ONNX 导出、厂商工具链转换和上板验收尚未执行。

完成重要阶段后只更新本文，不把当前状态写回通用流程或 Quick Start。
