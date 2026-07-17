# HLML 当前训练与数据状态

更新时间：2026-07-18

本文只记录会变化的项目事实。通用原理和命令见 [完整训练流程](HLML_training_workflow.md)，最短命令清单见 [Quick Start](HLML_quick_start.md)。

## 1. 代码版本

- HLML 当前契约：3.0，Git tag：`HLML-3.0`。
- HLMF 当前契约：2.0，Git tag：`HLMF-2.0`。
- 两套系统均不兼容 HLML-2.0/HLMF-1.x 的旧配置和工作区。
- HLML 服务器验收：70 个 Python 文件通过编译，155 项测试通过；TensorFlow 2.9 已实际验证结构损失、Gold-only mask 和 Keras sample-weight 契约。
- HLMF 服务器验收：27 个 Python 文件通过编译，6 项测试通过。
- 当前服务器验收时未挂载 `/dev/nvidia0`，因此没有运行正式训练；GPU 训练仍需在 GPU 可见后执行 `make doctor`。

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
| Dragon Gold trainable/ignored | 5,189 / 2 ROI |

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

## 5. 尚未完成

- `v3-pretrain-r1` 尚未执行人工负样本复核和 geometry/multitask 正式训练。
- `v3-finetune-r1` 目前只有 Dragon Gold；新录制和 disagreement 的 600/800 总预算任务尚未冻结和标注。
- `data_only`、`structure`、可选 `structure_roi_aug` 候选尚未正式训练。
- 新模型的 ONNX 导出、厂商工具链转换和上板验收尚未执行。

完成重要阶段后只更新本文，不把当前状态写回通用流程或 Quick Start。
