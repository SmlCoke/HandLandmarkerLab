# HandLandmarkerLab（HLML 3.0）

HLML 训练适用于 A1 板端部署的轻量 Hand Landmarker。Palm Detector 是冻结的外部资产；本仓库训练并导出 Hand ROI 上的 21 点、hand presence 和 handedness 三个输出。

训练分两阶段：

```text
pretrain: geometry → multitask
finetune: human Gold + pretrain replay
```

HLML 3.0 是新的数据和实验契约，不兼容旧版工作区。可再生 ROI 直接从 `/root/autodl-tmp/DatesetFab` 读取；`/root/autodl-tmp/TrainFab/HLML-3.0` 只保存聚合标签、审计清单、Gold 工作区和训练结果。

## 文档入口

- [HLML 完整训练流程](docs/training_system/HLML_training_workflow.md)：权威操作流程、原理、人工/程序分工和排错。
- [HLML Quick Start](docs/training_system/HLML_quick_start.md)：熟悉系统后直接照着运行。
- [当前训练状态](docs/training_system/HLML_current_training_status.md)：服务器目录、数据和实验状态；该文档允许随项目推进更新。
- [下一步计划](docs/training_system/HLML_next_step_plan.md)：当前两天人工 Gold 与候选实验安排。
- [专项工具文档](docs/training_system/tools/)：环境、数据/训练、评估和部署接口的深入参考。

HLMF 数据制作说明位于 `/root/HandLandmarksFab/docs/annotating_system/`。实际命令以两个仓库的 Makefile 为准。

## 最短检查

```bash
cd /root/HandLandmarkerLab
git pull --ff-only
conda activate hand-landmarker-tf29
make paths
make compile
make test-unit
make doctor
```

模型固定接口：`float32 NCHW (B,1,256,256)`；输出顺序为 42 个 landmark 坐标、`hand_flag`、`handedness`。正式导出仍执行 ONNX 数值一致性和厂商算子/体积契约检查。
