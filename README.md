# HandLandmarkerLab（HLML 4.0）

HLML 是 Hand Landmarker v2 的训练、固定 ROI 评估和 ONNX/A1 导出系统。4.0 直接读取 HLMF 3.0 在 `HAND_DATASET_ROOT` 发布的 manifest；`HAND_TRAIN_ROOT` 只保存零拷贝索引快照、报告、checkpoint 与导出物。

本次升级不兼容旧配置、旧数据契约或旧命令。模型结构、ROI 几何、训练损失、checkpoint 以及 ONNX/A1 接口保持 v2 契约，不引入辅助 head 或结构实验。

公共配置保持最少且单一职责：`datasets.yaml` 管数据成员，`training.yaml` 管三阶段训练，`evaluation.yaml` 管固定 ROI Val/Test，`inference.yaml` 管原图文件夹推理，`deploy.yaml` 只管 ONNX/A1 模型导出。

## 边界

- 训练成员由 dataset、negative dataset 和 selection ID 控制，不手工拼 JSONL、不复制 ROI。
- geometry 只用可靠 positive；multitask 加入审核发布的真负样本；multi-finetune 混合困难 positive 与强制 pretrain replay。
- 困难样本挖掘只读取 Train。
- Val/Test 只读取 HLMF 已生成并经 CVAT 复核的固定 Hand ROI，不运行 Palm Detector，也不从原图重建 ROI。
- 当前不报告 Palm 漏检、部分双手召回率或原图级联性能。

## 文档

- [完整训练工作流](docs/training_system/HLML_training_workflow.md)
- [Quick Start](docs/training_system/HLML_quick_start.md)
- [当前状态](docs/training_system/HLML_current_training_status.md)
- [下一阶段计划](docs/training_system/HLML_next_step_plan.md)
- [数据与训练契约](docs/training_system/tools/data_and_training.md)
- [固定 ROI 评估](docs/training_system/tools/evaluation.md)
- [部署契约](docs/training_system/tools/deployment_contract.md)

## 公共入口

```bash
make help
make config-check
make data-audit HLML_STAGE=geometry
make geometry
make multitask
make mine-hard
make multi-finetune
make val HLML_STAGE=multi_finetune
make freeze-winner HLML_STAGE=multi_finetune
make locked-test
make export HLML_STAGE=multi_finetune
make compile test
```
