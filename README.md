# HandLandmarkerLab（HLML 4.0）

HLML 是 Iris Hand Landmarker 的训练、固定 ROI 评估、文件夹级联推理和 ONNX/A1 导出系统。系统直接读取 HLMF 3.0 在 `HAND_DATASET_ROOT` 发布的 manifest；`HAND_TRAIN_ROOT` 只保存零拷贝索引快照、报告、checkpoint 与导出物。

Iris v3 提供 `v3-pro`、`v3-max`、`v3-lite` 三档结构，并保留 `v2` 不变。`v3-pro` 与 v2 同构；`v3-max` 使用四个 Conv/Depthwise+BN 训练分支以及可融合的 1x1/identity BN 分支，部署时折叠回与 v2 相同的单分支参数量；`v3-lite` 缩减通道。三档均保持固定输入、三输出语义、ROI 几何、loss 和 A1 接口。

公共配置保持最少且单一职责：`datasets.yaml` 管数据成员，`training.yaml` 管三阶段训练，`evaluation.yaml` 管固定 ROI Val/Test，`inference.yaml` 管原图文件夹推理，`deploy.yaml` 管 ONNX/A1 模型与配套转换数据导出。

## 边界

- 训练成员由 dataset、negative dataset、CVAT-reviewed hard dataset 和可选 recorded Gold dataset ID 控制，不手工拼 JSONL、不复制 ROI。
- geometry 只用可靠 positive；multitask 加入审核发布的真负样本；multi-finetune 混合困难/Gold positive-negative 与强制 pretrain replay。
- 困难样本挖掘只读取 Train，并按 snapshot ledger 保证多轮不重复。
- Val/Test 只读取 HLMF 已生成并经 CVAT 复核的固定 Hand ROI，不运行 Palm Detector，也不从原图重建 ROI。
- 当前不报告 Palm 漏检、部分双手召回率或原图级联性能。

## 文档

- [完整训练工作流](docs/training_system/HLML_training_workflow.md)
- [Quick Start](docs/training_system/HLML_quick_start.md)
- [当前状态](docs/training_system/HLML_current_training_status.md)
- [下一阶段计划](docs/training_system/HLML_next_step_plan.md)
- [常见问题与解答](docs/annotating_system/HLML_qa.md)
- [数据与训练契约](docs/training_system/tools/data_and_training.md)
- [固定 ROI 评估](docs/training_system/tools/evaluation.md)
- [部署契约](docs/training_system/tools/deployment_contract.md)

## Iris v3 选择与训练前导出

配置默认使用 `v3-pro`。三台克隆服务器分别设置：

```bash
export HLML_MODEL_VERSION=v3-pro   # 另两台使用 v3-max / v3-lite
export HLML_SNAPSHOT_ID=iris-v3-data-r1
export HLML_EXPERIMENT_ID=iris-v3-pro-r1
make config-check
make data-audit HLML_STAGE=geometry
make export-preflight HLML_STAGE=geometry
make geometry
```

`make export-preflight` 在没有正式 checkpoint 时生成未训练 ONNX、contract 和 `model_conversion/datasets.zip`，只用于 A1 图/算子兼容性检查，不代表精度模型。文件夹推理默认使用 `palm_detector/eos-2.1/model_opt.onnx` 和 `/root/autodl-tmp/DatesetFab/InferSource/0718/images`。

## 公共入口

```bash
make help
make config-check
make export-preflight HLML_STAGE=geometry
make data-audit HLML_STAGE=geometry
make geometry
make infer HLML_STAGE=geometry
make multitask
make mine-hard MINING_ARGS='--round-id r01 --max-rois 1000'
make multi-finetune
make val HLML_STAGE=multi_finetune
make freeze-winner HLML_STAGE=multi_finetune
make locked-test
make export HLML_STAGE=multi_finetune
make compile test
```
