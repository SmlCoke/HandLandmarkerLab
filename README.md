# Hand Landmarker Training System

本仓库提供 AetherSign Hand Landmarker 的完整训练闭环：读取 HandLandmarkerFab 的 canonical JSONL、两阶段训练、在现成 `256×256` Hand ROI 上做人工金标自动评估、任意图片文件夹可视化推理，以及经过接口和数值校验的 ONNX 导出。

本系统**只训练 Hand Landmarker，不训练 Palm Detector**。Val/Test 输入已经是 `256×256` Hand ROI，评估只加载并运行 Hand Landmarker，绝不加载或调用 Palm。只有 `make infer` 处理任意外部图片时才使用 `preminilary/palm/model_opt.onnx`，执行 Palm → ROI → Hand；其阈值、anchor 解码、NMS、ROI 参数和坐标变换均以 A1 板端程序为准。

## 固定接口

| 项目 | 契约 |
|---|---|
| 输入 | `float32`、NCHW、`(B,1,256,256)`、灰度值除以 255 |
| 输出 0 | 21 个二维点，42 个值，顺序 `x0,y0,...,x20,y20` |
| 输出 1 | `hand_flag` sigmoid 概率 |
| 输出 2 | `handedness` sigmoid 概率，`Left=0`、`Right=1` |
| 外部图片 Hand ROI | `scale=(1.8,1.8)`、`shift=(0,-0.1)`、wrist→middle MCP 旋转 |
| 外部图片 Palm | score `0.50`、14-head NMS `0.30`、selected-candidate suppression `0.35`、最多 2 个 |

模型内部结构可以通过新增 `models/hand_landmarker/<version>.py` 演进，但 Hand I/O 接口和外部图片/板端端到端路径的几何不得改变。后两行 Palm/ROI 契约不参与 Val/Test。第一版 `models/hand_landmarker/v1.py` 是 `preminilary/hand/model.py` 的独立副本。

## 快速开始

所有命令均在仓库根目录执行。先确认服务器实际数据根；默认是 `/root/autodl-tmp`，也可覆盖：

```bash
export HAND_DATA_ROOT=/root/autodl-tmp
```

创建环境（本仓库不会自动安装依赖，以下命令由使用者执行）：

```bash
make env
conda activate hand-landmarker-tf29
make doctor
make inspect
```

`make inspect` 会依次审计 pretrain Train、finetune Train、Val 与 Test，并在训练配置的检查中把当前 Train、Val、锁定 Test 两两比较。`global_crop_id`、`source_group_id`、解析后的 crop 路径或图像 SHA-256 任一精确跨集合重叠都会使检查失败；同名文件仅告警。pretrain 与 finetune 之间允许 pseudo replay，因而不把两个训练阶段彼此比较视为数据泄漏门禁。

训练：

```bash
make train-pretrain
make train-finetune
```

也可用 `make train` 顺序执行两阶段。采样器在每个 batch 中用整数配额保持 70% positive、25% runtime negative、5% low-Palm negative；第二阶段还固定约 40% Gold 与 60% pseudo replay。`sampling_weight` 只在已选定的 supervision×sample-type 单元内选择记录。Test 不参与 checkpoint、阈值、增强或量化方案选择。

评估、人工复核推理和导出：

```bash
make eval-val
make eval-test
make infer
make export
```

- `eval-val`：直接读取 canonical `256×256` Hand ROI，只运行 Hand；可做 presence threshold sweep，结果只用于 Val 选型。
- `eval-test`：直接读取 canonical `256×256` Hand ROI，只运行 Hand；使用已冻结阈值，不做 threshold tuning。
- `infer`：实际执行 Palm → rotated ROI → Hand，并输出叠加骨架图和 `predictions.jsonl`。
- `export`：导出静态 batch=1 ONNX，检查输入/输出顺序、shape、FLOAT32 类型、A1 算子及属性约束，并用 zeros、ones 和随机输入做 Keras/ONNX 数值一致性验证。

修改默认配置或临时指定另一份配置：

```bash
make train-pretrain TRAIN_PRETRAIN_CONFIG=configs/train_pretrain.yaml
make infer INFER_CONFIG=/path/to/my_infer.yaml
```

## 数据入口

正式 loader 只读取以下 canonical 文件，不读取 manifest，也不会把图片目录中的旧文件 glob 成训练样本：

```text
Stage 1  train_pretrain_merged/05_labels/hand_training_labels_pretrain.jsonl
Stage 2  train_finetune_merged/05_labels/hand_training_labels_finetune.jsonl
Val      val_merged/05_labels/hand_validation_labels.jsonl
Test     test_merged/05_labels/hand_test_labels.jsonl
```

图片优先严格读取每行的 `crop_path`。若数据整体搬迁，只能通过 YAML 中显式的 `data_root/image_roots` 做可审计 rebase；不会静默回退到历史 `source_crop_path`。

## 评估口径

Val/Test canonical JSONL 的每一行已经对应一张 `256×256` Hand ROI。`eval-val` 与 `eval-test` 只解析该行的 `crop_path`，把 ROI 直接送入 Hand Landmarker；它们不读取原图、不构造 ROI，也不实例化 Palm Detector。

因此报告中的 presence、landmark 和 handedness 都是 **Hand ROI 级指标**，不包含 Palm detection、原图级 recall 或端到端级联指标。需要在任意外部图片上人工复核完整路径时，使用独立的 `make infer`：Palm → rotated ROI → Hand。该推理结果不得混入 Val/Test 指标。

Val/Test 默认在 `${HAND_DATA_ROOT}/hand_landmarker_runs/v1/eval/{val,test}` 写入 `metrics.json` 与 `predictions.jsonl`。逐 ROI 记录保存 21 个归一化预测点、Gold positive 的逐点像素误差、输出范围健康字段、是否触发板端 `/256` 兼容缩放，以及 Gold/模型 SHA-256；汇总报告另保存配置 SHA-256。它们用于复现 Hand ROI 评估，不代表 Palm 或整图级联性能。

默认训练产物位于 `${HAND_DATA_ROOT}/hand_landmarker_runs/v1/{pretrain,finetune}`，ONNX 与契约报告位于 `${HAND_DATA_ROOT}/hand_landmarker_runs/v1/export`。精确文件清单与恢复语义见[数据、权重与两阶段训练](docs/training_system/data_and_training.md)和[A1 板端与 ONNX 部署契约](docs/training_system/deployment_contract.md)。

## 目录结构

```text
configs/                    # 两阶段训练、Val/Test、推理、导出配置
hand_landmarker/            # 数据、训练、Hand ROI 评估、外部图片推理和导出实现
models/hand_landmarker/     # 可版本化模型定义，v1 为初始结构
scripts/                    # 命令行入口
tests/                      # 不需要训练数据的契约/几何/采样测试
docs/training_system/       # 详细操作和设计文档
preminilary/                # 初始模型、板端程序和冻结 Palm 资产（依据）
```

## 文档

- [环境创建与服务器检查](docs/training_system/environment.md)
- [数据、权重与两阶段训练](docs/training_system/data_and_training.md)
- [评估与人工可视化复核](docs/training_system/evaluation.md)
- [A1 板端与 ONNX 部署契约](docs/training_system/deployment_contract.md)
- [标注与数据制作总流程](docs/annotation/dataset_preparation_workflow.md)

## 安全检查

```bash
make compile
make test
```

本地没有 TensorFlow 2.9 环境时，仍可运行纯 Python/NumPy/OpenCV 契约测试；训练、ONNX 导出和真实模型推理必须在文档规定的环境中完成。系统不会自行安装依赖、启动远端训练或修改冻结数据。
