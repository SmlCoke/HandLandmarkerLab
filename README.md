# Hand Landmarker Training System

本仓库提供 AetherSign Hand Landmarker 的完整训练闭环：读取 HandLandmarkerFab 的 canonical JSONL、可独立交付的 pseudo pretrain、可选的 Gold + pseudo finetune、在现成 `256×256` Hand ROI 上做人工金标自动评估、任意图片文件夹可视化推理，以及经过接口和数值校验的 ONNX 导出。**finetune 不是运行系统的前置条件**：没有 finetune 数据时，pretrain 模型可以独立完成训练、Val/Test 评估、外部图片推理和 ONNX 导出；数据就绪后再切换到 finetune 路线。

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

`MODEL_STAGE` 的默认值是 `pretrain`。因此不带阶段参数的 `make inspect` 只审计 pretrain Train、Val 与锁定 Test，完全不会读取 finetune JSONL；当前没有 finetune 数据时也应正常完成。检查会在当前 Train、Val、Test 间两两比较，`global_crop_id`、`source_group_id`、解析后的 crop 路径或图像 SHA-256 任一精确跨集合重叠都会使检查失败；同名文件仅告警。

只使用现有 pretrain 数据即可完成默认闭环：

```bash
make inspect
make train
make eval-val
# 在 Val 上选定并冻结 pretrain 阶段的 presence threshold 后：
make eval-test
make infer
make export
```

以上通用目标全部解析到 pretrain 配置、pretrain checkpoint 和带 `pretrain` 的独立输出目录。`make train` **只训练一个当前阶段**，不会隐式启动 finetune。pretrain 采样器在每个 batch 中用整数配额保持 70% positive、25% runtime negative、5% low-Palm negative；`sampling_weight` 只在已选定的 supervision×sample-type 单元内选择记录。Test 不参与 checkpoint、阈值、增强或量化方案选择。

finetune 数据准备完成后，可在每条命令上显式切换：

```bash
make MODEL_STAGE=finetune inspect
make MODEL_STAGE=finetune train
make MODEL_STAGE=finetune eval-val
# 在 Val 上独立冻结 finetune 阶段的 presence threshold 后：
make MODEL_STAGE=finetune eval-test
make MODEL_STAGE=finetune infer
make MODEL_STAGE=finetune export
```

也可使用不会受 `MODEL_STAGE` 影响的显式目标：`inspect-pretrain`/`inspect-finetune`、`inspect-val-pretrain`/`inspect-val-finetune`、`inspect-test-pretrain`/`inspect-test-finetune`、`train-pretrain`/`train-finetune`（训练别名为 `pretrain`/`finetune`）、`eval-val-pretrain`/`eval-val-finetune`、`eval-test-pretrain`/`eval-test-finetune`、`infer-pretrain`/`infer-finetune` 和 `export-pretrain`/`export-finetune`。只有确认 finetune canonical 数据已经存在时，才使用 `make inspect-all` 或 `make train-all`；它们才会顺序处理两个阶段。

- `eval-val`：直接读取 canonical `256×256` Hand ROI，只运行当前阶段 Hand；可做 presence threshold sweep，结果只用于该阶段的 Val 选型。
- `eval-test`：直接读取 canonical `256×256` Hand ROI，只运行当前阶段 Hand；使用该阶段已冻结的阈值，不做 threshold tuning。
- `infer`：对外部图片实际执行 Palm → rotated ROI → 当前阶段 Hand，并输出叠加骨架图和 `predictions.jsonl`。
- `export`：导出当前阶段静态 batch=1 ONNX，检查输入/输出顺序、shape、FLOAT32 类型、A1 算子及属性约束，并用 zeros、ones 和随机输入做 Keras/ONNX 数值一致性验证。

pretrain 与 finetune 必须各自在 Val 上选择并冻结 threshold；不得把一个阶段的 threshold 直接用于另一个阶段。Val/Test 的阶段 wrapper 各自显式保存 `evaluation.hand_flag_threshold`：完成 `eval-val-<stage>` 后，把选定值写入对应的 `eval_test_<stage>.yaml`，再运行 Test。配置中的 `model.checkpoint_stage` 显式声明 checkpoint 来源，运行时会把它写入 provenance；该值不会根据文件路径或 finetune 数据是否存在来猜测。Make 的默认路由会让阶段、checkpoint 和输出目录保持一致，评估、推理、ONNX 和 contract 默认输出目录也都包含阶段名，两个阶段不会静默覆盖彼此的产物。

修改默认配置、切换阶段或临时覆盖模型与输出：

```bash
make train-pretrain TRAIN_PRETRAIN_CONFIG=configs/train_pretrain.yaml
make MODEL_STAGE=finetune infer
python scripts/evaluate.py --config configs/eval_val.yaml \
  --model-path /path/to/checkpoint.weights.h5 --output-dir /path/to/eval-output
python scripts/infer_folder.py --config configs/infer.yaml \
  --model-path /path/to/checkpoint.weights.h5 --output-dir /path/to/infer-output
python scripts/export_onnx.py --config configs/export.yaml \
  --weights-path /path/to/checkpoint.weights.h5 \
  --output-path /path/to/model.onnx --contract-path /path/to/model.contract.json
```

三个入口都支持 `--overwrite`，但只有在确认目标产物可以替换时才使用。CLI 覆盖只改变本次运行，不会改写 YAML，也不会根据自定义路径自动改变 `model.checkpoint_stage`；因此必须选用与权重真实来源一致的阶段配置。完整参数以各脚本 `--help` 为准。

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

Val/Test 默认在 `${HAND_DATA_ROOT}/hand_landmarker_runs/v1/eval/<stage>/{val,test}` 写入 `metrics.json` 与 `predictions.jsonl`，其中 `<stage>` 是 `pretrain` 或 `finetune`。逐 ROI 记录保存 21 个归一化预测点、Gold positive 的逐点像素误差、输出范围健康字段、是否触发板端 `/256` 兼容缩放，以及 Gold/模型 SHA-256；汇总报告另保存配置 SHA-256，并把配置的 `model.checkpoint_stage` 记为 `model_checkpoint_stage`。它们用于复现某个阶段的 Hand ROI 评估，不代表 Palm 或整图级联性能。

默认训练产物位于 `${HAND_DATA_ROOT}/hand_landmarker_runs/v1/{pretrain,finetune}`，推理结果位于 `${HAND_DATA_ROOT}/inference/output/<stage>`，ONNX 与契约报告位于 `${HAND_DATA_ROOT}/hand_landmarker_runs/v1/export/<stage>`。精确文件清单与恢复语义见[数据、权重与可选两阶段训练](docs/training_system/data_and_training.md)和[A1 板端与 ONNX 部署契约](docs/training_system/deployment_contract.md)。

## 目录结构

```text
configs/                    # 可选两阶段训练、按阶段路由的 Val/Test、推理、导出配置
hand_landmarker/            # 数据、训练、Hand ROI 评估、外部图片推理和导出实现
models/hand_landmarker/     # 可版本化模型定义，v1 为初始结构
scripts/                    # 命令行入口
tests/                      # 不需要训练数据的契约/几何/采样测试
docs/training_system/       # 详细操作和设计文档
preminilary/                # 初始模型、板端程序和冻结 Palm 资产（依据）
```

## 文档

- [环境创建与服务器检查](docs/training_system/environment.md)
- [数据、权重与可选两阶段训练](docs/training_system/data_and_training.md)
- [评估与人工可视化复核](docs/training_system/evaluation.md)
- [A1 板端与 ONNX 部署契约](docs/training_system/deployment_contract.md)
- [标注与数据制作总流程](docs/annotation/dataset_preparation_workflow.md)

## 安全检查

```bash
make compile
make test
```

本地没有 TensorFlow 2.9 环境时，仍可运行纯 Python/NumPy/OpenCV 契约测试；训练、ONNX 导出和真实模型推理必须在文档规定的环境中完成。系统不会自行安装依赖、启动远端训练或修改冻结数据。
