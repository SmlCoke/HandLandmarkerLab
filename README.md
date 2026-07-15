# Hand Landmarker Training System

本仓库训练 AetherSign Hand Landmarker。当前交付范围是 v2 pretrain，不包含 finetune，也不训练 Palm Detector。

## I. 两阶段训练流程总览

当前任务是训练一个适合 A1 板端部署的轻量 Hand Landmarker，使其在项目采集域内尽可能接近 Google MediaPipe Hand Landmarker (pretrain stage)，并通过少量人工金标准数据纠正教师模型的系统误差 (finetune stage)。

采用两个阶段：

1. **第一阶段：教师—学生伪标签学习**。使用 Google MediaPipe 生成的大规模伪标签训练学生模型，获得覆盖手型、光照、姿态、ROI 偏移和背景变化的基座模型。这一步无需人工精细标注训练集，因此可以在有限的时间内利用教师模型 Google MediaPipe 制造出大量训练集。这一步的目的是让我们自己的模型训练得非常像 Google MediaPipe Hand Landmarker，尽可能覆盖训练域的多样性。由于教师模型的局限性，学生模型在某些情况下可能会出现漏检、关键点偏差和 presence 错误。
2. **第二阶段：人工金标准数据微调**。从训练域中选择少量高价值 ROI 进行精细人工复核，再与第一阶段的高质量伪标签混合训练，纠正教师漏检、关键点偏差和 presence 错误。

整个过程中，验证集 val 和测试集 test 不变，并且都是人工精细标注的金标准数据。

这条路线属于教师—学生伪标签学习或模型模仿。由于当前 MediaPipe Python API 没有导出 hand presence 的连续分数，也没有逐点置信度，因此目前主要使用离散 presence、handedness 和 21 点坐标监督，不是完整的 soft-logit 知识蒸馏。

**数据集角色**

| 数据集 | 当前规模（以 Hand ROI 计） | 标签质量 | 是否更新模型参数 | 主要用途 |
|---|---:|---|---|---|
| Train pseudo | 10w 张左右 | Google MediaPipe 伪标签，存在噪声 | 是，第一阶段 | 学习教师行为和数据域多样性 |
| Train gold | 未制作 | 人工复核的近似金标准 | 是，第二阶段 | 纠正伪标签偏差和困难样本 |
| Val | 1000 张左右 | 人工金标准 | 否 | 选 checkpoint、调 presence 阈值、early stopping |
| Test | 1000 张左右 | 人工金标准 | 否 | 最终冻结方案的独立评测 |

### 1.1 第一阶段：pretrain

pretrain 分为两个显式阶段：

1. `geometry`：只使用具有完整 21 点伪标签的 positive，先学稳定手部几何；
2. `multitask`：从 geometry best 初始化，只加入人工确认的 true negative，学习 `hand_flag` 并轻量学习 handedness。

未经人工复核的 `NEG_*_CANDIDATE` 是 teacher abstention，不是可信无手样本。系统会把它们及其 ROI 持久化到审查包，但 `make pretrain-multitask` 会 fail-closed，绝不会自动把它们作为负例训练。

详细的分阶段原理、删除式负例复核和逐步命令见 [Pretrain 数据与分阶段训练操作手册](docs/training_system/data_and_training.md)。历史故障证据见 [两次 pretrain 失败分析与恢复方案](docs/training_history/2026-07-14_pretrain_failure_analysis_and_recovery.md)、[v2 smoke 失败与架构恢复](docs/training_history/2026-07-15_v2_smoke_failure_and_architecture_recovery.md)、[preflight 量化失败分析](docs/training_history/2026-07-15_preflight_quantization_failure.md) 和 [训练来源负例及 ROI 域审计](docs/training_history/2026-07-15_train_source_negative_and_roi_audit.md)。

## II. 固定接口

| 项目 | 契约 |
|---|---|
| 输入 | `float32`、NCHW、`(B,1,256,256)`、灰度 `/255` |
| 输出 0 | 42 个 landmark 值，顺序 `x0,y0,...,x20,y20` |
| 输出 1 | `hand_flag` sigmoid |
| 输出 2 | `handedness` sigmoid，Left=0、Right=1 |
| 外部原图 ROI | `scale=(1.8,1.8)`、`shift=(0,-0.1)`、wrist→middle MCP 旋转 |

Val/Test 直接读取 canonical 256×256 Hand ROI，只运行 Hand Landmarker。只有 `make infer-geometry` 或 `make infer-multitask` 才对外部原图执行 Palm → rotated ROI → Hand。

## III. 模型定义

### 3.1 当前版本：v2

当前 registry 只提供 `models/hand_landmarker/v2.py`：

- 使用普通 ReLU，不再产生官方转换工具已拒绝的 LeakyReLU；
- 训练期使用稳定的 Conv+BN 残差块，残差尾部零初始化，避免深层输出爆炸；
- landmark head 从归一化坐标 `0.5` 起步，两个概率 head 使用极小的确定性权重、从接近 `0.5` 起步；
- 导出前把每个 BN 精确折叠为单个带 bias 的 Conv；
- depthwise/grouped Conv 的 group 保守限制为不超过 128；
- ONNX 严格算子集合为 `Conv/Add/Relu/MaxPool/Sigmoid/Identity/Reshape`；
- 导出器拒绝全零 Conv、恒定输出和超过 group 上限的图，并对实际 ONNX 强制执行 `15 MiB` 大小上限；
- 输入、输出名称、顺序和 shape 不变。

multitask checkpoint 使用 geometry-first 的 `val_multitask_score`，同时考虑 landmark MAE、presence accuracy 和 handedness accuracy，但分类误差只占较小权重。

导出 contract 同时记录训练图→部署图、部署 Keras→ONNX 两级数值一致性。

## IV. 快速开始

实验路径和 ID 直接写在 Makefile 顶部。每次服务器实验先修改并提交这些值，然后执行：

```bash
git pull
conda activate hand-landmarker-tf29
make paths
make compile
make pretrain-curate
make test
make doctor
make inspect-geometry
make pretrain-geometry-smoke
make pretrain-geometry
make eval-val-geometry
make eval-test-geometry
make infer-geometry
```

`make test` 需要当前 ID 的 curated Train 以及已冻结 Val/Test，因此放在 `make pretrain-curate` 之后。它先运行单元测试，再生成一个不可用于精度评估的 v2 ONNX 和真实 Train/Val/Test 转换输入。preflight 使用确定性的非零量化探针权重，避免“全零卷积、恒定输出”让 INT8 工具无法计算量化点；这些一次性权重不会改变正式训练初始化。该产物用于在正式训练前提交官方工具链验证结构兼容性：

```text
hand_landmarker_runs/<HAND_PRETRAIN_ID>/export/preflight/
├── hand_landmarker_v2_untrained.onnx
├── hand_landmarker_v2_untrained.contract.json
├── untrained.weights.h5
└── model_conversion/datasets.zip
```

Curate 生成的训练 JSONL 直接引用 `${HAND_TRAIN_ROOT}/train_sources/` 中的 ROI；`train_pretrain_curated/` 只保存索引、决策、哈希清单和报告。人工审查工作区中的候选图是唯一额外产生的 ROI 文件，不作为训练输入。

`make pretrain-curate` 会自动创建 `hand_landmarker_reviews/<HAND_PRETRAIN_ID>/negative_candidates/`。三人分工浏览其中图片，删除所有“有手或不确定”的 ROI，只保留明确无手背景；不要新增、改名、移动或编辑图片。完成后执行：

```bash
make pretrain-curate-reviewed
make check-multitask-data
make pretrain-multitask
make eval-val-multitask
make eval-test-multitask
make infer-multitask
make export-multitask
```

Makefile 不再提供 `train`、`pretrain`、`multitask`、`inspect` 等含义不完整的别名；每个公开目标都显式标出 pretrain 子阶段和动作。

## V. 配置与目录

```text
configs/                    9 个当前 pretrain 配置
hand_landmarker/            数据、训练、评估、推理、导出实现
models/hand_landmarker/     v2 模型与部署融合
scripts/                    CLI 与数据门禁
tests/                      数据、路由、导出和模型契约测试
docs/training_system/       当前操作手册
docs/training_history/      历史故障证据与恢复记录
docs/annotation/            仅说明与独立 HandLandmarkerFab 仓库的边界
preminilary/                冻结 Palm 资产与原始参考实现
```

其他文档：

- [环境创建与服务器检查](docs/training_system/environment.md)
- [评估与人工可视化复核](docs/training_system/evaluation.md)
- [A1 板端与 ONNX 部署契约](docs/training_system/deployment_contract.md)
- [模型转换数据制作说明](docs/model_conversion/conversion_method.md)
