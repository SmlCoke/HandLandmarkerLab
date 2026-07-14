# Hand Landmarker Training System

本仓库训练 AetherSign Hand Landmarker。当前交付范围是 v2 pretrain，不包含 finetune，也不训练 Palm Detector。

pretrain 分为两个显式阶段：

1. `geometry`：只使用具有完整 21 点伪标签的 positive，先学稳定手部几何；
2. `multitask`：从 geometry best 初始化，只加入人工确认的 true negative，学习 `hand_flag` 并轻量学习 handedness。

未经人工复核的 `NEG_*_CANDIDATE` 是 teacher abstention，不是可信无手样本。系统会把它们及其 ROI 持久化到审查包，但 `make multitask` 会 fail-closed，绝不会自动把它们作为负例训练。

详细的分阶段原理、人工负例判定、JSONL 格式和逐步命令见 [Pretrain 数据与分阶段训练操作手册](docs/training_system/data_and_training.md)。历史故障证据见 [两次 pretrain 失败分析与恢复方案](docs/training_history/2026-07-14_pretrain_failure_analysis_and_recovery.md)。

## 固定接口

| 项目 | 契约 |
|---|---|
| 输入 | `float32`、NCHW、`(B,1,256,256)`、灰度 `/255` |
| 输出 0 | 42 个 landmark 值，顺序 `x0,y0,...,x20,y20` |
| 输出 1 | `hand_flag` sigmoid |
| 输出 2 | `handedness` sigmoid，Left=0、Right=1 |
| 外部原图 ROI | `scale=(1.8,1.8)`、`shift=(0,-0.1)`、wrist→middle MCP 旋转 |

Val/Test 直接读取 canonical 256×256 Hand ROI，只运行 Hand Landmarker。只有 `make infer` 才对外部原图执行 Palm → rotated ROI → Hand。

## v2

当前 registry 只提供 `models/hand_landmarker/v2.py`：

- 使用普通 ReLU，不再产生官方转换工具已拒绝的 LeakyReLU；
- 训练期使用 Conv+BN 多分支重参数化以增加优化自由度；
- 导出前把 BN 和辅助分支融合为单个 Conv；
- ONNX 严格算子集合为 `Conv/Add/Relu/MaxPool/Sigmoid/Identity`；
- 输入、输出名称、顺序和 shape 不变。

multitask checkpoint 使用 geometry-first 的 `val_multitask_score`，同时考虑 landmark MAE、presence accuracy 和 handedness accuracy，但分类误差只占较小权重。

导出 contract 同时记录训练图→部署图、部署 Keras→ONNX 两级数值一致性。

## 快速开始

实验路径和 ID 直接写在 Makefile 顶部。每次服务器实验先修改并提交这些值，然后执行：

```bash
git pull
conda activate hand-landmarker-tf29
make paths
make compile
make test
make curate
make doctor
make inspect
make smoke
make train
make eval-val
make eval-test
make infer
```

完成负例人工审查后：

```bash
make curate-reviewed
make check-multitask
make multitask
make eval-val HAND_PRETRAIN_PHASE=multitask
make eval-test HAND_PRETRAIN_PHASE=multitask
make infer HAND_PRETRAIN_PHASE=multitask
make export HAND_PRETRAIN_PHASE=multitask
```

`make pretrain` 是 geometry 的首次顺序入口：`curate → doctor → inspect → smoke → train`；它不会跳过人工复核去自动启动 multitask。

## 配置与目录

```text
configs/                    8 个当前 pretrain 配置
hand_landmarker/            数据、训练、评估、推理、导出实现
models/hand_landmarker/     v2 模型与部署融合
scripts/                    CLI 与数据门禁
tests/                      数据、路由、导出和模型契约测试
docs/training_system/       当前操作手册
docs/training_history/      历史故障证据与恢复记录
preminilary/                冻结 Palm 资产与原始参考实现
```

其他文档：

- [环境创建与服务器检查](docs/training_system/environment.md)
- [评估与人工可视化复核](docs/training_system/evaluation.md)
- [A1 板端与 ONNX 部署契约](docs/training_system/deployment_contract.md)
- [模型转换数据制作说明](docs/model_conversion/conversion_method.md)
