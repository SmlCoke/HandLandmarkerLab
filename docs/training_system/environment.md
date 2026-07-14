# 训练环境创建与服务器检查

目标环境是 AutoDL Ubuntu 20.04、Python 3.8、TensorFlow 2.9.0、CUDA 11.2、cuDNN 8。`environment.yml` 创建 `hand-landmarker-tf29`；不要额外安装 `tensorflow-gpu`，也不要在 Conda 环境中再安装一份会抢占宿主动态库的 CUDA/cuDNN。

## 创建或更新环境

```bash
cd /path/to/HandLandmarkerLab
make env-create
conda activate hand-landmarker-tf29
python -m pip check
```

已有专用环境需要按仓库定义对齐时：

```bash
make env-update
```

该命令带 `--prune`，只应用于本项目专用环境。

## 使用镜像已有环境

先只读检查，不要直接升级：

```bash
which python
python --version
python -m pip show tensorflow numpy protobuf h5py
nvidia-smi
nvcc --version
python -c "import tensorflow as tf; print(tf.__version__); print(tf.sysconfig.get_build_info()); print(tf.config.list_physical_devices('GPU'))"
```

`nvidia-smi` 正常只说明驱动可见，不代表 TensorFlow 已经成功加载 CUDA/cuDNN。

## 项目门禁

服务器代码更新后执行：

```bash
conda activate hand-landmarker-tf29
make paths
make compile
make test
```

`make paths` 会打印 Makefile 内固定的训练系统根、pretrain ID，以及由两者派生的 review、curated 和 run 路径。它们不再主要依赖 shell 临时环境变量。

数据完成提纯后继续：

```bash
make doctor
make inspect-geometry
```

- `doctor` 使用 `configs/train_geometry.yaml` 检查 Python/TensorFlow 版本、TensorFlow build info、GPU 可见性和关键依赖；
- `compile` 只做 Python 语法检查；
- `test` 运行标准库 unittest；服务器 TensorFlow 环境还会运行 v2 构图与融合数值测试；
- `inspect-geometry` 审计 geometry Train、Val、锁定 Test，并计算图片 SHA-256 与跨 split 泄漏。

第一次还没有 curated 快照时先运行 `make pretrain-curate`；不要创建空 JSONL 来绕过 doctor/inspect。

## 固定实验路径

Makefile 顶部直接定义：

```make
HAND_TRAIN_ROOT := /root/autodl-tmp/TrainFab/HLML-2.0
HAND_PRETRAIN_ID := v2-pretrain-r1
```

服务器路径改变或开始新实验时，修改并提交 Makefile。评估、推理、导出使用 `*-geometry` 或 `*-multitask` 显式目标选择子阶段，不需要操作者维护 phase 环境变量。正式训练的数据根和 ID 应留在仓库版本中，避免只存在于某次 shell 会话。

## TensorFlow 2.9 注意事项

- Python 3.8、TensorFlow/Keras 2.9.0、NumPy 1.23.5、protobuf 3.19.6、h5py 3.7.0、flatbuffers 1.12 是固定兼容组，不要单独升级其中一项；
- 训练配置默认关闭 deterministic GPU，因为 TensorFlow 2.9 没有 RTX 3090 depthwise backprop 的 deterministic 实现；
- 配置默认关闭 mixed precision，先建立可复现 FP32 基线；
- 日志中的 TensorFlow build CUDA/cuDNN 版本不是当前进程实际加载动态库的充分证明；最终以 GPU 可见、训练实际进入 cuDNN 且 smoke 通过为准；
- `make pretrain-geometry` 和 `make pretrain-multitask` 拒绝覆盖既有 run；新实验修改 `HAND_PRETRAIN_ID`，续训才使用 `training.resume_checkpoint`。

完整训练步骤见 [Pretrain 数据与分阶段训练操作手册](data_and_training.md)。
