# 训练环境与检查

生产训练环境保持 AutoDL Ubuntu 20.04、Python 3.8、TensorFlow 2.9.0、CUDA 11.2 与 cuDNN 8。使用仓库 `environment.yml` 创建专用环境，不额外安装第二套 TensorFlow/CUDA 动态库。

```bash
cd /path/to/HandLandmarkerLab
conda env create -f environment.yml
conda activate hand-landmarker-tf29
python -m pip check
make environment-check
```

已有专用环境时只需激活；依赖文件确实发生变化时才执行 `conda env update -f environment.yml --prune`，不要在每次训练前重复更新。

HLML 4.0 的只读检查入口：

```bash
make paths
make config-check
make environment-check
make compile
make test
```

`environment-check` 使用 `configs/training.yaml` 的 geometry profile 检查 Python/TensorFlow 版本、build metadata、GPU 可见性和关键依赖。`config-check` 解析三个 training profile、Val/Test evaluation profile，以及独立 inference/deploy 配置，不需要数据集存在。

开始训练前先在 `configs/datasets.yaml` 填写 HLMF 发布 ID，再执行相应 stage 的 data audit；不要创建空 JSONL 绕过门控。

```bash
make data-audit HLML_STAGE=geometry
make geometry
```

固定工作目录默认值：

```make
HAND_DATASET_ROOT ?= /root/autodl-tmp/DatesetFab
HAND_TRAIN_ROOT ?= /root/autodl-tmp/TrainFab/HLML-4.0
HLML_SNAPSHOT_ID ?= v4-r1
HLML_EXPERIMENT_ID ?= v4-r1
HLML_RELEASE_ID ?= v4-r1
```

TensorFlow 2.9 环境继续使用 FP32 默认配置。GPU 可见、实际 cuDNN 训练和 smoke 结果才是运行时可用性的证据；TensorFlow build metadata 本身不是动态库已正确加载的充分证明。
