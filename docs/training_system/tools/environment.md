# 训练环境与检查

生产训练环境为 AutoDL Ubuntu 20.04、Python 3.8、TensorFlow 2.9.0、CUDA 11.2、cuDNN 8 与 RTX 3090 24 GB。使用仓库 `environment.yml` 创建 `hand-landmarker-tf29`，不额外安装第二套 TensorFlow/CUDA 动态库。

当前服务器必须在激活环境后设置以下变量，TensorFlow 才能稳定枚举并使用 GPU：

```bash
conda activate hand-landmarker-tf29
readonly CUDA_LIBRARY_DIR=/usr/local/cuda-11.2/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="$CUDA_LIBRARY_DIR:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDNN_V8_API_DISABLED=1
python -m pip check
make environment-check
```

依赖文件确实发生变化时才执行 `conda env update -f environment.yml --prune`；本轮 v3 实现没有新增依赖。

只读检查入口：

```bash
make paths
make config-check
make environment-check
make compile
make test
```

`environment-check` 检查 Python/TensorFlow 版本、build metadata、GPU 可见性和关键依赖；GPU 可见与实际 smoke 才是动态库正确加载的证据。`config-check` 解析三个 training profile、Val/Test evaluation profile 以及独立 inference/deploy 配置。

默认工作身份：

```make
HAND_DATASET_ROOT ?= /root/autodl-tmp/DatesetFab
HAND_TRAIN_ROOT ?= /root/autodl-tmp/TrainFab/HLML-4.0
HLML_SNAPSHOT_ID ?= iris-v3-data-r1
HLML_EXPERIMENT_ID ?= iris-v3-pro-r1
HLML_RELEASE_ID ?= iris-v3-pro-r1
HLML_MODEL_VERSION ?= v3-pro
```

训练前先确认 `configs/datasets.yaml` 的发布 ID，执行对应 stage 的 `make data-audit`；不得创建空 JSONL 绕过门禁。
