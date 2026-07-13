# 训练环境创建与验证

本文面向 AutoDL Ubuntu 20.04 服务器。项目配置以 Python 3.8、TensorFlow 2.9.0、CUDA 11.2 和 cuDNN 8 为兼容基线；本次实现只提供环境定义和验证入口，**没有在本机或服务器安装任何依赖**。

## 1. 文件职责

- `environment.yml`：创建名为 `hand-landmarker-tf29` 的 Python 3.8 Conda 环境，并从仓库根目录安装 `requirements.txt`。
- `requirements.txt`：训练、评测、推理和 ONNX 导出的固定运行依赖。
- `requirements-dev.txt`：可选的本地测试、覆盖率和静态检查工具；训练不依赖这些包。
- `configs/*.yaml`：任务配置。默认数据根均为 `/root/autodl-tmp`。
- `Makefile`：只封装命令，不会在 `doctor`、`inspect`、`compile` 或 `test` 中启动训练。

`environment.yml` 故意不安装 Conda 版 `cudatoolkit` 或 `cudnn`。目标镜像已经提供 CUDA 11.2 和 cuDNN 8；在环境内再装一份 GPU 运行库可能抢先进入动态库搜索路径，使 TensorFlow 看见错误版本。

## 2. 创建新环境

所有命令都从仓库根目录执行。`environment.yml` 中的相对 requirements 路径依赖这一点。

```bash
cd /path/to/HandLandmarkerLab
make env
conda activate hand-landmarker-tf29
python -m pip check
```

`make env` 等价于 `conda env create -f environment.yml`。已有同名环境需要按锁文件对齐时可运行 `make env-update`；该目标包含 `--prune`，会移除未写入环境定义的 Conda 包，因此必须先确认该环境没有承载其他项目。

如果需要开发工具，再在该环境中执行：

```bash
python -m pip install -r requirements-dev.txt
```

不要同时安装 `tensorflow-gpu`。Linux 下本项目固定的 `tensorflow==2.9.0` 包本身包含 GPU 支持，实际能否加载 GPU 取决于宿主驱动、CUDA 11.2 和 cuDNN 8 动态库是否可见。

## 3. 使用镜像已有环境时

先只读盘点，不要直接升级镜像环境：

```bash
which python
python --version
python -m pip show tensorflow numpy protobuf h5py
nvidia-smi
nvcc --version
```

如果镜像环境已经是 Python 3.8 和 TensorFlow 2.9.0，可先用它运行 `make doctor`。长期训练仍建议创建独立环境，避免后续 notebook 或其他项目升级 NumPy、protobuf、Keras 后破坏 TensorFlow 2.9。

如果必须保留镜像内额外的厂商包，可先克隆镜像环境，再只修改克隆：

```bash
conda env list
conda create -n hand-landmarker-tf29 --clone <镜像环境名>
conda activate hand-landmarker-tf29
python -m pip install -r requirements.txt
```

克隆后仍必须运行下一节的完整验证。不要仅凭 `nvidia-smi` 正常就判断 TensorFlow GPU 可用；`nvidia-smi` 验证的是驱动，而不是 TensorFlow 能否加载 CUDA/cuDNN 动态库。

## 4. 安装后验证

先验证解释器和依赖解析：

```bash
python --version
python -m pip check
python -c "import tensorflow as tf; print(tf.__version__); print(tf.sysconfig.get_build_info()); print(tf.config.list_physical_devices('GPU'))"
```

然后运行项目门禁：

```bash
make doctor
make compile
make test
make inspect
```

预期含义：

1. `make doctor` 使用 `configs/train_pretrain.yaml`，读取 `environment.tensorflow`、`environment.cuda`、`environment.cudnn_major` 与 `environment.require_gpu`，检查 Python/TensorFlow 版本、TensorFlow build info、GPU 可见性及关键依赖。配置要求 GPU，因此 TensorFlow 不支持 CUDA 或没有可见物理 GPU 都应视为失败，而不是静默回退 CPU。
2. `make compile` 在内存中检查 `hand_landmarker/`、`models/`、`scripts/` 和 `tests/` 下的 Python 源码，不生成 `__pycache__`。
3. `make test` 使用标准库 `unittest discover`，不要求安装 pytest。
4. `make inspect` 依次运行 `inspect-pretrain`、`inspect-finetune`、`inspect-val` 和 `inspect-test`。两个训练检查都会自动加载其 Train、配置的 Val 与锁定 Test，计算图像 SHA-256 并做两两泄漏检查；ID、source group、解析路径或内容哈希任一精确重叠都会失败。独立的 Val/Test 检查再验证各自 schema、图片和标签。检查不会遍历图片目录来自动补样本，也不会启动训练。需要定位问题时才单独运行某个 `make inspect-*` 目标。

数据尚未放到默认路径时，先运行不带配置的 `python scripts/check_environment.py` 只检查解释器和依赖。`make doctor` 会同时检查默认训练 JSONL，因此此时应明确报告路径缺失；`make inspect` 同样应失败。不要为了让门禁通过而创建空 JSONL。

`tf.sysconfig.get_build_info()` 给出的 `cuda_version`/`cudnn_version` 是 TensorFlow **构建目标**，不是当前进程实际加载的动态库版本。Doctor 会把两者分开报告：构建字段存在时与配置期望比较，构建版本不匹配会失败；字段缺失时 `build_status` 为 `unknown` 并给出 warning。CUDA/cuDNN 的精确运行时版本无法通过该跨版本诊断可靠读取，因此 `runtime_version` 和 `runtime_status` 明确保持 `unknown`，并给出 warning；不会用构建元数据伪造运行时通过。warning 本身不改变退出码，但 `require_gpu: true` 时，`built_with_cuda != true`、GPU 查询失败或物理 GPU 列表为空都会失败。

可以覆盖 Make 变量而不修改 Makefile：

```bash
make doctor PYTHON=/opt/conda/envs/hand-landmarker-tf29/bin/python
make inspect-finetune
```

## 5. 配置与服务器路径

六个入口配置如下：

| 任务 | 配置 | 默认 canonical 数据/输入 |
|---|---|---|
| 第一阶段训练 | `configs/train_pretrain.yaml` | `/root/autodl-tmp/train_pretrain_merged/05_labels/hand_training_labels_pretrain.jsonl` |
| 第二阶段微调 | `configs/train_finetune.yaml` | `/root/autodl-tmp/train_finetune_merged/05_labels/hand_training_labels_finetune.jsonl` |
| Val | `configs/eval_val.yaml` | `/root/autodl-tmp/val_merged/05_labels/hand_validation_labels.jsonl` |
| Test | `configs/eval_test.yaml` | `/root/autodl-tmp/test_merged/05_labels/hand_test_labels.jsonl` |
| 文件夹推理 | `configs/infer.yaml` | `/root/autodl-tmp/inference/input` |
| ONNX 导出 | `configs/export.yaml` | finetune 的 `best.weights.h5` |

Val/Test 的 canonical 行直接指向现成 `256×256` Hand ROI；这两个入口只运行 Hand Landmarker，不需要 Palm 模型或原图路径。只有“文件夹推理”入口对任意外部图片执行 Palm → ROI → Hand。

所有数据、checkpoint 和 run 输出都使用 `${HAND_DATA_ROOT:-/root/autodl-tmp}`。配置加载器会先读取 `HAND_DATA_ROOT`，变量不存在时才使用 `/root/autodl-tmp`。因此服务器路径确认后可以统一覆盖，而不必逐项改 YAML：

```bash
export HAND_DATA_ROOT=/实际的数据盘路径
make doctor
make inspect
```

该变量必须在同一个 shell 会话或作业脚本中导出；只修改 `data_root` 字段不会自动重写其他绝对路径。本机 PowerShell 烟测可使用 `$env:HAND_DATA_ROOT = 'D:\path\to\data'`。

数据 loader 必须以配置指定的 canonical JSONL 为样本集合，并读取每行 `crop_path`；不得直接 glob `02_roi_crops/images`。`source_crop_path` 只用于溯源，不作为 canonical 路径的静默替代品。

## 6. Make 任务

```bash
make train-pretrain
make train-finetune
make eval-val
make eval-test
make infer
make export
```

也可以使用下划线别名 `train_pretrain`、`train_finetune`、`eval_val` 和 `eval_test`。`make train` 会先完整执行第一阶段，再执行第二阶段。Test 只能在 checkpoint、presence 阈值和量化/导出方案冻结后运行。

命令行始终只向脚本传入一个任务配置。需要确认入口参数时，以对应脚本的 `--help` 为准，不要假设训练或评估脚本支持未声明的附加参数。

## 7. GPU 与旧版 TensorFlow 注意事项

- 版本必须一起看：Python 3.8、TensorFlow/Keras 2.9.0、NumPy 1.23.5、protobuf 3.19.6、h5py 3.7.0 和 flatbuffers 1.12 是一组固定兼容基线。不要单独升级其中一个包。
- 配置默认关闭 mixed precision，先建立可复现 FP32 基线并确保 ONNX/板端输出一致，再单独评估精度和性能收益。
- 配置启用 TensorFlow GPU memory growth，避免启动时一次占满 RTX 3090 的 24 GB 显存。
- `onnxruntime==1.13.1` 使用 CPU 包，仅用于导出后的数值一致性检查，不会与 TensorFlow 抢占 CUDA 运行库。
- 服务器无图形桌面，因此使用 `opencv-python-headless`；不要再安装 `opencv-python`，否则容易出现重复二进制或 GUI 动态库问题。
- 系统盘只有约 30 GB，数据和 run 输出应保留在 `/root/autodl-tmp` 数据盘。创建环境前后都应检查 `df -h`，并避免把 checkpoint、TensorBoard 日志或 ONNX 产物写进 Conda 环境目录。

TensorFlow 没有发现 GPU 时，按以下顺序排查：

1. `nvidia-smi` 是否能看到 RTX 3090 和驱动；
2. `tf.sysconfig.get_build_info()` 是否显示 CUDA 11.2/cuDNN 8 构建信息；
3. `ldconfig -p | grep -E 'libcuda|libcudnn|libcublas'` 是否能找到动态库；
4. 当前 `LD_LIBRARY_PATH` 是否被另一套 Conda CUDA 覆盖；
5. `python -m pip check` 是否报告 NumPy/protobuf/Keras 冲突。

不要在未定位原因前直接升级 CUDA、TensorFlow 或显卡驱动；这会改变板端导出基线并使问题更难复现。
