# HLML 4.0 训练工作流

## 0. 环境依赖

本轮 HLML 4.0 更新没有改变训练环境：继续使用 Conda 环境 `hand-landmarker-tf29`，其依赖由仓库根目录的 `environment.yml` 和 `requirements.txt` 共同定义。目标服务器环境仍为 Ubuntu 20.04、Python 3.8、TensorFlow/Keras 2.9.0、CUDA 11.2 和 cuDNN 8；CUDA/cuDNN 使用 AutoDL 系统提供的动态库，不在 Conda 环境中重复安装。

首次创建环境：

```bash
cd /path/to/HandLandmarkerLab
conda env create -f environment.yml
conda activate hand-landmarker-tf29
python -m pip check
make environment-check
```

输入：仓库根目录的 `environment.yml` 与 `requirements.txt`。

处理：Conda 创建 Python 3.8 环境，随后由 `environment.yml` 中的 pip 条目安装 TensorFlow 2.9、Keras 2.9、ONNX、ONNX Runtime、OpenCV、Pillow 和 YAML 等固定版本依赖。

输出：名为 `hand-landmarker-tf29` 的 Conda 环境；`pip check` 应无依赖冲突，`make environment-check` 应输出环境、TensorFlow build metadata 和 GPU 可见性检查结果。已经创建该环境时只需执行 `conda activate hand-landmarker-tf29`，不要重复创建。

## 1. 系统边界与工作目录

HLML 直接读取 HLMF 3.0 已发布的 manifest，并从 `HAND_DATASET_ROOT` 原位加载 `256×256` Hand ROI。`HAND_TRAIN_ROOT` 只保存零拷贝索引快照、训练报告、checkpoint、评估结果和导出物，不复制数据集图片。

```bash
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_TRAIN_ROOT=/root/autodl-tmp/TrainFab/HLML-4.0
export HLML_SNAPSHOT_ID=v4-r1
export HLML_EXPERIMENT_ID=v4-r1
export HLML_RELEASE_ID=v4-r1
cd /path/to/HandLandmarkerLab
```

长期输入目录：

```text
HAND_DATASET_ROOT/
  PretrainSource/<dataset_id>/dataset_manifest.json
  EValSource/<dataset_id>/dataset_manifest.json
  GoldSource/NegativeSamples/<negative_dataset_id>/published/manifest.json
  Selections/<selection_id>/published/manifest.json
  Registry/registry.sqlite3
```

运行产物目录：

```text
HAND_TRAIN_ROOT/
  snapshots/<snapshot_id>/<stage>/
  runs/<experiment_id>/<stage>/
  mining/<snapshot_id>/
  releases/<release_id>/
  inference/<experiment_id>/<stage>/
```

评估严格限制在 HLMF 已生成并经 CVAT 复核的固定 Hand ROI：Val/Test 不运行 Palm Detector、不从原图重建 ROI、不统计 Palm 漏检，也不报告原图级联准确率。`make infer` 是独立的文件夹级联推理工具，不属于 Val/Test 协议。

## 2. 阶段零：通过 ID 选择数据

阶段名：Dataset Selection。

操作：编辑 `configs/datasets.yaml`，只填写 HLMF 已发布的 `dataset_id`、`negative_dataset_id`、`selection_id`、`proposal_variant` 和权重。不要手工拼接 JSONL，也不要把 ROI 复制到 `HAND_TRAIN_ROOT`。

输入位置与选择键：

- `stages.*.datasets` → `PretrainSource/<dataset_id>/dataset_manifest.json`。
- `stages.multitask.negative_datasets` → `GoldSource/NegativeSamples/<id>/published/manifest.json`。
- `stages.multi_finetune.selections` → `Selections/<id>/published/manifest.json`。
- `stages.multi_finetune.new_datasets` → 新录制的 Train dataset manifest。
- `evaluation.val/test` → `EValSource/<dataset_id>/dataset_manifest.json` 中相应 split。

可以直接编辑 YAML，也可以使用其中的环境变量占位：

```bash
export HLML_PRETRAIN_DATASET_ID=FullEnhance0801
export HLML_NEGATIVE_DATASET_ID=background-neg-0801
export HLML_SELECTION_ID=hard-positive-0801
export HLML_EVAL_DATASET_ID=national-eval-0801
export HLML_PROPOSAL_VARIANT=palm-v1
```

输出：本阶段只确定成员关系，不写新数据。

## 3. 阶段一：公共配置解析检查

阶段名：Config Check。

命令：

```bash
make config-check
```

输入：`configs/datasets.yaml`、`configs/training.yaml`、`configs/evaluation.yaml`、`configs/inference.yaml`、`configs/deploy.yaml` 及当前环境变量。

处理：解析 datasets contract、三个训练 profile（geometry、multitask、multi_finetune）、两个固定 ROI 评估 profile（val、test）、独立文件夹推理配置和独立 ONNX/A1 导出配置，检查 YAML 结构和变量展开。

输出：终端 JSON，`status=ok` 表示全部 profile 可解析。该命令不读取图片、不生成 snapshot，也不训练。

## 4. 阶段二：零拷贝数据审计

阶段名：Data Audit。

单独审计某个阶段：

```bash
make data-audit HLML_STAGE=geometry
make data-audit HLML_STAGE=multitask
make data-audit HLML_STAGE=multi_finetune
```

输入：`datasets.yaml` 选择的 HLMF manifest、`Registry/registry.sqlite3` 以及 manifest 引用的 ROI 图片。

处理：

1. 解析新来源名 `<background>-<distance>-<lighting>-<condition>-<split>-<session>-<performer>`。
2. 检查 capture source 和 raw image 不跨 Train/Val/Test。
3. 检查一次运行中同一 `capture_source_id` 只选择一个 `proposal_variant`。
4. 核对 `roi_id`、registry 记录、相对路径和 dataset/source/split 字段。
5. 限制图片必须位于 `HAND_DATASET_ROOT` 内，并解码为单通道 `256×256` ROI。
6. 人员跨 split 默认写 warning；`policies.performer_cross_split=fail` 可升级为硬错误。

完整性检查使用稳定 ID、SQLite、文件路径、尺寸与解码，不对数据集图片反复计算 SHA-256。

输出：

```text
HAND_TRAIN_ROOT/snapshots/<snapshot_id>/<stage>/train.jsonl
HAND_TRAIN_ROOT/snapshots/<snapshot_id>/<stage>/val.jsonl
HAND_TRAIN_ROOT/snapshots/<snapshot_id>/<stage>/test.jsonl
HAND_TRAIN_ROOT/snapshots/<snapshot_id>/<stage>/snapshot.json
```

JSONL 中 `crop_path` 指向 `HAND_DATASET_ROOT` 原图，没有图片副本。默认拒绝覆盖已有 snapshot；确实需要重建同一 ID 时使用 `DATA_ARGS=--overwrite`，但正式运行更推荐换新的 `HLML_SNAPSHOT_ID` 保留可追溯性。

## 5. 阶段三：Geometry 训练

阶段名：Geometry。

命令：

```bash
make geometry
```

输入：

- `configs/datasets.yaml` 的 `stages.geometry.datasets`。
- HLMF 发布的可靠 Train positive。
- `configs/training.yaml` 的 geometry profile。
- 自动审计得到的 `snapshots/<snapshot_id>/geometry/{train,val}.jsonl`。

处理：命令先运行 geometry data audit，再调用现有 v2 训练器。Geometry 只学习关键点几何，配置任何 `negative_datasets` 都会被硬拒绝；本阶段不使用真负样本或 candidate negative。

输出：

```text
HAND_TRAIN_ROOT/runs/<experiment_id>/geometry/checkpoints/best.weights.h5
HAND_TRAIN_ROOT/runs/<experiment_id>/geometry/checkpoints/last.weights.h5
HAND_TRAIN_ROOT/runs/<experiment_id>/geometry/checkpoints/final.weights.h5
HAND_TRAIN_ROOT/runs/<experiment_id>/geometry/history.json
```

训练器还会按配置保存周期 checkpoint。后续 multitask 默认从 geometry 的 `best.weights.h5` 初始化。

## 6. 阶段四：Multitask 训练

阶段名：Multitask。

命令：

```bash
make multitask
```

输入：

- geometry winner：`runs/<experiment_id>/geometry/checkpoints/best.weights.h5`。
- `stages.multitask.datasets` 选择的 Train positive。
- `stages.multitask.negative_datasets` 选择的 HLMF 已发布真负样本。
- multitask profile 和审计生成的 snapshot。

处理：从 geometry winner 初始化，保留 v2 的 landmarks、presence 和 handedness 输出与现有损失；真负样本的 `negative_dataset_id` 权重进入 `sampling_weight`，用于 presence 多任务训练。未经过 HLMF 删除式复核的 candidate negative 不可使用。

输出：

```text
HAND_TRAIN_ROOT/snapshots/<snapshot_id>/multitask/*
HAND_TRAIN_ROOT/runs/<experiment_id>/multitask/checkpoints/best.weights.h5
HAND_TRAIN_ROOT/runs/<experiment_id>/multitask/history.json
```

## 7. 阶段五：Train-only 困难来源挖掘

阶段名：Hard Source Mining。

默认使用 multitask winner 推理：

```bash
make mine-hard
```

显式 checkpoint 或已有 student prediction 也可以通过 Make 透传：

```bash
make mine-hard MINING_ARGS='--checkpoint /abs/path/best.weights.h5 --batch-size 64'
make mine-hard MINING_ARGS='--predictions /abs/path/student_predictions.jsonl'
```

输入：`snapshots/<snapshot_id>/multitask/train.jsonl`、MediaPipe 标签，以及 multitask checkpoint 或预计算 student prediction。

处理：只读取 Train 固定 ROI，对比 student 与 MediaPipe label，按 `capture_source_id` 聚合样本数、mean/median/P90 像素误差、PCK、collapse、距离、亮度和姿态跨度，并生成供 HLMF 删除式复核的 ROI 请求。代码会硬拒绝 Val/Test；Test 结果也不能反向进入采样权重、训练配置、checkpoint 或阈值选择。

输出默认不可覆盖：

```text
HAND_TRAIN_ROOT/mining/<snapshot_id>/student_predictions.jsonl
HAND_TRAIN_ROOT/mining/<snapshot_id>/source_ranking.json
HAND_TRAIN_ROOT/mining/<snapshot_id>/hlmf_review_request.jsonl
```

把 `hlmf_review_request.jsonl` 交给 HLMF 执行 `hard-review` / `hard-publish`。HLMF 发布后，将新的 `selection_id` 写回 `configs/datasets.yaml` 的 `stages.multi_finetune.selections`。

## 8. 阶段六：Multi-finetune 训练

阶段名：Multi-finetune。

命令：

```bash
make multi-finetune
```

输入：

- multitask winner：`runs/<experiment_id>/multitask/checkpoints/best.weights.h5`。
- `selections` 中经 HLMF 删除明显教师错误后发布的困难 positive。
- 可选 `new_datasets` 新录制 Train positive。
- `negative_datasets` 中的已发布真负样本。
- `datasets` 中的 mandatory pretrain replay pool。

处理：困难 selection、新录制数据和真负样本组成 hard/new 侧；未被这些成员占用的 PretrainSource positive 组成 replay 侧。默认 hard/new 55%、replay 45%。两者必须都大于零且总和为 1；因此不能关闭 replay。每个 dataset/selection/negative dataset 的 `weight` 继续参与侧内采样。

输出：

```text
HAND_TRAIN_ROOT/snapshots/<snapshot_id>/multi_finetune/*
HAND_TRAIN_ROOT/runs/<experiment_id>/multi_finetune/checkpoints/best.weights.h5
HAND_TRAIN_ROOT/runs/<experiment_id>/multi_finetune/history.json
```

## 9. 阶段七：固定 Hand ROI Val

阶段名：Fixed-ROI Val。

命令：

```bash
make val HLML_STAGE=multi_finetune
```

输入：

- `snapshots/<snapshot_id>/<stage>/val.jsonl`。
- `runs/<experiment_id>/<stage>/checkpoints/best.weights.h5`。
- `configs/evaluation.yaml` 的 val profile。

处理：直接读取 HLMF 已生成、已复核的 `crop_path`，只运行 Hand Landmarker。Val 可以执行 `threshold_sweep` 来选择 presence threshold。不会读取原图或运行 Palm。

输出：

```text
HAND_TRAIN_ROOT/runs/<experiment_id>/eval/<stage>/val/predictions.jsonl
HAND_TRAIN_ROOT/runs/<experiment_id>/eval/<stage>/val/metrics.json
```

指标包括 mean/median/P90/P95 像素误差、PCK、landmark collapse、presence 和 handedness，并按 dataset、capture source、label origin、annotation style、distance 和 lighting 分组。

## 10. 阶段八：冻结唯一 winner

阶段名：Winner Freeze。

命令：

```bash
make freeze-winner HLML_STAGE=multi_finetune HLML_RELEASE_ID=v4-r1
```

输入默认是：

```text
runs/<experiment_id>/eval/multi_finetune/val/metrics.json
runs/<experiment_id>/multi_finetune/checkpoints/best.weights.h5
```

如需指定其他已完成的 Val 结果和 checkpoint，可使用：

```bash
make freeze-winner HLML_STAGE=multi_finetune FREEZE_ARGS='--metrics /abs/metrics.json --checkpoint /abs/best.weights.h5'
```

处理：确认 metrics 的 split 为 Val、scope 为 fixed Hand ROI，冻结模型版本、stage、checkpoint、snapshot 和 Val 选定的 presence threshold。一个 release ID 只能创建一次。

输出：

```text
HAND_TRAIN_ROOT/releases/<release_id>/winner.json
```

## 11. 阶段九：Locked Test

阶段名：Locked Fixed-ROI Test。

命令：

```bash
make locked-test HLML_STAGE=multi_finetune HLML_RELEASE_ID=v4-r1
```

输入：`releases/<release_id>/winner.json`、其中锁定的 checkpoint、`snapshots/<snapshot_id>/<stage>/test.jsonl` 和 deploy test profile。

处理：只允许 winner descriptor 指定的 checkpoint 和 Val 冻结的 presence threshold；禁止 threshold sweep、checkpoint 切换和覆盖已有结果。Test 不运行 Palm，也不能被 mining 或训练代码读取。

输出一次且不可覆盖：

```text
HAND_TRAIN_ROOT/releases/<release_id>/test/predictions.jsonl
HAND_TRAIN_ROOT/releases/<release_id>/test/metrics.json
```

## 12. 阶段十：文件夹级联推理

阶段名：Folder Inference，和固定 ROI 评估分离。

命令：

```bash
export HLML_INFER_INPUT=/path/to/images
make infer HLML_STAGE=multi_finetune
```

输入：`configs/inference.yaml` 指定的任意支持格式原图文件夹、Palm ONNX 和 Hand checkpoint。

处理：运行 Palm → canonical Hand ROI → v2 Hand Landmarker，并按配置绘制可视化。这是部署前人工检查工具，不能把结果当 Val/Test fixed-ROI 指标。

输出：

```text
HAND_TRAIN_ROOT/inference/<experiment_id>/<stage>/
```

其中包含 JSONL、summary 和按配置生成的标注图；默认拒绝覆盖。

## 13. 阶段十一：ONNX/A1 导出

阶段名：Export。

命令：

```bash
make export HLML_STAGE=multi_finetune
```

输入：所选 stage 的 v2 checkpoint 和只负责模型导出的 `configs/deploy.yaml`。

处理：导出静态 NCHW `[1,1,256,256]`、opset 11 ONNX，保持输出顺序 `[landmarks, hand_flag, handedness]`，审计 A1 允许算子、模型大小、depthwise group 和 Keras/ONNXRuntime 数值一致性。

输出默认位于：

```text
HAND_TRAIN_ROOT/runs/<experiment_id>/export/<stage>/hand_landmarker_v2.onnx
```

同时在导出物旁写入 contract/report；默认拒绝覆盖。

## 14. 阶段十二：环境、测试和双仓验收

检查服务器 TensorFlow/CUDA/GPU 环境：

```bash
make environment-check
```

输入为 training environment contract，输出为终端环境报告。

语法、完整单元测试和公共命令检查：

```bash
make compile
make test
make help
```

双仓库合成验收：

```bash
make acceptance-smoke
```

该命令运行 HLMF contract 测试、HLML 合成 warehouse 的三阶段/固定 ROI 测试，并解析全部公共配置；不使用真实 Test 选择模型。

## 15. `configs/datasets.yaml` 参数说明

- `dataset_id`：HLMF 发布的数据集逻辑 ID，不是目录绝对路径。
- `proposal_variant`：选择同一来源的哪一版 Palm/ROI 结果。一次运行中一个 `capture_source_id` 只能选一个 variant。
- `weight`：正数，写入该来源记录的 `sampling_weight`。它控制相对抽样权重，不复制样本。
- `negative_dataset_id`：只能引用 HLMF `GoldSource/NegativeSamples/<id>/published/`。
- `selection_id`：只能引用 HLMF `Selections/<id>/published/` 的零拷贝困难样本集合。
- `new_datasets`：multi-finetune 可选新录制 Train positive，格式与 `datasets` 相同。
- `hard_fraction/replay_fraction`：默认 `0.55/0.45`，必须均大于 0 且总和为 1。
- `evaluation.val/test`：只选择 EValSource 中已复核 fixed ROI；两者按 manifest 的 split 字段过滤。
- `policies.performer_cross_split`：默认 `warn`，需要严格人员隔离可设为 `fail`。

`image_integrity`、`image_sha256` 和 `test_may_feed_training_or_mining` 是公开策略声明，应分别保持稳定 ID/registry/decode、图片 SHA 禁用和 Test 禁止回流。

## 16. `configs/training.yaml` 参数说明

### 16.1 模型与环境

- `environment`：锁定 Python、TensorFlow、CUDA 和 cuDNN 版本；服务器不匹配时先处理环境，不应为绕过检查而随意改版本。
- `model.version/num_iterations/output_order`：现有 v2 网络和三输出契约。当前重构不做辅助 head 或结构实验。
- `data.image_size/channels/input_layout/input_scale`：与 HLMF `256×256` 灰度 ROI 和 A1 NCHW 契约一致。

### 16.2 通用训练参数

- `training.epochs`、`batch_size`：训练轮数与批大小；显存不足时优先降低 batch size。
- `optimizer.learning_rate`：geometry 默认较大，multitask/multi-finetune profile 会覆盖为更小学习率。
- `initial_checkpoint`：阶段初始化权重；multitask 必须指向 geometry winner，multi-finetune 必须指向 multitask winner。
- `resume_checkpoint`：仅用于同一阶段中断恢复，不能代替跨阶段初始化。
- `gradient_clip_norm`、`mixed_precision`：数值稳定与性能选项；开启 mixed precision 前应验证目标环境和导出一致性。
- `checkpoint.monitor/mode`、`early_stopping`、`learning_rate_schedule`：只使用 Val 指标；Test 禁止参与。
- `max_wall_time_hours`、`periodic_checkpoint`：服务器时间预算和恢复点策略。

### 16.3 采样、损失和增强

- `sampling.sample_type_fractions`：控制 positive/negative 类型抽样；geometry 的 negative 必须为 0，multitask profile 才启用真负样本。
- `sampling.epoch_size`、`replacement`：每 epoch 抽样量及是否有放回；过大可能反复抽到少数来源。
- `honor_record_sampling_weight`：必须保持开启，才能使用 datasets.yaml 中的权重。
- `losses.*.coefficient`：landmarks、presence、handedness 的相对损失系数。修改后应在固定 Val 上比较，不能用 Test 调参。
- `augmentation`：只作用于训练 ROI。旋转、缩放和平移过大可能破坏 canonical ROI 几何；调整后先做 smoke 和可视化抽查。
- `profiles`：仅放相对基础配置的阶段覆盖。不要复制三份完整配置，避免阶段契约漂移。

## 17. `configs/evaluation.yaml`、`configs/inference.yaml` 与 `configs/deploy.yaml`

三个文件不使用跨语义 profile：`evaluation.yaml` 只做固定 ROI Val/Test，`inference.yaml` 只做原图文件夹级联推理，`deploy.yaml` 只做模型导出和 A1 审计。

### 17.1 `evaluation.yaml`：Val/Test

- `data.labels`：固定指向相应 snapshot 的 `val.jsonl` 或 `test.jsonl`。
- `evaluation.mode`：Val/Test 必须是 `roi`。
- `hand_flag_threshold`：Val 初始 presence 阈值。
- `threshold_sweep`、`tune_thresholds`：Val 可开启；Test profile 必须关闭并使用 winner threshold。
- `pck_thresholds`：以 ROI 尺寸归一化的 PCK 阈值列表。
- `output.overwrite`：正式 Val/Test 建议为 `false`，locked Test 强制为 `false`。

### 17.2 `inference.yaml`：Folder Inference

- `input.images_dir/extensions/recursive`：任意文件夹推理的输入范围。
- `palm.*`：只在文件夹推理配置中运行 Palm；调整 score/NMS/ROI 几何不会改变固定 ROI 评估。
- `hand_roi.*`：应与部署端和 HLMF 使用的 ROI contract 对齐。
- `output.write_annotated_images/write_jsonl/draw_*`：控制检查产物，不影响模型数值。

### 17.3 `deploy.yaml`：ONNX/A1 Export

- `export.opset`、`input_name/output_names/dynamic_batch`：A1 部署接口，当前为 opset 11、固定 batch 和固定输出顺序。
- `maximum_model_size_mb`、`maximum_depthwise_group`、`a1_allowed_operators`：硬件审计门槛。
- `validate.random_samples` 与容差：Keras/融合 ONNX/ONNXRuntime 数值一致性检查。放宽容差前必须定位具体算子误差。
- `metadata`：部署端预处理和输出解释契约，必须和训练配置一致。

## 18. 数据泄漏与 Test 锁定原则

Train、Val、Test 的 capture source 和 raw image 必须完全隔离；人员隔离至少保留警告。困难挖掘仅使用 Train。唯一 winner 只能根据固定 Val 冻结；Test 只执行一次该 winner，Test 输出不得被数据审计、采样、训练、mining、threshold 或 checkpoint 选择代码读取。
