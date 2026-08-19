# HLML 4.0 训练工作流

## 0. 环境依赖

HLML 4.0 继续使用 Conda 环境 `hand-landmarker-tf29`，其依赖由仓库根目录的 `environment.yml` 和 `requirements.txt` 共同定义。目标服务器环境为 Ubuntu 20.04、Python 3.8、TensorFlow/Keras 2.9.0、CUDA 11.2 和 cuDNN 8；CUDA/cuDNN 使用 AutoDL 系统提供的动态库，不在 Conda 环境中重复安装。训练器默认启用 `tqdm` 的 epoch/batch 进度条，因此更新本仓库后需确保环境已安装 requirements 中固定版本的 `tqdm`。

首次创建环境：

```bash
cd /path/to/HandLandmarkerLab
conda env create -f environment.yml
conda activate hand-landmarker-tf29
readonly CUDA_LIBRARY_DIR=/usr/local/cuda-11.2/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="$CUDA_LIBRARY_DIR:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDNN_V8_API_DISABLED=1
python -m pip check
make environment-check
```

输入：仓库根目录的 `environment.yml` 与 `requirements.txt`。

处理：Conda 创建 Python 3.8 环境，随后由 `environment.yml` 中的 pip 条目安装 TensorFlow 2.9、Keras 2.9、ONNX、ONNX Runtime、OpenCV、Pillow 和 YAML 等固定版本依赖。

输出：名为 `hand-landmarker-tf29` 的 Conda 环境；`pip check` 应无依赖冲突，`make environment-check` 应输出环境、TensorFlow build metadata 和 GPU 可见性检查结果。已经创建该环境时只需执行 `conda activate hand-landmarker-tf29`，不要重复创建。

## 1. 系统边界与工作目录

HLML 直接读取 HLMF 3.0 已发布的 manifest，并从 `HAND_DATASET_ROOT` 加载 `256×256` Hand ROI。PretrainSource/EValSource 与新录制的 `GoldSource/ReviewedDatasets` 使用各自来源 ROI；GoldSource 负样本和困难样本使用 HLMF 在 published 目录内生成的独立图片副本。`HAND_TRAIN_ROOT` 只保存索引快照、训练报告、checkpoint、评估结果和导出物，不再复制任何数据集图片。

HLMF 内部的 Palm/HCF/landmark teacher 版本不属于训练数据边界。当前 v3 数据选择使用 Eos-2.1 + HaMeR + `v1-mobilenet_v3_large` 发布变体；`multitask` 明确复用独立审核发布的 `neg-eos_2.0-hcf0813-hp0.5`。发布给 HLML 的接口仍是 `hlmf_dataset_v1`、单通道 `256×256` ROI、manifest/JSONL/Registry 与教师溯源字段。正样本 proposal 域和已发布负样本域分别执行单 variant 门禁，因此二者可保留各自生成时的 Eos 版本；split、raw image、ROI ID、Registry 与图片解码仍统一严格审计。文件夹级联 `make infer` 默认使用 Eos-2.1。

```bash
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_TRAIN_ROOT=/root/autodl-tmp/TrainFab/HLML-4.0
export HLML_MODEL_VERSION=v3-pro
export HLML_SNAPSHOT_ID=iris-v3-data-r1
export HLML_EXPERIMENT_ID=iris-v3-pro-r1
export HLML_RELEASE_ID=iris-v3-pro-r1
cd /path/to/HandLandmarkerLab
```

长期输入目录：

```text
HAND_DATASET_ROOT/
  PretrainSource/<dataset_id>/dataset_manifest.json
  EValSource/<dataset_id>/dataset_manifest.json
  GoldSource/NegativeSamples/<negative_dataset_id>/published/{manifest.json,negative_labels.jsonl,images/}
  GoldSource/HardSamples/<hard_dataset_id>/published/{manifest.json,hard_labels.jsonl,images/}
  GoldSource/ReviewedDatasets/<dataset_id>/dataset_manifest.json
  Selections/<selection_id>/published/{manifest.json,selection.jsonl,images/}  # legacy 已发布只读
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

## 1.1 Iris v3 模型系列与选择

所有版本输入均为 NCHW `[1,1,256,256]`，输出顺序均为 `landmarks[42]`、`hand_flag[1]`、`handedness[1]`。配置文件通过 `HLML_MODEL_VERSION` 统一选择训练、固定 ROI 评估、推理和导出所用结构：

- `v3-pro`：直接复用未修改的 v2 构建和 Conv/Depthwise+BN 融合实现；训练参数 1,951,756，部署参数 1,912,324。
- `v3-max`：每个普通卷积使用 4 个 Conv+BN 分支并在适用处增加 identity BN；每个 3x3 depthwise 使用 4 个 3x3+BN、一个 1x1+BN 和 identity BN。训练参数 7,629,268，部署时数学融合为单分支 1,912,324，训练/部署参数比 3.99。
- `v3-lite`：保持拓扑、非线性、下采样和 head 不变，将 stage 通道改为 `16,24,40,80,112,160,224`；训练参数 878,272，部署参数 852,832。
- `v2`：历史结构入口继续保留且代码不改动。

三台克隆服务器分别设置 `HLML_MODEL_VERSION=v3-pro|v3-max|v3-lite`，并使用不同 `HLML_EXPERIMENT_ID`；`HLML_SNAPSHOT_ID`、数据配置、stage 和训练超参数保持一致，才能进行结构对照。该选择不改变训练阶段和数据合同。

## 2. 阶段零：通过 ID 选择数据

阶段名：Dataset Selection。

操作：编辑 `configs/datasets.yaml`，只填写 HLMF 已发布的 `dataset_id`、`negative_dataset_id`、`hard_dataset_id`、`proposal_variant`、可选的 `capture_source_ids` 白名单和权重。不要手工拼接 JSONL，也不要把 ROI 复制到 `HAND_TRAIN_ROOT`。

`datasets`、`negative_datasets`、`hard_datasets`、`gold_datasets` 和 `evaluation.val/test` 都是成员列表，可各自配置一个或多个已发布 ID；每个 dataset 类成员独立填写 variant 和 `weight`。省略 `capture_source_ids` 时消费该 split 中所有已发布所选 variant 的 source；提供非空白名单时，每个 ID 必须存在于 dataset manifest、属于目标 split 并发布所选 variant，否则审计失败。该白名单用于冻结正式 Val/Test，也可用于确有需要的 Train 子集。`selections` 仅保留对历史已发布资产的兼容读取，不用于新困难复核。

输入位置与选择键：

- `stages.*.datasets` → `PretrainSource/<dataset_id>/dataset_manifest.json`。
- `stages.multitask.negative_datasets` → `GoldSource/NegativeSamples/<id>/published/manifest.json`；训练图片取每行 `published_relpath`。
- `stages.multi_finetune.hard_datasets` → `GoldSource/HardSamples/<id>/published/manifest.json`；用 `source_crop_relpath` 核对 registry，用 `published_relpath` 读取 CVAT 精修后的独立发布副本。每个 hard dataset 是独立 proposal variant 域，因此可复用于使用更新 Palm variant 的后续训练流程，但其自身仍只能包含一个 variant。
- `stages.multi_finetune.gold_datasets` → `GoldSource/ReviewedDatasets/<dataset_id>/dataset_manifest.json`；只能是新录制、按 Eval 同款自动标注 + CVAT 人工复核后发布的 train Gold，允许 positive 与 negative。
- `evaluation.val/test` → `EValSource/<dataset_id>/dataset_manifest.json` 中相应 split。

以下 Iris-1.1/Eos-2.0 片段仅是历史选择示例，不是当前 v3 配置；当前成员必须以 `configs/datasets.yaml` 和当次 data audit 为准：

```yaml
stages:
  geometry:
    datasets:
      - {dataset_id: FullEnhance0801, proposal_variant: eos_2.0-rtmpose-hcf0813-gate, weight: 1.0}
      - {dataset_id: FullEnhance0803, proposal_variant: eos_2.0-rtmpose-hcf0813-gate, weight: 1.0}
      - {dataset_id: FullEnhance0810, proposal_variant: eos_2.0-rtmpose-hcf0813-gate, weight: 1.0}
evaluation:
  val:
    - dataset_id: FullEnhanceVal0801
      proposal_variant: eos_2.0-rtmpose-hcf0813-gate
      capture_source_ids: [complex-mid-bright-random-val-s01-peak,
                           complex-mid-bright-random-val-s01-soar,
                           complex-near-bright-random-val-s01-peak]
    - dataset_id: FullEnhanceVal0801
      proposal_variant: eos_2.0-rtmpose-gate
      capture_source_ids: [complex-mid-dark-random-val-s01-peak,
                           white-mid-bright-random-val-s01-soar]
  test:
    - dataset_id: FullEnhanceVal0801
      proposal_variant: eos_2.0-rtmpose-gate
      capture_source_ids: [complex-mid-bright-random-test-s01-peak,
                           complex-near-bright-random-test-s01-peak]
```

Eos-1.0 ROI 来自只覆盖手掌的旧 Palm 几何，只能作为独立 legacy/stress 回放；它不能与完整手掌和手指 Anchor 产生的 Eos-2.0 ROI 合并成主 Val/Test headline 指标。同一原图已进入 Eos-2.0 主集时，也不得再以旧 variant 重复进入 legacy 集。`FullEnhanceVal0803`、`RTMPose-Finetune-Test-0812` 不属于本轮成员。

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
2. 已审核发布的 negative dataset 保留其生成时的 proposal variant，并与 PretrainSource positive 分属两个 variant 唯一性域；同一域内部仍禁止同一 capture source 混用多个 variant。这一例外不放宽 split、raw image、ROI ID、Registry 或图片检查。
3. 检查 capture source 和 raw image 不跨 Train/Val/Test。
4. 对每个 dataset 只读取包含所选 `proposal_variant` 的 capture source；同一 manifest 中仅发布其他历史 variant 的 source 会被跳过。配置 `capture_source_ids` 后则要求白名单逐项精确命中，不允许缺失 source、错误 split 或未发布目标 variant。目标 split 至少要有一个所选 variant，且同一 source 不得重复发布同名 variant。
5. 要求 dataset、negative dataset 与 selection manifest 使用当前 HLMF `hlmf_dataset_v1` schema，并核对 `roi_id`、registry 记录、相对路径和 dataset/source/split 字段。
6. negative dataset 与 selection 必须声明 `image_policy=copied_review_and_published_images`；negative 直接读取 `published_relpath`，selection 先用 `source_crop_relpath` 验证原 ROI 身份，再从 `published_relpath` 读取独立副本。源 ROI 图片被删除或源 variant 退役后，已发布 selection 仍可单独读取。
7. 限制实际训练/评估图片必须位于 `HAND_DATASET_ROOT` 内，并解码为单通道 `256×256` ROI。
8. 人员跨 split 默认写 warning；`policies.performer_cross_split=fail` 可升级为硬错误。

HLMF 行中的教师溯源字段会原样保留到 snapshot，包括 `source`、`label_origin`、`annotation_style`、`teacher_model_id`、`handedness_teacher_model_id`、`hand_presence_teacher_model_id`，以及可选的 `rtmpose_geometry_rescue`、`hamer_inference`、`hamer_geometry_rescue`。HaMeR direct 行使用 `hamer/hamer_openpose21_v1`，TFLite rescue 行仍使用 `mediapipe/mediapipe_tflite_rescue_v1`；当前 HCF teacher ID 为 `hand-classifier-v1-mobilenet_v3_large`，既有 snapshot 仍可保留 0814、0813 或 0809。读取器不硬编码 teacher/HCF 版本，因此版本更新不会在 HLML 入库时丢失 provenance。

HLMF 当前有两种等价的辅助 crop-pixel 表示：常规 MediaPipe/RTMPose/HaMeR 行满足 `crop_px = crop_norm × 255`，`mediapipe_tflite_rescue_v1` 行满足连续 crop extent 约定 `crop_px = crop_norm × 256`。两者的实际训练目标都是同一 `landmarks_crop_norm`。HLML warehouse 审计要求整行严格匹配其中一种约定，不匹配任一种即失败；随后记录 `warehouse_crop_pixel_convention`，并把内部 snapshot 的辅助 `landmarks_crop_px` 统一规范化为 `crop_norm × 255`。这不会改写 HLMF 发布物，也不会改变归一化训练目标或 image-space 坐标。

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

处理：命令先运行 geometry data audit，再由 registry 按 `HLML_MODEL_VERSION` 构建 v3-pro、v3-max 或 v3-lite（也可显式选择历史 v2）。Geometry 只学习关键点几何，配置任何 `negative_datasets` 都会被硬拒绝；本阶段不使用真负样本或 candidate negative。

输出：

```text
HAND_TRAIN_ROOT/runs/<experiment_id>/geometry/checkpoints/best.weights.h5
HAND_TRAIN_ROOT/runs/<experiment_id>/geometry/checkpoints/last.weights.h5
HAND_TRAIN_ROOT/runs/<experiment_id>/geometry/checkpoints/final.weights.h5
HAND_TRAIN_ROOT/runs/<experiment_id>/geometry/history.json
```

训练器还会按配置保存周期 checkpoint。后续 multitask 默认从 geometry 的 `best.weights.h5` 初始化。

Geometry 训练结束后必须立即运行固定 ROI Val 与文件夹级联 infer：

```bash
make val HLML_STAGE=geometry
export HLML_INFER_INPUT=/path/to/representative/images
make infer HLML_STAGE=geometry
```

`make val` 输入 geometry snapshot 的 `val.jsonl` 与 geometry winner，输出 `runs/<experiment>/eval/geometry/val/`；`make infer` 输入代表性原图、所选 Eos 和同一 winner，输出 `inference/<experiment>/geometry/`。前者用于可比较的固定 ROI 指标，后者用于发现 Palm→ROI→Hand 级联的可视化异常；两种结果不能混为同一指标。

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

处理：从同一模型版本的 geometry winner 初始化，保持 landmarks、presence、handedness 三输出与既有损失。当前 v3 配置明确使用完整已发布负样本集 `neg-eos_2.0-hcf0813-hp0.5`；其 16,910 条 Eos-2.0 负样本与 82,902 条 Eos-2.1/HaMeR r4 正样本属于两个独立 proposal 审计域，但共同保持 Train split、ROI/Registry 和图片发布合同。2026-08-18 实测 multitask snapshot 为 Train 99,812、Val 14,411、Test 5,343，membership errors 为 0。采样比例仍由 `training.yaml` 的 multitask profile 控制，正式训练前以该 snapshot 的实际 sample-type 计数为准。

输出：

```text
HAND_TRAIN_ROOT/snapshots/<snapshot_id>/multitask/*
HAND_TRAIN_ROOT/runs/<experiment_id>/multitask/checkpoints/best.weights.h5
HAND_TRAIN_ROOT/runs/<experiment_id>/multitask/history.json
```

Multitask 训练结束后必须执行 Val、infer 和 export：

```bash
make val HLML_STAGE=multitask
export HLML_INFER_INPUT=/path/to/representative/images
make infer HLML_STAGE=multitask
make export HLML_STAGE=multitask
```

Val/infer 分别输出 `eval/multitask/val/` 与 `inference/<experiment>/multitask/`。Export 同时输出 ONNX/A1 审计文件和 `model_conversion/datasets.zip`；转换数据包只读抽样当前 multitask snapshot 的 Train/Val/Test ROI。

## 7. 阶段五：Train-only 困难来源挖掘

阶段名：Hard Source Mining。

每轮必须给出独立 `round_id` 和最大 ROI 数量；例如第一轮 1000：

```bash
make mine-hard MINING_ARGS='--round-id r01 --max-rois 1000'
```

显式 checkpoint 或已有 student prediction 也可以通过 Make 透传：

```bash
make mine-hard MINING_ARGS='--round-id r02 --max-rois 1500 --checkpoint /abs/path/best.weights.h5 --batch-size 64'
make mine-hard MINING_ARGS='--round-id r02 --max-rois 1500 --predictions /abs/path/student_predictions.jsonl'
```

输入：`snapshots/<snapshot_id>/multitask/train.jsonl`、HLMF 当前发布的教师标签（通常来自 RTMPose/HCF/TFLite rescue 链路），以及 multitask checkpoint 或预计算 student prediction。

处理：只读取 Train positive 固定 ROI。每行困难度由关键点误差排序 80%、presence 误差 10%、handedness 误差 10% 组成；错误分类和低置信 head 因此可进入候选，但关键点仍是主信号。若人工真值中的 handedness 为 `unknown`，该 ROI 仍参与关键点与 presence 排序，只把 handedness 分量记为 0，并从 handedness 错误率分母排除。报告按 `capture_source_id` 聚合像素误差、PCK、collapse、presence/handedness 错误率、距离、亮度和姿态跨度。请求按综合困难度排序并截断到 `max_rois`。代码会硬拒绝 Val/Test；Test 结果也不能反向进入采样权重、训练配置、checkpoint 或阈值选择。

同一个 `snapshot_id` 代表一次完整的 geometry + multitask + multi-finetune 数据流程。`mining/<snapshot_id>/selection_ledger.json` 记录各轮已经筛选的 ROI；后续轮自动排除这些 ID，只要求在该 snapshot 内不重复，不扫描整个 DatesetFab 或所有历史困难集。`round_id` 只能使用一次。

输出默认不可覆盖：

```text
HAND_TRAIN_ROOT/mining/<snapshot_id>/selection_ledger.json
HAND_TRAIN_ROOT/mining/<snapshot_id>/rounds/<round_id>/student_predictions.jsonl
HAND_TRAIN_ROOT/mining/<snapshot_id>/rounds/<round_id>/source_ranking.json
HAND_TRAIN_ROOT/mining/<snapshot_id>/rounds/<round_id>/hlmf_review_request.jsonl
```

把当轮 `hlmf_review_request.jsonl` 交给 HLMF 执行 `hard-review`，上传 ROI 与 CVAT 1.1 草标并精修，随后执行 `hard-import` / `hard-publish`。HLMF 的 `hard_dataset_id` 必须是通用数据身份，不得含 snapshot/run/round 语义。发布后，将一个或多个 ID 写回 `stages.multi_finetune.hard_datasets`。

## 8. 阶段六：Multi-finetune 训练

阶段名：Multi-finetune。

命令：

```bash
make multi-finetune
```

输入：

- multitask winner：`runs/<experiment_id>/multitask/checkpoints/best.weights.h5`。
- `hard_datasets` 中经 HLMF CVAT 1.1 精修后发布的困难 positive/negative。
- 可选 `gold_datasets`：新录制、自动标注并经人工 CVAT 复核的通用 Gold positive/negative；不能从既有 PretrainSource/EValSource 冒充。
- `negative_datasets` 中的已发布真负样本。
- `datasets` 中的 mandatory pretrain replay pool。

处理：困难数据集、人工复核 Gold 和真负样本组成 hard/gold 侧；未被这些成员占用的 PretrainSource positive 组成 replay 侧。整个 CVAT-reviewed hard release 均派生为 `human_gold`，包括人工确认但未移动关键点的行和 `*_human_corrected` 行。默认 hard/gold 55%、replay 45%。两者必须都大于零且总和为 1；因此不能关闭 replay。每个 hard/gold/negative dataset 的 `weight` 继续参与侧内采样。gold 侧保持总负样本比例 10%，其中 `NEG_RUNTIME_CANDIDATE=0.05` 消费 CVAT 确认的 hard negative，`NEG_LOW_PALM_CANDIDATE=0.05` 消费已发布普通真负样本，避免人工 hard negative 被零比例静默排除。默认 `sampling.epoch_size=3000`；在当前 379 条 Gold `POS_RUNTIME` 的规模下，其精确 epoch 配额为 1436，平均每条期望抽取约 3.79 次，低于 rare-cell 门禁的 4 次上限，并保留少量余量。若更换 hard/Gold 成员，必须重新以实际 snapshot 运行采样门禁，不能通过放宽门禁来容纳过大的 epoch。

输出：

```text
HAND_TRAIN_ROOT/snapshots/<snapshot_id>/multi_finetune/*
HAND_TRAIN_ROOT/runs/<experiment_id>/multi_finetune/checkpoints/best.weights.h5
HAND_TRAIN_ROOT/runs/<experiment_id>/multi_finetune/history.json
```

Multi-finetune 训练结束后同样必须执行 Val、infer 和 export：

```bash
make val HLML_STAGE=multi_finetune
export HLML_INFER_INPUT=/path/to/representative/images
make infer HLML_STAGE=multi_finetune
make export HLML_STAGE=multi_finetune
```

三条命令必须使用同一 snapshot、experiment 和 stage；输出分别位于 `eval/multi_finetune/val/`、`inference/<experiment>/multi_finetune/` 与 `export/multi_finetune/`。完成 Val 后再冻结 winner，locked Test 仍只运行一次冻结结果。

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
make freeze-winner HLML_STAGE=multi_finetune HLML_RELEASE_ID=iris-v3-pro-r1
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
make locked-test HLML_STAGE=multi_finetune HLML_RELEASE_ID=iris-v3-pro-r1
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

输入：`configs/inference.yaml` 指定的原图文件夹（默认 `/root/autodl-tmp/DatesetFab/InferSource/0718/images`）、`palm_detector/eos-2.1/model_opt.onnx` 和所选 `HLML_MODEL_VERSION` 的 Hand checkpoint。Eos-2.1 文件来自 HLMF 的 `models/palm_detector/eos-2.1/model_384x224_opt.onnx`，部署到 HLML 时改名为 `model_opt.onnx`；ONNX 文件由执行环境单独部署，不提交 Git。

处理：Eos-2.1 将灰度原图缩放为 `384×224`，形成 float32 NCHW `[1,1,224,384]`；按 `14×24`、`7×12` 两层和各自两组 Anchor 解码共 840 个 Anchor，合并后执行一次全局 NMS（score `0.25`、IoU `0.10`、最多 2 手），再以 scale `1.8/1.8`、shift `0/-0.1` 构造 `256×256` Hand ROI 并运行选定 Iris 模型。这是部署级 smoke，不能替代固定 ROI Val/Test。

输出：

```text
HAND_TRAIN_ROOT/inference/<experiment_id>/<stage>/
```

其中包含 JSONL、summary 和按配置生成的标注图；默认拒绝覆盖。

## 12.1 阶段十点五：训练前 ONNX/A1 预检查

正式训练前，对三档结构分别执行：

```bash
export HLML_MODEL_VERSION=v3-pro  # 依次替换为 v3-max、v3-lite
export HLML_STAGE=geometry
make export-preflight
```

输入：所选架构、geometry snapshot 的 Train/Val/Test JSONL 和 `configs/deploy.yaml`。不需要训练 checkpoint；命令以固定 seed 生成非零探针权重，使零初始化 head 和 zero-gamma BN 不会掩盖算子、量化动态范围或融合问题。

处理：融合所有训练专用分支，导出 opset 11 静态 ONNX；检查训练图/部署图数值一致性、A1 算子白名单、depthwise group、模型大小、Keras/ONNX Runtime 一致性，并从 Train 抽 100、Val/Test 各抽 25 个 ROI 生成转换数据。

输出：`HAND_TRAIN_ROOT/preflight/<model_version>/<stage>/` 下的 `*_untrained.onnx`、`*.contract.json`、探针 `*.weights.h5` 和 `model_conversion/datasets.zip`。这些文件只用于官方工具链兼容性验证，`preflight.untrained=true`，不得作为精度模型。

## 13. 阶段十一：ONNX/A1 导出

阶段名：Export。

命令：

```bash
make export HLML_STAGE=multi_finetune
```

输入：所选 stage、所选 `HLML_MODEL_VERSION` 的正式 checkpoint、同一 stage 的 Train/Val/Test snapshot，以及 `configs/deploy.yaml`。

处理：导出静态 NCHW `[1,1,256,256]`、opset 11 ONNX，保持输出顺序 `[landmarks, hand_flag, handedness]`，审计 A1 允许算子、模型大小、depthwise group 和 Keras/ONNXRuntime 数值一致性。ONNX 验证通过后，只读当前 stage snapshot：从 Train 分层稳定抽取 100 个 calibration ROI，从 Val/Test 各抽取 25 个 evaluation ROI，保存为 `np.save` 生成的 `float32 (1,1,256,256)` NCHW 数组，像素为灰度 `uint8/255`。

Finetune 导出（`HLML_STAGE=multi_finetune`）额外要求 calibration 的 `sources.train` 以 `config_path` 引用 `configs/training.yaml`（profile 由 `HLML_STAGE` 选择）；导出 contract 会记录并校验 finetune 训练配置、curation manifest 与 multitask 初始 checkpoint 的 SHA256，作为 finetune 导出溯源。Pretrain 导出（geometry/multitask）解析同一训练配置的 pretrain profile，输入数据不变，也没有溯源要求。

输出默认位于：

```text
HAND_TRAIN_ROOT/runs/<experiment_id>/export/<stage>/
  hand_landmarker_<model_version>.onnx
  hand_landmarker_<model_version>.contract.json
  model_conversion/
    datasets/calibrate_datasets/*.npy
    datasets/evaluate_datasets/*.npy
    datasets.zip
    datasets_manifest.json
    datasets_report.json
```

`datasets.zip` 内只含 `datasets/` 及两类 `.npy`；calibration 不读取 Val/Test，评测集必须同时包含 Val 与 Test。模型、contract 和数据包均默认拒绝覆盖。

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
make acceptance-smoke \
  HLMF_REPO=/root/HandLandmarksFab \
  HLMF_PYTHON=/root/miniconda3/envs/anfab/bin/python
```

该命令使用 `HLMF_PYTHON` 在 HLMF 自己的 `anfab` 环境运行 contract 测试，再用当前 HLML 环境运行合成 warehouse 的三阶段/固定 ROI 测试并解析全部公共配置；两仓依赖不混装。合成接口覆盖部分发布的 EOS 2.0 proposal variant、同一 dataset 按 source 白名单跨 variant 组集、HCF 0813/0814 双头 teacher ID、HaMeR direct provenance 与 `hamer_inference`、HaMeR/RTMPose TFLite rescue 的 `norm×256` 上游 crop-pixel 约定及 canonical 规范化、HLMF 独立 published 图片、`source_crop_relpath`/`published_relpath` 与两类可选几何补救字段，不使用真实 Test 选择模型。Palm 解码回归测试另行校验 `[1,1,224,384]`、840 Anchor、矩形输出映射和全局 NMS。

## 15. `configs/datasets.yaml` 参数说明

- 成员列表：`stages.*.datasets`、`negative_datasets`、`hard_datasets`、`gold_datasets`、`evaluation.val/test` 均支持多个条目；构建 snapshot 时合并所有成员，并保留每个成员的 ID、variant 和权重。重复 ROI、跨 split 或同一 capture source 混用多个 variant 仍会失败。
- `dataset_id`：HLMF 发布的数据集逻辑 ID，不是目录绝对路径。
- `proposal_variant`：选择 dataset 中已发布的哪一版 Palm/ROI 结果。每个列表成员独立选择，因此同一 dataset 可用多个成员在不同 source 上消费不同 variant。HLMF 允许一个 variant 只发布到 dataset 的部分 source；没有 source 白名单时 HLML 会跳过未发布该 variant 的 source，但目标 split 若一个匹配 source 都没有则失败。
- `capture_source_ids`：dataset 类成员的可选非空、无重复 source ID 白名单。提供后不再宽泛消费该 variant 的其他 source；每个 ID 都必须存在于 manifest、匹配当前 Train/Val/Test split 且发布该成员的 `proposal_variant`。
- `weight`：正数，写入该来源记录的 `sampling_weight`。它控制相对抽样权重，不复制样本。
- `negative_dataset_id`：只能引用 HLMF `GoldSource/NegativeSamples/<id>/published/`；HLML 读取其独立 `published_relpath` 图片副本。
- `hard_dataset_id`：引用 HLMF `GoldSource/HardSamples/<id>/published/` 的 CVAT 精修困难集；来源身份由 `source_crop_relpath` 与 registry 核对，实际输入为独立 `published_relpath` 图片，允许 positive/negative。
- `gold_datasets`：multi-finetune 可选 `GoldSource/ReviewedDatasets` 中新录制且人工复核的 train Gold，格式与 `datasets` 相同，允许 positive/negative。
- `selections`：只为既有 `Selections/<id>/published/` 保留兼容读取，不是新流程配置项。
- `hard_fraction/replay_fraction`：默认 `0.55/0.45`，必须均大于 0 且总和为 1。
- `evaluation.val/test`：只选择 EValSource 中已复核 fixed ROI；两者按 manifest 的 split 字段过滤。
- `policies.performer_cross_split`：默认 `warn`，需要严格人员隔离可设为 `fail`。

`image_integrity`、`image_sha256` 和 `test_may_feed_training_or_mining` 是公开策略声明，应分别保持稳定 ID/registry/decode、图片 SHA 禁用和 Test 禁止回流。

## 16. `configs/training.yaml` 参数说明

### 16.1 模型与环境

- `environment`：锁定 Python、TensorFlow、CUDA 和 cuDNN 版本；服务器不匹配时先处理环境，不应为绕过检查而随意改版本。
- `model.version`：由 `HLML_MODEL_VERSION` 选择 `v3-pro`、`v3-max`、`v3-lite` 或历史 `v2`；训练、评估、推理、导出必须一致。
- `model.num_iterations/output_order`：保持七阶段迭代数和三输出契约；不同 v3 档位不增加辅助 head。
- `data.image_size/channels/input_layout/input_scale`：与 HLMF `256×256` 灰度 ROI 和 A1 NCHW 契约一致。

### 16.2 通用训练参数

- `training.epochs`、`batch_size`：训练轮数与批大小；显存不足时优先降低 batch size。
- `optimizer.learning_rate`：geometry 默认较大，multitask/multi-finetune profile 会覆盖为更小学习率。
- `initial_checkpoint`：阶段初始化权重；multitask 必须指向 geometry winner，multi-finetune 必须指向 multitask winner。
- `resume_checkpoint`：仅用于同一阶段中断恢复，不能代替跨阶段初始化。
- `gradient_clip_norm`、`mixed_precision`：数值稳定与性能选项；开启 mixed precision 前应验证目标环境和导出一致性。
- `progress_bar`：默认 `tqdm`，显示 epoch 与 batch 进度；也可设为 `keras` 或 `none`。正式训练保持 `tqdm`。
- `checkpoint.monitor/mode`、`early_stopping`、`learning_rate_schedule`：只使用 Val 指标；Test 禁止参与。
- `max_wall_time_hours`、`periodic_checkpoint`：服务器时间预算和恢复点策略。

### 16.3 采样、损失和增强

- `sampling.sample_type_fractions`：控制 positive/negative 类型抽样；geometry 的 negative 必须为 0，multitask profile 才启用真负样本。当前 v3 geometry 使用四个 Eos-2.1 + HaMeR r4 Train dataset，并按 `POS_RUNTIME=1.0`、其他类型为 0 抽样；当前 multitask 完整加入 `neg-eos_2.0-hcf0813-hp0.5`，按 `POS_RUNTIME=0.90`、`NEG_LOW_PALM_CANDIDATE=0.10`、其他类型为 0 抽样。multi-finetune 的 gold tier 使用 `POS_RUNTIME=0.70`、`POS_LOW_PALM=0.20`、`NEG_RUNTIME_CANDIDATE=0.05`、`NEG_LOW_PALM_CANDIDATE=0.05`，同时消费 hard negative 与普通真负样本。正式训练前仍须以 snapshot audit 的实际单元计数为准；不存在的 cell 只能按各 profile 明示的 missing-cell policy 处理，不能伪造 sample type。
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
- unknown handedness：合法 positive 仍参与 presence 与 landmarks（pixel error、NME、PCK）统计，只从 handedness 指标中排除；报告的 `excluded_unknown_label_count` 给出排除数量。经人工保留且略超固定 ROI 边界的有限关键点不裁剪，按原坐标计算误差并在 data contract 中报告 warning。
- `output.overwrite`：正式 Val/Test 建议为 `false`，locked Test 强制为 `false`。

### 17.2 `inference.yaml`：Folder Inference

- `input.images_dir/extensions/recursive`：任意文件夹推理的输入范围。
- `palm.models_root/model_id/model_filename`：默认解析为 `palm_detector/eos-2.1/model_opt.onnx`；`--palm-model-id` 只改变模型目录，所选模型必须满足同一 `384×224` 矩形输入、输出和 Anchor 契约。
- `palm.input_width/input_height/feature_levels`：固定为 `384/224`、`14×24` 与 `7×12` 两层；每层两组 Anchor 尺寸必须与 HLMF EOS 2.0 一致。`score_threshold=0.25`、`nms_iou_threshold=0.10`，所有层合并后只执行一次全局 NMS。
- `hand_roi.*`：应与部署端和 HLMF 使用的 ROI contract 对齐。
- `output.write_annotated_images/write_jsonl/draw_*`：控制检查产物，不影响模型数值。

### 17.3 `deploy.yaml`：ONNX/A1 Export

- `export.opset`、`input_name/output_names/dynamic_batch`：A1 部署接口，当前为 opset 11、固定 batch 和固定输出顺序。
- `maximum_model_size_mb`、`maximum_depthwise_group`、`a1_allowed_operators`：硬件审计门槛。
- `validate.random_samples` 与容差：Keras/融合 ONNX/ONNXRuntime 数值一致性检查。放宽容差前必须定位具体算子误差。
- `metadata`：部署端预处理和输出解释契约，必须和训练配置一致。
- `conversion_datasets`：保持启用；定义当前 snapshot 的 Train/Val/Test 输入、100/25/25 抽样数、分层字段和独立 `model_conversion` 输出目录。Calibration 的 `sources.train` 使用 `config_path: "configs/training.yaml"` 引用训练配置（finetune 导出依赖它认证训练溯源，见阶段十一），Val/Test 两个 evaluation source 仍直接给出 `labels` 路径。

## 18. 数据泄漏与 Test 锁定原则

Train、Val、Test 的 capture source 和 raw image 必须完全隔离；人员隔离至少保留警告。困难挖掘仅使用 Train。唯一 winner 只能根据固定 Val 冻结；Test 只执行一次该 winner，Test 输出不得被数据审计、采样、训练、mining、threshold 或 checkpoint 选择代码读取。
