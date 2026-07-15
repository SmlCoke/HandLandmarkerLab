# Pretrain 数据与分阶段训练操作手册

本文是当前 v2 pretrain 的主操作手册。当前范围不包含 finetune，也不需要 Gold finetune 数据。

## 1. 为什么 pretrain 分成两个阶段

模型始终有三个输出：

1. `landmarks`：21 个二维关键点，共 42 个数；
2. `hand_flag`：ROI 中是否有手；
3. `handedness`：Left/Right。

训练代码一直能够计算三个 head 的 loss，但旧流程没有一个安全、正式的 multitask 阶段入口。原因不是缺少三头网络，而是原始 `NEG_RUNTIME_CANDIDATE` 和 `NEG_LOW_PALM_CANDIDATE` 只是 Google MediaPipe 没有给出结果的 teacher abstention。已有审计证明其中包含大量肉眼可见的手；直接将它们当作 `hand_flag=0` 会迫使 backbone 把真实手特征学习成背景，并通过共享特征反过来破坏 landmarks。

当前 pretrain 因而明确分成两阶段：

| 阶段 | 配置 | 数据 | 目的 |
|---|---|---|---|
| Geometry | `configs/train_geometry.yaml` | 只有经过提纯、具有完整 21 点的 positive | 先学稳定的手部几何与关键点拓扑 |
| Multitask | `configs/train_multitask.yaml` | 同一批 positive + 人工确认的 true negative | 在尽量保持 geometry 的同时学习 presence，并轻量学习 handedness |

Geometry 阶段不是“模型只能训练 landmarks”。它仍保留三个输出，只是 loss 配置有意设为 landmarks 主导、positive-only presence 辅助、handedness 关闭。Multitask 阶段会从 geometry 的 `best.weights.h5` 初始化，启用人工 true negative 和较小的 presence/handedness loss。

`make pretrain-multitask` 默认 fail-closed：没有足够人工确认负例时会拒绝训练，不会把未复核 teacher abstention 自动降格成负例。这也是先前没有把 multitask 与 geometry 一起自动执行的根本原因。

## 2. 当前文件与固定实验身份

`configs/` 只保留 9 个文件：

```text
curate_pretrain.yaml   持久化提纯与负例审查队列
train_smoke.yaml       128 ROI 过拟合门禁
train_geometry.yaml    第一阶段 geometry
train_multitask.yaml   第二阶段 multitask
eval_val.yaml          Val ROI 评估
eval_test.yaml         锁定 Test ROI 评估
infer.yaml             外部原图 Palm → ROI → Hand 推理
export.yaml            v2 融合、ONNX 与转换数据导出
export_preflight.yaml  训练前随机权重 ONNX 与转换数据预检
```

服务器路径与实验 ID 固定写在 `Makefile` 顶部：

```make
HAND_TRAIN_ROOT := /root/autodl-tmp/TrainFab/HLML-2.0
HAND_PRETRAIN_ID := v2-pretrain-r1
```

`HAND_TRAIN_ROOT` 是当前训练系统版本的唯一数据盘根；`HAND_PRETRAIN_ID` 同时标识提纯快照、geometry/multitask run、评估、推理和导出。review decisions 路径写在 `configs/curate_pretrain.yaml`，由程序生成，不是 Makefile 变量，也不需要人工创建目录或编写 JSONL。不再维护 curated ID、run ID、phase 三套外部变量。每次正式实验前直接修改并提交这两行。不要只在 shell 中临时 `export` 一个新值，否则后续难以仅凭仓库版本复核训练使用了哪套数据与输出目录。执行以下命令可核对本次 Make 实际使用的值：

```bash
make paths
```

目标数据盘布局如下。`TrainFab` 只属于本仓库的训练、评估和推理系统；`DatasetFab` 是另一个仓库的数据集制作系统，两者不得互相写入。

```text
/root/autodl-tmp/
├── DatasetFab/                         # 独立的数据集制作系统
└── TrainFab/
    └── HLML-2.0/                       # HAND_TRAIN_ROOT
        ├── eval_sources/
        ├── hand_landmarker_inference/
        ├── hand_landmarker_runs/
        ├── hand_landmarker_reviews/       # curate 自动创建的可视化审查工作区
        ├── peak_train_data/
        ├── soar_train_data/
        ├── test_merged/
        ├── train_pretrain_merged/
        ├── train_pretrain_curated/
        └── val_merged/
```

新的数据提纯或训练不得复用已有 ID。建议按 `v2-pretrain-r1`、`v2-pretrain-r2` 递增；系统默认拒绝覆盖非空训练目录。

## 3. 持久化提纯产物

原始入口固定为：

```text
${HAND_TRAIN_ROOT}/train_pretrain_merged/05_labels/hand_training_labels_pretrain.jsonl
```

运行 `make pretrain-curate` 后生成：

```text
train_pretrain_curated/<HAND_PRETRAIN_ID>/
├── images/                         # 真正允许进入训练的独立 ROI
├── review_images/                  # 冻结后的待审负例 ROI，不参与训练
├── 05_labels/
│   ├── hand_training_labels_pretrain_landmarks.jsonl
│   ├── hand_training_labels_pretrain_multitask.jsonl
│   └── hand_training_labels_pretrain_smoke.jsonl
├── audit/
│   ├── pretrain_curation_catalog.jsonl
│   ├── included_landmarks.jsonl
│   ├── excluded_and_held.jsonl
│   ├── negative_review_queue.jsonl
│   ├── image_manifest.jsonl
│   └── review_image_manifest.jsonl
└── qc/
    ├── curation_report.json
    └── sha256_manifest.json
```

`landmarks.jsonl` 只包含合格 positive；`multitask.jsonl` 初次提纯时也只有这些 positive。`train_pretrain_curated/.../review_images` 是冻结的内部审计副本，`hand_landmarker_reviews/.../negative_candidates` 才是人工删除式复核工作区。人工完成后，程序仅把工作区中仍存在、manifest 匹配且 SHA-256 未改变的图片写成 `CONFIRMED_NEGATIVE`；自动重叠门禁仍可拒绝其中存在冲突的样本。

训练入口会验证 labels、materialized ROI 和 manifest 的 SHA-256。提纯不是训练时的内存过滤，因此训练结束后仍能复核当时真正使用的 JSONL 和图片。

执行命令 `make pretrain-curate` 还会自动创建负样本人工复核工作区：

```text
hand_landmarker_reviews/<HAND_PRETRAIN_ID>/
├── negative_candidates/              # 删除有手/不确定图片，只保留明确无手图片
│   ├── NEG_RUNTIME_CANDIDATE/<dataset_id>/*.png
│   └── NEG_LOW_PALM_CANDIDATE/<dataset_id>/*.png
├── review_manifest.jsonl              # 程序生成；禁止手工修改
├── review_report.json
├── REVIEW_INSTRUCTIONS.md
└── negative_review_decisions.jsonl    # 完成复核后由程序自动生成
```

## 4. 人工完整操作流程

以下命令均在服务器仓库根目录执行。

### 4.1 更新代码和确认环境

```bash
git pull
conda activate hand-landmarker-tf29
make paths
make compile
make test-unit
```

检查 `make paths` 的数据根、curated ID 和 run ID。若目标目录已经用于旧实验，先修改并提交 Makefile 中的 ID，不要打开 `overwrite` 覆盖旧实验。

### 4.2 第一次提纯并产生人工审查包

```bash
make pretrain-curate
make test
```

`make test` 的导出预检需要刚生成的 curated geometry Train 和已冻结的 Val/Test。若只检查 Python 单测而尚未提纯数据，使用 `make test-unit`。

首先检查：

```text
<curated_root>/qc/curation_report.json
<curated_root>/audit/negative_review_queue.jsonl
<curated_root>/audit/review_image_manifest.jsonl
<curated_root>/review_images/
${HAND_TRAIN_ROOT}/hand_landmarker_reviews/<HAND_PRETRAIN_ID>/negative_candidates/
${HAND_TRAIN_ROOT}/hand_landmarker_reviews/<HAND_PRETRAIN_ID>/review_report.json
```

`curation_report.json` 中至少需要核对：

- `included_landmark_positives` 是否与预期接近；
- `negative_review_queue` 是否等于待人工处理的规模；
- `negative_overlap_confirmed_hand` 是否异常偏高；
- `reason_counts` 中是否出现新的未知原因；
- `included_by_dataset`、`included_by_sample_type` 是否被单一来源完全支配。

### 4.3 人工复核负例

直接递归打开 `hand_landmarker_reviews/<HAND_PRETRAIN_ID>/negative_candidates/`：

- 图片中只要存在手、手指、手腕或疑似手部区域，就删除该图片；
- 遮挡、模糊、过暗、过曝、边缘局部或无法确定的图片也删除；
- 只有明确无手的背景 ROI 才保留；
- 不要增加、改名、移动或编辑图片，否则 manifest/哈希门禁会拒绝；
- `NEG_RUNTIME_CANDIDATE`、`NEG_LOW_PALM_CANDIDATE` 和各 `dataset_id` 已分目录，可由三人直接按目录分工；无需记录逐图 reviewer，也无需手写 JSONL。

配置中的 `review.reviewer: hlml-visual-review-team` 表示这是一批由三人团队共同完成的可视化复核。`make pretrain-curate-reviewed` 会为所有保留图片统一生成 `reviewer`、`reviewed_at`、`review_method` 和图片 SHA-256。运行该命令本身就是“全部目录已审完”的显式确认，因此未完成前不要运行。

### 4.4 从保留图片自动生成决策并重建快照

```bash
make pretrain-curate-reviewed
```

该命令扫描 `negative_candidates/` 中仍存在的图片，对照 `review_manifest.jsonl` 检查身份和 SHA-256，自动生成 YAML 所指定的 `negative_review_decisions.jsonl`，再显式重建同一个 curated ID。被删除的图片继续保持 HOLD，不会进入训练；自动检测到与已确认手重叠的候选即使被保留，也仍不会进入 multitask。重建后检查：

```bash
make check-multitask-data
```

默认 multitask 门禁要求：

- confirmed negative 总数不少于 500；
- `NEG_RUNTIME_CANDIDATE` 不少于 100；
- `NEG_LOW_PALM_CANDIDATE` 不少于 100；
- 每个进入 multitask 的 negative 都带 `CONFIRMED_NEGATIVE`、团队 reviewer、时间、review method 和审查图片 SHA-256；
- 不允许任何未复核 negative 混入。

门槛写在 `configs/train_multitask.yaml`，变更门槛必须作为一次明确的配置变更提交，不能通过删掉 gate 绕过。数量不足时仍可训练 geometry，但不可启动 multitask。

`hand_landmarker_reviews/<HAND_PRETRAIN_ID>/review_report.json` 记录原候选数、保留确认数和删除数；`qc/sha256_manifest.json` 与 `qc/curation_report.json` 记录自动生成 decisions 文件的路径、SHA-256 和决策数。确认后的 multitask JSONL 和图片全部保留在磁盘。

### 4.5 Geometry 阶段

```bash
make doctor
make inspect-geometry
make pretrain-geometry-smoke
make pretrain-geometry
```

顺序含义：

1. `doctor` 检查 Python 3.8、TensorFlow 2.9、CUDA/cuDNN 和 GPU；
2. `inspect-geometry` 审计 geometry Train、Val、锁定 Test，并检查跨 split 泄漏；
3. `pretrain-geometry-smoke` 在固定 128 ROI 上训练，并逐条前向验证是否真的可过拟合；
4. `pretrain-geometry` 会再次验证 smoke gate，然后启动完整 geometry。

系统不提供聚合的 `make pretrain` 或缩写 `make train`；这是为了让日志中的每条命令都能独立说明实际执行的阶段。首次训练仍按上述四条显式命令顺序执行。

Geometry 输出：

```text
${HAND_TRAIN_ROOT}/hand_landmarker_runs/<PRETRAIN_ID>/geometry/
```

正式候选权重是 `checkpoints/best.weights.h5`。checkpoint、ReduceLROnPlateau 和 EarlyStopping 都以 `val_landmark_mae/min` 为准；训练结束还会核验 best 状态是否真的对应 history 中的最低验证 MAE。

### 4.6 Geometry 评估与推理

```bash
make eval-val-geometry
# 根据 Val 报告确定 hand_flag threshold，并写回 eval_test.yaml / infer.yaml。
make eval-test-geometry
make infer-geometry
```

Val/Test 直接读取已有 256×256 Hand ROI，只运行 Hand Landmarker；它们不运行 Palm。`make infer-geometry` 才会对外部原图执行 Palm → rotated ROI → Hand，因此两者不能混为同一评估口径。

Test 只能在 Val 已完成 checkpoint、threshold 和训练方案选择后运行。Test 结果不得反向用于调学习率、epoch、增强、loss coefficient 或 threshold。

### 4.7 Multitask 阶段

人工负例门禁通过且 geometry best checkpoint 已存在后执行：

```bash
make check-multitask-data
make inspect-multitask
make pretrain-multitask
```

`make pretrain-multitask` 本身也会先运行门禁与 inspect。它从：

```text
<RUN_ID>/geometry/checkpoints/best.weights.h5
```

初始化，并写入：

```text
<RUN_ID>/multitask/
```

默认 batch 采样为 90% positive、10% confirmed negative：

```text
POS_RUNTIME               72%
POS_LOW_PALM              18%
NEG_RUNTIME_CANDIDATE      8%
NEG_LOW_PALM_CANDIDATE     2%
```

Landmark coefficient 仍为 20，presence 为 0.25，handedness 为 0.05；学习率降到 `5e-5`。checkpoint、降学习率与 early stopping 统一最小化 geometry-first 指标：

```text
val_multitask_score = val_landmark_mae
                    + 0.02  × (1 - val_hand_flag_accuracy)
                    + 0.005 × (1 - val_handedness_accuracy)
```

分类误差会参与选 best，但权重明显低于 landmark MAE，避免 presence 变好却让 geometry 再次崩坏。该派生指标也会写入 history 和 checkpoint state，训练结束时按 history 复核 best epoch。

如果 hand_flag 指标上升但 Val landmark MAE、PCK 或可视化骨架明显恶化，不应接受该 checkpoint。优先降低 presence coefficient/negative fraction，而不是增加错误负例。

### 4.8 Multitask 评估、推理和导出

每个命令直接在目标名中选择 multitask 产物，不再要求人工传入 phase 环境变量：

```bash
make eval-val-multitask
# 冻结 multitask 自己的 Val threshold 后：
make eval-test-multitask
make infer-multitask
make export-multitask
```

Geometry 和 multitask 必须各自在 Val 上确定 threshold，不能沿用另一个阶段的值。

## 5. v2 模型与导出约束

所有训练、Val/Test、infer 和 export 配置都固定使用 `model.version: v2`。v1 已从当前 registry 移除。

v2 的改动：

- 所有 `LeakyReLU` 替换为普通 `ReLU`；
- stage 深度固定为 `[2,2,3,4,4,6,6]`，通道为 `[24,32,64,128,192,256,384]`；
- pointwise、depthwise、stem 和 stride-2 主干均使用训练期 `Conv + BatchNorm`；
- 每个 residual/downsample 主分支末端 BN 的 gamma 零初始化，使网络初始接近稳定 shortcut，而不是连续累加随机残差；
- landmark head 使用零 kernel、`0.5` bias，hand flag/handedness 使用零 kernel、零 logits bias；
- 导出前把每个 BN 精确折叠为一个带 bias 的 Conv，部署图不携带 BN；
- 部署图约 230 万参数；导出器按真实 ONNX 文件检查并强制不超过 15 MiB；
- 输入和三个输出的名称、顺序、shape、数据类型完全不变。

严格导出白名单为：

```text
Conv, Add, Relu, MaxPool, Sigmoid, Identity, Reshape
```

`LeakyRelu` 不再在白名单中。`make export-geometry` 或 `make export-multitask` 会依次验证：

1. 训练图与融合后 Keras 图的数值一致性；
2. 融合后的 Keras 与 ONNX 数值一致性；
3. 固定输入/三输出接口；
4. ONNX opset 11；
5. 算子名称和 Conv/MaxPool 属性限制；
6. 转换校准/评测 NPY 数据集。

`make test` 还会调用 `configs/export_preflight.yaml`，用固定随机种子生成未经训练的 ONNX 和真实转换数据包。这只验证融合、I/O、算子、15 MiB 大小门槛以及官方转换兼容性，绝不能用于精度判断。若只想运行代码单测或只重建预检产物，可分别运行：

```bash
make test-unit
make test-export-preflight
```

融合报告写入 `hand_landmarker_v2.contract.json` 的 `reparameterization_parity`，其中记录训练参数量、部署参数量和逐输出最大误差。只要出现 `LeakyRelu`、未折叠 BN 或其他白名单外算子，默认导出就会失败，不应使用 `--force` 交付官方转换。

## 6. Canonical 标签与 loss 规则

一个 canonical JSONL 行对应一张 256×256 Hand ROI。Positive 必须有且仅有 ID `0..20` 的 21 点；loader 按 ID 排序。Negative 的 landmarks 必须为空，handedness 必须为 `unknown`。

每个 head 的记录级有效权重为：

```text
presence   = hand_presence_loss_weight × supervision_loss_weight × presence_quality_weight
landmark   = landmark_loss_weight × supervision_loss_weight × landmark_quality_weight
handedness = handedness_loss_weight × supervision_loss_weight × handedness_quality_weight
```

Negative 的 landmark/handedness 权重自动为 0；unknown handedness 的 handedness 权重为 0。`sampling_weight` 只在已经选定的 `supervision_tier × sample_type` 单元内抽样，不乘入 loss。

增强会同步变换图片与 21 点，不会静默 clamp 越界点。水平翻转仍关闭，因为真实手与镜像 handedness 约定尚未完成独立验证。

## 7. 输出保护、恢复与复核

新训练默认拒绝写入非空 `outputs.run_dir`。重跑时优先修改 Makefile 的 run ID；只有明确续训才在对应 YAML 中填写 `training.resume_checkpoint`。`initial_checkpoint` 只表示从某权重开始新的阶段，multitask 正是这种语义。

每个训练目录包含：

```text
checkpoints/best.weights.h5
checkpoints/last.weights.h5
checkpoints/final.weights.h5
checkpoints/*.state/
checkpoints/*.state.json
experiment_metadata.json
history.json
training_report.json
logs/history.csv
logs/tensorboard/
model_summary.txt
```

复核一次实验时至少保存并对应检查：

- 训练所用 Git commit 与 dirty 状态；
- Makefile 中的 `HAND_TRAIN_ROOT`、`HAND_PRETRAIN_ID` 和所执行的显式阶段目标；
- curation manifest、review manifest/report、自动 decision SHA-256、训练 labels SHA-256、逐图 SHA-256；
- best checkpoint SHA-256 与 best epoch；
- Val 选择的 threshold；
- Test、infer 与 export 的模型 SHA-256；
- ONNX contract 中的 operator list 与两级 parity 报告。

## 8. 最短命令清单

Geometry：

```bash
make pretrain-curate
make doctor
make inspect-geometry
make pretrain-geometry-smoke
make pretrain-geometry
make eval-val-geometry
make eval-test-geometry
make infer-geometry
```

人工负例与 multitask：

```bash
# 删除 negative_candidates 中所有有手或不确定图片后
make pretrain-curate-reviewed
make check-multitask-data
make pretrain-multitask
make eval-val-multitask
make eval-test-multitask
make infer-multitask
make export-multitask
```

只生成官方转换 NPY 数据：

```bash
make conversion-data-multitask
```
