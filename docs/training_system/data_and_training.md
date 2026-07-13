# 数据、权重与两阶段训练

## 1. Canonical JSONL

一个 JSONL 行对应一个已经裁好的 `256×256` Hand ROI。训练仅接受 07A 输出的 `hand_training_labels_pretrain.jsonl` 或 `hand_training_labels_finetune.jsonl`；Val/Test 仅接受 07B 输出的 Gold 文件。manifest、catalog、excluded、ignored 和旧版 `hand_training_labels.jsonl` 都不是正式 loader 输入。

关键字段如下：

- 唯一身份：`global_crop_id`、`dataset_id`、`source_group_id`；
- 图片：`crop_path`、`width=256`、`height=256`；
- 标签：`hand_presence.present`、`handedness.label`、三组 landmarks；
- ROI 溯源元数据：`palm_valid`、`palm_score`、`roi_rect`、`roi_corners_px`；这些字段不会让 Val/Test 加载或运行 Palm；
- Train：`training_stage`、`supervision_tier`、`sample_type`、质量/采样/loss 权重；
- Eval：`split`、`ground_truth_valid`、evaluation source/partition/owner。

Positive 必须有且仅有 ID `0..20` 的 21 点；loader 按 `id` 排序，不能依赖 JSON 数组顺序。Negative 的三组 landmarks 必须为空，handedness 必须为 `unknown`。当前 schema 没有逐点 visibility；一个 Gold 点无法可靠标注时应忽略整个 ROI，而不是发明逐点 mask。

模型监督使用 `landmarks_crop_norm`，像素换算固定为：

```text
x_px = x_norm × 255
y_px = y_norm × 255
```

Pseudo 可能包含轻微越界点，inspect 会报告；不会静默 clamp。Gold 点必须在 `[0,1]`。

## 2. Fail-closed 过滤

Train 必须满足：

- `selection_action=include`（字段必须存在）；
- `ignore_for_training != true`；
- `training_stage` 与配置阶段一致；
- 所有采样和 loss 权重有限且非负。

Val/Test 必须满足：

- `ground_truth_valid=true`；
- `split` 与配置一致；
- `palm_valid=true`；
- 不在 ignored 文件中。

满足上述条件后，Val/Test loader 直接读取每行 `crop_path` 指向的 `256×256` Hand ROI，并且只调用 Hand Landmarker。`palm_valid` 是 canonical 数据的质量/溯源字段，不是运行 Palm Detector 的开关。

Canonical 文件已完成质量裁决，因此 loader 不会再次按 `needs_review`、Palm score、teacher confidence 或学生预测过滤。`palm_valid=false` 也不表示 Hand negative：`POS_LOW_PALM` 是合法正样本。这里的 Palm 名称来自数据制作阶段，不代表训练或 Val/Test 会执行 Palm 推理。

图片只从 canonical `crop_path` 解析。`source_crop_path` 是搬迁前溯源信息，不可作为隐式 fallback。图片目录中可能残留未被 canonical 文件引用的旧 crop，禁止用目录 glob 产生样本。

## 3. Head 权重

每个 head 的有效 loss 权重分别是：

```text
presence = hand_presence_loss_weight
         × supervision_loss_weight
         × presence_quality_weight

landmark = landmark_loss_weight
         × supervision_loss_weight
         × landmark_quality_weight

handedness = handedness_loss_weight
           × supervision_loss_weight
           × handedness_quality_weight
```

Negative 的 landmark/handedness 权重为 0；unknown handedness 的 handedness 权重为 0。`sampling_weight` **只用于 sampler 在一个已经选定的 `supervision_tier × sample_type` 单元内选择记录**；它不决定单元配额，也不乘入 loss。

训练 step 对三个 head 分别计算：

```text
head_loss = sum(sample_weight × per_sample_loss) / sum(sample_weight)
```

分母为 0 时该 head loss 为 0。这样不同 batch 的 positive/negative 构成不会意外改变 head 的整体尺度。

默认损失：landmark 使用 Huber，presence 和 handedness 使用 BCE；三者再乘 YAML 的全局 coefficient 后组成 total loss。

## 4. Stage 1：Pseudo pretrain

输入为 `train_pretrain_merged` canonical 文件，且活动 supervision tier 必须只有 `pseudo`。每个实际 batch（包括最后一个不足 batch）先用 largest-remainder 构造四类整数配额：`POS_RUNTIME=56%`、`POS_LOW_PALM=14%`、`NEG_RUNTIME_CANDIDATE=25%`、`NEG_LOW_PALM_CANDIDATE=5%`，即合计 70% positive、25% runtime negative、5% low-Palm negative。余数相同时按上述固定顺序裁决。选定 `supervision_tier × sample_type` 单元后，才按该单元内的 `sampling_weight` 抽行。

所有正比例 sample type 都必须在每个活动 supervision tier 中存在，且对应单元的正 `sampling_weight` 总和必须大于 0。缺少交叉单元、`sampling_bucket != supervision_tier:sample_type` 或把桶权重静默重归一化到其他类别都会 fail-closed。

默认启用轻量、同步增强：亮度、对比度、噪声和小幅仿射。仿射会同步更新 21 点；点不会被静默裁到边界。水平翻转默认关闭。若后续确认真实手/镜像约定，启用翻转时必须同时执行 `x→1-x` 并交换 Left/Right。

主要输出：

```text
<run_dir>/checkpoints/best.weights.h5
<run_dir>/checkpoints/last.weights.h5
<run_dir>/checkpoints/final.weights.h5
<run_dir>/experiment_metadata.json
<run_dir>/history.json
<run_dir>/training_report.json
```

其中 `<run_dir>` 的默认值分别是 `${HAND_DATA_ROOT}/hand_landmarker_runs/v1/pretrain` 与 `${HAND_DATA_ROOT}/hand_landmarker_runs/v1/finetune`。另外还会写入：

```text
<run_dir>/checkpoints/best.weights.h5.state/
<run_dir>/checkpoints/best.weights.h5.state.json
<run_dir>/checkpoints/last.weights.h5.state/
<run_dir>/checkpoints/last.weights.h5.state.json
<run_dir>/logs/history.csv
<run_dir>/logs/tensorboard/
<run_dir>/model_summary.txt
```

`best.weights.h5`、`last.weights.h5` 和 `final.weights.h5` 都是仅含 Hand backbone 权重的 HDF5 文件。`.state/` 与 `.state.json` 是 best/last 对应的 optimizer、完成 epoch 和 monitor 状态；`training_report.json` 与 `experiment_metadata.json` 保存数据检查、标签/配置哈希和产物 SHA-256。`final.weights.h5` 是训练正常结束后当前内存中的权重；启用 `restore_best_weights` 时通常对应 early-stopping 恢复后的最佳权重。

新训练默认拒绝写入非空 `outputs.run_dir`，避免静默覆盖已有实验。续训应设置 `training.resume_checkpoint`；确认需要重新使用同一路径时，才显式设置 `outputs.overwrite: true`。更稳妥的做法是为每次新实验使用独立 run 目录。

## 5. Stage 2：Gold + pseudo replay

输入为 `train_finetune_merged` canonical 文件，并从 Stage 1 最佳权重初始化。启动时必须确认至少存在一条 `supervision_tier=gold`。

默认每个 batch 先用 half-up 得到约 40% Gold 和 60% pseudo replay 的整数行配额，同时用 largest-remainder 得到四类 sample type 的整数列配额，再构造满足两组边际总数的 `Gold/pseudo × sample_type` 交叉配额。默认 batch=32 时，Gold 为 13 条，四类总数依次为 `18/4/8/2`；Gold 四类为 `7/2/3/1`，pseudo 四类为 `11/2/5/1`。八个交叉单元均须存在。不要只用少量 Gold 长时间训练，否则容易遗忘第一阶段覆盖的姿态、背景和光照。

Stage 2 使用更低学习率，在人工 Val 上 early stopping。Test 不允许参与学习率、epoch、presence threshold、增强或量化选择。

Keras `fit` 固定使用 `shuffle=False`：batch 内容和 epoch 随机流完全由 `CanonicalSequence` 管理，Keras 不再独立打乱 Sequence 的 batch 顺序。每个正常 epoch 结束时 Sequence 自增 epoch；从 `training.resume_checkpoint` 恢复时，训练入口先读取完成 epoch，并调用 `CanonicalSequence.set_epoch(initial_epoch)`。同一 labels、seed、配置、绝对 epoch 和 batch index 会复现相同采样与增强随机流；恢复不会重放 epoch 0。该契约只覆盖 epoch 边界恢复，不声明支持 batch 中途恢复。

完整恢复依赖所选 best/last 权重旁的 `.state/` TensorFlow checkpoint；存在时会恢复 optimizer 和 epoch。若 `.state/` 缺失但 `.state.json` 仍在，只能恢复模型权重和记录的 epoch，报告会明确给出 optimizer 未恢复警告。`training.initial_checkpoint` 只加载初始权重（Stage 2 默认指向 Stage 1 的 `checkpoints/best.weights.h5`），不是续训语义；它与 `training.resume_checkpoint` 互斥。

## 6. 数据检查

```bash
make inspect
```

检查内容包括 JSONL/schema、ID 重复、stage/split、权重、图片存在性/可读性/尺寸、分布和文件 SHA-256。`make inspect` 会分别以 pretrain 与 finetune 训练配置审计当前 Train、Val 和锁定 Test，并在每次审计内两两检查 `global_crop_id`、`source_group_id`、解析路径和内容哈希；任一精确跨集合重叠都会失败，文件名相同本身只告警。pretrain 与 finetune 训练集允许 pseudo replay，因此不把两个训练阶段彼此比较作为泄漏门禁。图片内容哈希只能发现完全相同文件，无法替代采集 session 元数据。

训练入口不会假定操作者已单独运行 `make inspect`：`create_sequences` 会再次对实际参加训练的 Train/Val 计算图像 SHA-256，并对 `global_crop_id`、`source_group_id`、解析后的 crop 路径和图像内容哈希做跨 split 检查；任一精确重叠都会在创建 Sequence 前拒绝训练。该检查结果嵌入 run 下的 `experiment_metadata.json` 与 `training_report.json`，不会伪造一个不存在的独立 `data_report.json`。
