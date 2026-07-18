# HLML 3.0 完整训练流程

## 1. 系统目标

HLML 训练 A1 板端 Hand Landmarker。输入是 Palm 阶段产生的 `256×256` 灰度 Hand ROI；输出是 21 个二维关键点、手存在概率和左右手概率。Palm Detector 不在本仓库训练。

流程：

```text
HLMF 数据制作与认证
  → pretrain geometry
  → 人工删除式负样本复核
  → pretrain multitask
  → 自动困难样本选择 + 少量人工 Gold
  → finetune data_only / structure / optional structure_roi_aug
  → Val/infer 配对比较
  → locked Test
  → ONNX / 厂商工具链 / 上板
```

本文只描述通用流程。服务器当前完成到哪里见 [当前训练状态](HLML_current_training_status.md)。

## 2. 根目录与零拷贝原则

Makefile 默认：

```text
HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
HAND_TRAIN_ROOT=/root/autodl-tmp/TrainFab/HLML-3.0
HAND_PRETRAIN_ID=<pretrain-id>
HAND_FINETUNE_ID=<finetune-data-id>
```

`DatesetFab` 是与训练版本无关的可再生数据仓库：

```text
DatesetFab/
├── PretrainSource/   # 原图、00～03 和 pseudo 标签
├── GoldSource/       # domain/source-id；task 是待标态，published 是发布态
└── eval_sources/     # 固定 Val/Test 真源
```

HLMF/HLML 的聚合标签把 `crop_path` 直接指向该仓库；不再建立 `HLML-3.0/train_sources` 或逐版本 Gold 副本。

允许物化图片的情况只有：

- 人工删除式负样本审核树；
- CVAT Gold task；
- 可视化/overlay；
- 模型转换输入包。

同一数据盘内优先硬链接；所有可训练清单仍绑定内容 SHA-256。

## 3. 初始化和代码门控

```bash
cd /root/HandLandmarksFab
git pull --ff-only
conda activate anfab
make compile
make test

cd /root/HandLandmarkerLab
git pull --ff-only
conda activate hand-landmarker-tf29
make paths
make compile
make test-unit
make doctor
```

任何门控失败都先修复，不开始人工 CVAT 或正式训练。

## 4. HLMF 聚合 pretrain、Val 和 Test

```bash
cd /root/HandLandmarksFab
conda activate anfab
make finalize_train_pretrain
make build_pretrain_source_registry
make finalize_val
make finalize_test
```

程序负责：

- 从 DatesetFab 校验每个来源 manifest/标签/图片；
- namespace、去重、质量分层和训练权重；
- 生成 HLML 可用的绝对 `crop_path`；
- 发布 source registry 和 SHA 报告。

人工不拼 JSONL、不改 ID、不写 SHA。

输出：

```text
HLML-3.0/train_pretrain_merged/
HLML-3.0/val_merged/
HLML-3.0/test_merged/
```

## 5. Pretrain curation 和负样本人工复核

### 5.1 第一次 curation

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
make pretrain-curate
```

程序生成 geometry 正样本快照、128 ROI smoke 子集和负样本候选审核树。MediaPipe 没输出手只表示“教师放弃判断”，不是可信背景；这些 `NEG_*_CANDIDATE` 在人工确认前不能进入 multitask。

### 5.2 人工删除式复核

审核说明以生成目录中的 README/manifest 为准。操作模式：

1. 完整复制 `negative_candidates/` 为 `negative_reviewed/`，保持相对路径。
2. 在 `negative_reviewed/` 中逐图查看。
3. 删除所有含手、手指、手腕、模糊或无法确认的图片。
4. 只保留明确背景；不新增、不重命名、不移动、不编辑图片。

可以 7z/zip 压缩后上传网盘、下载到本地复核、重新压缩上传。普通压缩/解压不会改变文件内容，所以图片 SHA-256 不变；不要使用会重编码图片的软件。

复核完成：

```bash
make pretrain-curate-reviewed
```

程序自动比较候选全集和保留集合，生成审核决策、removed/quarantine 清单，逐图验证 SHA，并冻结 multitask 数据。人工不写删除清单。

## 6. Geometry

geometry 只训练具有完整 21 点的 positive，先让模型学习稳定几何；hand_flag/handedness 不是该阶段重点。

```bash
make inspect-geometry
make inspect-geometry-smoke
make pretrain-geometry-smoke
make pretrain-geometry
make eval-val-geometry
make infer-geometry
```

smoke 的意义是证明模型、loss、梯度、checkpoint 和数据管线能过拟合小集合，不代表泛化效果已经足够。

结果：

```text
hand_landmarker_runs/<HAND_PRETRAIN_ID>/smoke/
hand_landmarker_runs/<HAND_PRETRAIN_ID>/geometry/
hand_landmarker_runs/<HAND_PRETRAIN_ID>/eval/geometry/val/
hand_landmarker_inference/<HAND_PRETRAIN_ID>/geometry/
```

## 7. Multitask

multitask 从 geometry best 初始化，加入人工确认 true negative，训练 landmark、hand_flag 和轻量 handedness。

```bash
make check-multitask-data
make inspect-multitask
make pretrain-multitask
make eval-val-multitask
make infer-multitask
make export-multitask
```

checkpoint 使用 geometry-first 的 `val_multitask_score`。若门控报告确认负例数量/类型不足，返回第 5 节，不绕过。

## 8. Finetune 数据：Gold + replay

### 8.1 数据 ID 和实验 ID

- `HAND_FINETUNE_ID`：冻结的数据快照，决定 Gold、replay、curated labels。
- `FINETUNE_EXPERIMENT_ID`：训练/评估/推理/导出目录，可在同一数据上运行多个候选。
- `FINETUNE_PROFILE`：`data_only`、`structure`、`structure_roi_aug`。

因此比较模型候选时保持同一个 `HAND_FINETUNE_ID`，只更换 `FINETUNE_EXPERIMENT_ID`；不复制图片或标签，也不让候选使用不同数据。

### 8.2 两类训练数据分别解决什么问题

Finetune 不是“只用少量人工数据再训练”，而是把两类监督合成一个冻结快照：

- **Gold**：人工确认的 presence、21 点和 handedness。它纠正教师漏检、关键点偏差、错误手势形状和 presence 错误。
- **replay（回放数据）**：从 pretrain 来源中确定性抽出的高质量伪标签和背景样本。它不是 Gold；作用是保留原有场景覆盖、presence 和 handedness 能力，避免模型只记住少量人工场景。

Gold 可以来自多个不可变 source：每批外部 Dragon Gold、新录制 Gold、student–teacher disagreement Gold，以及确有需要时由历史 hard case 转成的 reviewed Gold。多个同类 source 会一起聚合；不能覆盖或删除旧 source 来“腾位置”。

数据流如下：

```text
DatesetFab/GoldSource/*/*/published ──HLMF finalize──> hmlf_gold_merged ─┐
                                                                        ├─ 显式批次选择 ─> finetune-curate
pretrain registry ──prepare-finetune-sources──> mandatory replay ───────┘
                              └───────────────> disagreement score pool
```

### 8.3 在长期 GoldSource 建立 Gold

```bash
cd /root/HandLandmarksFab
conda activate anfab

# 每批 Dragon 单独运行一次，批次 ID 不得复用
make prepare_dragon_gold \
  HAND_FINETUNE_ID=<finetune-data-id> \
  DRAGON_SOURCE_ROOT=$HAND_DATASET_ROOT/GoldSource/dragon/<dragon-batch-id>/source \
  DRAGON_BATCH_ID=<unique-dragon-batch-id>

# 所有 Gold source 发布/导入后重新聚合
make finalize_train_finetune HAND_FINETUNE_ID=<finetune-data-id>
```

`DatesetFab/GoldSource/<domain>/<source-id>/published/finetune_source.json` 是单个批次的认证描述符。GoldSource 与训练版本无关；新 finetune 不再 seed 或复制旧 Gold，而是直接发现所有 published 子批次。HLMF 的 `hmlf_gold_merged` 是本次训练版本可重建的全仓认证聚合。

Gold 的“发布”和“本次训练启用”不是一回事：

- 发布把一批人工真值变成长期、不可变、跨实验复用的 GoldSource；
- HLMF finalize 认证并聚合仓库中 **全部** published 批次，用于统一做重复、冲突和 SHA 检查；
- HLML 随后通过 `gold_selection.yaml` 对每个 source ID 明确写 `enabled: true/false`；只有 true 的批次进入本次训练。

因此发布 Dragon 或新录制 Gold 不需要在语义上绑定某个 finetune ID。命令里的 `HAND_FINETUNE_ID` 只是让 selection 类任务找到对应 mining request，以及指定本次聚合快照的输出工作区；published Gold 未来仍可被其他数据 ID 直接复用。disabled 也只是本次不用，不会删除或贬低该批数据。

### 8.4 在 HLML 建立 replay 和 disagreement score pool

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
make prepare-finetune-sources HAND_FINETUNE_ID=<finetune-data-id>
```

程序自动完成：

1. 认证 HLMF 的 pretrain source registry、curated multitask 标签和 checkpoint，并读取 GoldSource 中所有历史/pending Gold 身份用于排重；
2. 从广泛来源中确定性选择 replay，保存其来源、标签和 SHA；
3. 用当前 student checkpoint 对可选 positive 推理；
4. 把 student 预测与 MediaPipe teacher 伪标签比较，生成逐 ROI disagreement 分数池。

程序不是按目录名猜 replay，也不要求人工列出 replay source ID。默认规则写在 `configs/prepare_finetune_sources.yaml` 的 `selection.pretrain_replay`：

1. 输入只能来自已经认证的 pretrain source registry 和当前 pretrain ID 的 curated multitask 标签；registry 把每行恢复到 DatesetFab 的真实来源路径并校验图片 SHA；
2. 所有人工确认的背景负样本都必须保留；若其数量已经超过 `max_records`，程序直接失败，不能偷偷丢负例；
3. 在剩余容量中确定性抽取 positive。默认总上限为 10000，positive 的 `POS_RUNTIME` / `POS_LOW_PALM` 目标比例为 0.75 / 0.25，并在各原始 dataset 间分配；
4. 排序和抽样使用固定 salt，相同配置、标签与 SHA 会得到相同 replay；
5. replay descriptor 保存父 pretrain ID、标签、真实图片路径和内容 SHA，之后由 finetune 门控再次认证。

如确实要改变 replay 上限或正样本比例，应先修改上述配置，再用一个全新的 `HAND_FINETUNE_ID` 运行；不要修改已经生成的 replay JSONL。replay 是强制来源，不能像 Gold 一样通过选择清单关闭。

这里的 **student** 是正在训练的自研 Hand Landmarker，**teacher** 是 MediaPipe 伪标签。分歧大只表示两者对同一 ROI 的关键点差别大，因此适合人工复核；不代表 teacher 必然正确。查看：

```text
finetune/<id>/sources/replay/
finetune/<id>/mining/teacher_student/disagreement_scores.jsonl
finetune/<id>/mining/disagreement_gold/selection_report.json
finetune/<id>/mining/prepare_finetune_sources_report.json
```

这些产物绑定 checkpoint 和输入 SHA。输入或 checkpoint 变化时应使用新的 finetune 数据 ID 重新建立，不覆盖已有快照。

### 8.5 本轮没有时间标 disagreement/negative-removed 时

这两类都是可选的增量 Gold，不是 finetune 的硬前置。仍然运行一次 `prepare-finetune-sources` 来建立 mandatory replay 和 disagreement 分数池，但可以：

- 不运行 `prepare-finetune-round`；
- 不在 HLMF 导出新的 disagreement/negative-removed task；
- 只导入已经完成的 CVAT task；
- 在 Gold 选择中复用历史 published 困难样本 Gold，或把这些领域全部设为 disabled。

之后照常让 HLMF finalize 全部 published Gold，再冻结本次选择。只要至少一个启用的 Gold 来源满足 Gold 门控，mandatory replay 完整，finetune 就可以继续。自动生成的 disagreement 分数池保留在当前 finetune 工作区，未来要人工处理时应创建新的 round/source ID，不能把未冻结的分数列表手改成训练标签。

## 9. ROI 多轮 Gold

### 9.1 为什么要分 round/source

一个 `round-id` 表示一次冻结选样，一个 `source-id` 表示一份不可变 CVAT/Gold 来源。以后想从同一原始来源继续取样时，创建新 round/source ID；程序会保留并排除历史 ROI，因此旧人工标注不会浪费，也不需要删除。

### 9.2 先制作新录制任务

1. 在 `GoldSource/new_recorded_gold/<source-id>/source/` 放入无损图片流，并在该目录跑完 HLMF 00～03；
2. HLMF 使用 `native_existing` 和本轮计划给出的显式限额确定性抽样；
3. 任务生成后，查看 `task_descriptor.json` 得到实际任务数；
4. 不要人工从原始目录随意挑图，也不要在任务冻结后改限额。

HLMF 命令：

```bash
make export_finetune_gold \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_SOURCE_ID=new_recorded_gold_<round-id> \
  FINETUNE_SOURCE_MODE=native_existing \
  FINETUNE_RAW_SOURCE_ROOT=$HAND_DATASET_ROOT/GoldSource/new_recorded_gold/new_recorded_gold_<round-id>/source \
  FINETUNE_MAX_ITEMS=<new-recorded-limit>
```

### 9.3 冻结本轮 disagreement selection

新录制任务存在后，在 HLML 执行：

```bash
make prepare-finetune-round \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_ROUND_ID=<round-id> \
  FINETUNE_GOLD_BUDGET=<round-budget> \
  NEW_RECORDED_SOURCE_IDS=new_recorded_gold_<round-id>[,new_recorded_gold_<other-id>]
```

`FINETUNE_GOLD_BUDGET` 是本轮两个 CVAT task 的合计上限，必须由执行计划显式指定。程序读取：

- GoldSource 内所有已发布 Gold 标签和所有 pending task manifest；
- 所有历史 selection request；
- 当前/历史 CVAT task manifest；
- Val/Test labels 与 ignored sidecar。

并按 `parent_global_crop_id`、`global_crop_id`、来源图片身份、ROI SHA256、归一化像素 SHA256 排除重复。disagreement 上限等于“本轮总预算减新录制实际任务数”；候选不足时任务可以少于预算，但不会用低价值重复项硬凑。

输出：

```text
finetune/<id>/mining/rounds/<round-id>/disagreement_gold_<round-id>/
├── selection_request.jsonl
├── selection_report.json
└── ranked_eligible.jsonl
```

`selection_report.json` 必须确认冻结预算、新录制计数、排除文件和最终选中数都符合预期。

### 9.4 HLMF 导出、人工标注和导入

回到 HLMF 导出 disagreement：

```bash
make export_finetune_gold \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_SOURCE_ID=disagreement_gold_<round-id> \
  FINETUNE_SOURCE_MODE=selection_subset
```

HLMF 为每个批次生成 `GoldSource/<domain>/<source-id>/task/qc/cvat_job_plan.json`。人工按计划在 CVAT 完成完整 21 点和 handedness，或明确标 `no_hand` / `ignore_for_training`；返回的完整 XML 放入同一批次的 `task/reviewed.xml`。严格 import 成功后，必要审计文件移入 `published/audit/`，task 自动删除；此后该批次只以 published 身份参与聚合和训练。

```bash
make import_finetune_gold HAND_FINETUNE_ID=<finetune-data-id>
make finalize_train_finetune HAND_FINETUNE_ID=<finetune-data-id>
```

每次增加一轮 Gold 后都重新 finalize；最终聚合会保留所有历史 published 来源并跨轮去重。`source` 是原始真源、`task` 是暂存人工任务、`published` 是认证训练来源，三者不能因文件名相似而混用。Dragon 原始整图/标注与生成 ROI 不同，长期保留 `source + published`；新录制批次也可保留不可再生的 source，但不会长期保留已完成 task。然后返回 HLML。

### 9.5 negative-removed 的多轮做法

如果计划新增这类 Gold，先在 **新的 finetune 数据 ID** 下修改 `configs/prepare_finetune_sources.yaml`：

```yaml
selection:
  negative_removed:
    enabled: true
    max_items: <本轮数量>
```

再运行 `make prepare-finetune-sources`。程序从当前 pretrain 的 `negative_removed_manifest.jsonl` 选择候选，并排除历史 Gold、pending task、Val/Test 及同轮其他 request。随后用新的批次 ID 在 HLMF 执行：

```bash
make export_finetune_gold \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_SOURCE_ID=negative_removed_gold_<round-id> \
  FINETUNE_SOURCE_MODE=selection_subset
```

CVAT/import/published 流程与 disagreement 完全相同。每次新增批次必须换 source ID；旧 `negative_removed_gold_*` 保留并参与未来排重，绝不能为了重新抽样而删除宝贵的历史 Gold。默认配置关闭该候选任务，是为了避免每次 finetune 都无意中产生额外人工工作，而不是禁止复用历史 published 数据。

## 10. Finetune curate 和门控

### 10.1 按 Gold source_id 决定是否参与训练

HLMF 的 `hmlf_gold_merged` 始终认证 GoldSource 中全部 published Gold。每次 finetune 必须先让 HLML 生成一份逐批选择清单：

```bash
make prepare-finetune-gold-selection \
  HAND_FINETUNE_ID=<finetune-data-id> \
  GOLD_ENABLE_SOURCE_IDS=<source-id-a>,<source-id-b>
```

- 输出 `finetune/<id>/gold_selection.yaml`，列出每个领域下的每个 published 子批次；
- 命令列出的批次为 `enabled: true`，其余批次明确写成 `false`，没有隐式默认；
- 每项同时锁定 source kind、领域、descriptor 相对路径和 descriptor SHA256；
- GoldSource 后来新增、遗漏、改名或 descriptor 变化时，门控会失败，不会静默加入训练；
- disabled source 仍验证 descriptor、标签和聚合 SHA，但其行不进入训练、Gold 权重或 smoke；
- replay 不出现在选择清单中，代码强制它保持 `enabled: true`、`required: true`，且工作区中必须恰好有一个 replay source。

选择清单存在即拒绝重建。只有在所有计划 Gold 已发布、HLMF 聚合完成后生成；需要改变来源组合时使用新的 `HAND_FINETUNE_ID`。输出报告的 `source_selection_manifest`、`source_selection` 和 `disabled_source_rows` 会说明每个批次的最终决定。

### 10.2 从 GoldSource 到最终训练 JSONL 的完整聚合顺序

按下面的固定顺序操作，不能把“聚合”和“选择”颠倒：

1. HLMF 对每个已完成人工复核的 task 执行 import，得到各自 `published/finetune_source.json`；未标完的 task 不发布，也不参与训练。
2. HLMF 执行 `make finalize_train_finetune HAND_FINETUNE_ID=<id>`，扫描并认证 GoldSource 中全部 published descriptor，把去重后的全仓 Gold 写到 `finetune/<id>/hmlf_gold_merged/`。
3. HLML 已通过 `make prepare-finetune-sources` 在同一 `<id>` 下建立唯一 mandatory replay。
4. HLML 执行 `make prepare-finetune-gold-selection ... GOLD_ENABLE_SOURCE_IDS=...`，为聚合中每个 Gold 批次冻结 true/false 决定。
5. `make check-finetune-sources` 同时认证全仓 Gold 聚合、逐批选择清单和 replay；来源缺失、新增、descriptor 改动或 SHA 不一致都会失败。
6. `make finetune-curate` 只抽取 enabled Gold，再和 replay 合并；跨来源相同标签去重，Gold 与 replay 重合时保留 Gold。
7. `check-finetune-data` 和 `inspect-finetune` 通过后，生成的 `train_finetune_merged/<id>/05_labels/hand_training_labels_finetune.jsonl` 才是训练直接读取的冻结标签。

对应数据流：

```text
GoldSource 全部 published --HLMF认证--> hmlf_gold_merged --逐source启用/禁用--┐
pretrain curated + registry --确定性选择--> mandatory replay --------------┤
                                                                           v
                                                            finetune-curate 冻结训练集
```

### 10.3 Curate、检查和可视化

```bash
make finetune-curate HAND_FINETUNE_ID=<finetune-data-id> FINETUNE_PROFILE=<profile>
make check-finetune-data HAND_FINETUNE_ID=<finetune-data-id>
make inspect-finetune HAND_FINETUNE_ID=<finetune-data-id>
```

`finetune-curate` 自动认证 HLMF 聚合、合并 Gold 与 replay、执行身份去重，并把角色权重和 batch 中 Gold 比例解析进冻结报告。具体数值由当前 profile/config 决定，不属于通用流程。每个 role 内先跨轮去重，再按 source 规模分配；Gold 与 replay 重复时保留 Gold、删除 replay 重复行。

重点查看：

```text
train_finetune_merged/<id>/qc/curation_report.json
train_finetune_merged/<id>/05_labels/hand_training_labels_finetune.jsonl
train_finetune_merged/<id>/05_labels/hand_training_labels_finetune_smoke.jsonl
```

`check-finetune-data` 验证 Gold/replay 数量、role 覆盖、训练权重、结构 loss mask、图片/SHA 和 Val 隔离；`inspect-finetune` 生成抽样可视化。任何门控失败都应回到数据来源修正，不跳过。

Finetune smoke 对冻结的 256 ROI 使用固定的正/负均衡 overfit 抽样和较快的 smoke-only 学习率；这是为了让小集合同时检验 presence、landmarks、handedness，而不是复制正式训练的类别先验。正式 `sample_type_fractions_by_tier` 和学习率没有被改写，仍由 `check-finetune-data` 单独认证。门控会严格检查这两个 smoke-only 覆盖参数，不能在命令行随意改变。

### 10.4 区分“抽样占比”和“Loss 权重占比”

这两个概念互相独立，不能把其中一个当成另一个：

- `training.gold_fraction` 决定每个训练 batch 抽取多少 Gold ROI；它是**数据抽样占比**。
- `sampling_weight` 只决定同一个抽样格子内某行被抽到的概率；它也不是 Loss 权重。
- `FINETUNE_GOLD_LOSS_WEIGHT` 与 `FINETUNE_PSEUDO_LOSS_WEIGHT` 是 Gold/pseudo 的**真实 Loss 倍率**。默认都是 `1.0`，即不额外偏向任一监督层。

对某个训练样本和输出 head，最终 sample weight 为：

```text
监督层倍率 × supervision_loss_weight × 该 head 的 loss_weight × 该 head 的 quality_weight
```

`sampling_weight` 不进入这个乘式。训练器再用这些 sample weight 计算该 head 的加权平均 Loss；`landmarks`、`hand_flag`、`handedness` 各自的 `coefficient` 最后决定不同 head 之间的组合比例。因此，仅看 Gold/pseudo 目录中的图片数量，不能推出它们对梯度的实际影响。

查看当前设置和 epoch 0 的有效权重质量分数：

```bash
make check-finetune-data \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_EXPERIMENT_ID=<experiment-id> \
  FINETUNE_PROFILE=<profile> \
  FINETUNE_GOLD_LOSS_WEIGHT=<gold-multiplier> \
  FINETUNE_PSEUDO_LOSS_WEIGHT=<pseudo-multiplier>

python -m json.tool \
  /root/autodl-tmp/TrainFab/HLML-3.0/hand_landmarker_runs/<experiment-id>/finetune_data_gate.json
```

重点看 `sampling.loss_weighting`：

- `configured_supervision_tier_weights`：命令传入的真实 Loss 倍率；
- `nominal_tier_loss_mass_fraction`：只考虑 `gold_fraction` 和层倍率得到的直观比例；
- `epoch0_effective_head_weight_mass_fraction`：把实际抽样行的 record/head/quality 权重也算进去后，各 head 的 Gold/pseudo 权重质量分数；
- `epoch0_mean_per_batch_head_weight_fraction`：考虑训练器逐 batch 归一化后的等误差基准比例。

这些仍是“权重质量”而不是训练结束后的真实梯度占比，因为真实贡献还取决于每行预测误差。若 `gold_fraction=f`、Gold 倍率为 `g`、pseudo 倍率为 `p`，忽略行级权重时 Gold 的名义 Loss 质量分数为 `f*g / (f*g + (1-f)*p)`。

开启一个不同 Loss 比例的候选时，数据快照可以复用，但必须换新的 `FINETUNE_EXPERIMENT_ID`，并在 `check-finetune-data`、`finetune-smoke`、`check-finetune-smoke`、`finetune-train` 的每条命令中传入完全相同的两个倍率。smoke 会把倍率绑定进 resolved config；改变倍率后不能复用旧 smoke。

```bash
# 示例：Gold 每个有效样本的 Loss 倍率为 pseudo 的 2 倍；这不是 Gold 50% 抽样。
COMMON="HAND_FINETUNE_ID=<finetune-data-id> FINETUNE_EXPERIMENT_ID=<new-experiment-id> FINETUNE_PROFILE=data_only FINETUNE_GOLD_LOSS_WEIGHT=2.0 FINETUNE_PSEUDO_LOSS_WEIGHT=1.0"

make check-finetune-data $COMMON
make finetune-smoke $COMMON
make check-finetune-smoke $COMMON
make finetune-train $COMMON
```

两个倍率必须为有限正数；不能把 replay 倍率设为 `0` 来绕过 mandatory replay。倍率变化属于新的训练实验，不要求重新制作 Gold/replay 或重新 `finetune-curate`。

## 11. 三种 profile

### 11.1 data_only

只验证新 Gold 和来源重平衡，不增加结构 loss。

```bash
make finetune-smoke HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_EXPERIMENT_ID=<data-only-experiment-id> FINETUNE_PROFILE=data_only
make finetune-train HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_EXPERIMENT_ID=<data-only-experiment-id> FINETUNE_PROFILE=data_only
```

### 11.2 structure

在同一冻结数据上增加：

- `bone_vector_loss`：20 条 MediaPipe 真正骨连接的预测/Gold 向量 Huber。
- `spread_ratio_loss`：以 wrist 为基准的整体 spread，计算预测/Gold log-ratio Huber。

结构 mask 只在人工 Gold、presence=true、landmark loss 有效时非零。pseudo replay、negative、ignored 均为 0；没有固定骨长、固定关节方向或手势模板。

```bash
make finetune-smoke HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_EXPERIMENT_ID=<structure-experiment-id> FINETUNE_PROFILE=structure
make finetune-train HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_EXPERIMENT_ID=<structure-experiment-id> FINETUNE_PROFILE=structure
```

### 11.3 structure_roi_aug

只在 structure 确有改善、仍对 ROI 偏移敏感且时间允许时运行。它把增强调整为 rotation ±10°、scale 0.90～1.10、translation 0.05。

## 12. 评估、自动分析和选择

每个候选：

```bash
make eval-val-finetune FINETUNE_EXPERIMENT_ID=<candidate>
make infer-finetune FINETUNE_EXPERIMENT_ID=<candidate>
```

配对分析：

```bash
make analyze-finetune-errors \
  BASELINE_FINETUNE_ID=<baseline> \
  CANDIDATE_FINETUNE_ID=<candidate>

make compare-finetune-runs \
  BASELINE_FINETUNE_ID=<baseline> \
  CANDIDATE_FINETUNE_ID=<candidate>
```

输出包含 overall/per-dataset/per-landmark、PCK、presence、handedness、骨向量误差、spread、infer 检出/塌缩数、配对差值和最多 40 张 overlay。overlay 颜色：Gold 绿、baseline 红、candidate 蓝。

结果目录内的 `summary.json` 是总览，`per_roi_metrics.jsonl` 是逐 Val ROI 指标，`inference_paired_metrics.jsonl` 是逐 infer 图片的 Palm/Hand/塌缩和逐点变化，`paired_comparison.json` 保存完整配对明细，`overlays/` 只保留四类代表图且合计不超过 40 张。

先使用 Val 和固定 infer 决策；只对冻结 winner 运行：

```bash
make eval-test-finetune FINETUNE_EXPERIMENT_ID=<winner>
```

## 13. 导出与上板

```bash
make export-finetune FINETUNE_EXPERIMENT_ID=<winner>
make conversion-data-finetune FINETUNE_EXPERIMENT_ID=<winner>
```

导出器验证模型输入输出、checkpoint stage、BN folding、ONNX 数值一致性、允许算子、group 上限和体积。之后执行厂商工具链转换和固定手势上板回归。

## 14. 结果目录

```text
hand_landmarker_runs/<pretrain-id>/geometry/
hand_landmarker_runs/<pretrain-id>/multitask/
hand_landmarker_runs/<experiment-id>/finetune_smoke/
hand_landmarker_runs/<experiment-id>/finetune/
hand_landmarker_runs/<experiment-id>/eval/finetune/{val,test}/
hand_landmarker_runs/<experiment-id>/analysis/
hand_landmarker_inference/<experiment-id>/finetune/
hand_landmarker_exports/<experiment-id>/finetune/
```

每个训练 run 保存 resolved config、标签 SHA、Git commit、数据报告、checkpoint 和 history。不要手工覆盖已完成实验；使用新的 `FINETUNE_EXPERIMENT_ID`。

## 15. 常见失败

- `Checkpoint monitor ... was not produced`：配置 monitor 必须是日志实际键；3.0 smoke 使用 `total_loss`。
- Gold 点超出 ROI：如果无法可靠表达，HLMF 标 `ignore_for_training`；不要削弱门控让异常点进入训练。
- 重复 selection：检查 `selection_report.json` 的 occupied files/tokens；不要删除历史 Gold 规避。
- 数据路径缺失：确认 DatesetFab 目录和 README；不要复制到临时 train_sources 掩盖问题。
- profile 输出已存在：换 `FINETUNE_EXPERIMENT_ID`；不要 overwrite 正式 run。
- structure loss 无收益：停止，不运行 `structure_roi_aug`；以配对 Val/infer 证据为准。
