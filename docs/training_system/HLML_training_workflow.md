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

`DatesetFab` 是可再生数据仓库，包含来源级 manifest、伪标签和 ROI。HLMF/HLML 的聚合标签把 `crop_path` 直接指向该仓库；不再建立 `HLML-3.0/train_sources` 副本。

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
HLMF sources/gold/* ──finalize──> hmlf_gold_merged ─┐
                                                     ├─ finetune-curate ─> 冻结训练 JSONL
pretrain registry ──prepare-finetune-sources──> replay┘
                              └───────────────> disagreement score pool
```

### 8.3 在 HLMF 建立或继承 Gold

```bash
cd /root/HandLandmarksFab
conda activate anfab

# 每批 Dragon 单独运行一次，批次 ID 不得复用
make prepare_dragon_gold \
  HAND_FINETUNE_ID=<finetune-data-id> \
  DRAGON_SOURCE_ROOT=$HAND_DATASET_ROOT/<dragon-batch-root> \
  DRAGON_BATCH_ID=<unique-dragon-batch-id>

# 所有 Gold source 发布/导入后重新聚合
make finalize_train_finetune HAND_FINETUNE_ID=<finetune-data-id>
```

`sources/gold/<source-id>/finetune_source.json` 是单个来源的认证描述符；`hmlf_gold_merged/qc/finalize_finetune_report.json` 是所有 Gold 的聚合报告。只有聚合报告通过，HLML 才会接受数据。

需要从同一数据契约的上一 finetune 快照继承认证 Gold 时：

```bash
make seed_finetune_gold \
  BASE_FINETUNE_ID=<old-data-id> \
  HAND_FINETUNE_ID=<new-data-id>
```

目标存在即失败；程序硬链接并验证全部历史 source。随后只能用新的 source/round ID 增加 Gold，不删除旧 disagreement、new-recorded 或 reviewed Gold。

### 8.4 在 HLML 建立 replay 和 disagreement score pool

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
make prepare-finetune-sources HAND_FINETUNE_ID=<finetune-data-id>
```

程序自动完成：

1. 认证 HLMF 的 pretrain source registry、curated multitask 标签和 checkpoint；
2. 从广泛来源中确定性选择 replay，保存其来源、标签和 SHA；
3. 用当前 student checkpoint 对可选 positive 推理；
4. 把 student 预测与 MediaPipe teacher 伪标签比较，生成逐 ROI disagreement 分数池。

这里的 **student** 是正在训练的自研 Hand Landmarker，**teacher** 是 MediaPipe 伪标签。分歧大只表示两者对同一 ROI 的关键点差别大，因此适合人工复核；不代表 teacher 必然正确。查看：

```text
finetune/<id>/sources/replay/
finetune/<id>/mining/teacher_student/disagreement_scores.jsonl
finetune/<id>/mining/disagreement_gold/selection_report.json
finetune/<id>/mining/prepare_finetune_sources_report.json
```

这些产物绑定 checkpoint 和输入 SHA。输入或 checkpoint 变化时应使用新的 finetune 数据 ID 重新建立，不覆盖已有快照。

## 9. ROI 多轮 Gold

### 9.1 为什么要分 round/source

一个 `round-id` 表示一次冻结选样，一个 `source-id` 表示一份不可变 CVAT/Gold 来源。以后想从同一原始来源继续取样时，创建新 round/source ID；程序会保留并排除历史 ROI，因此旧人工标注不会浪费，也不需要删除。

### 9.2 先制作新录制任务

1. 人工录制当前模型薄弱的姿态，并在 HLMF 跑完 00～03；
2. HLMF 使用 `native_existing` 和本轮计划给出的显式限额确定性抽样；
3. 任务生成后，查看 `task_descriptor.json` 得到实际任务数；
4. 不要人工从原始目录随意挑图，也不要在任务冻结后改限额。

HLMF 命令：

```bash
make export_finetune_gold \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_SOURCE_ID=new_recorded_gold_<round-id> \
  FINETUNE_SOURCE_MODE=native_existing \
  FINETUNE_RAW_SOURCE_ROOT=$HAND_DATASET_ROOT/<source-id> \
  FINETUNE_MAX_ITEMS=<new-recorded-limit>
```

### 9.3 冻结本轮 disagreement selection

新录制任务存在后，在 HLML 执行：

```bash
make prepare-finetune-round \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_ROUND_ID=<round-id> \
  FINETUNE_GOLD_BUDGET=<round-budget> \
  NEW_RECORDED_SOURCE_ID=new_recorded_gold_<round-id>
```

`FINETUNE_GOLD_BUDGET` 是本轮两个 CVAT task 的合计上限，必须由执行计划显式指定。程序读取：

- 所有已发布 Gold source；
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

HLMF 为 new-recorded 和 disagreement 分别生成 `cvat/<source-id>/qc/cvat_job_plan.json`。人工按计划在 CVAT 完成完整 21 点和 handedness，或明确标 `no_hand` / `ignore_for_training`；返回的完整 XML 放入各自 `cvat/<source-id>/reviewed.xml`。

```bash
make import_finetune_gold HAND_FINETUNE_ID=<finetune-data-id>
make finalize_train_finetune HAND_FINETUNE_ID=<finetune-data-id>
```

每次增加一轮 Gold 后都重新 finalize；最终聚合会保留所有历史来源并跨轮去重。然后返回 HLML。

## 10. Finetune curate 和门控

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
