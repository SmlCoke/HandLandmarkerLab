# HLML 当前下一步优化计划

更新时间：2026-07-18。可用时间约两天，人工最多标 800 个 Hand ROI，可多人分工。本文只记录本轮任务；通用操作见 [完整训练流程](HLML_training_workflow.md)。

## 1. 当前事实与本轮决策

- 数据根：`/root/autodl-tmp/DatesetFab`；工作根：`/root/autodl-tmp/TrainFab/HLML-3.0`。
- pretrain ID：`v3-pretrain-r1`；finetune 数据 ID：`v3-finetune-r1`。
- geometry smoke 曾跑满并生成 best checkpoint。此前 `pretrain-geometry` 并未真正开始，而是在它的前置 smoke gate 中因新数据结构兼容错误退出；本次服务器部署会保留旧 smoke 备份，并在新 commit 下重跑 smoke/gate。人工只需启动正式 geometry。
- 现有 Gold 是 `dragon_gold_0716_v1`：5,191 ROI，5,189 可训练、2 ignored。
- 本轮新增人工 Gold 优先 800；若参与人数不足，在任何 disagreement task 生成前统一冻结为 600。800 方案中 new-recorded 最多 300、disagreement 补余量；600 方案中 new-recorded 最多 200、disagreement 补余量。
- 必做候选：`data_only` 与 `structure`。只有 structure 明确减少塌缩且 Val 不退化时，才做 `structure_roi_aug`。

## 2. 先恢复 geometry 正式训练

服务器更新后：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
git pull --ff-only
make doctor
make check-geometry-smoke
```

部署完成时标准 `smoke/` 已在当前 commit 下重新生成并通过 128 ROI 门控；这里再运行 `check-geometry-smoke` 只是快速复验，不会重新训练。随后人工在 `screen`/`tmux` 中启动：

```bash
make pretrain-geometry
```

观察：

```text
/root/autodl-tmp/TrainFab/HLML-3.0/hand_landmarker_runs/v3-pretrain-r1/geometry/training_report.json
.../geometry/history.json
.../geometry/checkpoints/best.weights.h5
```

训练完成后：

```bash
make eval-val-geometry
make infer-geometry
```

查看：

```text
.../hand_landmarker_runs/v3-pretrain-r1/eval/geometry/val/metrics.json
.../hand_landmarker_inference/v3-pretrain-r1/geometry/
```

人工只需确认 loss 正常下降、无 NaN、Val/infer 不再“漫天飞舞”；不要为了追求几张样例提前改训练结构。

## 3. geometry 训练期间并行完成两项人工工作

### 3.1 复核 true negative

候选目录：

```text
/root/autodl-tmp/TrainFab/HLML-3.0/hand_landmarker_reviews/v3-pretrain-r1/negative_candidates/
```

完整复制为同级 `negative_reviewed/`，逐图删除所有含手、疑似手、手指、手腕、模糊或无法确认的图片，只保留明确背景。可 7z/zip 往返传输；不编辑、不改名、不移动保留图片。

```bash
make pretrain-curate-reviewed
make check-multitask-data
make inspect-multitask
```

报告必须证明保留图片 SHA 与候选一致，并存在足够的人工确认 negative；不要绕过门控。

### 3.2 录制 source e

录制 20～40 分钟，分短 session，覆盖握拳、数字 1、侧向张掌、从张到握/从握到张、遮挡或两手靠近、远近变化、画面边缘、左右手、不同参与者/背景/光线。横屏 `1280×720`，避免长时间静止。

原视频与稀疏 TIFF：

```text
/root/autodl-tmp/DatesetFab/finetune_source_e_r01/raw_videos/
/root/autodl-tmp/DatesetFab/finetune_source_e_r01/images/
```

详细抽帧、00～03 和 CVAT 命令见 HLMF 的 `docs/annotating_system/HLMF_next_step_plan.md`。这部分可由未参与 negative 复核的人并行完成。

## 4. Multitask

geometry best 和 true negative 都就绪后：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
make check-multitask-data
make inspect-multitask
make pretrain-multitask
```

程序从 geometry best 初始化，同时训练 landmarks、hand_flag 和 handedness，并按 `val_multitask_score` 保存 best。训练完成：

```bash
make eval-val-multitask
make infer-multitask
make export-multitask
```

查看：

```text
.../hand_landmarker_runs/v3-pretrain-r1/multitask/training_report.json
.../hand_landmarker_runs/v3-pretrain-r1/eval/multitask/val/metrics.json
.../hand_landmarker_inference/v3-pretrain-r1/multitask/
.../hand_landmarker_exports/v3-pretrain-r1/multitask/
```

人工抽看张掌、握拳、数字 1、侧掌与背景拒绝。multitask 的作用是提供可部署 pretrain 模型和后续 replay；它不替代 Gold finetune。

## 5. 自动建立 replay 与 disagreement 分数池

```bash
make prepare-finetune-sources \
  HAND_PRETRAIN_ID=v3-pretrain-r1 \
  HAND_FINETUNE_ID=v3-finetune-r1
```

程序自动认证 curated pretrain、geometry checkpoint 和 source registry，生成最多配置量的 replay，并对 positive 计算 student–teacher 分歧。查看：

```text
.../finetune/v3-finetune-r1/sources/replay/pretrain_replay/finetune_source.json
.../finetune/v3-finetune-r1/mining/teacher_student/disagreement_scores.jsonl
.../finetune/v3-finetune-r1/mining/prepare_finetune_sources_report.json
```

如果这些目录已存在且报告绑定旧 checkpoint，不覆盖；使用新的 `HAND_FINETUNE_ID` 重建。若它们尚不存在，就保持 `v3-finetune-r1`。

## 6. 冻结本轮 600/800 Gold

先由 HLMF 导出并冻结 `new_recorded_gold_r01`。其实际数量记录在：

```text
.../finetune/v3-finetune-r1/cvat/new_recorded_gold_r01/task_descriptor.json
```

800 方案：

```bash
make prepare-finetune-round \
  HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_ROUND_ID=r01 \
  FINETUNE_GOLD_BUDGET=800 \
  NEW_RECORDED_SOURCE_ID=new_recorded_gold_r01
```

人手不足时在本命令第一次执行前改用 `FINETUNE_GOLD_BUDGET=600`。同一 `r01` 生成后不可改变预算；需要重选必须换新 round ID。程序会用 disagreement 补足余量并排除 Dragon、历史 Gold/CVAT/request、Val/Test 和像素重复项。

查看：

```text
.../finetune/v3-finetune-r1/mining/rounds/r01/disagreement_gold_r01/selection_report.json
.../finetune/v3-finetune-r1/mining/rounds/r01/disagreement_gold_r01/selection_request.jsonl
```

然后 HLMF 导出 `disagreement_gold_r01`。人工对两个 task 分工标注，完整做 21 点、handedness、`no_hand` 或 `ignore_for_training`；HLMF 导入并 `finalize_train_finetune`。人工不用手工合并 JSONL。

## 7. Curate、门控和 smoke

HLMF 聚合通过后：

```bash
cd /root/HandLandmarkerLab
make finetune-curate HAND_FINETUNE_ID=v3-finetune-r1 FINETUNE_PROFILE=data_only
make check-finetune-data HAND_FINETUNE_ID=v3-finetune-r1
make inspect-finetune HAND_FINETUNE_ID=v3-finetune-r1
```

程序自动让 Gold 覆盖重复 replay、按 role 重平衡并生成冻结标签。检查：

```text
.../train_finetune_merged/v3-finetune-r1/qc/curation_report.json
.../train_finetune_merged/v3-finetune-r1/05_labels/hand_training_labels_finetune.jsonl
```

先运行两个 profile 的 smoke；两者都必须通过再正式训练：

```bash
make finetune-smoke HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-r1a FINETUNE_PROFILE=data_only
make check-finetune-smoke HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-r1a FINETUNE_PROFILE=data_only

make finetune-smoke HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-r1b FINETUNE_PROFILE=structure
make check-finetune-smoke HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-r1b FINETUNE_PROFILE=structure
```

## 8. 两个正式候选与自动比较

候选 A 只验证数据改进：

```bash
make finetune-train HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-r1a FINETUNE_PROFILE=data_only
make eval-val-finetune FINETUNE_EXPERIMENT_ID=v3-finetune-r1a
make infer-finetune FINETUNE_EXPERIMENT_ID=v3-finetune-r1a
```

候选 B 在完全相同数据上增加 Gold-only 骨向量与整体 spread 结构约束：

```bash
make finetune-train HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_EXPERIMENT_ID=v3-finetune-r1b FINETUNE_PROFILE=structure
make eval-val-finetune FINETUNE_EXPERIMENT_ID=v3-finetune-r1b
make infer-finetune FINETUNE_EXPERIMENT_ID=v3-finetune-r1b
```

```bash
make analyze-finetune-errors \
  BASELINE_FINETUNE_ID=v3-finetune-r1a \
  CANDIDATE_FINETUNE_ID=v3-finetune-r1b
make compare-finetune-runs \
  BASELINE_FINETUNE_ID=v3-finetune-r1a \
  CANDIDATE_FINETUNE_ID=v3-finetune-r1b
```

程序生成 overall/per-dataset/per-landmark、PCK、presence、handedness、spread、塌缩统计和最多 40 张代表 overlay。人工只查看摘要、overlay 和固定 infer 手势，不制作额外表格。

选择优先级：塌缩明显减少；握拳/数字 1/侧掌改善；Val 误差和 PCK 不明显退化；presence 不恶化。只有 B 达到这些条件且仍对 ROI 偏移敏感，才用新的实验 ID 运行可选 `structure_roi_aug`。

## 9. 冻结 winner

只对 winner 运行：

```bash
make eval-test-finetune FINETUNE_EXPERIMENT_ID=<winner>
make export-finetune FINETUNE_EXPERIMENT_ID=<winner>
make conversion-data-finetune FINETUNE_EXPERIMENT_ID=<winner>
```

然后完成官方工具链转换和板端固定手势回归。locked Test 不用于反复调参。
