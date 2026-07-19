# HLML 最后一轮自动伪标签训练计划

更新时间：2026-07-19。目标是在不新增人工标点的前提下，用部署同域无损 TIFF 和 MediaPipe 伪标签完成最后一次 pretrain + finetune。根因证据见 [当前训练状态](HLML_current_training_status.md)。

## 0. 立即停止的工作

- 不再跑 Gold=1.5/3/5；P0 已证明 Gold/pseudo 倍率不是主瓶颈。
- 不再调 structure Loss；它会展开骨架，但没有提高定位精度。
- 不把 H.264、I420、JPEG 或视频抽帧数据加入最后一轮。
- 不在提交前重写模型或扩大参数量。
- 不做新人工 Gold、不重新人工复核海量负样本。

建议 ID：

```text
新 pretrain ID:       v3-pretrain-final-autopseudo-r1
新 finetune 数据 ID:  v3-finetune-final-autopseudo-r1
finetune 实验 ID:     v3-finetune-final-autopseudo-dataonly-g20-r1
```

## 1. 人工只负责录制和快速 QC

### 1.1 必须满足

- 直接保存开发板/部署链路的 `1280×720` 灰度无损 TIFF。
- 一张图只放一只手；不要让一个 Hand ROI 同时含两只手。
- 每名队员使用不同的人、背景和录制时段。
- 不要用连续静止帧凑数量。建议每秒保留 3～5 张，或只有姿态/位置发生可见变化时才保存。
- 每批单独建源目录和 `dataset_id`，不要把几名队员的数据混成一个无法追踪的大目录。

### 1.2 录制优先级

1. 完全握拳、半握拳、握拳到张掌的慢过渡。
2. 数字 1、弯曲食指、拇指内收/外展。
3. 侧掌、手背/手心旋转、手腕旋转。
4. 手指互相遮挡、手接近 ROI 边缘。
5. 近/中/远尺度，偏左/偏右/偏上/偏下位置。
6. 亮/暗环境和不同背景，左右手尽量平衡。

每种失败姿态都录制一段缓慢连续变化。这样即使 MediaPipe 在最困难一帧弃权，相邻较容易角度仍可能提供监督。

### 1.3 留出纯自动 holdout

每名队员至少留出一个完整录制批次，约占其数据的 10%。该批同样跑 MediaPipe，但**不登记到 `configs/finalize_train.yaml`**。不能从同一段连续序列随机抽 10%，必须整段/整场留出。

holdout 只用于检查学生是否学到教师，不需要人工标点。提交时间不够时，至少保留目录，不能把所有图片都塞入 Train。

## 2. 每个训练批次分别运行 HLMF

服务器初始化：

```bash
cd /root/HandLandmarksFab
git pull --ff-only
conda activate anfab

export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_WORK_ROOT=/root/autodl-tmp/TrainFab/HLML-3.0
export HAND_PRETRAIN_ID=v3-pretrain-final-autopseudo-r1

make compile
make test
```

每批原始 TIFF 放在独立目录的 `images/` 下，例如：

```text
$HAND_DATASET_ROOT/PretrainSource/HandViolenceFinal0720/peak/fist_side_r01/images/
$HAND_DATASET_ROOT/PretrainSource/HandViolenceFinal0720/soar/rotation_scale_r01/images/
```

对每批分别运行：

```bash
export HLMF_SOURCE_ROOT=$HAND_DATASET_ROOT/PretrainSource/HandViolenceFinal0720/<member>/<batch>

make autolabel \
  HLMF_SOURCE_ROOT="$HLMF_SOURCE_ROOT" \
  AUTOLABEL_ROLE=train
```

`negative_candidate_threshold` 只影响 Palm 低分负样本候选，不会提高 MediaPipe landmark 教师质量。最后一轮以 geometry positive 为主，除非 Palm 统计明确要求，否则不要临时乱改阈值。

每批必须存在：

```text
$HLMF_SOURCE_ROOT/qc/image_validation_report.json
$HLMF_SOURCE_ROOT/qc/palm_detection_stats.json
$HLMF_SOURCE_ROOT/qc/mediapipe_roi_stats.json
$HLMF_SOURCE_ROOT/02_roi_crops/hand_roi_crops_manifest.jsonl
$HLMF_SOURCE_ROOT/02_roi_crops/hand_landmarks_autolabel_draft.jsonl
```

快速计数：

```bash
grep -c '"present": true' \
  "$HLMF_SOURCE_ROOT/02_roi_crops/hand_landmarks_autolabel_draft.jsonl"
grep -c '"present": false' \
  "$HLMF_SOURCE_ROOT/02_roi_crops/hand_landmarks_autolabel_draft.jsonl"
```

人工只随机查看少量 `02_roi_crops/images/` 和 downsample/可视化结果，确认：ROI 真的是单手、方向没有批量错误、关键点落在实际手上。无需逐张修点。

出现以下任一情况就重录该批，不把它交给 HLML：

- 大量 ROI 同时含两只手；
- MediaPipe 正样本率极低；
- 大量手被裁断或只剩局部手指；
- 输入不是部署同域无损 TIFF；
- 几万张图实际上只有少量静止姿态。

## 3. HLMF 聚合新的 pretrain

在 `/root/HandLandmarksFab/configs/finalize_train.yaml` 中为每个训练批次增加一条 source。每条必须使用不同 `dataset_id`；holdout 批次不要登记。

示例：

```yaml
- dataset_id: peak_final_0720_fist_side_r01
  contributor: Peak
  root: ${HAND_DATASET_ROOT}/PretrainSource/HandViolenceFinal0720/peak/fist_side_r01
  manifest: 02_roi_crops/hand_roi_crops_manifest.jsonl
  pseudo_labels: 02_roi_crops/hand_landmarks_autolabel_draft.jsonl
  crop_images_dir: 02_roi_crops/images
```

登记完成后：

```bash
cd /root/HandLandmarksFab
conda activate anfab

make finalize_train_pretrain
make build_pretrain_source_registry
```

检查：

```text
/root/autodl-tmp/TrainFab/HLML-3.0/train_pretrain_merged/
├── 05_labels/hand_training_labels_pretrain.jsonl
└── qc/
```

必须确认：

- 新的每个 `dataset_id` 都出现在 finalize 报告；
- positive 数量确实增加；
- `crop_path` 指向 `/root/autodl-tmp/DatesetFab/PretrainSource/...`；
- Val/Test 泄漏检查为 0；
- 没有为了追求“数十万”而让一个来源占据绝大多数有效正样本。

## 4. HLML curate：复用旧负样本，不做新人工复核

```bash
cd /root/HandLandmarkerLab
git pull --ff-only
conda activate hand-landmarker-tf29

export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_TRAIN_ROOT=/root/autodl-tmp/TrainFab/HLML-3.0
export HAND_PRETRAIN_ID=v3-pretrain-final-autopseudo-r1

make compile
make test-unit
make doctor
make pretrain-curate
```

`pretrain-curate` 已生成 geometry positive；它同时会建立本轮负样本候选。不要人工浏览数万张新负样本。只把 `v3-pretrain-r1` 中已经人工确认的 1,049 张背景图，与本轮 candidate 按相对路径和 SHA256 做交集并硬链接复用：

```bash
python - <<'PY'
from pathlib import Path
import hashlib, os

root = Path('/root/autodl-tmp/TrainFab/HLML-3.0/hand_landmarker_reviews')
old = root / 'v3-pretrain-r1' / 'negative_reviewed'
candidate = root / 'v3-pretrain-final-autopseudo-r1' / 'negative_candidates'
reviewed = root / 'v3-pretrain-final-autopseudo-r1' / 'negative_reviewed'

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

reused = 0
for old_file in old.rglob('*'):
    if not old_file.is_file():
        continue
    rel = old_file.relative_to(old)
    candidate_file = candidate / rel
    if not candidate_file.is_file():
        continue
    if sha256(old_file) != sha256(candidate_file):
        raise RuntimeError(f'SHA mismatch: {rel}')
    target = reviewed / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.link(candidate_file, target)
    reused += 1

print('reused reviewed negatives:', reused)
if reused < 500:
    raise SystemExit('Too few reviewed negatives were reusable; stop before multitask')
PY

make pretrain-curate-reviewed
```

这一步不会把新 teacher-abstain 样本冒充负例：新候选全部被排除，只有历史上人工确认过且 SHA 完全相同的背景进入 multitask。

## 5. Geometry：最后一轮的核心

训练前查看本轮真实 positive 数量：

```bash
make inspect-geometry
make inspect-geometry-smoke
```

数据小于约 120,000 个有效正 ROI 时，保持 `configs/train_geometry.yaml` 的 `sampling.epoch_size: null`。若远超 120,000，先把连续近重复数据在 HLMF source 层减量；不要靠盲目增大 epoch 消化所有相邻帧。

不改模型结构，按原链路运行：

```bash
make pretrain-geometry-smoke
make check-geometry-smoke
make pretrain-geometry
make eval-val-geometry
make infer-geometry
```

结果：

```text
$HAND_TRAIN_ROOT/hand_landmarker_runs/v3-pretrain-final-autopseudo-r1/
├── geometry/checkpoints/best.weights.h5
├── geometry/history.json
├── eval/geometry/val/metrics.json
└── ...
```

### 5.1 用未入 Train 的 MediaPipe holdout 检查蒸馏

把 `HOLDOUT_ROOT` 指向第 1.3 节保留并已运行 HLMF autolabel 的整批目录。以下检查不需要人工标签，也不写入数据集：

```bash
export HOLDOUT_ROOT=/root/autodl-tmp/DatesetFab/PretrainSource/HandViolenceFinal0720/<member>/<holdout-batch>

python - <<'PY'
from pathlib import Path
import json, os, random, statistics
import cv2
import numpy as np
from hand_landmarker.runtime import KerasHandPredictor

root = Path(os.environ['HOLDOUT_ROOT'])
labels = root / '02_roi_crops/hand_landmarks_autolabel_draft.jsonl'
rows = []
for line in labels.open(encoding='utf-8'):
    row = json.loads(line)
    if (row.get('hand_presence') or {}).get('present') and len(row.get('landmarks_crop_norm') or []) == 21:
        rows.append(row)

random.Random(20260719).shuffle(rows)
rows = rows[:min(2000, len(rows))]
if len(rows) < 200:
    raise SystemExit(f'too few positive holdout rows: {len(rows)}')

model = KerasHandPredictor(
    '/root/autodl-tmp/TrainFab/HLML-3.0/hand_landmarker_runs/'
    'v3-pretrain-final-autopseudo-r1/geometry/checkpoints/best.weights.h5',
    'v2', (2, 2, 3, 4, 4, 6, 6),
)
errors = []
for start in range(0, len(rows), 64):
    batch = rows[start:start + 64]
    images = []
    for row in batch:
        path = Path(row['crop_path'])
        if not path.is_absolute():
            path = root / path
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f'unreadable crop: {path}')
        images.append(image)
    predictions = model.predict(images, batch_size=64)
    for row, prediction in zip(batch, predictions):
        target = np.array([
            [point['x'], point['y']]
            for point in sorted(row['landmarks_crop_norm'], key=lambda point: point['id'])
        ])
        output = np.array(prediction.landmarks_norm)
        errors.append(float(np.linalg.norm(output - target, axis=1).mean() * 255.0))

print('holdout positives:', len(errors))
print('student vs MediaPipe mean px:', statistics.mean(errors))
print('student vs MediaPipe median px:', statistics.median(errors))
print('student vs MediaPipe P90 px:', float(np.percentile(errors, 90)))
PY
```

- mean ≤ 8 px：学生已经较好复现同域教师，可继续。
- mean 8～12 px：可继续，但必须结合 Gold Val 决定。
- mean > 15 px：学生连未见过的教师正标签都没有学好，停止 multitask/finetune；增加阶段不会自动解决。

### Geometry 继续门槛

旧 geometry 基线：mean 20.73、P90 37.32、PCK@0.10 0.334。

- **推荐继续**：mean ≤ 18 px，且 P90 ≤ 34 px，固定 infer 中的握拳/侧掌明显减少塌缩。
- **强结果**：mean ≤ 15 px，且 P90 ≤ 30 px。
- **停止下游**：mean ≥ 20 px，或 P90 不降，或只降低训练 Loss 而 Val 不变。此时再跑 multitask/finetune 不会把根因自动修好。

MediaPipe 在自己能检出的 Val 子集上是 4.38 px。学生不可能仅凭“更多图片”就被保证达到该值；以上门槛用于确认最后一轮至少产生了真实、可交付的提升。

## 6. Multitask：只做 head 校准，不把它当第二次大预训练

旧 multitask 在第 1 epoch 即得到 best，mean 从 20.73 退化到 22.01。新数据主要应在 geometry 中发挥作用；multitask 的目标是补 presence/handedness，同时尽量保住 landmarks。

确认复用负样本门控后运行：

```bash
make check-multitask-data
make inspect-multitask
make pretrain-multitask
make eval-val-multitask
make infer-multitask
```

`sampling.epoch_size_upper_bound=6400` 在这里是有意的：只有约 1,000 张已人工确认负样本，不能为了“吃完所有新正样本”把少量负样本重复几十次。大规模正样本已经由 geometry 使用。

若 multitask 相比新 geometry 的 mean 恶化超过 0.5 px，或 P90 恶化超过 2 px，明确记录 geometry 为 landmark winner；仍可让 multitask 作为后续 finetune 初始化候选，但不能默认“阶段更晚就更好”。

## 7. 自动重建 replay，并只跑一轮 finetune

不新增 Gold；复用已有 published Gold。先生成新的 replay：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29

export HAND_PRETRAIN_ID=v3-pretrain-final-autopseudo-r1
export HAND_FINETUNE_ID=v3-finetune-final-autopseudo-r1

make prepare-finetune-sources
```

HLMF 为新 finetune 数据 ID 认证现有 Gold：

```bash
cd /root/HandLandmarksFab
conda activate anfab
export HAND_FINETUNE_ID=v3-finetune-final-autopseudo-r1
make finalize_train_finetune HAND_FINETUNE_ID=$HAND_FINETUNE_ID
```

回到 HLML，只启用现有无损/有效批次：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29

export HAND_PRETRAIN_ID=v3-pretrain-final-autopseudo-r1
export HAND_FINETUNE_ID=v3-finetune-final-autopseudo-r1

make prepare-finetune-gold-selection \
  HAND_FINETUNE_ID=$HAND_FINETUNE_ID \
  GOLD_ENABLE_SOURCE_IDS=disagreement_gold_hlml2.0,negative_removed_gold_hlml2.0,dragon_gold_0718_v1,new_recorded_gold_0718_r01,new_recorded_gold_0718_r02

make finetune-curate \
  HAND_FINETUNE_ID=$HAND_FINETUNE_ID \
  FINETUNE_PROFILE=data_only

make check-finetune-data \
  HAND_FINETUNE_ID=$HAND_FINETUNE_ID \
  FINETUNE_EXPERIMENT_ID=v3-finetune-final-autopseudo-dataonly-g20-r1 \
  FINETUNE_PROFILE=data_only \
  FINETUNE_GOLD_LOSS_WEIGHT=2.0 \
  FINETUNE_PSEUDO_LOSS_WEIGHT=1.0
```

只跑 data-only、Gold/pseudo Loss=2:1。P0 中这是 finetune mean 最好的点；不要再扫比例和 structure：

```bash
make finetune-smoke \
  HAND_FINETUNE_ID=$HAND_FINETUNE_ID \
  FINETUNE_EXPERIMENT_ID=v3-finetune-final-autopseudo-dataonly-g20-r1 \
  FINETUNE_PROFILE=data_only \
  FINETUNE_GOLD_LOSS_WEIGHT=2.0 \
  FINETUNE_PSEUDO_LOSS_WEIGHT=1.0

make finetune-train \
  HAND_FINETUNE_ID=$HAND_FINETUNE_ID \
  FINETUNE_EXPERIMENT_ID=v3-finetune-final-autopseudo-dataonly-g20-r1 \
  FINETUNE_PROFILE=data_only \
  FINETUNE_GOLD_LOSS_WEIGHT=2.0 \
  FINETUNE_PSEUDO_LOSS_WEIGHT=1.0

make eval-val-finetune \
  HAND_FINETUNE_ID=$HAND_FINETUNE_ID \
  FINETUNE_EXPERIMENT_ID=v3-finetune-final-autopseudo-dataonly-g20-r1

make infer-finetune \
  HAND_FINETUNE_ID=$HAND_FINETUNE_ID \
  FINETUNE_EXPERIMENT_ID=v3-finetune-final-autopseudo-dataonly-g20-r1
```

## 8. 最终选型规则

比较新 geometry、新 multitask、新 finetune，不按阶段名称决定 winner。

1. 先看 mean、P90、PCK@0.10。
2. 再看固定 infer 的握拳、侧掌、数字 1 和遮挡姿态。
3. 同等 landmark 精度下，选择 presence/handedness 更好的模型。
4. 若 finetune mean/P90 明显差于 multitask，就交付 multitask；若 multitask 也明显差于 geometry，则保留 geometry 作为关键点精度证明，同时评估 multitask/finetune 是否因 presence 能力更适合整机演示。
5. 只对 winner 运行 Test、export、conversion 和上板。

```bash
export WINNER_ID=<winner-experiment-id>
export WINNER_PHASE=<geometry-or-multitask-or-finetune>
```

若 winner 为 finetune：

```bash
make eval-test-finetune FINETUNE_EXPERIMENT_ID="$WINNER_ID"
make export-finetune FINETUNE_EXPERIMENT_ID="$WINNER_ID"
make conversion-data-finetune FINETUNE_EXPERIMENT_ID="$WINNER_ID"
```

若 winner 为 pretrain，分别使用对应的 `eval-test-geometry/export-geometry` 或 `eval-test-multitask/export-multitask`。

## 9. 人与程序分工

人工只做：

- 录制多样、无损、单手 TIFF；
- 为每批建立目录并在 `finalize_train.yaml` 登记唯一 ID；
- 随机浏览少量 ROI/教师 overlay；
- 启动命令，按门槛决定继续或停止；
- 查看固定 infer、完成厂商转换、上板和交付材料。

程序自动做：

- Palm 检测、ROI、MediaPipe 伪标签、质量分层和聚合；
- SHA、路径、重复、Val/Test 泄漏检查；
- 复用历史已确认负样本并排除全部新未审核 negatives；
- 在未进入 Train 的自动 holdout 上计算学生相对 MediaPipe 的误差；
- smoke、geometry、multitask、finetune、best checkpoint；
- Val/Test 指标、infer、ONNX parity 和算子审计。

最后一轮不需要人工标 21 点，也不需要人工复核新的负样本。
