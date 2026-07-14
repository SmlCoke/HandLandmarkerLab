# 评估与人工可视化复核

## 1. Hand ROI Gold 评估

```bash
make eval-val
make eval-test
```

默认评估 Makefile 中 `HAND_PRETRAIN_PHASE := geometry` 对应的 checkpoint。multitask 完成后使用 `make eval-val HAND_PRETRAIN_PHASE=multitask` 和 `make eval-test HAND_PRETRAIN_PHASE=multitask`。

Val/Test 的 07B Gold 中，每行 `crop_path` 已经指向一张 `256×256` Hand ROI。评估器直接把这些 ROI 批量送入 Hand Landmarker，执行路径固定为：

```text
canonical Hand ROI → Hand Landmarker → 三个 head 的指标
```

评估过程不会读取原图、运行 Palm、重新裁 ROI 或做 ROI matching。报告只表示 Hand ROI 级性能，不包含 Palm detection、原图级 recall 或端到端级联性能。

### Presence

在所有未忽略 ROI 上输出 TP/FP/TN/FN、accuracy、precision、recall、F1、FPR 和 FNR。Val 可以扫 threshold；Test 配置关闭调阈值，必须使用同一 pretrain phase 在 Val 上冻结的值。geometry 与 multitask 的 threshold 必须独立选择、记录和冻结。

具体冻结流程：运行 `make eval-val HAND_PRETRAIN_PHASE=<phase>`，根据 Val 报告选择 threshold，把值写入 `configs/eval_test.yaml` 的 `evaluation.hand_flag_threshold`，并在实验记录中注明 phase，再运行相同 phase 的 Test。外部推理还应同步更新 `configs/infer.yaml` 的 `inference.hand_flag_threshold`。

### Landmarks

主指标覆盖所有 GT positive。即使模型 presence 判为 false，只要 landmark head 有输出，仍计算 21 点误差，避免幸存者偏差。

- 每 ROI 的 21 点平均像素误差及 mean/median/P90/P95；
- 每个 landmark ID 的平均误差；
- NME：除以 GT 21 点外接框对角线；
- PCK@0.05/0.10/0.15；
- landmark prediction coverage。

若某个 GT positive 没有可用 landmark prediction，landmark coverage 记为缺失，PCK 对应点按 failure 计数；不会伪造一个有限像素误差。

### Handedness

只在 GT positive 且 Left/Right 明确的样本上报告 overall accuracy、Left recall、Right recall 和 confusion matrix。

默认输出目录分别是 `${HAND_DATA_ROOT}/hand_landmarker_runs/<RUN_ID>/eval/<phase>/val` 与 `${HAND_DATA_ROOT}/hand_landmarker_runs/<RUN_ID>/eval/<phase>/test`，目录内包含 `metrics.json` 和逐 ROI `predictions.jsonl`。

评估默认不覆盖已有的这两个文件；重复运行前应换一个 `output.dir`，或在确认后显式设置 `output.overwrite: true`。

`predictions.jsonl` 每行保存：

- ROI 身份、canonical `crop_path` 与审计后解析路径；
- Gold/predicted presence、handedness 及两个 score；
- `landmarks_roi_norm`：固定 21 个 `[x,y]` 归一化预测点，不因 presence threshold 未通过而丢弃；
- 对 GT positive 的 `landmark_errors_px_by_id`（ID 0..20）、平均像素误差和 NME；negative 的误差字段为 `null`；
- `landmark_raw_max_abs`、`normalized_out_of_range_coordinate_count` 与 `board_landmark_scale_divisor`，用于识别输出范围异常以及板端 `max_abs>2` 时触发的整手 `/256` 兼容缩放；
- 本次 Gold JSONL 的 `labels_sha256`、实际 Hand 模型的 `hand_model_sha256`，以及从配置 `model.checkpoint_stage` 复制的 `model_checkpoint_stage`。

`metrics.json` 汇总上述指标和 `landmark_output_health`，并记录 Gold JSONL 路径/SHA-256、实际评估 backend、模型路径/SHA-256、`model_checkpoint_stage`、配置路径/SHA-256 及 canonical 数据审计结果。逐 ROI 与汇总 provenance 让不同训练路线可以按同一 Gold 和模型字节复核；这些哈希不表示 Palm 或端到端整图评估。

需要临时评估另一份同阶段 checkpoint 或输出到新目录时，不必修改 YAML：

```bash
python scripts/evaluate.py --config configs/eval_val.yaml \
  --model-path /path/to/checkpoint.weights.h5 \
  --output-dir /path/to/eval-output
```

确认可替换既有结果时再追加 `--overwrite`。CLI 覆盖不会改写配置；自定义模型路径与输出目录必须由操作者放在正确的 geometry/multitask 命名空间中。

## 2. 任意文件夹人工复核

编辑 `configs/infer.yaml` 的输入路径，然后运行：

```bash
make infer
```

该目标默认加载 geometry checkpoint。multitask 使用 `make infer HAND_PRETRAIN_PHASE=multitask`，输出按 phase 隔离在 `${HAND_DATA_ROOT}/hand_landmarker_inference/<RUN_ID>/<phase>`。

这是本系统中处理任意外部图片的独立入口，始终执行：

```text
任意外部图片 → Palm → rotated ROI → Hand
```

它会输出：

```text
<output>/rendered/**/<原文件名（含扩展名）>.annotated.png
<output>/predictions.jsonl
<output>/summary.json
```

叠加图包含 Palm bbox、旋转 ROI、Hand skeleton、Palm/presence/handedness 分数。板端本身不会按 `hand_flag` 门控 landmark 输出；PC 图中 threshold 仅控制是否绘制骨架，JSONL 始终保存三个 head 的原始结果。

`make infer` 的级联输出用于人工复核，不得写入或替代 `eval-val`/`eval-test` 的 Hand ROI 指标。

输入应是标注系统约定的 `1280×720 upright` 灰度/可转灰度图。若输入是板端传感器的 `720×1280` 竖屏原始方向，显式设置：

```yaml
input:
  source_orientation: sensor_portrait_rotate_clockwise
```

输出目录默认不覆盖已有结果；确认后设置 `output.overwrite: true`。

也可对单次运行覆盖模型和输出目录：

```bash
python scripts/infer_folder.py --config configs/infer.yaml \
  --model-path /path/to/checkpoint.weights.h5 \
  --output-dir /path/to/infer-output
```

确认可覆盖时追加 `--overwrite`。推理摘要会连同模型路径/SHA-256 记录 `model.checkpoint_stage`；该字段来自配置而不是路径猜测，因此自定义模型必须与所选阶段 wrapper 一致。
