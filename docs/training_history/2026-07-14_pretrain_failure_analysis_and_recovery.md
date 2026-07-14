# 两次 pretrain 失败分析与恢复方案（2026-07-14）

本文只讨论 pseudo-label pretrain。没有使用、要求或修改 finetune / Gold Train 路线。

## 结论

两次结果差并非单一原因：

1. 第一轮的 landmark loss 量级过小，`val_total_loss` 又主要由分类头决定，epoch 1 被错误地当成关键点最佳权重。
2. 第二轮虽然提高了 landmark 权重，却触发了确定性的 checkpoint 方向 bug：`val_landmark_mae` 被自定义 callback 按 `max` 保存，`best.weights.h5` 实际是最差 MAE，而推理正是加载了它。
3. 第二轮由 Keras EarlyStopping 恢复出的 `final.weights.h5` 明显好于错误的 `best`，但 Test 仍只有 32.41 px，说明修正 checkpoint 不能单独解决学习质量。
4. 原 pretrain 每个 batch 强制使用 30% teacher-negative candidate。服务器数据的几何交叉检查证明，至少 48.2% 的这些 candidate ROI 明确覆盖了同一帧中已经确认的手。它们是假负例或高度歧义样本，却持续用 `hand_flag=0` 更新共享 backbone。
5. MediaPipe 检出的 positive 伪标签大多合理；坐标顺序、NCHW/NHWC、sample-weight 顺序和同步仿射代码未发现错位。当前优先级应是 checkpoint、数据语义和可过拟合性门禁，而不是用 sigmoid 掩盖散点输出。

## 服务器实测证据

### 第一轮 `hand_landmarker_runs/v1`

| 权重 | Test mean pixel error | PCK@0.10 | 越界手 |
|---|---:|---:|---:|
| `best`（epoch 1） | 103.79 px | 0.0095 | 774 / 985 |
| `last`（epoch 13） | 40.47 px | 0.0825 | 0 / 985 |

第一轮 epoch 1 的 Val 总损失为：

```text
0.002442 landmark + 0.226654 hand_flag + 0.1 * 0.77937 handedness
= 0.307033
```

landmark 只贡献约 0.8%，所以该 `best` 不是关键点最佳。

### 第二轮 `hand_landmarker_runs/v1-landmarks-enhanced`

服务器 `best.weights.h5.state.json`：

```json
{
  "monitor": "val_landmark_mae",
  "mode": "max",
  "completed_epoch": 3,
  "value": 0.11780638992786407
}
```

实际 history 的最小值是：

```text
epoch 32: val_landmark_mae = 0.0838252157
```

所以第二轮默认 `best` 是错误 checkpoint。三类权重 SHA-256 也不同：

```text
best  27f45e785b6e53df5900856fdc50b40c6c89c505ca709ff80106c8c09a36f722
last  79d780e24d534070ae5dffa36f45ab88e9d858d5331d2fdd413e4a2cda893c6d
final 5e512b3b60dfbf179793cafd71f08a93529e60dd03a17a72a78d341a2f738b2b
```

重新用现有代码直接评估 `final`（EarlyStopping 恢复的 epoch 32）后：

| 权重 | Test mean pixel error | PCK@0.10 | Mean NME | 越界手 |
|---|---:|---:|---:|---:|
| 错误 `best` | 47.16 px | 0.0470 | 0.4367 | 1 / 985 |
| 恢复后的 `final` | 32.41 px | 0.1288 | 0.3040 | 0 / 985 |

用 Train positive 的逐点平均坐标构造一个完全不看图像的 constant-pose baseline，在同一 Test 上也能得到约 **36.83 px**。`final` 仅比它好约 4.42 px，这与推理图中“关键点收成一个偏小的平均拳形”一致：模型学到了一点条件信息，但大部分仍是均值回归，并没有充分学会图像到姿态的映射。

`final` 已不再是完全随机散点，但仍远未达到可部署标准。也就是说：用户看到的第二轮“漫天飞”首先是选错权重；真正的训练结果仍然差，但程度没有 `best` 展示得那么严重。

### teacher-negative 污染

源 canonical JSONL：

```text
总记录                        25,831
有 21 点的 positive           17,721
teacher-negative candidate      8,110
```

对每个 negative ROI，使用同一 `source_group_id` 下 positive 的掌部核心点 `0,1,5,9,13,17` 与 ROI 四边形做包含测试：

```text
同帧存在 positive                           6,738 / 8,110
negative ROI 覆盖至少 1 个已确认掌部点       4,255 / 8,110
negative ROI 覆盖至少 3 个已确认掌部点       3,908 / 8,110 = 48.2%
negative ROI 覆盖全部 6 个已确认掌部点       3,543 / 8,110
```

`3,908` 是能自动证明的下界；没有同帧 positive 的 candidate 仍可能是 teacher 全帧漏检，不能据此认定为真负样本。这 3,908 条冲突样本按来源分为 **Peak 2,149 / Soar 1,759**，按候选类型分为 **NEG_LOW_PALM 2,964 / NEG_RUNTIME 944**。因此 Peak 的漏检确实严重，但并非只发生于 Peak，也并非删除一个来源就能解决。

旧采样器在每个 64 batch 中固定抽：

```text
36 POS_RUNTIME
 9 POS_LOW_PALM
16 NEG_RUNTIME_CANDIDATE
 3 NEG_LOW_PALM_CANDIDATE
```

因此一个典型 batch 约有 9 个可以直接证明与手重叠的错误 negative。它们的 landmark weight 确实是 0，不会直接把 42 个坐标拉到零；但 `hand_flag=0` 的 BCE 会穿过 landmarks 共用的 backbone，而且 hard pose 不但得不到关键点监督，反而被教成“无手”。

### positive 标签与域差异

positive 中：

```text
HIGH quality                    17,391
MEDIUM（主要是 handedness 低置信） 330
含越界 landmark 的记录              147
非有限坐标                              0
```

抽查可视化支持“positive 的 21 点大部分正确”。现有代码审计也排除了以下全局错位：

- label/model/runtime 都使用 `x0,y0,...,x20,y20`；
- 输入均为 `(B,1,256,256)`，模型的 `Permute((2,3,1))` 正确转 NHWC；
- landmark/presence/handedness sample weight 顺序一致；
- 图像和点使用同一 OpenCV 仿射矩阵；
- negative 的零 landmark target 被 landmark weight 0 掩码。

但 Train/Val/Test 仍存在明显域差异：

| 数据 | 平均灰度 | 黑色像素比例 | 黑色边界比例 |
|---|---:|---:|---:|
| Train positive | 122.15 | 4.47% | 7.04% |
| Val | 144.78 | 0.48% | 1.24% |
| Test | 148.61 | 0.03% | 0.10% |

这会在 checkpoint 与负例问题修好后继续限制泛化，需要靠更多 session、姿态和一致 ROI 分布的可靠 positive 解决。

## 已实施的本地修复

### 1. checkpoint 方向和训练后自证

`hand_landmarker/training.py` 现在：

- `loss/MAE/MSE/RMSE/error/NME/distance` 默认 `min`；
- `accuracy/AUC/precision/recall/F1/PCK/IoU` 默认 `max`；
- 无法判断的指标必须显式配置 mode，否则拒绝训练；
- checkpoint、ReduceLROnPlateau、EarlyStopping 分别使用明确的 monitor/mode；
- 相同 monitor 下强制 LR patience 小于 early-stopping patience；
- 训练结束验证 `best.state.json` 的 mode、epoch 和 value 等于 history 中相应的最优值，否则训练报告不能标记为 complete。

默认配置为：

```yaml
checkpoint:             {monitor: val_landmark_mae, mode: min}
learning_rate_schedule: {monitor: val_landmark_mae, mode: min, patience: 5}
early_stopping:          {monitor: val_landmark_mae, mode: min, patience: 20}
```

第二轮的 LR patience 和 early patience 都是 20；第一次降低 LR 的同一轮就停止，低 LR 没有学习窗口。新配置消除了该问题。

### 2. 统一训练/评估/推理/导出路径

所有 pretrain 路由共同使用：

```text
${HAND_PRETRAIN_RUN_ID:-v1-pretrain-geometry}
```

不再出现训练写入 `v1-landmarks-enhanced`，而默认 eval/infer 仍读取旧 `v1/pretrain` 的情况。每次仍须在 `metrics.json` / `summary.json` 核对模型绝对路径和 SHA-256。

### 3. 持久化 pretrain 提纯

新增：

```text
configs/curate_pretrain.yaml
scripts/curate_pretrain.py
hand_landmarker/pretrain_curation.py
make curate-pretrain
```

默认输出：

```text
${HAND_DATA_ROOT}/train_pretrain_curated/${HAND_PRETRAIN_CURATED_ID:-v1}/
├── 05_labels/
│   ├── hand_training_labels_pretrain_landmarks.jsonl
│   ├── hand_training_labels_pretrain_multitask.jsonl
│   └── hand_training_labels_pretrain_smoke.jsonl
├── images/<dataset_id>/...             # 独立 copy，保留实际训练像素
├── audit/
│   ├── pretrain_curation_catalog.jsonl # 源文件每一行都有决策
│   ├── included_landmarks.jsonl
│   ├── excluded_and_held.jsonl
│   ├── negative_review_queue.jsonl
│   └── image_manifest.jsonl
└── qc/
    ├── curation_report.json
    └── sha256_manifest.json
```

规则：

- positive 必须有完整、有限、ID 0..20 的三套坐标；geometry snapshot 要求 norm 在 `[0,1]`；
- LOW handedness confidence 不否定几何，因此 HIGH/MEDIUM 都可保留，geometry 阶段关闭 handedness loss；
- 所有未经独立确认的 `NEG_*_CANDIDATE` 默认 HOLD；
- 同帧 overlap 检查结果和排除原因写入每行 `pretrain_curation`；
- 只有 review JSONL 中带 reviewer 的 `CONFIRMED_NEGATIVE` 才能进入 multitask JSONL；若它和已确认手重叠，仍会 fail closed 留在 queue；
- 原始 canonical JSONL 不修改；included 图片物化到版本目录；输入/输出/图片哈希全部留盘。

按当前服务器数据，首次 curation 预期大约得到：

```text
landmark positives       17,574  # 17,721 - 147 越界记录
confirmed negatives           0
negative review queue      8,110
confirmed-overlap lower bound 3,908
smoke subset                 128
```

实际数字必须以生成后的 `qc/curation_report.json` 为准。

训练 loader 还会读取 `qc/sha256_manifest.json`，验证本次 labels 位于该 snapshot 内且 SHA-256 完全一致；数据审计还会将每张 materialized ROI 的实际 SHA-256 与标签中已认证的哈希逐一比较。训练时不能悄悄替换、追加、换图或内存过滤数据。

### 4. geometry-first pretrain

默认 `train_pretrain.yaml` 只读取持久化的 landmark positive：

```yaml
sample_type_fractions:
  POS_RUNTIME: 0.75
  POS_LOW_PALM: 0.25
  NEG_RUNTIME_CANDIDATE: 0.0
  NEG_LOW_PALM_CANDIDATE: 0.0

losses:
  landmarks:  {name: huber, delta: 0.05, coefficient: 20.0}
  hand_flag:  {coefficient: 0.1}  # 只学习 ROI 中 positive=1，防止级联拒绝
  handedness: {coefficient: 0.0}
```

这仍是 pretrain，不是 finetune。代价是当前模型没有学到可信的 no-hand 分类边界：在可靠 negative 准备完成前，Palm proposal 基本都会进入 landmarks。不能用受污染的 candidate 换取表面上的 hand_flag accuracy。

## 服务器重训顺序

本次本地代码合入后，由使用者提交、push，并在服务器 `git pull`。建议为每次实验显式使用新 run ID：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
export HAND_DATA_ROOT=/root/autodl-tmp
export HAND_PRETRAIN_CURATED_ID=v1-geometry-r1
export HAND_PRETRAIN_RUN_ID=v1-pretrain-geometry-r1

make test
make compile

# 每次实验使用新的 snapshot ID；不要用 --overwrite 覆盖需要保留审计的旧版本。
make curate-pretrain
python -m json.tool \
  /root/autodl-tmp/train_pretrain_curated/${HAND_PRETRAIN_CURATED_ID}/qc/curation_report.json

# doctor 会验证当前训练 labels，首次运行必须放在 curation 之后。
make doctor
make inspect-pretrain
make smoke-pretrain-overfit
python -m json.tool \
  /root/autodl-tmp/hand_landmarker_runs/${HAND_PRETRAIN_RUN_ID}/smoke/smoke_gate_report.json
```

`smoke-pretrain-overfit` 使用落盘的固定 128 个 positive，把该子集的采样权重统一为 1，并关闭 augmentation 和两个分类 loss。训练结束后，门禁会核对完成状态、resolved config、Git/labels/checkpoint/history 哈希，再加载 `best` 对 128 张图按原顺序各做一次无增强、无重复前向。默认要求 mean MAE ≤ 0.01、sample p90 MAE ≤ 0.02、sample max MAE ≤ 0.05（分别约 2.55 / 5.10 / 12.75 px）。失败时不要启动完整训练：此时 negative 污染已被隔离，问题应继续定位到模型优化/容量或 ROI 标签契约。`make train-pretrain` 和默认 `make train` 都会再次执行该门禁，不能静默绕过。

smoke 通过后：

```bash
make train-pretrain

# 必须看到 mode=min，且 value 等于 history 的最小 val_landmark_mae。
cat /root/autodl-tmp/hand_landmarker_runs/${HAND_PRETRAIN_RUN_ID}/pretrain/checkpoints/best.weights.h5.state.json

make eval-val-pretrain
```

只有 Val 达到门禁，才运行锁定 Test 和整图推理：

```bash
make eval-test-pretrain
make infer-pretrain
```

建议 Val 门禁：

```text
mean pixel error <= 15 px
PCK@0.10 >= 0.80
normalized out-of-range hand < 1%
```

如果 Val 未达到门禁，不应反复查看 Test 选配置。

## negative 人工复核

不要直接编辑源 canonical。另建 review decisions JSONL，例如：

```json
{"crop_id":"...","decision":"FALSE_NEGATIVE_HAND_VISIBLE","reviewer":"name","notes":"clear hand"}
{"crop_id":"...","decision":"CONFIRMED_NEGATIVE","reviewer":"name","notes":"background only"}
{"crop_id":"...","decision":"HOLD","reviewer":"name","notes":"uncertain"}
```

把路径写入 `configs/curate_pretrain.yaml` 的 `curation.negative_review_decisions`，然后对一个新 snapshot 版本运行 curation。当前 geometry 阶段不要求完成这项工作；在有足够可靠 negatives 以前，不启动 presence multitask 阶段。

## 如果 geometry Val 仍不合格

先看 smoke：

- smoke 不通过：训练系统仍无法记住 128 张图，应检查梯度、容量和单 batch 可视化，不要归因于泛化。
- smoke 通过但全量 Val 差：主要是 pseudo-positive 覆盖和域差异。特别是 teacher 漏检的握拳、遮挡、边缘手没有 21 点监督，单纯删除假负例也不能凭空补标签。

下一步应在 pretrain 数据制作端增加可靠 positive，而不是恢复 candidate negatives：

1. 对同一 ROI 使用更低 teacher threshold、旋转/尺度 TTA 和相邻帧时序重试；
2. 只有任一可靠分支给出完整且一致的 21 点时才加入 landmark snapshot；
3. 对握拳、侧手、遮挡、运动模糊、画面边缘按 pose/session 分层统计；
4. 保持 Train/Val/真实 infer 的 ROI scale、旋转、padding 和亮度分布一致；
5. 每批新增数据都生成独立 curation snapshot、报告和 SHA-256。

在上述门禁通过前，不建议先加 sigmoid、骨长正则或更复杂的结构 loss。它们可以约束输出外观，却会掩盖 checkpoint/data pipeline 是否真正学会图像到关键点的映射。
