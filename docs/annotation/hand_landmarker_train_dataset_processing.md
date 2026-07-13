# Hand Landmarker 训练集处理方案

> 文档定位：详细说明约 66000 个训练候选 ROI 如何进入第一阶段伪标签训练和第二阶段人工微调。  
> 更新时间：2026-07-10。  
> 本文中的权重和阈值是首版起点，最终只允许根据人工验证集校准。

## 1. 当前训练数据与处理目标

当前训练候选池由两部分组成：

- 本方约 5000 张 A1 原始 TIFF，经 `00-03` 生成约 22000 个 ROI；
- 队友使用相同流程和配置生成约 46000 个 ROI；
- 合计约 66000 个 ROI。

这些 ROI 不是 66000 个独立、等质量样本。同一张原图可能产生：

- 最多 2 个超过阈值的 `detections`；
- 多个低分 `negative_candidates`；
- 多个实际指向同一只手的相邻 anchor；
- 同一物理手经过轻微不同 ROI 几何得到的重复 crop。

当前 ONNX decode 中，`max_detections=2` 只限制正式 detections；低分候选会按分数排序后最多保留 `max_detections * 5 = 10` 个，而且这批 negative candidates 没有经过与正式 detections 相同的 NMS。因此每张原图的低分候选是训练 ROI 膨胀和同手重复的主要来源之一。

因此训练集处理目标不是“删除所有不完美数据”，而是：

1. 把 66000 行视为候选目录；
2. 先执行结构门禁；
3. 按 presence 与 Palm 来源分型；
4. 识别当前规则能发现的伪标签风险；
5. 按同一原图中的物理手实例聚类并归一化重复权重；
6. 控制低价值负样本数量；
7. 生成第一阶段和第二阶段各自的训练清单；
8. 保留全部排除原因和有效样本量统计。

整体训练流程见 [两阶段训练流程总览](hand_landmarker_training_workflow.md)，后续脚本实现见 [处理系统修正计划](hand_landmarker_pipeline_revision_plan.md)。

> 当前仓库尚未实现本文所述 07A、global namespace、quality catalog 和 stage-specific sampler。这些是下一步代码需求，不是现在可以直接运行的既有功能。

## 2. 输入文件与权威字段

第一阶段每个数据源需要：

```text
02_roi_crops/images/*.png
02_roi_crops/hand_roi_crops_manifest.jsonl
02_roi_crops/hand_landmarks_autolabel_draft.jsonl
对应 autolabel 配置
原始 TIFF，或预先生成的 source-image SHA256 inventory
```

第二阶段还需要人工复核训练子集：

```text
03_reviewed/hand_landmarks_reviewed.jsonl
CVAT import QC 报告
```

字段语义：

| 字段 | 含义 | 是否决定 Hand 正负 |
|---|---|---|
| `hand_presence.present` | ROI 内是否有有效目标手 | 是 |
| `palm_valid` | ROI 来自正式 detection 还是低分 candidate | 否，只表示来源 |
| `palm_score` | 上游 Palm 分数 | 否，只用于分析和采样 |
| `landmarks_crop_norm/px` | ROI 内 21 点 | 只在 presence=true 时监督 landmark |
| `landmarks_image_px` | 反投影到原图的 21 点 | 用于重复实例聚类和 QC |
| `handedness` | Left/Right/unknown | 只在有效正样本上监督 handedness |
| `needs_review` | 当前有限 QC 是否触发警告 | 不是质量真值 |
| `ignore_for_training` | 人工明确排除 | 是，优先级最高 |

注意：脚本 `03` 的 draft 行目前不直接带 `needs_review`，它只把汇总写入 `mediapipe_roi_stats.json`；`ignore_for_training` 也只有经过 CVAT 与 `05` 的子集才存在。07A 必须对 draft 重新执行 QC 并生成 `quality_flags/needs_review`。draft 中缺少 ignore 只能解释为“未人工判定”，不能解释为“人工确认可用”。

必须坚持：

> Hand Landmarker 的正负只由 `hand_presence.present` 决定；`palm_valid=false, presence=true` 仍然是 Hand 正样本。

## 3. 两位成员数据合并前的命名空间

### 3.1 当前冲突风险

当前 ID 形式为：

```text
palm_det_id = image_stem:palm0 或 image_stem:neg0
crop_id = palm_det_id:crop
```

如果两位成员都包含 `000001.tiff`，会生成相同的 `palm_det_id`、`crop_id` 和 crop basename。当前 `index_by()` 与按 basename 建索引的逻辑遇到重复 key 时会以后一个静默覆盖前一个。

不得直接执行：

```text
拼接两个 JSONL
把两个 images 目录复制到同一个平面目录
```

### 3.2 推荐合并规范

为每份数据分配不可变 `dataset_id`：

```text
peak_train_v1
soar_train_v1
```

训练目录中使用：

```text
global_crop_id = dataset_id + ":" + original_crop_id
source_image_uid = dataset_id + ":" + original_image_name
```

保留：

```text
dataset_id
contributor
source_crop_id
source_image_uid
annotation_provenance
pipeline_version
config_sha256
palm_model_sha256
mediapipe_model_sha256
```

合并前必须验证：

- `global_crop_id` 唯一；
- crop basename 或目标路径唯一；
- 所有 `crop_path` 可读；
- 同一 ID 不同内容直接终止；
- 相同文件 SHA256 作为 exact duplicate 报告；
- 两份配置的 Palm、NMS、ROI 几何和 MediaPipe 模型一致。

还要检查路径可移植性。当前 Train 配置把数据放在仓库外，`crop_path` 可能保留生成机器上的绝对路径。合并工具必须解析每个 source 的原路径，把 crop copy/hardlink 到 combined dataset 的 namespaced 子目录，实际重命名冲突 basename，并把 `crop_path` 重写为 combined root 下的可移植相对路径。不能只新增 namespace JSON 字段，却继续引用另一台服务器的绝对路径。

## 4. 当前字段能判定与不能判定的内容

### 4.1 可以直接判定

利用 `palm_valid × hand_presence.present` 可以可靠分为四种语义来源：

| 类型 | 条件 | 含义 |
|---|---|---|
| `POS_RUNTIME` | `presence=true, palm_valid=true` | 板端运行阈值以上 detection 中的正样本 |
| `POS_LOW_PALM` | `presence=true, palm_valid=false` | 低分 Palm candidate 中 teacher 认为有手 |
| `NEG_RUNTIME_CANDIDATE` | `presence=false, palm_valid=true` | 板端真实会收到的 teacher-negative candidate；人工确认前不等于 Palm 误检 |
| `NEG_LOW_PALM_CANDIDATE` | `presence=false, palm_valid=false` | 阈值以下 teacher-negative candidate；可能是背景，也可能是 teacher 漏手 |

还能直接检查：

- presence 与点数是否冲突；
- 21 个点的 id、数值和边界是否合法；
- negative 是否错误携带 landmarks；
- handedness 是否 Left/Right/unknown；
- crop 是否存在、可读、尺寸是否为 `256x256`；
- manifest 与 label 的 ID、路径和 ROI 几何是否对应。

### 4.2 当前字段不能证明

仅凭现有 draft/final JSONL 不能可靠区分：

- 21 点完全正确；
- 少数点轻微偏差；
- 21 点整体落在错误位置但仍在图内；
- `presence=false` 且实际无手；
- `presence=false` 但 Google 漏掉真实手；
- ROI 中有两只手但 teacher 只返回其中一只；
- teacher 是否标到了错误的那只手；
- `source=cvat_reviewed` 是否真的被人工逐点修改过。

`needs_review=false` 只能表示没有触发现有有限规则，不能解释为标签 100% 正确。

另外，训练配置中 `mediapipe.num_hands=1`，因此 `mediapipe_num_hands_detected > 1` 基本不能发现真实双手 ROI。训练候选中的双手歧义只能依靠人工或后续额外分析识别。

## 5. 结构门禁：任何训练阶段之前必须执行

### 5.1 `DROP_EXPLICIT_IGNORE`

条件：

```text
ignore_for_training=true
```

该标记优先级最高，不参与任何训练阶段，但保留在排除清单和统计中。

### 5.2 `DROP_INVALID`

至少包括：

- crop 缺失、不可读或尺寸不正确；
- `crop_id`、global ID 或 crop basename 冲突；
- manifest orphan 或 label orphan；
- 坐标存在 NaN/Inf；
- positive 三组 landmarks 不是 21 点；
- point id 不是唯一完整的 `0..20`；
- norm、crop px 与 image px 严重不一致；
- negative 仍带有任一 landmarks；
- negative handedness 不是 unknown；
- ROI corners 缺失、数量不是 4 或无法完成投影；
- label 与 manifest 的 image、Palm ID 或 ROI 身份冲突。

结构错误必须 fail closed：进入 quarantine，不允许脚本通过清空、截断或猜测坐标把它修成可训练样本。

### 5.3 `HOLD_AMBIGUOUS`

结构合法但语义存在较大风险的样本，例如：

- `needs_review=true` 且无法按具体 flag 安全降权；
- 两只手同时进入 ROI 且目标手不明确；
- teacher negative 与同原图确认正手明显重叠；
- teacher 在轻微变换下 presence 或 landmarks 剧烈变化；
- 图片几乎无信息；
- 高 Palm 分数 negative，疑似 Google 漏手；
- 人工或可视化发现 21 点整体标错。

第一版训练默认不使用 `HOLD_AMBIGUOUS`，将其加入人工复核候选池。

## 6. 第一阶段样本分型与参与规则

样本应同时拥有三个独立维度：

1. `sample_type`：正负与 Palm 来源；
2. `quality_tier`：伪标签可信等级；
3. `duplicate_status`：是否为重复实例代表。

不能用一个 `needs_review` 布尔值替代这三个维度。

### 6.1 推荐类型

#### Positive `quality_tier=HIGH`

最低条件：

```text
presence=true
21 点结构完整且数值合法
needs_review=false
非已知多手歧义
handedness 为 Left/Right，或仅关闭 handedness loss
```

它仍然只是“高质量伪标签候选”，不是人工真值。高质量重复 ROI 仍可保持 `quality_tier=HIGH`；是否被当前阶段选中由独立的 `duplicate_status`、cluster 轮换和 `sampling_weight` 决定，不能因为重复而篡改质量等级。

#### Positive `quality_tier=MEDIUM`

结构正确，但存在轻度风险，例如：

- 轻度越界或边缘黑边；
- teacher 多变换一致性略低；
- 图像模糊但手仍清晰可见；
- 同一物理手不同 ROI 的 teacher 点位轻微不一致。

第一版 baseline 可以先不使用；基座稳定后再以较低 landmark 权重加入。

#### Positive `quality_tier=PRESENCE_ONLY`

有手基本可信，但 21 点不可信：

```text
presence 可训练
landmark loss = 0
handedness loss = 0
```

只有人工或额外一致性分析确认“确实有手”时才能生成该类，不能只根据一次 teacher 输出自动推断。

#### `NEG_RUNTIME_CANDIDATE`

```text
presence=false
palm_valid=true
```

它最接近板端真实运行输入，训练价值高；但人工确认前只能称为 pseudo runtime-negative candidate，Palm 可能是正确的而 Google 漏手。未经人工或一致性确认时应降低采样/质量权重，并优先进入审计队列；只有人工确认无手后才是 Gold hard negative。

#### `NEG_LOW_PALM_CANDIDATE`

```text
presence=false
palm_valid=false
```

只保留少量、多样化代表。它不是主要部署输入分布，不能因数量巨大而主导 presence head；低 Palm score 只是排序信号，不是已校准的“背景正确概率”。

### 6.2 推荐权重表达

必须分开四类控制量：

```text
sampling_weight
  只控制 sampler 中被抽中的概率，不乘入 loss

三个现有 *_loss_weight
  在新 schema 中作为 0/1 head applicability mask

supervision_loss_weight
  控制整条 pseudo/Gold 监督的总体可信度

presence/landmark/handedness_quality_weight
  控制具体 head 标签的可信度
```

训练时对每个 head 单独归一化：

```text
effective_head_weight_i = head_mask_i
                        * supervision_loss_weight_i
                        * head_quality_weight_i

L_head = sum(effective_head_weight_i * loss_i)
       / max(sum(effective_head_weight_i), eps)
```

`sampling_weight` 禁止再次乘入 loss，避免同一风险被 sampler 和 loss 无意中二次降权。负样本比例也不应改变 landmark/handedness loss 的数值尺度。

首版起点：

| 类别 | Stage 1 | Sampling | Supervision | Presence quality | Landmark quality | Handedness quality |
|---|---|---:|---:|---:|---:|---:|
| `sample_type=POS_RUNTIME, quality=HIGH` | 主训练 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 或关闭 mask |
| `sample_type=POS_LOW_PALM, quality=HIGH` | 限额保留 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 或关闭 mask |
| `quality=MEDIUM` | 后段加入 | 0.5 | 0.5～1.0 | 按 flag | 按 flag | 按 flag |
| `quality=PRESENCE_ONLY` | 可选 | 0.25 | 0.5～1.0 | 1.0 | mask=0 | mask=0 |
| pseudo `NEG_RUNTIME_CANDIDATE` | 审计/限额 | 0.25～0.5 | 0.25～0.5 | 1.0 | mask=0 | mask=0 |
| Gold-confirmed runtime negative | 重要 hard negative | 由 bucket 决定 | 1.0 | 1.0 | mask=0 | mask=0 |
| pseudo `NEG_LOW_PALM_CANDIDATE` | 少量采样 | 0.25 | 0.5 | 1.0 | mask=0 | mask=0 |
| HOLD/DROP | 不使用 | 0 | 0 | 0 | 0 | 0 |

`MEDIUM` 不能用一套固定权重覆盖不同风险：landmark 轻微问题只降低 landmark；handedness 冲突只关闭 handedness；presence TTA 不稳定则降低 presence 或直接转 HOLD。以上均为初始策略，禁止根据 Test 修改。

### 6.3 Batch 分层起点

推荐起点：

```text
70% positive
25% NEG_RUNTIME_CANDIDATE
5% NEG_LOW_PALM_CANDIDATE
```

允许范围：

```text
全部 positive：65%～80%
NEG_RUNTIME_CANDIDATE：15%～30%
NEG_LOW_PALM_CANDIDATE：0%～10%
```

`POS_LOW_PALM` 建议不超过 positive 配额的 10%～25%。实际比例应在人工 Val 上比较 presence 与 landmark 指标后冻结。

## 7. 重复 ROI 降采样详细流程

### 7.1 核心原则

不要简单永久删除所有重复 ROI。不同 ROI 的平移、旋转和尺度差异具有一定鲁棒性价值，但同一物理手实例的总训练贡献应接近 1，而不是每个 anchor 都贡献 1。

需要记录：

```text
source_group_id
duplicate_cluster_id
duplicate_cluster_size
duplicate_rank
duplicate_sample_weight
```

### 7.2 第一步：按数据源和原图分组

```text
source_group_id = dataset_id + ":" + image
```

不能只按文件名分组，也不能把同一原图中的两个真实手强行合并。

### 7.3 第二步：正样本物理手聚类

同一原图的不同 ROI crop 坐标不可直接比较，应使用 `landmarks_image_px`。

对两个 positive A、B，定义人工可校准的归一化距离：

```text
hand_scale = max(两者 reference/teacher landmark bbox 对角线平均值, epsilon)
D(A,B) = 21 点原图欧氏距离的 median / hand_scale
```

结合：

- handedness 是否兼容，但只作软特征；
- landmark bbox IoU；
- wrist 与 MCP 点距离；
- ROI 和 Palm 几何重叠。

初始可尝试：

```text
D(A,B) <= 0.15～0.20
且 landmark-derived bbox IoU 达到配置阈值
```

阈值必须用少量人工抽样校准。建议使用按 crop ID 稳定排序的 complete-linkage 聚类：候选与簇内所有成员都满足阈值时才加入，避免单链桥接相邻双手。空间强一致但 handedness 冲突时增加 flag，不因一次 teacher handedness 错误强行拆簇。当前采集通常每图 0～2 个主要手实例，但算法不硬编码最多两个 cluster。

### 7.4 第三步：选择或轮换代表 ROI

同一物理手簇：

1. 优先人工 Gold；
2. 再优先无 quality flag；
3. 再优先 `palm_valid=true`；
4. 再比较边界、图像有效区域、teacher 稳定性；
5. 最后使用 palm score 和稳定字典序打破平局。

训练方式二选一：

- 每个 epoch 从簇中随机选择 1 个代表；或
- 全部轮换，但每个 ROI 权重约为 `1 / cluster_size`。

允许额外保留第 2 个 ROI，前提是 ROI 几何差异明显，作为 crop jitter augmentation；其权重建议 0.25～0.5。其余记录保留在 catalog，但不进入该阶段 canonical 训练文件。

### 7.5 第四步：检查可疑 negative

如果同一原图存在确认 positive，而某个 negative ROI polygon 包含该手的大部分掌部点：

```text
p0, p1, p5, p9, p13, p17
```

则增加：

```text
NEGATIVE_OVERLAPS_CONFIRMED_HAND
```

并转入 `HOLD_AMBIGUOUS`，不能自动作为 presence=false 训练。这可以发现一部分 `presence=false` 但实际有手的 teacher 漏检。

### 7.6 第五步：负样本限额

每张原图建议最多：

- 1～2 个空间不同的 `NEG_RUNTIME_CANDIDATE`；
- 0～1 个 `NEG_LOW_PALM_CANDIDATE`。

高度重叠的 negative 可用 rotated ROI polygon IoU、中心/尺度和 crop pHash 聚类。低分 negative 优先保留接近正式 Palm 阈值、且不与确认手重叠的代表。

### 7.7 第六步：跨原图近重复

当前 schema 没有可靠的 `recording_id/session_id/frame_index`，因此不能通过猜文件名完成严格时序降采样。

首版只做：

- 原始 TIFF/crop SHA256 去除 exact duplicate；
- 在明确属于同一录制 session 时，用 pHash 或图像特征识别相邻近重复；
- 近重复簇使用逆簇大小采样；
- 后续采集时补充 session 和 frame 元数据。

### 7.8 推荐层级 sampler

```text
先选择成员/数据源
  → 再选择 source image 或物理手 cluster
    → 再选择 cluster 内某个 ROI
      → 最后满足 batch 的 positive/hard-negative/easy-negative 配额
```

第一版可以把两个成员接近 50:50 或限制大数据源不超过约 60%，但这只是防止 46000 行天然压过 22000 行的 baseline，不代表真实最优分布。最终来源权重应综合独立物理手 cluster 数、随机审计质量和目标部署来源确定，不能让原始行数自动决定贡献比例。

## 8. 第一阶段最小可实施流程

时间紧张时不必一开始就对全部 66000 ROI 执行昂贵的多次 teacher TTA。最低版本：

1. 给两个数据源增加 namespace；
2. 执行结构门禁和文件完整性检查；
3. 07A 对 draft 重新运行 QC，生成具体 `quality_flags` 和派生 `needs_review`；
4. 第一版只对 pseudo 排除所有派生 `needs_review=true`；人工 Gold 不应用 teacher/Palm heuristic；
5. 按同原图 landmark 投影聚类 positive；
6. 每原图限制 hard/easy negative；
7. 按四象限类型与数据源分层采样；
8. 每类随机可视化抽查；
9. 训练第一版基座；
10. 在人工 Val 上评估。

后续增强版本可以只对 `needs_review`、hard negative 和模型分歧样本执行 3～5 次轻微可逆变换，再根据 teacher 输出一致性升级或降级样本。多变换生成关键点共识的思路可参考 [Data Distillation](https://openaccess.thecvf.com/content_cvpr_2018/html/Radosavovic_Data_Distillation_Towards_CVPR_2018_paper.html)。

## 9. 第二阶段 Train gold 选择

### 9.1 随机审计集

随机抽取约 500～1000 个 ROI，但不能在 66000 行上直接均匀抽样，否则大 duplicate cluster 会被重复抽中。先按数据源、source image/物理手 cluster 分层，再从每个 cluster 选择一个 ROI，并覆盖：

- peak / soar 两个来源；
- `palm_valid × presence` 四类；
- Left/Right；
- Palm score 分桶；
- 正常、暗光、反光、模糊、黑边、边缘手；
- 不同 duplicate cluster size。

随机审计用于估计伪标签错误率，不能全部被主动学习困难样本替代。报告至少同时给出 ROI-weighted、cluster-weighted 和 source-image-weighted 错误率；主动挑错队列也要限制每个 cluster 的入选数量。

### 9.2 主动挑错集

基座模型训练后，再选约 1000～2000 个：

- teacher 与 student presence 不一致；
- handedness 不一致；
- 21 点 mean/max distance 较大；
- student presence 接近阈值；
- `needs_review=true`；
- `NEG_RUNTIME_CANDIDATE`，尤其高 Palm score；
- teacher negative 与同图 positive 重叠；
- 黑边、遮挡、交叉手、暗光和 ROI 明显偏移。

### 9.3 Train gold 的 CVAT 规则

维持现有 labels：

```text
no_hand
Left
Right
ignore_for_training
hand_landmarks
```

- 无手：删除 skeleton/handedness，只保留 `no_hand`；
- 有且可可靠标注一只目标手：一个 21 点 skeleton + 唯一 Left/Right；
- 有手但 21 点无法可靠完整确定、目标手歧义或数据损坏：`ignore_for_training`；
- 不允许 `no_hand` 与 skeleton 同时存在；
- 不允许 Left 和 Right 同时存在；
- 不允许保留两个 skeleton。

Train gold 中出现双手时，也必须复用 [验证集处理方案第 5 节](hand_landmarker_val_dataset_processing.md) 的 `crop_id → palm_det_id → bbox/p0/p9` 目标手判定；目标唯一才标一只，anchor 融合或归属不清时 ignore。

## 10. 第二阶段混合微调数据

Gold 必须按 global crop ID 覆盖对应 pseudo，不允许同一 ROI 同时以两套标签重复进入 batch。

覆盖还要扩展到 duplicate cluster：一个物理手 cluster 中只要已有 Gold，Stage 2 默认只使用 Gold 代表并排除同簇 pseudo siblings，避免精标一个 ROI 后又 replay 另外五个冲突伪标签。未来如要把 Gold 通过原图坐标反投影到 siblings，必须逐个验证 ROI 内边界和投影一致性后生成新 Gold，不能直接复制 crop 坐标。

推荐：

```text
30%～50% human Gold
50%～70% 第一阶段筛选后的 `quality_tier=HIGH` pseudo 与已确认 negative
```

- Gold positive：三个 head 全权重；
- Gold negative：只训练 presence；
- pseudo replay：继续使用降采样后的高质量版本；
- 不从 Val/Test 借样本；
- 使用小学习率和人工 Val early stopping；
- 可以尝试很短的 Gold-only 收尾，但只有 Val 持续提升时保留。

## 11. 建议输出字段与报告

训练 catalog 建议保留原始字段并新增：

```json
{
  "schema_version": "train_catalog_v1",
  "dataset_id": "peak_train_v1",
  "crop_id": "peak_train_v1:000001:palm0:crop",
  "global_crop_id": "peak_train_v1:000001:palm0:crop",
  "source_crop_id": "000001:palm0:crop",
  "source_image_uid": "peak_train_v1:000001.tiff",
  "annotation_provenance": "mediapipe_pseudo",
  "sample_type": "POS_RUNTIME",
  "quality_tier": "HIGH",
  "quality_flags": [],
  "selection_action": "include",
  "duplicate_cluster_id": "peak_train_v1:000001:hand0",
  "duplicate_cluster_size": 4,
  "training_stage": "pretrain",
  "sampling_bucket": "pseudo_positive_runtime",
  "sampling_weight": 0.25,
  "supervision_loss_weight": 1.0,
  "presence_quality_weight": 1.0,
  "landmark_quality_weight": 1.0,
  "handedness_quality_weight": 1.0,
  "hand_presence_loss_weight": 1.0,
  "landmark_loss_weight": 1.0,
  "handedness_loss_weight": 1.0
}
```

报告至少包含：

- 两个来源各自输入/保留/排除数量；
- 四种 `palm_valid × presence` 数量；
- quality tier 和 quality flag 分布；
- exact/near duplicate 数量；
- 每张原图 ROI 数直方图；
- 每个 positive cluster 大小；
- negative 限额前后数量；
- 每种排除原因；
- 原始行数、独立 cluster 数和各 sampling bucket 的抽样概率；`sampling_weight` 之和只有在明确归一化定义后才能称为有效样本量；
- Stage 1/Stage 2 canonical JSONL 的 SHA256。

## 12. 验收清单

- [ ] 两份来源都有唯一 `dataset_id`；
- [ ] 不存在重复 global crop ID 或 basename；
- [ ] 所有输入图片可读且为 `256x256`；
- [ ] 所有 positive 恰好 21 个合法点；
- [ ] 所有 negative 三组 landmarks 为空；
- [ ] `palm_valid=false, presence=true` 没有被误当 negative；
- [ ] 可疑 teacher negative 与同图 positive 重叠已转 HOLD；
- [ ] 每个物理手 cluster 的总权重受控；
- [ ] 每张原图的 easy negative 已限额；
- [ ] 两个数据源的采样贡献不由原始行数直接决定；
- [ ] Gold 通过 global ID 覆盖 pseudo；
- [ ] Val/Test 没有出现在任何训练清单；
- [ ] 所有排除样本和原因仍可追踪；
- [ ] 训练文件和报告均有版本与 SHA256。
