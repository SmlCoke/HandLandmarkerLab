# Hand Landmarker 验证集处理方案

> 文档定位：定义主验证集的人工复核、Gold 筛选、双手 ROI 和共享规则。  
> 当前组成：Peak shared `vals_data` + Soar shared `vals_data` + 当前路线 independent `vali_data`；最终由 07B 合并为一个 Val Gold。
> 更新时间：2026-07-10。

## 1. 验证集的角色

验证集是人工金标准 benchmark，支持两个训练阶段的模型选择，但绝不参与参数更新。

第一阶段使用 Val：

- 选择伪标签训练 checkpoint；
- 检查样本筛选和正负采样是否有效；
- 发现 presence、landmark、handedness 的主要失败类型；
- 初步选择 presence 阈值。

第二阶段使用同一 Val：

- early stopping；
- 选择 Gold/pseudo 混合比例；
- 选择最终 checkpoint；
- 冻结 presence 阈值；handedness 默认使用 0.5，若实验确需校准也只能在 Val 上完成并在 Test 前冻结；
- 比较 FP32、量化仿真和板端输出。

不得把 Val ROI 加入 Train gold，也不得根据 Val 样本重新生成训练伪标签。整体流程见 [两阶段训练流程总览](hand_landmarker_training_workflow.md)。

## 2. 人工复核前的流程

共享和独立部分分别运行相同的 00–04：

```text
共享 vals_data：configs/autolabel_val.yaml
独立 vali_data：configs/autolabel_vali.yaml
00_validate_images.py
01_export_palm_detections.py
02_build_hand_roi_crops.py
03_run_mediapipe_on_rois.py
04_export_cvat_xml.py
```

当前 Val 配置符合主评测要求：

```text
Palm score threshold = 0.50
NMS IoU = 0.30
cross-head suppress IoU = 0.35
max detections = 2
keep_low_score_candidates_for_negatives = false
ROI scale = 1.8 × 1.8
shift_y = -0.1
output = 256 × 256
```

Val 不导出低分 `negative_candidates`。所有正常生成的 ROI 都应来自板端阈值真正会送入 Hand Landmarker 的 detection，因此人工确认的 `palm_valid=true, presence=false` 必须保留为部署 hard negative。

不得为了得到更“漂亮”的指标，按 Google confidence、学生置信度、Palm score 或 teacher/student 一致性自动纯化 Val。

CVAT 复核双手或目标不明确样本前，建议额外用 draft 运行：

```powershell
python scripts/06_visualize_autolabels.py --config configs/autolabel_val.yaml --labels-jsonl <vals_draft.jsonl>
python scripts/06_visualize_autolabels.py --config configs/autolabel_vali.yaml --labels-jsonl <vali_draft.jsonl>
```

它生成原图 Palm bbox、p0/p9 和 rotated ROI overlay，只用于帮助确认当前 ROI 的 Palm anchor，不改变 `00-03` 输出或任何标签。

## 3. 人工复核前先冻结标注约定

开始批量复核前，两位标注者应共同检查一小批已知样例并冻结：

1. 当前摄像头和 ROI 是否镜像；
2. Google 的 Left/Right 与项目训练映射是否一致；
3. 双手 ROI 中如何沿 Palm anchor 识别目标手；
4. 何时认为 21 点仍可可靠定位；
5. 何时必须使用 `ignore_for_training`。

Left/Right 指真实目标手的 handedness，不能按它位于画面左侧还是右侧临时判断。约定冻结后不得在标注中途改变。

现有 CVAT labels 保持不变：

```text
no_hand
Left
Right
ignore_for_training
hand_landmarks
```

不需要新增“双手”或“遮挡”标签。双手歧义通过 `ignore_for_training` 排除，原因与数量由后续报告统计或人工清单补充。

现有 CVAT label 无法自动保存“为什么 ignore”或“目标明确双手”等上下文。若要生成本文要求的分组统计，必须同时维护一个不改变 CVAT schema 的 sidecar：

```text
review_context.csv
crop_id,palm_det_id,review_reason,context_group,target_hand_rule,reviewer,note
```

至少所有 ignored、双手和 anchor 异常样本必须填写。若没有 sidecar，后续系统只能报告 generic ignore 数量，不能可靠恢复 `multi_hand_target_ambiguous` 等原因。

## 4. CVAT 人工复核决策表

| ROI 实际情况 | CVAT 操作 | 是否进入主 Val |
|---|---|---:|
| 完全没有手 | 删除 skeleton 和 Left/Right，只保留 `no_hand` | 是，presence negative |
| 一只手，自动点全部正确 | 保留一个 skeleton，确认唯一 Left/Right | 是 |
| 一只手，少数或多数点错误，但人工可以准确修正 | 修正全部 21 点和 handedness | 是 |
| 暗光、模糊或局部遮挡，但 21 点仍可可靠确定 | 完整标注 21 点 | 是，属于有价值困难样本 |
| 实际有手，但完整 21 点无法可靠确定 | 添加 `ignore_for_training`，不要标 `no_hand` | 否 |
| 两只手，Palm anchor 明确指向其中一只，目标手可完整标注 | 只标目标手的一个 skeleton 和 handedness | 是，另一只手作为干扰 |
| 两只手，Google 标到非目标手，但 anchor 目标明确 | 把 skeleton 整体改到目标手 | 是 |
| 两只手交叉/重叠，目标归属或点位无法唯一确定 | 添加 `ignore_for_training`，不要标 `no_hand` | 否 |
| 图像损坏或异常到无法判断 | 添加 `ignore_for_training` | 否 |

### 4.1 合法正样本

必须满足：

- 没有 `no_hand`；
- 没有 `ignore_for_training`；
- 恰好一个 `hand_landmarks` skeleton；
- skeleton 恰好有 21 个点；
- 恰好一个 `Left` 或 `Right`；
- 21 点都对应同一目标手；
- 不把不可见点随意夹到 crop 边界。

当前 schema 没有逐点 visibility 或单点 loss mask，因此如果某些关键点无法可靠确定，不能猜点后作为 Gold；应忽略整个 ROI。

### 4.2 合法负样本

必须满足：

- 恰好一个 `no_hand`；
- 没有 skeleton；
- 没有 Left/Right；
- 没有 `ignore_for_training`。

Palm score 很高也不影响人工真值。人工确认无手就是合法 hard negative。

### 4.3 合法忽略样本

必须包含：

```text
ignore_for_training
```

不要同时添加 `no_hand`。为了节省时间，ignored ROI 不要求继续修正自动 skeleton；后续 07B 必须先识别 ignore，再跳过该样本的严格 Gold 校验。ignored 样本仍要进入 ignored 清单和比例报告，但不进入主指标。

ignored 输出必须明确 `ground_truth_valid=false`。其中未修正的自动 presence、handedness 和 landmarks 不具备 Gold 语义，不得被训练、主评测或 challenge landmark 指标读取。

### 4.4 禁止状态

- `no_hand` 与 skeleton 同时存在；
- `no_hand` 与 Left/Right 同时存在；
- Left 和 Right 同时存在；
- 非 ignored ROI 存在两个 skeleton；
- 非 ignored positive 缺少 handedness；
- 非 ignored positive 少于或多于 21 点。

## 5. 双手 ROI：目标手如何确定

当前 Hand Landmarker 每个 ROI 只输出一套 21 点，因此双手 crop 的 Gold 必须指向唯一目标手。Palm anchor 不是一个额外的类别标签，而是产生该 ROI 的具体 Palm detection 几何：

```text
palm_det_id
bbox_px = [xmin, ymin, xmax, ymax]
p0 = wrist 附近的 Palm keypoint
p9 = middle-finger MCP 附近的 Palm keypoint
方向向量 = p0 → p9
```

其中 bbox 负责指出掌部区域，p0/p9 负责指出这只手的 wrist—middle-MCP 轴向。ROI 的中心、旋转和尺度都由这三个信息生成，因此它们是双手场景中选择目标手的主要依据。

这里的 bbox 是 **Palm bbox，不是整只手的外接框**。指尖位于 bbox 外是正常现象；判断时应看 bbox 是否包住掌心/MCP 区域，不能因为某只手的全部手指更完整地落在 bbox 内就选它。p0/p9 也是检测模型的近似定位点，应按解剖邻域和整体轴向判断，不要求与人工 landmark 0/9 像素级重合。

### 5.1 第一步：从 CVAT crop 精确追溯到 Palm detection

不能只看 crop 中哪只手更大、更居中，也不能只沿用 Google 初始 skeleton。必须先完成 ID 追溯：

1. 用 CVAT 图片 basename 在 `hand_roi_crops_manifest.jsonl` 中找到对应 `crop_path`；
2. 读取该行的 `crop_id`、`image`、`palm_det_id`、`roi_corners_px`；
3. 在 `palm_detections.jsonl` 中找到同一 `image`；
4. 在该 image 的 `detections` 中按 `palm_det_id` 找到唯一 detection；Val/Test 正常情况下不应来自 `negative_candidates`；
5. 读取该 detection 的 `bbox_px`、`keypoints_px.p0` 和 `keypoints_px.p9`；
6. 打开该 crop 对应的原图 anchor reference，而不是凭 CVAT 中的自动 skeleton 猜目标。

当前 `06` 全图 overlay 会同时画出同一原图的多个 Palm bbox 和多个 ROI，Palm 图形本身只显示 score，不保证仅凭颜色就能分清某个 `palm_det_id`。人工确认时必须结合 manifest/JSON 的 ID；后续工具应生成每个 crop 独立的 anchor reference，见本文 5.8 和处理系统修正计划。

如果出现以下任一情况，不应继续凭肉眼猜：

- basename 在 manifest 中不唯一；
- `palm_det_id` 在 Palm JSONL 中找不到或出现重复；
- manifest 的 image 与 Palm record 不一致；
- 该 Val/Test ROI 实际来自 `negative_candidates`；
- bbox、p0 或 p9 缺失。

这些属于数据完整性 fatal error，必须先修复流水线并重新生成 reference，不能用 `ignore_for_training` 绕过。ignore 只处理图像语义歧义或无法可靠标点，不能把缺失/重复 ID、错误 split 来源或损坏 anchor 悄悄移出评测集。

### 5.2 第二步：理解原图 anchor 与 crop 的关系

`roi_corners_px` 顺序固定为：

```text
C0 = crop top-left
C1 = crop top-right
C2 = crop bottom-right
C3 = crop bottom-left
```

原图中的一点 `P` 可以映射到 crop 归一化坐标 `(u,v)`：

```text
ex = C1 - C0
ey = C3 - C0
u = dot(P - C0, ex) / dot(ex, ex)
v = dot(P - C0, ey) / dot(ey, ey)

x_crop = 255 * u
y_crop = 255 * v
```

该简式依赖当前 `roi_corners_px()` 生成的矩形，即 `C0→C1` 与 `C0→C3` 近似正交。辅助工具必须先检查两条边非零、正交误差在容差内并验证 round-trip；检查失败属于数据完整性错误，不能继续人工判定或用 ignore 绕过。

将 p0、p9 和 bbox 四个角都映射到 crop 后，可以在参考图上画出：

- p0：建议红点；
- p9：建议蓝点；
- p0→p9：建议带箭头的轴线；
- Palm bbox：映射后通常是一个旋转四边形；
- 当前 ROI 边界和 `crop_id/palm_det_id`。

这张参考图只用于标注辅助，不能作为训练输入上传。CVAT 中仍使用原始灰度 ROI。

### 5.3 第三步：分别为两只可见手建立候选对应

对 crop 中的每只手分别观察：

```text
W = 该手 wrist，对应 MediaPipe landmark 0
M = 该手 middle MCP，对应 landmark 9
Palm core = landmarks 0, 1, 5, 9, 13, 17 所围成的掌部区域
Hand axis = W → M
```

对每只候选手检查四类证据：

1. **p0 对应**：anchor p0 是否落在或接近该手 wrist，而不是另一只手的 wrist/手指/背景；
2. **p9 对应**：anchor p9 是否落在或接近同一只手的 middle MCP；
3. **轴向对应**：`p0→p9` 是否与该手 `W→M` 方向一致，而不是指向另一只手；
4. **掌区对应**：anchor bbox 是否主要覆盖该手的 palm core，而不是同时把两只掌心包成一个大框。

最重要的规则是：

> p0 与 p9 必须共同指向同一只手。p0 更像手 A、p9 更像手 B 时，不允许用“更居中”强行选择，这通常表示 anchor 融合了两只手。

ROI 中心、目标手大小和旋转后的“手指大致朝上”只能作为辅助证据，因为这些量本身就是由可能有误差的 bbox/p0/p9 推导得到，不能覆盖直接 anchor 对应。

### 5.4 唯一目标手的通过条件

满足下列任一组，可以判定目标唯一。

#### 强唯一

- p0 明确对应手 A 的 wrist；
- p9 明确对应同一手 A 的 middle MCP；
- p0→p9 与手 A 的 W→M 方向一致；
- bbox 主要覆盖手 A 的掌部；
- 手 B 不满足上述成套关系。

此时无论手 B 是否完整、是否更大、是否更靠 crop 中心，都只标手 A。

#### 可接受唯一

Palm 模型的某一个 keypoint 有轻微偏移时，仍必须至少有一个**直接解剖对应**，并满足下面一种模式：

- p0 和 p9 虽不像素级重合，但仍分别对应同一手的 wrist 与 middle-MCP 邻域；或
- p0/p9 中一个具有清晰解剖对应，同时 bbox 掌区明确只属于同一手，另一个 keypoint 的方向没有与之冲突。

p0→p9 轴向来自 p0/p9，ROI rotation 也来自 p0/p9；ROI center/scale 又主要来自 bbox。因此“轴向 + ROI rotation”或“bbox + ROI center”不能算两票独立证据。ROI 中心、旋转和尺度只用于否决明显矛盾或增强人工信心，不能在缺少直接解剖对应时单独使样本通过。

### 5.5 必须判为歧义的情况

以下任一情况使用 `ignore_for_training`，不要任意挑一只：

- p0 对应手 A、p9 对应手 B；
- bbox 同时覆盖两个 palm core，p0/p9 又不能提供明确区分；
- p0/p9 落在两手之间、交叉区域或背景，无法建立解剖对应；
- 两只手对 anchor 的 wrist/MCP/方向匹配程度相近；
- Palm bbox 或 p0/p9 明显融合了两只手；
- anchor 对应手明确，但该手的完整 21 点因 crop 截断、重叠或遮挡无法可靠标注；
- 需要依靠 Google 初始 skeleton 才能选出一只，而 anchor 几何本身不支持；
- 两位复核者按照同一规则仍选择不同目标手。

这些样本有手，因此不能标 `no_hand`。sidecar 建议记录：

```text
review_reason = multi_hand_target_ambiguous
target_hand_rule = anchor_ambiguous
```

### 5.6 Google skeleton、handedness 和中心位置的优先级

目标选择证据优先级固定为：

```text
同一 palm_det_id 的 p0/p9 + bbox
  > ROI 几何一致性
  > 只有一只完整可见手这一事实
  > Google 初始 skeleton
  > handedness、手大小或简单中心距离
```

- Google skeleton 标到另一只手：如果 anchor 明确，应把 21 点整体改到 anchor 目标手；
- Google handedness 不能用来决定目标，因为 teacher 可能先选错手再给出正确的“另一只手” handedness；
- “离中心最近”“面积最大”不能单独作为目标规则；
- 如果 anchor 不对应任何手，但 crop 中只有一个能构成合理、完整目标的手实例，presence 语义仍是 crop 内有手，可以标这唯一有效目标，并在 sidecar 记录 `target_hand_rule=single_visible_hand_fallback`；第二只手只有少量边缘像素、残缺片段且不可能构成有效目标时，不取消该 fallback；
- 如果 anchor 不对应任何手且 crop 中存在两只都能构成合理目标的手，则没有唯一 fallback，必须 ignore。

### 5.7 同一原图多个 detections 的交叉检查

同一原图可能产生两个 detections 和两个 ROI：

- 每个 ROI 必须独立沿自己的 `palm_det_id` 找目标；
- 两个 anchor 分别指向左右手时，各自标对应手；
- 两个 anchor 都指向同一只手时，两个 ROI 都可以标同一目标手，不能为了“凑左右手”把其中一个改标另一只；
- 一个 anchor 融合两手而另一个清晰时，只保留清晰 ROI；融合 anchor 对应 ROI ignore；
- 交叉检查只能发现矛盾，不能反过来覆盖单个 anchor 的直接证据。

### 5.8 推荐的人工记录与未来辅助工具

`review_context.csv` 对所有双手 ROI 增加：

```text
crop_id
palm_det_id
context_group = multi_hand_target_clear | multi_hand_target_ambiguous
target_hand_rule = anchor_p0_p9_bbox | anchor_bbox_axis | single_visible_hand_fallback | anchor_ambiguous
review_reason
reviewer
second_review_status
```

所有 `multi_hand_target_ambiguous` 必须 ignore。`multi_hand_target_clear` 可以进入主 Val/Test，但建议由第二人快速确认目标手，不必重复精修全部 21 点。

为了让这套规则可高效执行，后续 `06` 应增加 per-crop anchor reference：

- 原图只高亮当前 `palm_det_id` 的 bbox/p0/p9/ROI，其他 detections 灰显；
- crop 参考图显示映射后的 bbox、p0、p9 和箭头；
- 图片上明确写出 `crop_id` 与 `palm_det_id`；
- 输出 `target_anchor_review_index.csv`，保存映射后的 anchor 坐标与参考图路径；
- 不修改原始 ROI，不改变 CVAT labels，也不改变 `00-03`。

### 5.9 快速示例

#### 示例 A：目标明确，保留

```text
p0 落在左手 wrist 邻域
p9 落在左手 middle MCP 邻域
bbox 覆盖左手掌心
右手虽然更靠 crop 中心，但与 p0/p9 不匹配
```

结论：左手是目标；只标左手，右手作为干扰。

#### 示例 B：anchor 融合，忽略

```text
p0 接近左手 wrist
p9 接近右手 middle MCP
bbox 横跨两只掌心
```

结论：无法形成同一手的 wrist→middle-MCP 对应；添加 `ignore_for_training`，不标 `no_hand`。

#### 示例 C：teacher 选错手，人工纠正

```text
p0/p9/bbox 明确对应右手
Google 初始 skeleton 却画在左手
```

结论：删除/移动原 skeleton，完整标注右手 21 点和 Right；不能因为 teacher 已有标注而选左手。

#### 示例 D：anchor 本身是假定位，但只有一只可见手

```text
p0/p9 没有可靠落在手上
crop 中只有一只完整且可可靠标注的手
```

结论：因为没有第二个竞争目标，按 crop 内 presence 语义标唯一可见手，并记录 `single_visible_hand_fallback`。如果 crop 中有两只手，则必须 ignore。

## 6. 20%～30% 双手 ROI 是否需要重录

### 6.1 先审计原因，不要立即全部推倒

“ROI 中看见两只手”不等于“该 ROI 必须 ignore”。先统计：

```text
multi_hand_total
multi_hand_target_clear
multi_hand_target_ambiguous
multi_hand_anchor_or_bbox_failure
other_ignored
eligible_positive
eligible_negative
```

并抽查全图 overlay，区分：

1. ROI 正确围绕目标手，只是第二只手也进入 crop；
2. Palm bbox/p0/p9 混合了两只手，目标 anchor 本身错误；
3. ROI scale 与录制距离共同导致两手同框；
4. 只有第二只手的小块边缘进入，不影响目标定义。

### 6.2 推荐决策

| 审计结果 | 处理 |
|---|---|
| 大部分双手 ROI 目标清楚 | 不重录，按目标手规则标注 |
| 只有少量歧义 | 标 ignore，继续使用其余样本 |
| 真正 ambiguous/ignore 比例较高，导致有效样本不足 | 只补录被排除的大致数量，不重录全部 |
| 双手比例明显高于真实部署 | 补录更有代表性的主 Val；旧双手数据转 challenge |
| 大量 Palm anchor 融合两手或 ROI 目标不可定义 | 定向补录并单独报告上游失败，不能通过随意选手掩盖 |

建议保留现有歧义双手 ROI 文件，不进入主指标，但形成 `multi_hand_challenge` 清单。这样既不污染单目标 landmark Gold，也不掩盖真实双手系统风险。

歧义双手之所以 ignore，正是因为当前单 skeleton schema 无法定义唯一 landmark 真值。因此该 challenge 默认只能报告数量、覆盖率、Palm anchor ambiguity/failure rate 和定性失败案例。若未来需要双手召回或双手 landmark 的定量端到端指标，必须另外制作原图级双手实例 Gold；当前 ROI CVAT schema 不足以支持。

时间允许时，建议按真正 ignored 的数量近似一对一补录，使最终 eligible Val 尽量恢复到当前约 1000 ROI 的量级，并补足被排除后变少的 positive/negative、Left/Right 和关键光照场景。若无法补足，必须报告实际 eligible 数量和 ignore 比例，而不能仍把“1028 ROI”当作有效评测规模。

如果最终应用本身包含双手手语，不能把所有双手场景从系统测试中消失。正确做法是：

- 主 Hand Landmarker benchmark 使用目标唯一的 ROI；
- 歧义多手作为单独 challenge/end-to-end 指标；
- 补录时让 Palm 更容易为两只手分别产生独立 ROI，但仍保留真实干扰情况。

## 7. 是否与 Soar 共享验证集

可以，而且推荐共享。

训练流程独立应体现在：

- 训练数据选择；
- 伪标签质量策略；
- 去重和采样；
- loss、增强和优化器；
- 模型 checkpoint 和随机种子。

如果两人使用完全不同的 Val，就无法判断结果差异来自模型还是数据难度。共同 Val 才能公平比较。

当前组织固定为：

- `vals_data`：两人共享的主 Val 部分；
- `vali_data`：每条独立训练路线使用的独立 Val 部分；
- 两部分分别完成 CVAT 复核和 05 导入，再由 `configs/finalize_val.yaml` 一次性合并；
- 配置中不写死“80%/40%”，以实际 included ROI 数量为准并在报告中统计；
- 每个 source 配置唯一 `dataset_id`，07B 自动为 crop/Palm/hand/原图分组 ID 添加 namespace；不依赖原文件名中的 `peak_` / `soar_`；
- 两人先共同复核一批校准样本；
- 正式标注可以各负责一部分；
- 所有双手、ignore 和困难样本交叉复核；
- 随机约 10% 普通样本由两人独立复核并仲裁；
- 两人使用同一份最终 Val JSONL、ignored manifest 和评测脚本。

共享评测集的“一损俱损”风险主要来自共同标注错误，而不是共享本身。通过双人校准、抽样交叉复核和版本冻结，比各自重复精标约 2000 张更节省时间且更可靠。

## 8. 当前仓库导入与筛选流程

人工复核后：

```text
CVAT reviewed XML
  → 05_import_cvat_xml.py
  → hand_landmarks_reviewed.jsonl + cvat_import_stats.json
  → 07B_finalize_evaluation_labels.py --config configs/finalize_val.yaml --split val
  → hand_validation_labels.jsonl
```

### 8.1 当前 05 的重要行为

- `ignore_for_training` 会导入为布尔字段；
- 多个 skeleton 只产生 warning，仍可能读取第一个；
- `no_hand + skeleton` 会产生 error，并按 negative 路径导入；
- 缺少 skeleton 且没有 `no_hand` 会按 negative 导入并警告；
- 缺失于 CVAT XML 的图片可能把 draft 补回并设 `needs_review=true`。

所以 `05` 的输出不能未经严格完整性门禁直接作为 Gold Val。

### 8.2 Val 的 07B 规则

07B 同时读取 Peak shared、Soar shared、当前路线 independent 三个 source 的 reviewed JSONL、真实 images 目录、行级 CVAT diagnostics 和 `cvat_import_stats.json`，然后：

1. manifest、CVAT image 和 reviewed row 一一覆盖且唯一；
2. 先把 `ignore_for_training=true` 移入 ignored 输出；
3. 对所有非 ignored 行执行严格结构校验；
4. 任一 multiple skeleton、no_hand conflict、缺失 handedness、点数错误、非有限/越界坐标或缺失 XML image 都要求回 CVAT 修正；Gold norm 必须在 `[0,1]`，crop px 必须在图内；
5. included 行默认要求 `palm_valid=true`，否则说明配置或数据漂移；
6. 人工确认的高分 Palm negative 合法，不能被训练集 heuristic 删除；
7. 不按 teacher/student confidence 或 `needs_review` 自动改写人工真值；
8. 任一 fatal error 时不覆盖上一个 canonical Val 文件。

Val 需要的是“结构严格、语义由人工决定”，不是训练集式的伪标签质量筛选。

## 9. 验证集输出与报告

建议输出：

```text
../autodl-tmp/val_merged/05_labels/hand_validation_labels.jsonl
../autodl-tmp/val_merged/05_labels/hand_val_ignored.jsonl
../autodl-tmp/val_merged/qc/finalize_val_report.json
```

报告必须包含：

- 输入 manifest/reviewed/config hash；
- manifest/XML/reviewed 覆盖率；
- included、ignored、invalid 数量；
- ignore 比例；
- positive、negative、Left、Right；
- `palm_valid=true, presence=false` hard-negative 数量；
- 所有错误原因；
- canonical 输出 SHA256；
- `review_context.csv` 的覆盖率与 hash（如启用分组报告）；
- 标注约定版本和评测脚本版本。

20%～30% ignore 应产生显著 warning 并显示剩余有效样本量，但不应由脚本自动把 Gold 变成其他标签。

## 10. 验收清单

- [ ] Val 与 Train/Test 的原始 session 不重叠；
- [ ] 所有 ROI 都来自正式 Palm detections；
- [ ] 所有 CVAT image 都已人工处理；
- [ ] 所有非 ignored positive 恰好一个 21 点 skeleton 和唯一 handedness；
- [ ] 所有非 ignored negative 只有 `no_hand`；
- [ ] 所有 ambiguous 双手 ROI 已 ignore；
- [ ] 所有双手 ROI 都完成 `crop_id → palm_det_id → detection` 唯一追溯并写入 sidecar；
- [ ] 所有目标明确的双手 ROI 均由 p0/p9、轴向和 bbox 掌区证据确定，只标 Palm anchor 对应手；
- [ ] p0/p9 分属两手、anchor 融合或匹配接近的 ROI 均已 ignore；
- [ ] ignored 数量、比例和原因已统计；
- [ ] 真正 ambiguous 比例已用于决定是否定向补录；
- [ ] Val 没有参与任何训练清单；
- [ ] presence 阈值只在 Val 上选择；
- [ ] Peak 与 Soar 使用相同 canonical Val hash；
- [ ] canonical 输出不存在结构 warning/error。
