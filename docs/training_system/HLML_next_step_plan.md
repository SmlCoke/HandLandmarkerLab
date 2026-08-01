# HLML 全国总决赛阶段：学生蒸馏缺口诊断与优化计划

更新时间：2026-07-31。

本文只记录当前问题、验证顺序和下一轮实验。通用操作方法仍以
[HLML_training_workflow.md](HLML_training_workflow.md) 和
[HLML_quick_start.md](HLML_quick_start.md) 为准。

## 0. 先纠正一个关键判断

上一版计划把主要问题描述为：

```text
Google MediaPipe 全流程漏检
→ 困难手势没有进入 pretrain positive
→ 学生从未见过这些姿态
```

这个描述不符合 HLMF 已经使用的主流程，不能再作为下一步工作的前提。

HLMF 当前默认自动标注链路实际上是：

```text
1280×720 原始无损 TIFF
→ AetherSign Palm Detector ONNX 检测手掌
→ 根据 bbox、p0、p9 构造旋转 Hand ROI
→ 保存 256×256、uint8、单通道灰度 ROI
→ 将同一灰度 ROI 复制为三通道 RGB
→ Google MediaPipe Hand Landmarker 输出 21 点
→ finalize
→ HLML geometry pretrain
```

因此，Google 官方完整 Hand Landmarker 管线中的 Palm Detector 漏检，并不等于 HLMF 没有为该图生成
伪标签。AetherSign Palm Detector 已经绕过了 Google Palm Detector，并能把更多正确 Hand ROI 送给
Google 的 Hand Landmark 阶段。

本地三组对照进一步证明了这一点：

| 原图 | 全程 Google | AetherSign Palm + Google Hand Landmark |
|---|---|---|
| `720x1280_8bit_20260718172426637_413.tiff` | 只标出 1 只手 | 两个 ROI 都得到合理 21 点，handedness 分数约 0.941/0.999 |
| `720x1280_8bit_20260718172635100_273.tiff` | 只标出 1 只手 | 两个 ROI 都得到合理 21 点，handedness 分数约 0.961/0.991 |
| `720x1280_8bit_20260718172640766_763.tiff` | 两只手均未标出 | 两个 ROI 都得到合理 21 点，handedness 分数约 0.977/0.998 |

对照目录：

```text
全程 Google：
D:\CICIEC\datasets\HandLandmarkerFab\test_data\google

AetherSign Palm + Google Hand Landmark：
D:\CICIEC\datasets\HandLandmarkerFab\test_data\02_roi_crops\hand_landmarks_visualization
```

结论：

> HLMF 已经实现“高召回外部 Palm proposal + 高精度 Google landmark 教师”。
> 下一步不应把这条路线当成待实现的新功能，也不能继续把全部误差归因于 Google 全流程漏检。

Google Hand Landmark 在外部 ROI 上仍可能弃权，但那只是剩余盲区。当前更紧迫的问题是：

> 大量高质量伪标签已经存在，为什么约 1.95M 参数的学生仍只能得到约 17～21 px 的独立集误差？

## 1. 修正后的根因优先级

### P0：学生没有在独立 ROI 上复现高质量教师

旧结果已经给出最直接的证据：

```text
MediaPipe 在自己成功检出的 Gold Val ROI 上：
mean ≈ 4.38 px

geometry 学生在同一教师成功子集上相对教师：
mean ≈ 17.03 px

geometry 学生在伪标签 Train 抽样 ROI 上：
mean ≈ 6.87 px

geometry 学生在完整独立 Gold Val 上：
mean ≈ 20.73 px
```

这说明：

1. 伪标签坐标本身不是 20 px 级错误；
2. 模型和反向传播并非完全失效，因为它能把训练 ROI 拟合到约 6.9 px；
3. 主要缺口发生在“从训练 ROI 泛化到独立 ROI”；
4. 即使只看教师成功的 ROI，学生也远没有达到教师精度；
5. 教师剩余漏检会进一步恶化困难集，但不能解释全部误差。

### P0：Train 与人工 Gold Val/Test 的标注风格不完全一致

pretrain positive 的目标全部来自 Google MediaPipe，而现有 Val/Test 中很大一部分来自人工标注。两者在
以下位置可能采用不同的视觉解释：

- 手腕点 `0` 应落在腕关节中心、手掌边缘还是前臂中轴；
- 拇指根部 `1/2` 的转折位置；
- 四指 MCP 根点 `5/9/13/17` 应靠近可见皮肤褶皱还是解剖关节中心；
- 遮挡手指的不可见关节应如何估计；
- 侧掌或握拳时，重叠关键点的深度投影位置；
- 模糊边界和出 ROI 点的处理。

这会产生两个问题：

1. 模型即使准确复现 Google，也会在人工 Gold 指标中被计为误差；
2. finetune 若混入另一种人工风格，可能把 geometry 从稳定的教师风格拉向不一致的人类风格。

已有结果说明该风险真实存在，但不能把全部误差归因于它：

```text
Google 与人工 Gold（仅教师成功的 836 ROI）：
mean = 4.38 px
median = 0.63 px
P90 = 12.83 px

geometry 学生在同一 836 ROI：
对人工 Gold mean = 17.57 px
对 Google 教师 mean = 17.03 px
```

`median=0.63 px` 表示多数点/样本非常接近；`P90=12.83 px` 又说明少数姿态或关键点存在明显教师错误、
人工误差或风格分歧。学生对两种标签仍都约为 17 px，因此：

> 标注风格不一致会污染 Val/Test 和 finetune 判断，但不是当前 17～21 px 误差的唯一主因。

下一步必须把“学生是否复现教师”和“学生是否符合人工定义”拆成两套指标，不能再用一份混合风格 Gold
回答两个问题。

### P0：高质量困难样本可能“存在，但没有被充分学习”

“数据集里有这张图”不等于“模型充分学过这种姿态”。仍需逐层检查：

```text
原始 TIFF 是否被 Palm 检出
→ ROI 是否成功生成
→ Google 是否输出 21 点
→ 是否通过 finalize 门控
→ 是否进入 merged labels
→ 是否进入 curated geometry labels
→ 是否被划到 teacher holdout
→ geometry 每个 epoch 期望抽到多少次
→ 实际训练中累计抽到多少次
```

当前训练集很大，并包含大量相邻视频帧。握拳、侧掌、数字 1、交叉和遮挡等困难姿态即使已经有高质量
伪标签，也可能在几十万普通 ROI 中占比很低；固定 epoch 随机抽样可能主要反复看到容易姿态。

### P0：学生结构和二维直接回归可能形成定位瓶颈

当前 v2 的重要特征是：

```text
输入：1×256×256 灰度
参数量：约 1.95M
连续下采样后：2×2×384
输出：用 2×2 Conv 直接回归 42 个二维坐标
```

这不是 Google MediaPipe Hand Landmark 的等价小模型。官方教师还使用更大的表示能力、三维坐标和
world landmark 等额外任务。当前学生在最终只保留 `2×2` 空间特征后直接回归 21 点，可能：

- 丢失细指尖、遮挡边界和相邻手指之间的空间信息；
- 在不确定姿态上趋向训练集平均手形；
- 用逐点 Huber 产生“总体 Loss 不大，但关键点聚成一团”的解；
- 难以从相似外观中区分握拳、手指交叉和侧向重叠。

当前 `v2.py` 中的 reparameterization 主要是 Conv/Depthwise Conv 与 BN 的等价融合，并不是
“训练时多个并行卷积分支、部署时合并成一个卷积”的完整 RepVGG 式增强。后者仍可作为实验，但不能在
诊断之前直接认定它一定有效。

### P0：训练 ROI 与真实运行 ROI 仍可能存在契约漂移

HLMF 标注阶段和 HLML canonical ROI 训练阶段使用的是同一份已保存 crop，这部分输入一致性较好；
HLMF 教师和 HLML 学生也都从单通道灰度 crop 出发，因此“教师看到彩色、学生只看到灰度”不是当前
主解释。

但端到端 PC/A1 推理会根据 Palm 输出重新裁 ROI。以下任一微小差异都会改变 Hand Landmarker 输入：

```text
Palm box 中心和尺寸
p0/p9
旋转方向
scale_x/scale_y
shift_x/shift_y
边界 padding
仿射采样坐标
双线性插值和取整
新旧 Palm 模型
```

因此必须分别回答：

- 模型在“训练时保存的同一 canonical ROI”上是否已经很差？
- 还是只有“运行时重新裁出的 ROI”上明显变差？

旧 Keras Gold Val 已经约 21 px，说明 ROI 运行时漂移不是唯一原因；但它可能继续放大板端误差。

### P1：数据有效多样性不足，而不是文件数量不足

20 万张相邻帧不能自动等价为 20 万个独立姿态。真正决定泛化的是：

- 不同人员；
- 不同手型；
- 左右手；
- 姿态与视角；
- 近、中、远距离；
- 亮暗和背景；
- 手指遮挡与交叉；
- Palm ROI 中的位置、旋转和尺度；
- 相互独立的录制序列。

后续审计必须报告“独立来源/序列/姿态桶”，不能只报告 ROI 总数。

### P1：multitask/finetune 可能遗忘 geometry

已有结果中 geometry 的 mean/P90 反而最好，multitask 最佳点过早出现，finetune Train 继续变好而
Val 基本不动。下一轮必须先解决 geometry 的学生—教师蒸馏缺口，再进入后续阶段。

### 已基本排除：ONNX 或厂商转换把正确模型算坏

Keras Val 本身已经较差；Keras 到 deploy graph 的融合误差、ONNX parity、厂商转换和上板均已通过。
板端算子限制可能间接限制了我们能选择的网络结构，但不是“导出后数值突然错误”。

## 2. 下一次长训练前必须完成的五个 P0 诊断

这四项应由程序自动完成。人工只需查看报告和少量可视化。

### 2.1 P0-A：同一 ROI 上的教师—学生配对审计

程序对两个严格隔离的池分别使用完全相同的 256×256 canonical ROI，同时运行 Google 教师和当前学生：

```text
evaluation pool：
未进入 Train 的 teacher holdout + 固定 Gold Val，只用于正式指标

diagnostic-train pool：
本来就属于 Train 的伪标签 ROI，只用于训练成员审计和 hard-overfit
```

每个 ROI 保存：

```text
dataset_id
source_group_id
source_sequence_id
crop_id
crop_path
image_sha256
proposal_backend
palm_score
ROI 在原图中的面积比例
亮度/对比度
教师是否成功
教师 21 点
学生 21 点
student_teacher_mean_px
student_gold_mean_px（有 Gold 时）
student_spread_ratio
pose_bucket
distance_bucket
```

自动输出三类核心样本：

```text
A: EXTERNAL_PALM_TEACHER_OK_STUDENT_OK
B: EXTERNAL_PALM_TEACHER_OK_STUDENT_BAD
C: EXTERNAL_PALM_TEACHER_ABSTAIN
```

其中 B 类是当前最有价值的数据：

- 外部 Palm 已经找到手；
- Google 已经给出高质量 21 点；
- 不需要人工重新标点；
- 学生却仍然预测错误或塌缩；
- 可直接用于 hard-positive mining 和结构诊断。

不能再把 B 类称为“教师漏检”；它是明确的“学生蒸馏失败”。

### 2.2 P0-B：训练成员关系与实际抽样曝光审计

对 B 类样本，以及从各来源随机抽取的一批样本，自动追踪完整 lineage：

```text
raw image
→ palm_detections.jsonl
→ hand_roi_crops_manifest.jsonl
→ hand_landmarks_autolabel_draft.jsonl
→ HLMF finalized/merged labels
→ HLML curated geometry labels
→ teacher holdout/train assignment
→ sampler cell
→ sampling_weight
→ 每 epoch 期望抽样次数
→ 整个训练预计/实际抽样次数
```

最终报告必须区分：

| 状态 | 含义 | 下一步 |
|---|---|---|
| `NOT_PROPOSED` | AetherSign Palm 没产生 ROI | 改 Palm/多尺度 proposal |
| `TEACHER_ABSTAIN` | 有 ROI，但 Google 不出 21 点 | 进入剩余教师盲区 |
| `REJECTED_BY_FINALIZE` | 有正确标签但被门控排除 | 检查门控是否误杀 |
| `HOLDOUT_ONLY` | 被完整隔离为 holdout | 正常，不加入 Train |
| `TRAIN_LOW_EXPOSURE` | 进入 Train，但抽样过少 | 自动提高困难桶份额 |
| `TRAIN_WELL_EXPOSED_STUDENT_BAD` | 已充分训练仍预测差 | 优先检查结构/损失 |

同时输出：

- 每来源的原始图数、ROI 数和 positive 数；
- 每个完整序列的帧数与抽样份额；
- 姿态桶、距离桶、亮度桶的数量和期望抽样次数；
- 每个 epoch 的 unique ROI 数、重复率和最大来源占比；
- B 类样本在训练集中所占比例。

### 2.3 P0-C：困难伪标签小集合过拟合测试

这是区分“数据问题”和“模型问题”最快的实验。

自动从 `diagnostic-train pool` 的 B 类中选择 128、256 或 512 个高质量、非重复困难 ROI：

- Google 21 点可视化正确；
- 多来源/多序列；
- 覆盖握拳、数字 1、侧掌、手指下垂、遮挡和交叉；
- 不含双手混入或严重裁断；
- 暂时关闭数据增强；
- 只训练 landmark head；
- 对小集合反复训练，直接评估同一集合。

不得从 Gold Val 或 teacher holdout 抽取这个小集合。hard-overfit 是训练能力诊断，不得污染任何正式
evaluation pool。

建议门槛：

```text
mean ≤ 3～5 px
P90 ≤ 8 px
几乎无塌缩
```

解释：

- 如果当前 v2 连 128～512 个高质量困难 ROI 都无法拟合，停止新增几十万数据；优先修正模型 head、
  损失、坐标实现或容量。
- 如果小集合可以拟合，但独立困难 holdout 仍很差，主要问题是多样性、抽样曝光、正则化或结构泛化。
- 如果小集合拟合结果在 Keras 正常、ONNX/A1 异常，再检查导出和板端；现有证据认为这种可能性较低。

正式 geometry smoke 只能证明训练程序能运行；这个 hard-overfit gate 才能证明学生能学会最关心的困难姿态。

### 2.4 P0-D：HLMF、HLML 与 A1 的 ROI 像素一致性

选 50～100 个固定 Palm 检测结果，使用完全相同的 bbox、p0、p9，分别生成：

```text
HLMF 保存的 ROI
HLML Python runtime 重建的 ROI
A1/C++ runtime 导出的 ROI
```

逐像素报告：

```text
shape/dtype
SHA256
mean absolute pixel difference
max absolute pixel difference
超过 1/2/4 灰度级的像素比例
```

同时比较 ROI 四角、中心、宽高和旋转。若像素不一致，必须先修 ROI contract，不能用训练继续掩盖。

### 2.5 P0-E：Google 与人工 Gold 的标注风格审计

在 Google 成功输出 21 点、同时又有人工 Gold 的 ROI 上，逐点比较两套标签。不能只计算一个总体 mean，
必须输出：

```text
每个 landmark id 的 mean/median/P90 距离
每个 landmark id 的平均 dx/dy（是否存在固定方向偏移）
腕点、拇指根、四指 MCP、PIP/DIP、指尖的分组误差
按握拳/张掌/侧掌/遮挡/交叉分组
按 Left/Right 分组
按标注人员分组
按来源和录制序列分组
Google 与人工标签的骨长、掌宽和 spread 差异
```

自动生成三层可视化：

```text
green：Google
red：human Gold
yellow line：同一关键点的风格差异向量
```

人工只需查看每个高差异桶的少量样本，并把原因标记为：

```text
HUMAN_STYLE
HUMAN_ERROR
TEACHER_ERROR
AMBIGUOUS_OCCLUSION
ROI/PROJECTION_ERROR
```

建议再抽取约 100～200 个 ROI，由两名标注者独立标注。程序计算：

- 人—人误差；
- 人—Google 误差；
- 不同关键点的标注方差。

若人—人误差与人—Google 误差接近，说明该点本身定义模糊，不应把单个人工标签当成绝对真值；若两名
人工高度一致而 Google 系统性偏移，才可判定为教师风格或教师错误。

## 3. 数据优化：从“继续加图”改为“自动挖掘学生不会的高质量伪标签”

### 3.1 继续使用现有伪标签

现有 AetherSign Palm + Google Hand Landmark positive 是高价值训练资产，不应删除，也不需要人工重标。

下一轮 positive 分成三层：

| 层 | 定义 | 用途 |
|---|---|---|
| `pseudo_easy` | 教师成功、当前学生误差较低 | 保持常见姿态和基本泛化 |
| `pseudo_hard` | 教师成功、当前学生误差高或塌缩 | geometry 的重点训练对象 |
| `teacher_abstain` | 外部 Palm 已产生 ROI，但 Google 仍不输出 21 点 | 暂不进入 geometry positive/negative |

这里的 `pseudo_hard` 不表示标签质量差，而表示“标签可靠，但学生当前不会”。

### 3.2 自动生成 hard-positive bucket

程序根据教师 21 点和学生预测自动派生：

- 学生—教师 mean/P90；
- 关键点 spread；
- 五指伸展/弯曲模式；
- 指尖之间距离；
- 手掌朝向近似；
- 遮挡/交叉近似；
- ROI 中手部尺度；
- Palm score；
- 来源、人员和完整序列。

不要依靠文件夹名作为唯一姿态标签。文件夹名可作为先验，但最终分桶应由关键点几何和来源元数据共同决定。

### 3.3 geometry 采用“容易样本打底 + 困难样本保底”的分层采样

首轮建议从保守比例开始：

```text
pseudo_easy: 70%～80%
pseudo_hard: 20%～30%
```

在 `pseudo_hard` 内再次保证：

- 不同来源/序列都有上限；
- 拳头、侧掌、数字 1、下垂、遮挡、交叉等桶有最低份额；
- 近、中、远距离有最低份额；
- 不允许单个连续视频支配 epoch；
- 不复制图片或 JSONL，通过 sampler 权重实现。

比例应根据 P0-B 的实际数量修正。hard 数量很少时不能无限重复，以免记住少数人的手。

### 3.4 自动去除连续近重复

每个序列使用感知哈希、教师关键点距离和时间间隔自动聚类。相邻、姿态几乎不变的帧只保留少量代表，
其余仍可留在数据仓库，但不应获得与独立场景相同的采样权重。

审计报告同时给出：

```text
物理文件数
有效独立序列数
去近重复后的代表 ROI 数
每个姿态桶的独立序列数
```

### 3.5 剩余 teacher-abstain 的优先级降低

AetherSign Palm 已经恢复了大量 Google 全流程漏检，因此不能再把
`full Google miss` 直接当作待人工标注集合。

真正需要后续处理的是：

```text
AetherSign Palm 已产生合理 ROI
且 Google Hand Landmark 在该 ROI 上仍弃权
且人工/时序证据证明 ROI 中确实有手
```

在完成 P0-A～D 前，这些样本只标为 `teacher_abstain`，既不作 positive，也不作 negative。之后再决定是否：

- 多尺度/轻微平移旋转后重试教师；
- 使用相邻帧一致性恢复；
- 少量人工 Gold；
- 由更强的离线教师重新标注。

### 3.6 为所有标签记录 annotation style

后续 HLMF/HLML 聚合时，每条 positive 至少记录：

```text
label_origin: mediapipe | human | mediapipe_human_corrected
annotation_style: mediapipe_v1 | human_legacy_v1 | project_consensus_v1
teacher_model_id
teacher_detected
human_reviewed
human_modified_landmark_ids
annotator_id（人工时）
```

训练、finetune、Val 和 Test 报告都必须按 `annotation_style` 分组。禁止将不同风格标签混合后只输出一个
总体 mean。

对 finetune Gold 也执行同样审计。若 geometry 对 Google-style 指标较好，而 finetune 后
Google-style 退化、human-style 略好，说明发生了明确的风格迁移，而不一定是模型总体能力提升。

现阶段建议把 Google/MediaPipe 风格作为主要蒸馏目标，原因是：

- 绝大多数 pretrain 标签已经采用该风格；
- Google 在成功 ROI 上通常稳定且准确；
- 重新人工统一十万级 pretrain 不现实；
- Gloss Translator 更需要时间稳定、前后一致的骨架，而不是不同标注者各自理解的“绝对解剖点”。

人工只修正明显错误和教师弃权，不要对所有可接受 Google 点进行主观微调。确需修正时，应在统一的
`project_consensus_v1` 标注规范下进行，并记录修改了哪些 landmark。

## 4. 模型优化的进入顺序

### 4.1 第一对照：不改模型，只改 hard-positive 曝光

固定同一初始化、训练时长、增强和 Val：

```text
E0：当前 v2 + 当前 sampling
E1：当前 v2 + pseudo_easy/hard 分层采样
```

如果 E1 显著改善教师—学生 holdout 和困难 Gold Val，说明现有模型还有未释放能力，继续扩大模型不是
第一优先级。

### 4.2 第二对照：训练期空间辅助监督

若 E1 仍不足，但 hard-overfit gate 可以通过，优先给中间高分辨率特征增加只在训练时使用的辅助 head：

- 21 个低分辨率 heatmap；
- 每个关键点相对手腕/掌心的 offset；
- 教师提供的 z 坐标；
- 教师 world landmark 或骨向量；
- 中间层 deep supervision。

部署输出仍保持：

```text
42 个二维坐标 + presence + handedness
```

辅助 head 在导出前删除，不增加板端算子和推理耗时。重点不是再次加入简单 spread 约束；旧 structure
实验已经证明“强迫骨架展开”不等于“定位正确”。辅助任务必须直接帮助空间定位或三维姿态判别。

### 4.3 第三对照：训练时多分支、部署时融合

可实现真正的训练期多分支卷积，例如：

```text
3×3 Conv+BN
+ 1×1 Conv+BN
+ identity BN（尺寸和通道允许时）
→ 训练结束后精确融合为单个 3×3 Conv
```

注意：

- 当前 Conv+BN folding 仍需保留；
- 多分支融合必须有逐层和端到端数值测试；
- depthwise 分支需单独验证；
- 融合后只能出现厂商工具链已支持的算子；
- 使用相同数据与相同训练预算和 E1 比较。

### 4.4 第四对照：保留更高空间分辨率

如果前述方法仍无法降低困难姿态误差，调整最后两级下采样，让 landmark head 在 `4×4` 或更高空间特征
上预测，再设计为板端支持的 Conv-only 输出。

这项修改比单纯增加重复 block 更可能直接改善指尖定位，但会增加计算量，必须同时测：

- 参数量；
- MACs；
- ONNX；
- 官方转换；
- A1 P50/P90/P95 延迟。

### 4.5 最后才是整体扩宽

纯增大通道数可能提高拟合能力，也可能加重训练集记忆。只有满足以下条件才进入：

- hard-overfit gate 能通过；
- E1 与辅助监督仍不能达到目标；
- 增宽模型的独立 holdout 确实改善；
- 融合后模型和延迟仍满足 A1。

## 5. 新 Palm Detector 与 Hand Landmarker 联合适配

重新训练 Palm 后，anchor 变化本身不是 Hand ROI 域变化；真正影响 Hand Landmarker 的是解码后的 box、
p0/p9 和 ROI 参数。

新 Palm 冻结后，程序自动执行：

1. 在同一批 TIFF 上运行旧 Palm 和新 Palm；
2. 配对两者的 box、p0/p9；
3. 比较中心、尺度、旋转和召回率；
4. 分近、中、远距离统计；
5. 分别从旧 Palm 和新 Palm proposal 生成真实 ROI；
6. 对新 Palm ROI 再运行 Google Hand Landmark 生成伪标签；
7. 将“同一原图、不同 Palm ROI”保留为不同但有关联的训练记录；
8. 用实测差异确定 ROI jitter，而不是主观猜平移/缩放范围。

建议的最终 geometry 数据：

```text
旧 AetherSign Palm ROI + Google 21 点
+ 新 Palm ROI + Google 21 点
+ 少量由实测新旧差异生成的 ROI jitter
```

这样既保留历史数据的多样性，又直接适配最终部署 Palm 的 ROI 分布。

## 6. 重建为双轨 Val/Test

### 6.1 不删除现有人工 Gold

现有 Val/Test 已经投入大量人工成本，并且能暴露 Google 教师与人工语义之间的差异。应将其冻结为：

```text
human_legacy_val_v1
human_legacy_test_v1
```

不覆盖文件、不修改原标签、不继续反复查看 Test。它们的用途从“唯一绝对真值”调整为：

- 检查学生是否符合已有人工定义；
- 观察不同版本是否只学会 Google 风格；
- 计算 annotation-style gap；
- 保持与历史模型可比。

### 6.2 新建纯教师一致性 holdout

从与 Train 按人员、来源和完整序列隔离的原始无损 TIFF 生成：

```text
AetherSign/final Palm
→ canonical 256×256 ROI
→ Google MediaPipe Hand Landmark
→ 只保留教师成功且通过自动质量门控的 ROI
```

分别冻结：

```text
teacher_style_val_v1
teacher_style_test_v1
hard_teacher_style_val_v1
hard_teacher_style_test_v1
```

这些标签不做人工位置微调，只做少量质量抽检。用途是回答：

> 学生能否在未见过的来源/人员/序列上复现训练教师？

它不是独立于教师的“绝对准确率”，只能称为 student-teacher consistency 或 distillation fidelity。

### 6.3 新建 MediaPipe 预标注、人工只纠错的共识集

用户提出的重做方式是可行的，但应该作为第三套评估，而不是覆盖前两套。

流程：

1. 先按人员、来源和完整序列冻结 Val/Test split；
2. 对所有 canonical ROI 运行 Google Hand Landmarker；
3. Google 成功时，将 21 点作为 CVAT 初始标注；
4. 人工只修正明显错误，不为轻微个人偏好移动关键点；
5. Google 弃权时，由人工按统一规范补齐；
6. 每次人工修改记录 `human_modified_landmark_ids`；
7. 一名标注者完成后，由另一人只复核高风险样本；
8. finalize 后冻结 SHA，Test 不再用于调参。

形成：

```text
consensus_val_v1
consensus_test_v1
```

其中每条记录保留：

```text
MEDIAPIPE_UNCHANGED
MEDIAPIPE_CORRECTED
HUMAN_FROM_SCRATCH_TEACHER_ABSTAIN
```

三类必须分别报告指标，否则少量人工困难样本会与大量教师成功样本混成一个不可解释的总数。

人工标注规范至少明确：

- 腕点 `0` 的统一落点；
- `1/2/5/9/13/17` 根关节定义；
- 遮挡点的估计方式；
- 侧掌/握拳重叠点的处理；
- 可允许出 ROI 的条件；
- 模糊不可判定时使用 ignore，而不是猜点；
- 以邻近教师成功帧作为风格参考。

### 6.4 固定评估矩阵

下一轮至少固定以下评估：

| 评估集 | 标签风格 | 目的 |
|---|---|---|
| `teacher_style_val/test_v1` | 纯 Google | 衡量教师蒸馏与泛化 |
| `hard_teacher_style_val/test_v1` | 纯 Google 困难 positive | 衡量已有困难伪标签是否学会 |
| `consensus_val/test_v1` | Google 预标注 + 人工仅纠错 | 衡量较可靠的项目目标 |
| `human_legacy_val/test_v1` | 历史人工 | 保持历史可比并观察风格偏差 |
| `fixed_palm_roi_parity_v1` | 同一 Palm/ROI 输入 | 区分模型误差与 ROI 漂移 |

若时间有限，优先级是：

```text
P0 teacher_style_val
P0 hard_teacher_style_val
P0 保留 human_legacy_val
P1 consensus_val
P1 所有 Test 只在最终候选冻结后运行
```

### 6.5 报告方式

每套至少输出：

```text
mean/median/P90/P95 pixel error
PCK@0.05/PCK@0.10
collapse rate
presence/handedness
按来源、序列、姿态、距离、亮度分组
按 annotation_style/label_origin 分组
按 landmark id 分组
学生相对教师误差
```

模型选型时不得把不同标签风格的数值直接合成一个平均值。推荐同时报告：

```text
teacher-style score
consensus score
legacy-human score
annotation-style gap
```

端到端序列还需输出：

```text
Palm frame recall
longest miss streak
usable ROI rate
landmark collapse streak
Gloss accuracy/confusion matrix
A1 latency
```

### 6.6 checkpoint 与 early stopping 不能再只监控混合风格 Val

过去 geometry 使用 Google-style pseudo 训练，却用包含大量历史人工风格的 Val 选择 best checkpoint。
如果两种风格在腕点/MCP 等位置存在系统差异，可能出现：

```text
student-teacher 仍在改善
但 human-style Val 已停止改善
→ ReduceLROnPlateau/early stopping 提前触发
→ 保存的 best 并非最佳教师蒸馏 checkpoint
```

下一轮每个 epoch 同时评估三条曲线：

```text
teacher_style_val_landmark_mae
hard_teacher_style_val_landmark_mae
consensus_or_human_legacy_val_landmark_mae
```

阶段监控建议：

| 阶段 | checkpoint 主监控 | 不可退化门控 |
|---|---|---|
| geometry | `teacher_style_val_landmark_mae` | hard teacher-style 与 consensus/human-legacy |
| multitask | teacher-style landmark + presence/handedness | geometry landmark 不得明显遗忘 |
| finetune | 与启用 Gold 风格一致的 consensus/human 指标 | teacher-style 不得大幅退化 |

不能用 Test 选择 checkpoint。训练结束后保留少量候选 checkpoint，再依据预先写好的双轨 Val 规则选一个，
避免看到 Test 后反向挑 epoch。

## 7. 十天执行计划

### Day 1：完成 P0 诊断，不启动长训练

程序：

- 生成教师—学生同 ROI 配对报告；
- 输出 A/B/C 三类样本；
- 生成完整训练 lineage 和 sampling exposure；
- 构建 128/256/512 hard-overfit 集；
- 导出 HLMF/HLML/A1 ROI parity 输入与报告；
- 在教师成功的现有 Gold ROI 上生成逐关键点 annotation-style 报告。

人工：

- 抽看 B 类和 C 类各约 50～100 张；
- 确认 B 类教师点确实正确；
- 检查 ROI 是否单手、是否裁断；
- 抽看 Google/human 高分歧样本，区分人工风格、人工错误和教师错误；
- 根据 hard-overfit 结果决定进入数据路线还是结构路线。

### Day 2：固定双轨 Val，并实现自动 hard-positive curation

程序：

- 自动计算 student-teacher error；
- 自动生成 pose/distance/source/sequence bucket；
- 自动做连续近重复降权；
- 在 teacher holdout、Gold Val 与 Train 之间做 SHA/序列泄漏检查；
- 按完整人员/来源/序列冻结 `teacher_style_val_v1` 和 `hard_teacher_style_val_v1`；
- 将旧人工 Val 冻结为 `human_legacy_val_v1`；
- 让训练日志同时输出三套 Val，并按阶段选择 checkpoint 主监控；
- 生成 E0/E1 sampling audit。

人工只查看分布和少量可视化，不逐张标点。`consensus_val_v1` 可由队员并行使用 Google 预标注、人工只
纠错的方式制作，不阻塞 E0/E1。

### Day 3：运行 E0/E1 geometry 对照

要求：

- 相同初始化；
- 相同 epoch size；
- 相同训练时长；
- 相同增强；
- 只改变 hard-positive sampling；
- 每个 epoch 同时记录 teacher-style、hard teacher-style 和 human-legacy Val；
- 不把三种标注风格合成一个总体 mean。

只有 E1 在独立指标上改善才进入 multitask。

### Day 4：根据结果分流

```text
hard-overfit 失败
→ 立即做模型/head/损失检查，不再堆数据

hard-overfit 成功，E1 改善
→ 继续扩大高质量、多来源 hard pseudo

hard-overfit 成功，E1 不改善
→ 做训练期空间辅助监督

canonical ROI 好、端到端 ROI 差
→ 优先修 Palm/ROI contract
```

### Day 5～6：只做一个结构变量

优先实验：

```text
E2 = E1 + 训练期 heatmap/offset/z/world landmark 辅助监督
```

如果实现成本或工具链风险更低，也可选择：

```text
E2 = E1 + 真正多分支可融合卷积
```

一次只能改一个主要变量。E2 必须先通过 hard-overfit、smoke、融合 parity、ONNX 和转换预检。

### Day 6～7：新 Palm 域适配

Palm 团队冻结候选模型后：

- 生成旧/新 Palm 对照；
- 统计近、中、远召回；
- 生成新 Palm 真实 ROI；
- 运行 Google Hand Landmark；
- 将高质量 positive 加入 E1/E2 的数据层；
- 根据旧/新 ROI 差异自动配置 jitter。

### Day 7～8：geometry winner 的正式训练

只选择已通过以下门槛的配置：

- hard-overfit 成功；
- teacher-style Val 明显改善；
- hard teacher-style Val 明显改善；
- consensus Val 或 human-legacy Val 不出现无法解释的明显退化；
- collapse rate 下降。

不要同时启动大量无解释的学习率、Gold/pseudo 比例和 structure 扫描。

### Day 8～9：multitask 与一次 finetune

只对 geometry winner 运行：

```text
geometry winner
→ multitask
→ 一次 Gold + replay finetune
```

每阶段都与 geometry winner 比较 landmark 指标。若 multitask/finetune 明显遗忘，就保留更早阶段作为
Hand Landmarker winner，不能按“阶段越晚越好”选模型。

finetune 报告必须按 `annotation_style` 拆分。如果使用 legacy human Gold，需同时观察 teacher-style
是否被拉坏；优先新增 `mediapipe_human_corrected` 风格的 Gold，而不是继续混入未审计的人工风格。

### Day 9～10：ONNX、A1、Gloss 与冻结

- 固定 ROI 比较 Keras、deploy graph、ONNX、A1 原始输出；
- 使用最终 Palm 做完整视频序列测试；
- 统计漏检 streak 和 landmark collapse streak；
- 用真实学生骨架重新训练/校准 Gloss Translator；
- 加入缺帧、presence 和关键点噪声模拟；
- 冻结模型、Palm anchor/decoder、ROI contract、阈值、commit 和 SHA；
- 保留分赛区冠军模型作为回退版本。

## 8. 决策门槛

### 8.1 是否继续增加数据

满足以下条件才继续扩大伪标签：

- hard-overfit gate 成功；
- 新数据来自新的人员/序列/姿态/距离，而不是相邻重复帧；
- Google 在 AetherSign/new Palm ROI 上的 21 点抽检正确；
- hard bucket 的独立序列数确实增加；
- E1 比 E0 的 teacher-style 独立 holdout 更好。

否则继续增加普通 ROI 很可能只增加训练时间。

### 8.2 是否修改模型

出现以下任一情况，应把结构优化升为 P0：

- 128～512 个困难高质量伪标签都无法拟合到 3～5 px；
- `TRAIN_WELL_EXPOSED_STUDENT_BAD` 占比高；
- teacher-style holdout 长期停留在约 15～17 px；
- 塌缩主要集中在空间细节复杂的姿态；
- 增加 hard exposure 后 Train 改善、holdout 不改善。

### 8.3 是否进入 multitask/finetune

geometry 至少应同时满足：

- teacher-style holdout 相对当前基线明显下降；
- hard teacher-style holdout 的 mean/P90 或 PCK 有可重复改善；
- consensus/human-legacy Val 的变化能够用逐关键点风格报告解释；
- collapse rate 下降；
- 固定 infer 中握拳、侧掌、数字 1、遮挡的改善可见。

若 geometry 没有改善，multitask 和 finetune 不能被当作自动修复步骤。

### 8.4 如何解释双轨指标

| 结果 | 判断 |
|---|---|
| teacher-style 明显改善，consensus 同时改善 | 真正有效，优先继续 |
| teacher-style 改善，legacy-human 不变 | 多半是蒸馏改善，仍可继续 |
| teacher-style 改善，legacy-human 小幅退化 | 先看分歧是否集中于腕点/MCP 风格，不能立即否定 |
| teacher-style 很好，consensus 明显差 | 学生只复制了教师错误，需要 corrected Gold |
| teacher-style 和 consensus 都差 | 不是标注风格能解释，继续查模型/数据 |
| legacy-human 改善，teacher-style 明显退化 | finetune 发生风格迁移，需结合 Gloss 和项目定义选型 |

最终 Test 必须同时报告 teacher-style、consensus 和 legacy-human，不能只挑对某个模型最有利的一套。

## 9. 人与程序分工

人工负责：

- 录制真正多样的无损 TIFF；
- 抽查少量 B 类/C 类和教师 overlay；
- 查看少量 Google/human 高分歧可视化并判定原因；
- 制作 consensus Val/Test 时只修明显错误和教师弃权，不按个人偏好重画全部点；
- 查看自动报告并按门槛做决策；
- 完成最终转换、上板和真实手势/Gloss 测试；
- 如仍有余力，对少量样本做双人独立标注以估计人—人误差。

程序负责：

- 外部 Palm proposal、ROI 和 Google 伪标签；
- 教师—学生同 ROI 配对；
- Google/human 逐关键点标注风格审计；
- teacher-style/consensus/legacy-human 双轨数据构建与分组评估；
- 训练 lineage 和 sampling exposure；
- hard-positive 自动挖掘与分桶；
- 连续近重复检测和降权；
- hard-overfit gate；
- Train/Val/Test/holdout 泄漏检查；
- HLMF/HLML/A1 ROI parity；
- geometry/multitask/finetune 训练与分阶段评估；
- deploy graph/ONNX/A1 数值一致性；
- 新旧 Palm ROI 域比较；
- 最终端到端序列指标。

## 10. 最终优先级

```text
P0 同一 ROI 的教师—学生误差审计
P0 Google/human 逐关键点标注风格审计
P0 teacher-style 与 human/consensus 双轨 Val
P0 geometry checkpoint 改用 teacher-style 主监控并设置人工/共识门控
P0 高质量伪标签的训练成员关系和实际曝光审计
P0 困难伪标签 hard-overfit gate
P0 HLMF/HLML/A1 ROI parity
P0 teacher-success/student-fail 的自动 hard-positive 采样

P1 训练期 heatmap/offset/z/world landmark 辅助监督
P1 真正的多分支可融合卷积
P1 新 Palm 真实 ROI 伪标签与实测 jitter
P1 防止 multitask/finetune 遗忘 geometry

P2 剩余 teacher-abstain 的自动恢复或少量 Gold
P2 整体扩宽部署模型
P2 OLED/GPIO 等展示扩展
```

当前最重要的认识是：

> “全程 Google 漏检”并不代表 HLMF 没有标签；AetherSign Palm + Google Hand Landmark 已经恢复了大量
> 高质量伪标签。现在首先要追查的是：这些标签是否真正进入训练、困难姿态是否得到足够曝光，以及当前
> 学生结构是否有能力把高质量教师蒸馏下来。同时必须把 Google 风格和人工风格分开报告，避免把标注
> 定义差异误判成模型误差，也避免用纯教师标签掩盖教师本身的错误。
