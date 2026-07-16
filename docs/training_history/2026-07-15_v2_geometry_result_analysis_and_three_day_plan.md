# v2 geometry 完整训练结果分析与三天优化执行手册

> 历史文档：本文保留 r2/r3 分析证据，不再作为当前操作入口。后续统一按 [HLMF + HLML Hand Landmarker 完整训练流程 v1.0](../training_system/end_to_end_training_workflow_v1_0.md) 执行。

初版日期：2026-07-15

操作手册更新：2026-07-16

对象：`v2-pretrain-r2` 结果分析，以及修复 Peak dark 后的 `v2-pretrain-r3` 后续操作
结论性质：分析、人工操作说明和下一步执行建议

## 当前进度和本文边界

截至 2026-07-16：

- `peak_train_0714_dark` 的 ROI 交付错误已经修复；
- HLMF 已重新完成 `finalize_train_pretrain`；
- HLML 已重新完成 `pretrain-curate` 并通过 geometry smoke test；
- r3 curated geometry 共包含 59,952 个 positive，其中 `peak_train_0714_dark` 实际进入 7,428 个；相对 r2 正好增加 7,428 个；
- r3 geometry 由操作者按现有正式流程训练；本手册不讨论进程管理。

因此，本文前半部分保留 r2 的问题证据，用于解释为什么需要 r3；后半部分从“r3 geometry 正常完成”开始，详细说明负例复核、multitask、Gold 数据制作和 finetune 的操作。

必须先明确一条边界：

- **HLMF（HandLandmarkerFab）** 是上游数据制作系统，仓库位于 `D:\CICIEC\datasets\HandLandmarkerFab`。它负责原图→Palm→256×256 Hand ROI→MediaPipe 伪标签→CVAT→Gold→07A finalize。
- **HLML（本仓库）** 是下游训练系统，仓库位于 `D:\CICIEC\MediaPipe\HandLandmarkerLab`。它负责 pretrain curate、geometry、人工确认负例、multitask、评估、推理和导出。
- `make finalize_train_pretrain`、`make finalize_train_finetune` 是 **HLMF 命令**；`make pretrain-curate`、`make pretrain-multitask` 是 **HLML 命令**。不要在错误的仓库执行。
- 当前 HLML **没有正式 finetune YAML、Make 目标、finetune smoke/eval/infer/export 入口**。底层训练器支持 finetune，不等于现在已有可交付的 `make finetune`。本文会把已经能执行的步骤和仍需实现的入口明确分开。

## 术语解释

下面这些词会在全文反复出现。第一次阅读时应先理解本节。

| 术语 | 本项目中的准确含义 |
|---|---|
| ROI（Region of Interest，感兴趣区域） | 从 1280×720 原图按 Palm 检测结果旋转、平移和放大后裁出的 256×256 小图。HLML 的 Hand Landmarker 只接收这种小图，不直接接收整张原图。 |
| Palm Detector | 上游手掌检测器。它先在原图中找到手并给出构造 ROI 所需的框和方向点。Palm 没给出 ROI 时，Hand Landmarker 根本没有机会预测 21 点。 |
| Hand Landmarker | 本项目实际训练的模型。输入一个 256×256 ROI，输出 21 个关键点、`hand_flag` 和 handedness。 |
| landmark / 21 点 | MediaPipe 约定的 21 个手部关键点：手腕 1 点，拇指 4 点，其余四指各 4 点。模型输出坐标使用 ID 0～20。 |
| teacher（教师模型） | 生成自动标签的 Google MediaPipe Hand Landmarker。它不是本项目最终部署模型。 |
| student（学生模型） | HLML 正在训练的自有 Hand Landmarker，即 r2/r3 模型。 |
| pseudo label（伪标签） | teacher 自动生成、未经逐图人工确认的标注。它可以用于大规模 pretrain，但不等同于真值。canonical 中对应 `annotation_provenance=mediapipe_pseudo`、`supervision_tier=pseudo`。 |
| Gold / Gold label（人工金标准） | 人在 CVAT 中逐图确认并修正后的标签，作为本项目当前能获得的高质量真值。canonical 中对应 `annotation_provenance=human_gold`、`supervision_tier=gold`。 |
| CVAT | 浏览器中的人工标注平台。本项目把 HLMF 04 生成的初始 XML 导入 CVAT，由人修正后再把 CVAT for images 1.1 XML 交给 HLMF 05 导回 JSONL。CVAT 不负责生成 HLML 可训练 canonical。 |
| Gold ROI | **不是新裁一张图片，也不一定是正样本。**它是“某个训练 source 中已经存在的 ROI 图片 + 同一 `crop_id` 的原 manifest 记录 + CVAT 人工复核后的 label 记录”。Gold 可以是有手、无手或 `ignore_for_training`。 |
| candidate（候选） | 尚未得到最终训练资格的样本。例如 negative candidate 只是“teacher 没输出手、疑似背景”，必须人工确认后才能成为 confirmed negative。 |
| positive（正样本） | ROI 中存在目标手；用于训练 21 点。 |
| negative（负样本） | ROI 中明确没有手；用于让 `hand_flag` 学会拒绝假 ROI。只有人工确认后才是 true/confirmed negative。 |
| teacher abstain（教师放弃输出） | teacher 没有输出一套可用的 21 点。它只表示“Google 没有给出结果”，不表示图中一定无手。 |
| Peak teacher-abstain | 不是现成字段名。分析中它指 Peak 场景里“人工 Gold 确认有手，但 Google 没输出”的样本；在 Train 中最接近的可操作对象，是 `NEG_*_CANDIDATE` 中肉眼实际看见手的图片。 |
| Peak pseudo positive | Peak 来源中 teacher 给出了 21 点、`hand_presence.present=true` 的伪标签正样本。它有手，但点位仍可能偏。 |
| student–teacher disagreement（学生—教师分歧） | 对同一 ROI，学生预测与 teacher 伪标签差很多，例如 21 点平均距离很大、一个判有手另一个判无手。当前 HLMF/HLML **没有现成的分歧排序命令**，不能把它写成已经可运行的筛选步骤。 |
| hard positive（困难正样本） | 明确有手，但因暗光、遮挡、手很小、旋转大、边缘、弯指等原因难以检测或标注的 ROI。 |
| hard negative（困难负样本） | Palm 很像检测到手、但人工确认 ROI 中没有手的背景。它比随机纯背景更适合训练 `hand_flag`。 |
| `POS_RUNTIME` | `palm_valid=true` 且有手：部署时正常由 Palm 阈值内 proposal 产生的正样本。 |
| `POS_LOW_PALM` | `palm_valid=false` 但有手：Palm 分数低于正式阈值，teacher 或人工却确认有手。它表示 Palm 容易漏掉的困难手。 |
| `NEG_RUNTIME_CANDIDATE` | Palm 正式接受了 ROI，但 teacher 没输出手。未经人工复核前只是“疑似负例”；复核后才可能成为 hard negative。 |
| `NEG_LOW_PALM_CANDIDATE` | 低 Palm 分数产生的疑似背景 ROI。仍必须人工确认，不能因为分数低就自动当负样本。 |
| `hand_flag` / presence | Hand Landmarker 对“这个 ROI 中是否有手”的输出。geometry 使用 positive-only 数据时它几乎只学会输出 1；multitask 才通过 confirmed negative 让它学习拒绝。 |
| handedness | 左手/右手分类输出。geometry 阶段没有有效训练；multitask 只以很小权重训练。 |
| geometry | pretrain 第一子阶段，只用有完整 21 点的 positive，优先学习关键点几何。 |
| multitask | pretrain 第二子阶段，从 geometry best 初始化，加入人工确认 negative，同时继续训练 landmarks、`hand_flag` 和少量 handedness。它不是 Palm 训练，也不是 Gold finetune。 |
| finetune | 从 geometry 或 multitask checkpoint 出发，将少量 Gold 与大量 pseudo 混合，以较低学习率纠正 teacher 的漏标和点位偏差。 |
| finalize | HLMF 将多个来源的 manifest、pseudo 和可选 Gold 统一校验、加 namespace、去重、分型，并生成 HLML loader 可直接读取的 canonical JSONL。 |
| canonical JSONL | 最终训练接口文件；一行对应一个 ROI，包含图片路径、21 点、样本类型、来源、采样权重和 loss 权重。不要手工拼接或修改。 |
| crop ID / source-local crop ID | 单个 HLMF source 内的 ROI 身份。04/05 CVAT 往返使用这种本地 ID。 |
| global crop ID | 07A 在本地 ID 前加 `dataset_id:` 后形成的全局身份，用于多来源合并。不能把 global ID 直接当 source-local ID 交给 HLMF 04/05。 |
| best / last checkpoint | `best.weights.h5` 是验证指标最好的权重；`last.weights.h5` 是最后一个完成 epoch 的权重。正式评估和下一阶段初始化通常使用 best。 |
| collapse（塌缩） | 模型把本应展开的 21 点预测成中间的一小团、短线或过小骨架。 |
| PCK | 在给定归一化距离阈值内的关键点比例，越高越好。PCK@0.10 表示误差小于规定手部尺度 10% 的点所占比例。 |
| NME | 归一化平均误差，越低越好。归一化用于减少 ROI 中手部大小差异的影响。 |
| Val / Test | Val 用于模型选择和调参；Test 只能在方案冻结后做最终评估。不能根据 Test 结果反向改训练。 |
| dev / development set（开发集） | 不参与训练、专门用于开发期比较方案和选阈值的小评估集。`presence dev` 是同时有明确有手和无手 ROI、用于评估 `hand_flag` 的开发集；`challenge-dev` 是集中困难场景的开发集。它们不能与 Train 同 session。 |
| shared / independent Val | shared 是团队共同使用的验证来源；independent 是当前路线单独保留的验证来源。它们都是 Val，不是训练数据。 |
| session 泄漏 | 同一段视频或相邻帧同时进入训练和评估，即使文件名不同也会让评估虚高。必须按整段采集 session 隔离。 |
| source / domain（来源 / 数据域） | source 是 07A 配置中的一个具体数据来源；domain 是具有相似相机、光照、人员、姿态或采集流程的数据分布，例如 Peak 域。domain gap 表示 Train 与 Val/Test 的分布不同。 |
| generalization / overfitting（泛化 / 过拟合） | 泛化是模型在没见过的数据上仍然准确；过拟合是 Train 越来越好、Val 反而变差。本文的 Train–Val gap（训练—验证差距）就是判断过拟合的重要证据。 |
| teacher selection bias（教师选择偏差） | 只有 teacher 成功输出的手才能自动成为 pseudo positive，teacher 最容易漏掉的困难手反而缺席，导致训练集系统性偏向“Google 容易识别的手”。 |
| bbox（bounding box） | 包围目标或全部预测点的最小水平矩形。文中的 bbox 面积过小，是筛查骨架塌缩的一个信号。 |
| convex hull / 凸包 | 能包住全部 21 点的最小凸多边形。凸包面积很小表示点很集中，但闭拳也可能天然较小，所以只能用于筛查，不能单独判错。 |
| Gold-relative（相对人工真值） | 在同一个 ROI 内，用预测几何量除以人工 Gold 的对应量。例如预测骨架长度 / Gold 骨架长度 `<0.7` 表示预测骨架不到人工真值的 70%。它不是 HLMF/HLML 现成字段。 |
| P10 / P25 / median | 一组数从小到大排序后的第 10%、25% 和 50% 分位数；median 就是中位数。 |
| threshold（阈值） | 分数达到多少才接受检测。例如 Palm threshold=0.50 表示 Palm 分数至少 0.50 才产生正式 ROI。 |
| threshold sweep | 在一组阈值上重复计算结果，观察召回和误报如何变化，再冻结一个阈值。 |
| recall（召回率） | 所有真实手中成功检测到的比例；越高表示漏检越少。 |
| precision（精确率） | 所有模型报出的手中真正是手的比例；越高表示假检越少。 |
| FPR（false-positive rate） | 真实无手样本被错误接受为有手的比例；越低越好。 |
| false positive / false negative（假阳性 / 假阴性） | false positive 是“实际无手却报有手”，俗称假检；false negative 是“实际有手却没检出”，俗称漏检。二者必须依靠人工真值判断，不能只看模型输出数量。 |
| quality tier / quality flag | HLMF 对样本的自动质量等级和原因标记。当前 HIGH 主要表示结构通过，不代表人工确认 21 点准确。 |
| loss | 训练时衡量预测与标签差异的数值；optimizer 根据 loss 更新参数。landmark、presence、handedness 可以有不同 coefficient。 |
| Huber loss | 对小误差近似平方、对大误差近似线性的回归损失，比纯平方误差对极端错误更稳健。当前 landmarks 使用它。 |
| coefficient | 某项 loss 在总 loss 中的系数。系数为 0 表示该输出不通过该 loss 训练。 |
| sampler / sampling fraction | 决定每个 batch 从哪些 supervision tier 和 sample type 抽多少样本的采样器与比例。它不等同于原始数据占比。 |
| batch / epoch / draw（批次 / 轮次 / 抽样一次） | batch 是一次梯度更新读取的一组样本；epoch 是训练配置定义的一轮抽样；draw 是 sampler 抽到某条样本一次。同一 Gold 可在一个 epoch 被重复 draw 多次。 |
| backbone / head | backbone 是共享图像特征主干；head 是从共享特征产生 landmarks、hand flag 或 handedness 的输出分支。 |
| checkpoint | 训练过程中保存的模型权重。带 `.state` sidecar 时还可保存 optimizer 和已完成 epoch。 |
| NaN / Inf | 训练数值异常：NaN 表示“不是有效数字”，Inf 表示正/负无穷。出现它们通常说明 loss、梯度或输入已经数值崩溃。 |
| initial checkpoint | 新阶段只读取起点权重，重新创建 optimizer，从该阶段 epoch 0 开始。geometry→multitask/finetune 应使用这种语义。 |
| resume checkpoint | 同一次训练意外停止后续跑，恢复最近权重，并尽量恢复 optimizer 和 epoch。不能拿它代替“开始新阶段”。 |
| from scratch / early stopping / ReduceLROnPlateau | from scratch 是随机初始化、完全不载入旧权重；early stopping 是 Val 长期不改善就提前结束；ReduceLROnPlateau 是 Val 停滞后自动降低学习率。 |
| smoke test | 在很小的固定数据上快速验证模型是否能过拟合、数据是否能读、loss/梯度/checkpoint 是否正常；通过不代表完整集精度好。 |
| QC / gate / include / drop / HOLD | QC 是质量检查及其报告；gate 是不满足条件就拒绝进入下一阶段的硬门禁；include 表示进入该阶段 canonical；drop 表示明确排除；HOLD 表示证据不足而暂缓使用，既不作为正例，也不作为已确认负例。 |
| fail-closed | 证据不足或校验失败时直接拒绝继续，而不是猜测或自动放宽。例如负例数量不足就拒绝 multitask。 |
| provenance | 一个标签或 checkpoint 的来源证据，例如 human Gold、MediaPipe pseudo、从哪个 best 初始化以及文件 SHA。 |
| fallback | 已经验证、随时可用于提交的保底模型；新实验失败时不影响最终交付。 |
| anchor / 稳定样本 | 少量普通、清晰、容易正确标注的 Gold，用于防止 finetune 只适应极端困难样本而遗忘常见姿态。这里不是 Palm anchor。 |
| ONNX / contract | ONNX 是交付给厂商工具链的模型格式；contract 是导出时保存的输入输出、shape、算子、checkpoint/ONNX SHA 和数值一致性证据。能完成格式转换只证明工具链接口可用，不证明模型精度足够。 |

## 1. 结论先行

当前模型已经证明“模型定义和训练链路能正常工作”：训练可以稳定收敛，checkpoint、早停、验证、推理都能完整执行，输出无 NaN、Inf 或越界，“关键点漫天飞舞”的数值故障已经消失。

但当前 `best` 还不适合作为最终效果模型。它的主要问题不是训练轮数不够，也没有证据表明首先应扩大模型，而是以下四项问题叠加：

1. **明显的 Train–Val 泛化差距**：训练误差持续下降，而 Gold Val 在 epoch 20 后恶化；按原数据原配置继续 geometry 很可能只会更贴合伪标签。
2. **Peak 困难域覆盖不足和 teacher selection bias**：Google 在大量 Peak Gold positive 上完全不输出 Hand Landmarker 结果，这些困难姿态不会进入 positive 伪标签训练。
3. **r2 存在一项确定的数据交付错误**：整批 `peak_train_0714_dark` 没有进入训练，其中包含 18,525 个原始 positive 候选。该问题已经在 r3 修复，最终为 r3 geometry 增加 7,428 个合格 positive。
4. **端到端漏检主要发生在 Palm Detector，而非当前 Hand Landmarker 的 `hand_flag`**；geometry 的 `hand_flag` 已饱和为 1，既不能恢复无 ROI 的手，也不能过滤降低 Palm 阈值后出现的假 ROI。

因此，三条路线不应纯选其一。三天内推荐的主线是：

> **r3 geometry → 人工复核 negative candidates → multitask；同时在 HLMF 制作 500～1000 个可用 Gold positive。只有 HLML 正式 finetune 入口补齐并通过 smoke 后，才启动短程 finetune。**

人工复核和 Gold 数据制作可以并行，但模型权重不会自动合并：真正的训练仍必须按 `geometry → multitask → finetune`，或从 geometry 分叉得到两个独立候选。人工负例与 multitask 用于让 `hand_flag` 可用并支持降低 Palm 阈值；Gold finetune 才直接纠正 landmark。当前不建议单纯延长 r2、随机扩大同类伪标签数据或扩大模型。

## 2. 审计范围与基线

服务器结果：

- geometry run：`/root/autodl-tmp/TrainFab/HLML-2.0/hand_landmarker_runs/v2-pretrain-r2/geometry`
- 独立推理：`/root/autodl-tmp/TrainFab/HLML-2.0/hand_landmarker_inference/v2-pretrain-r2/geometry`
- r2 数据 finalize 报告：`/root/autodl-tmp/TrainFab/HLML-2.0/train_pretrain_merged/qc/finalize_train_pretrain_report.json`

本次使用完整训练报告、history、best/last/final checkpoint、Gold Val/Test 预测、逐来源指标、独立推理记录和 examples 做交叉检查。独立推理所用原图与现有 Train/Val/Test 独立，但它目前没有人工关键点和期望手数真值，所以只能用于错误归因与定性挑战，不能把“52 张零 ROI”直接等同于 52 个 false negative。

## 3. 模型与训练链路是否正常

### 3.1 健康项

- 从 scratch 启动，Adam，初始学习率 `3e-4`，batch size 64。
- 计划最多 160 epoch，best 在 epoch 20，最终在 epoch 40 正常早停。
- epoch 26、31、36 的 ReduceLROnPlateau 正常降低学习率，但没有刷新 best。
- `best.weights.h5` 与 `final.weights.h5` 的 SHA-256 相同；Val/Test 使用的是正确的 best，而不是退化的 last。
- Val/Test 输出覆盖率 100%，没有 NaN/Inf，没有归一化坐标越界。
- 训练报告记录 git clean，commit 为 `08ca042443543bed09212cd6d808a381a251c0ed`。

这说明当前模型定义、前向/反向、loss、优化器、checkpoint 选择和推理链路没有结构性故障。

### 3.2 真正的问题是泛化，而不是欠拟合

| 指标 | epoch 1 | epoch 20（best） | epoch 40（停止） |
|---|---:|---:|---:|
| Train landmark MAE | 0.06521 | 0.01941 | 0.01382 |
| Val landmark MAE | 0.07929 | **0.05841** | 0.06167 |

epoch 20 后 Train 继续明显改善，Val 却回退；连续三次降低学习率也没有带来新的 Val best。因此：

- 不应恢复 r2 last 继续训练；
- 不应在相同数据上简单增加 epoch；
- 扩大模型参数量大概率会扩大当前拟合差距，而不是补齐缺失的困难域；
- 当前 best 应保留为后续 geometry/multitask/finetune 的基线与紧急交付 fallback。

### 3.3 Gold 评估基线

| Split | mean / median / P90 / P95 pixel error | mean NME | PCK@0.05 / 0.10 / 0.15 |
|---|---|---:|---|
| Val（1,226 positive） | 23.56 / 19.56 / 42.14 / 47.79 px | 0.2227 | 13.04% / 30.85% / 44.37% |
| Test（985 positive） | 20.34 / 19.41 / 35.59 / 39.39 px | 0.1928 | 17.11% / 35.12% / 49.92% |

Test 已经被本次查看过，后续不要再用它做筛选、阈值或超参数迭代。新实验先只依据固定 Val 和另建的 challenge-dev 做决策，方案冻结后再做一次最终 Test。

Presence 在 Val/Test 上为 1.0 不是有效成绩，因为两个 split 全是 positive，`tn=0`。Handedness 在 geometry 中 loss coefficient 为 0，输出约为 0.5，当前也没有评估意义。

## 4. “塌缩”是可量化的真实问题

视觉上的“小点团、骨架聚在中间”不是少数偶然图。以 20 条标准手骨边的总长度和关键点包围面积对 Gold 做相对比较：

- Val 预测中心化坐标方差只有 Gold 的约 59.7%，Test 约 64.7%；
- Val 有 `459/1226 = 37.4%` 的样本，预测骨架总长度不足 Gold 的 70%；其中 222 个低于 50%；
- Test 有 `335/985 = 34.0%` 低于 Gold 的 70%；
- Val 有 836 个预测关键点 bbox 面积不足 Gold 的 70%；
- 小预测面积与大 pixel error 有明显负相关，说明这不是单纯闭拳造成的小面积。

独立推理没有 Gold，但 217 个 Palm ROI 的 21 点凸包也显示明显收缩长尾：

- 凸包面积 P10 / P25 / median 为 `0.01034 / 0.01841 / 0.03158`；
- `20/217 = 9.2%` 小于 0.01；
- `66/217 = 30.4%` 小于 0.02；
- 对最小的 12 个 ROI 逐图检查，crop 内均存在目标手，而预测确实收成短线或点团。

凸包只是筛查指标，不应把闭拳、侧手自动判坏；真正的验收应使用 Gold-relative 骨架长度、bbox 面积和 pixel/PCK 联合判断。

## 5. 根因分析

### 5.1 问题高度集中在 Peak 域

| Split / source | mean pixel error |
|---|---:|
| Val Peak independent | 27.57 px |
| Val Peak shared | 28.20 px |
| Val Soar | 15.03 px |
| Test Peak | 24.66 px |
| Test Soar | 12.38 px |

当前 geometry Train 约 27.4% 来自 Peak，而 Val/Test 约 65% 来自 Peak。r2 的新 ROI 尺度已经比旧版本更接近 Val/Test，历史上的巨大 ROI-scale gap 不再是首要解释；剩余问题更像是 Peak 场景、困难姿态、teacher abstention 和实际采样占比共同造成的域差异。

### 5.2 不能把问题简单归结为“Google 坐标多数很差”

将 Val 的 MediaPipe draft 与人工 Gold 对齐：

- 1,226 个 Gold positive 中，teacher 有完整 21 点输出的为 836 个；这部分 teacher→Gold mean error 约 4.40 px；
- 其余 `390/1226 = 31.8%` 上 teacher 完全 abstain；
- Peak 的 806 个 Gold positive 中有 382 个 abstain，占 47.4%；
- Soar 只有 `8/420 = 1.9%` abstain；
- 学生在 teacher-success 子集上的 mean error 仍为 20.17 px，在 teacher-abstain 子集上进一步恶化到 30.84 px。

所以更准确的判断是：

1. 训练伪标签确实存在坏坐标尾部；
2. 但 teacher 一旦成功输出，多数坐标远好于当前学生；
3. 最大的数据问题之一是 teacher **不为困难 Peak 手生成 positive**，而不是大多数已有伪标签都错误；
4. 随机录制更多数据后继续使用同一 teacher，会再次遗漏最需要学习的场景。

### 5.3 现有 quality tier 没有真正筛出几何坏标签

r2 geometry 的 52,524 个样本全部是 `mediapipe_pseudo`：

- HIGH：51,558；
- MEDIUM：966；
- 唯一的 MEDIUM 原因是 `low_handedness_score`；
- 所有样本的伪标签监督权重相同为 0.7。

因此 HIGH 不能被解释为“21 点几何经过高质量检查”。以 Val Gold 的低分位几何量仅作排序阈值：

- 训练中 3,064 个样本骨架总长度低于 Val Gold P1；
- 2,550 个 bbox 面积低于 P1；
- 2,970 个 palm width 低于 P1；
- bbox 与骨架长度同时低于 P1 的有 1,248 个，其中 1,042 个来自 `POS_LOW_PALM`。

抽查最极端样本可见部分手、遮挡手、严重偏 ROI，甚至几乎无可见手的 ROI，却仍被当作 positive 学习。这些应成为 Review 队列，但阈值只应用于“优先级排序”，不能自动删除合法闭拳或透视缩短姿态。

### 5.4 r2 确定漏掉整批 Peak 暗光数据

`finalize_train_pretrain_report.json` 明确记录：

- source：`peak_train_0714_dark`；
- manifest / pseudo：57,962；
- included：0；
- excluded：57,962；
- quality tier：全部 INVALID；
- structural error：`crop_image_missing`。

服务器目录复核发现：

- HLMF 上游 `.../DatesetFab/HandViolenceEnhanced0714/peak/peak_dark/02_roi_crops/images` 有 **57,962 张正确 ROI PNG**；
- HLML 的 `.../TrainFab/HLML-2.0/train_sources/HandViolenceEnhanced0714/peak_dark/02_roi_crops/images` 却是 **6,851 张原始 TIFF**；
- 这批 57,962 个记录中原始 sample type 包含 14,325 个 `POS_LOW_PALM` 和 4,200 个 `POS_RUNTIME`，合计 18,525 个 positive 候选；实际重新去重、提纯后的新增量会更少。

这是当时最值得先修的确定性问题。该问题现已按正确方式修复：没有覆盖 r2，而是以 r3 重新 finalize、curate、inspect 和 smoke。r3 最终纳入 7,428 个 `peak_train_0714_dark` positive；18,525 是修复前的原始 positive 候选数，二者不同是因为 07A 与 curate 还会执行结构校验、去重和提纯。

### 5.5 端到端“漏检”必须与 geometry 分开处理

独立推理共 192 张原图：

| 每帧 Palm ROI 数 | 帧数 |
|---:|---:|
| 0 | 52 |
| 1 | 63 |
| 2 | 77 |

共 217 个 Palm ROI，Hand 推理失败数为 0。所有 ROI 的 `hand_flag_score` 都在 `0.9999505～0.9999999`，所以当前 examples 中的整手漏检不是 `hand_flag_threshold=0.5` 拒绝造成的，而是 Palm 没产生 ROI，或者双手画面只产生一个 ROI。

examples 能清楚区分两类故障：

- `...200759922_1230`、`...200802805_1470`、`...200923626_180`：Palm 已给出两手 ROI，但 Hand 骨架明显塌缩或偏移；
- `...200805738_1710`、`...200950688_2400`：画面有两只手，但 Palm 只给一只；已检测手的 geometry 相对更正常；
- `...200809304_2010`：两手都检测到，骨架没有极端缩小，但手指拓扑和关节位置仍有较大误差。

同一 Palm 模型的只读阈值复算结果：

| Palm threshold | 0 ROI 帧 | 1 ROI 帧 | 2 ROI 帧 | ROI 总数 |
|---:|---:|---:|---:|---:|
| 0.50 | 52 | 63 | 77 | 217 |
| 0.45 | 34 | 66 | 92 | 250 |
| 0.40 | 21 | 59 | 112 | 283 |
| 0.35 | 11 | 49 | 132 | 313 |

0.40 能目视恢复不少真实手，但也已出现明确背景假框。当前 geometry 的 `hand_flag≈1` 会接受所有新增假框，因此不能把“直接降低 Palm 阈值”当作最终方案。正确顺序是：

1. 先给这 192 张或新的 session 独立标注每帧期望手数，计算 Palm recall/precision；
2. 低阈值用于挖掘 hard positive 与 hard negative；
3. 完成 confirmed-negative multitask，让 `hand_flag` 具有拒绝能力；
4. 再联合 sweep Palm threshold 与 Hand flag threshold；
5. 视频场景可增加短时 ROI tracking/carry-forward，缓解连续帧短暂 Palm 丢失。

另一个值得单独排查的上游信号是 Palm `head7`。Palm 模型有 7×7 与 14×14 两个检测特征图输出分支，日志中简称 head7/head14；不同分支通常负责不同尺度的候选。head7 ROI 的塌缩率显著高于 head14，可能因为它承担了更困难/更小的手，也可能存在 p0→p9 方向和 ROI 旋转不稳定；目前只能作为分层复核条件，不能仅凭相关性断言 decoder（把网络输出解码成检测框和方向点的程序）有 bug。

## 6. 对三条路线的判断

| 路线 | 三天内优先级 | 建议 |
|---|---:|---|
| Review：批量删除低质量 positive | 低 | 不建议。会进一步删除困难场景并强化 teacher-success 偏差。 |
| Review：三分流复核并把困难样本改为 Gold | **最高** | 明确错 ROI/无手可 drop；可见手但伪标签差的转人工 Gold；不确定样本 HOLD。 |
| Enlarge：修复遗漏的 Peak dark | **最高** | 数据已经存在，新增成本小，是最确定的近期收益。 |
| Enlarge：随机录更多、仍全部用 Google 伪标 | 低 | teacher-abstain 困难手仍不进入 positive，连续视频还会产生大量近重复。 |
| Enlarge：扩大模型 | 不建议 | 已有明显过拟合；同时会重新引入参数量、延迟、量化和板端验证风险。ONNX 可转换并不改变这一泛化判断。 |
| Enlarge：加入结构 loss | 中低 | 只适合数据修复后的单变量小实验，不能替代正确标签。 |
| Advance：确认负例→multitask | 中高 | 对 presence、降低 Palm 阈值和假框过滤重要，但不直接修复关键点塌缩。 |
| Advance：高价值 Gold finetune | 数据价值最高；训练入口未完成 | 直接覆盖 teacher-abstain 和坐标偏差；先在 HLMF 制作 Gold，HLML 正式入口和 smoke 完成后才能训练。 |

推荐组合路线：

```text
修复 Peak dark 交付
        ↓
新 ID：finalize → curate → inspect → smoke → geometry
        ↓
固定 Val 按来源/困难组验收
        ↓
可选 multitask（若确认负例门禁已完成）
        ↓
500～1000 个定向 Gold positive
        ↓
HLML 正式 finetune 入口 + smoke（当前待实现）
        ↓
短程 finetune
        ↓
最终 Test → 导出真实 best → 厂商转换与板端验收
```

## 7. 三天执行安排

Peak dark 修复、r3 finalize/curate/smoke 已经完成，不再列为待办。剩余时间按以下顺序安排。

### r3 geometry 训练期间

1. 让 r3 geometry 独占其已经冻结的 curated 输入。
2. 可以并行下载并人工查看 `hand_landmarker_reviews/v2-pretrain-r3/negative_candidates/` 的副本；这里不是训练输入。
3. 可以并行在 HLMF 中整理 Gold 候选 ID、制作每个 source 的 subset 和 CVAT task。
4. **不要在 geometry 正在运行时执行 `make pretrain-curate-reviewed`**。该命令会用 `--overwrite` 重建 r3 curated 目录；即使 positive 理论上不变，也没有必要让训练进程与数据快照替换并发发生。

### r3 geometry 正常完成后

1. 检查 `geometry/training_report.json` 和 `experiment_metadata.json` 均为 complete，并保留 `best.weights.h5`。
2. 运行 `make eval-val-geometry`，与 r2 的 Val overall、Peak/Soar 和 collapse 指标比较。
3. 暂时不要根据 Test 继续调参；Test 留到方案冻结。
4. 将 r3 geometry best 导出并完成一次厂商转换，作为新的可交付 fallback。
5. 完成人工负例工作区回传，再运行 `make pretrain-curate-reviewed` 和 `make check-multitask-data`。

### multitask 与 Gold 的关系

- 人工工作可以并行：一组人删除式复核负例，另一组人在 CVAT 修正 Gold 21 点。
- 模型训练不能“并行后自动合并权重”。推荐主链是：

```text
r3 geometry best
        ↓
multitask best
        ↓
finetune（仅在正式入口完成后）
```

- 也可以从 geometry 分叉得到一个 multitask 候选和一个 landmark-first finetune 候选，但它们是两个模型，必须分别评估，不能把结果相加。
- 如果负例门禁不达标，保留 r3 geometry；不要把未经确认的 negative 强塞进 multitask。
- 如果 finetune 正式入口没有及时完成，最终 fallback 是“r3 geometry 或经 Val 证明不损伤 landmarks 的 multitask”，不要在提交前临时自制不可审计的训练命令。

后文第 12～16 节给出每一步的实际命令、目录、人工职责和程序检查。

## 8. 结构约束与模型扩大的具体建议

三天内不建议采用固定骨长、固定关节方向或预测 spread 硬下限。二维图像中的弯指、握拳、遮挡、透视缩短和交叉手都会破坏这些固定先验，硬约束可能把合法手势推向错误姿态。现有 coordinate Huber 本身已经惩罚预测点远离 GT；当前塌缩首先应通过数据域、伪标签纠错和 Gold 补齐解决。

如果修复 Peak dark 和 Gold finetune 后仍有明显塌缩，只做一个低风险单变量实验：

```text
L_edge = mean Huber((pred_j - pred_i) - (gt_j - gt_i))
```

其中 `(i,j)` 是 20 条标准手骨边，可按 GT palm scale 归一化并使用较小权重。这约束的是“相对于目标的边向量”，而不是强迫所有手满足固定方向/长度；训练时增加 loss 不会改变 ONNX 输出 contract。是否保留必须依据 Gold Val 的 pixel error、PCK 和 collapse 比例，而不是只看骨架视觉上更展开。

扩大 backbone/head、改成 heatmap 或大改网络，在三天窗口内都不划算：它们不能补出 teacher 从未提供的困难标签，还会重新打开收敛、量化、延迟和板端稳定性风险。

## 9. 新实验验收门槛

以 r2 Gold Val 为基线，建议在开始前固定以下门槛：

其中 pixel error、NME、PCK 可直接从标准 `metrics.json` 查看；Gold-relative 骨架比例和 teacher-abstain 子组是本次审计的附加分析指标，当前没有对应 Make target。若没有把这段分析正式实现为脚本，不要假装它们是自动 gate，至少用固定 examples 和标准 Val 指标完成主验收。

### Geometry / finetune

- overall mean pixel error 从 23.56 px 相对下降至少 10%；
- P90 从 42.14 px 相对下降至少 10%；
- PCK@0.10 从 30.85% 至少提高 5 个百分点；
- 两个 Peak Val 来源 mean error 都改善，Soar 回退不超过 5%；
- Gold-relative 骨架长度 `<0.7` 的比例从 37.4% 降到约 28% 或更低；
- teacher-abstain 子组 mean error 至少下降 15%；
- 固定 examples 中 Palm 已命中 ROI 的塌缩图有一致改善，而不是只改善个别图片。

### Multitask / 端到端

- Val landmark mean/P90 相对 geometry best 恶化不超过 3%；
- PCK 下降不超过 2 个百分点；
- presence dev 报告真实 positive recall、negative FPR 和阈值；
- 在有期望手数真值的端到端 dev 上联合选择 Palm/Hand threshold；
- 不以 `predicted_hand_count` 或 positive-only presence accuracy 代替召回率。

如果新实验没有达到这些门槛，继续使用已冻结的 r2 fallback 比在提交前换入未经证明的新模型更稳妥。

## 10. 最终建议

当前结果证明 HLML 的模型定义和训练实现可用，但 r2 的 Gold 泛化不足，不能通过“继续相同 geometry”解决。最优的三天决策不是在 Review、Enlarge、Advance 中单选，而是按根因组合：

1. **Peak dark 已经补回，先用 r3 geometry 的固定 Val 证明这次数据修复是否有效。**
2. **按既定删除式规则复核 negative：只保留明确背景；其中肉眼有手的候选在删除前另外记录 ID，转入 HLMF Gold 队列。**
3. **启动当前已经正式支持的 multitask；同时制作 Gold 数据。只有 HLML finetune 正式入口补齐并通过 smoke 后，才启动 Gold finetune。**
4. **不延长 r2、不优先扩模、不先加硬骨骼先验。**
5. **把 Palm recall 与 Hand geometry 分开验收，否则会把上游无 ROI 与下游关键点偏差混为一个问题。**

这条路线同时保留了困难场景的泛化价值，又能纠正 teacher 的系统性漏标和 landmark 偏差，并且把三天内的实验变量控制在可验证范围内。

## 11. 两个仓库的命令边界

后续所有命令块都在标题中标明仓库。进入服务器后应先 `cd` 到正确仓库，再激活对应环境。

本次服务器实际代码路径为：

```bash
HLML_REPO=/root/HandLandmarkerLab
HLMF_ROOT=/root/HandLandmarksFab
```

本地 HLMF 仓库名是 `HandLandmarkerFab`，服务器目录实际叫 `HandLandmarksFab`；不要凭名称猜路径。

| 动作 | 仓库 | 环境 | 当前真实命令 |
|---|---|---|---|
| 原图检查、Palm、ROI、MediaPipe 伪标 | HLMF | `anfab` | `make validate_images_train` 等 00～03 目标 |
| 导出/导入 CVAT | HLMF | `anfab` | `scripts/04_export_cvat_xml.py`、`scripts/05_import_cvat_xml.py` |
| 生成 pretrain canonical | HLMF | `anfab` | `make finalize_train_pretrain` |
| 生成 Gold+pseudo finetune canonical | HLMF | `anfab` | `make finalize_train_finetune` |
| pretrain 提纯与负例工作区 | HLML | `hand-landmarker-tf29` | `make pretrain-curate` |
| 完成删除式负例复核 | HLML | `hand-landmarker-tf29` | `make pretrain-curate-reviewed` |
| geometry | HLML | `hand-landmarker-tf29` | `make pretrain-geometry-smoke`、`make pretrain-geometry` |
| multitask | HLML | `hand-landmarker-tf29` | `make check-multitask-data`、`make pretrain-multitask` |
| finetune | HLML | `hand-landmarker-tf29` | **当前没有正式 Make/YAML 入口** |

当前 HLMF `finalize_train.yaml` 的 source 与实际子目录对应如下。制作 Gold 时必须按 source 分开，不能把多个 source-local ID 混进一个 CVAT task。

| `dataset_id` | `train_sources/` 下的 source 子目录 |
|---|---|
| `peak_train_v1` | `HandViolence0708/peak_train_data` |
| `soar_train_v1` | `HandViolence0708/soar_train_data` |
| `peak_train_0714_bright` | `HandViolenceEnhanced0714/peak_bright` |
| `peak_train_0714_dark` | `HandViolenceEnhanced0714/peak_dark` |
| `soar_train_0714_bright` | `HandViolenceEnhanced0714/soar_bright` |
| `soar_train_0714_dark` | `HandViolenceEnhanced0714/soar_dark` |
| `dragon_train_0714` | `HandViolenceEnhanced0714/dragon` |

以下看似自然的命令当前都不存在，不得照字面执行：

```text
make gold-select
make gold-export
make finetune
make pretrain-finetune
make check-finetune-data
make finetune-smoke
```

## 12. 删除式负样本人工复核完整操作

本节严格落实 `docs/training_system/data_and_training.md` 第 170 行附近定义的删除式流程：人工删除所有含手或无法确信为背景的图片，然后由 `make pretrain-curate-reviewed` 根据“仍保留的文件”自动生成确认负例。这里没有引入另一套复核算法。

### 12.1 复核对象与判断规则

HLML `make pretrain-curate` 已生成：

```text
/root/autodl-tmp/TrainFab/HLML-2.0/
└── hand_landmarker_reviews/v2-pretrain-r3/
    ├── negative_candidates/
    │   ├── NEG_RUNTIME_CANDIDATE/<dataset_id>/*.png
    │   └── NEG_LOW_PALM_CANDIDATE/<dataset_id>/*.png
    ├── review_manifest.jsonl
    ├── review_report.json
    └── REVIEW_INSTRUCTIONS.md
```

人工只修改 `negative_candidates/` 的**成员集合**，方法是删除图片：

- 只要看见手、手指、手腕或疑似手部区域：删除；
- 模糊、过暗、过曝、遮挡、边缘局部或无法确信“没有手”：删除；
- 只有明确的纯背景 ROI 才保留；
- 不新增、不重命名、不移动、不编辑、不旋转、不重新保存图片；
- 不修改 `review_manifest.jsonl`；
- 不手写 `negative_review_decisions.jsonl`；
- 绝对不删除 `train_sources/` 中的原始 ROI。`negative_candidates/` 只是 curate 生成的复核副本。

被删除图片的含义只是“不能确认成负例”。程序会让它继续保持 HOLD，不会把它当 positive，也不会自动加入 Gold。

为了不浪费其中的困难手，建议在删除前额外做一项记录：

- 清楚看见手、且愿意在 CVAT 标完整 21 点：记录该图的 `dataset_id` 和 source-local `crop_id`，进入 Gold 候选表；
- 只是模糊或无法可靠标点：删除，但不进入 Gold，或以后在 CVAT 标 `ignore_for_training`；
- 明确背景：保留，等待程序确认为 negative。

### 12.2 生成供人工查 ID 的映射表

`review_manifest.jsonl` 里的 `crop_id` 是 global ID，格式为 `dataset_id:source-local-crop-id`。下面的只读命令生成 CSV，方便在本地按图片相对路径查到 source-local ID。

在服务器 **HLML 仓库根目录**执行：

```bash
cd /root/HandLandmarkerLab

ROOT=/root/autodl-tmp/TrainFab/HLML-2.0
ID=v2-pretrain-r3
REVIEW_ROOT="$ROOT/hand_landmarker_reviews/$ID"

python - \
  "$REVIEW_ROOT/review_manifest.jsonl" \
  "$REVIEW_ROOT/gold_candidate_mapping.csv" <<'PY'
import csv
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])

with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8-sig", newline="") as dst:
    writer = csv.DictWriter(dst, fieldnames=[
        "candidate_relative_path",
        "dataset_id",
        "source_crop_id",
        "sample_type",
        "global_crop_id",
    ])
    writer.writeheader()
    for line in src:
        if not line.strip():
            continue
        row = json.loads(line)
        dataset_id = str(row["dataset_id"])
        global_id = str(row["crop_id"])
        prefix = dataset_id + ":"
        if not global_id.startswith(prefix):
            raise SystemExit("global crop_id prefix mismatch: " + global_id)
        writer.writerow({
            "candidate_relative_path": row["candidate_relative_path"],
            "dataset_id": dataset_id,
            "source_crop_id": global_id[len(prefix):],
            "sample_type": row["sample_type"],
            "global_crop_id": global_id,
        })
print(target)
PY
```

该 CSV 只是人工记录辅助文件，不参与 `pretrain-curate-reviewed`。不要把它放进 `negative_candidates/`。

### 12.3 7z 与网盘是否改变 SHA-256

ZIP 和 7z 都是无损归档格式。只要图片只是被压缩、解压和查看，没有经过图片编辑器重新编码，图片内容的 SHA-256 应保持不变。上传夸克网盘和下载压缩包也不会主动改变压缩包内部图片的字节。下文用 7z 举例；若使用 ZIP，staging、SHA 预检和“不能覆盖解压”的规则完全相同。

需要区分三件事：

- **图片 SHA-256**：必须保持不变；
- **复核前后 `.7z` 文件的 SHA-256**：因为成员被删除，通常会改变，这是正常的；
- 文件时间、权限：可能变化，但不影响内容 SHA。

常见的破坏 SHA 行为包括：用图片编辑器保存、自动旋转、格式转换、截图替换、压缩图片、云盘预览后“另存为”。只使用文件管理器查看和删除。

最终不依赖人工口头保证：`make pretrain-curate-reviewed` 会按服务器原始 `review_manifest.jsonl` 逐图重算 SHA；任一保留图片字节变化都会 fail-closed。

### 12.4 服务器打包

先确认服务器安装的是 `7z` 还是 `7zz`：

```bash
command -v 7z || command -v 7zz
```

以下以 `7z` 为例：

```bash
ROOT=/root/autodl-tmp/TrainFab/HLML-2.0
ID=v2-pretrain-r3
REVIEW_ROOT="$ROOT/hand_landmarker_reviews/$ID"
TRANSFER_DIR="$ROOT/review_transfer/$ID"

mkdir -p "$TRANSFER_DIR"
cd "$REVIEW_ROOT"

7z a -t7z -mx=1 \
  "$TRANSFER_DIR/${ID}-negative-review-original.7z" \
  negative_candidates \
  review_manifest.jsonl \
  review_report.json \
  REVIEW_INSTRUCTIONS.md \
  gold_candidate_mapping.csv

sha256sum "$TRANSFER_DIR/${ID}-negative-review-original.7z"
```

如果实际命令是 `7zz`，只替换程序名。将压缩包上传网盘；本地下载后可先核对压缩包 SHA，确认传输无损。

### 12.5 本地解压与多人分工

Windows 示例：

```powershell
7z x .\v2-pretrain-r3-negative-review-original.7z -o.\v2-pretrain-r3-review
```

只在解压目录的 `negative_candidates/` 内删除图片。`review_manifest.jsonl` 等文件只读。

多人复核时不要让每人拿完整副本，最后用其中一个人的压缩包覆盖其他人的结果；这样会把其他人删除的图片恢复。推荐：

1. 一个权威目录顺序复核；或
2. 按 `sample_type/dataset_id` 把互不重叠的子目录分给不同人，最后合并到唯一权威目录。

复核完成后，从解压根目录重新压缩，保证压缩包最外层直接是 `negative_candidates/`，不要额外多套一层同名目录：

```powershell
Set-Location .\v2-pretrain-r3-review
7z a -t7z -mx=1 ..\v2-pretrain-r3-negative-review-reviewed.7z .\negative_candidates
```

### 12.6 回传时绝对不能覆盖解压

一个严重陷阱：把复核后的 7z 直接解压覆盖到服务器原 `negative_candidates/`，不会删除服务器上那些“本地已经删除”的文件。它们仍会存在，并被错误确认成 negative。

正确方法是上传压缩包后解压到全新的 staging 目录：

```bash
ROOT=/root/autodl-tmp/TrainFab/HLML-2.0
ID=v2-pretrain-r3
REVIEW_ROOT="$ROOT/hand_landmarker_reviews/$ID"
STAGING="$ROOT/review_transfer/${ID}-returned-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$STAGING"
7z x "/上传目录/${ID}-negative-review-reviewed.7z" -o"$STAGING"
test -d "$STAGING/negative_candidates"
```

不要用本地副本覆盖服务器上的 `review_manifest.jsonl`；服务器原文件才是身份和 SHA 依据。

### 12.7 staging SHA 预检

以下只读脚本允许“少文件”，因为少掉的正是人工删除项；它拒绝未知图片和被修改的保留图片：

```bash
python - \
  "$REVIEW_ROOT/review_manifest.jsonl" \
  "$STAGING/negative_candidates" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
candidate_root = Path(sys.argv[2])
extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}

expected = {}
with manifest_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        relative = row["candidate_relative_path"]
        if relative in expected:
            raise SystemExit("duplicate manifest path: " + relative)
        expected[relative] = row["sha256"]

actual = {
    path.relative_to(candidate_root).as_posix(): path
    for path in candidate_root.rglob("*")
    if path.is_file() and path.suffix.lower() in extensions
}

unknown = sorted(set(actual) - set(expected))
modified = []
for relative, path in actual.items():
    if relative not in expected:
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected[relative]:
        modified.append(relative)

print("original candidates:", len(expected))
print("retained candidates:", len(actual))
print("deleted candidates:", len(expected) - len(actual))
print("unknown image paths:", len(unknown))
print("modified retained images:", len(modified))

if unknown:
    print("unknown examples:", unknown[:10])
if modified:
    print("modified examples:", modified[:10])
if unknown or modified:
    raise SystemExit(1)
PY
```

只有退出码为 0 才进行目录交换。先把原目录移到备份，不立即删除：

```bash
BACKUP="${REVIEW_ROOT}.negative_candidates.before-finalize.$(date +%Y%m%d-%H%M%S)"

mv "$REVIEW_ROOT/negative_candidates" "$BACKUP"

if ! mv "$STAGING/negative_candidates" "$REVIEW_ROOT/negative_candidates"; then
  mv "$BACKUP" "$REVIEW_ROOT/negative_candidates"
  exit 1
fi
```

备份至少保留到 `pretrain-curate-reviewed` 和 multitask gate 都成功。

### 12.8 让程序生成正式复核决策

等 r3 geometry 正常结束后，在服务器 **HLML 仓库根目录**执行：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
make paths
make pretrain-curate-reviewed
make check-multitask-data
```

`make pretrain-curate-reviewed` 实际执行：

```bash
python -B scripts/curate_pretrain.py \
  --config configs/curate_pretrain.yaml \
  --finalize-retained-review \
  --overwrite
```

程序负责：

1. 扫描仍存在的图片；
2. 对照 `review_manifest.jsonl` 检查相对路径；
3. 对每张保留图片重算 SHA-256；
4. 自动写 `negative_review_decisions.jsonl`，决策为 `CONFIRMED_NEGATIVE`；
5. 自动写 reviewer、时间、review method 和图片 SHA；
6. 重建同一个 r3 curated 快照；
7. 对与已知 positive 重叠的候选继续 HOLD；
8. 不把人工删除的图片误当 negative。

其中 `reviewer` 不是程序猜出的个人姓名，而是读取 `configs/curate_pretrain.yaml` 的 `review.reviewer`；当前值为 `hlml-visual-review-team`。如果多人协作需要追溯到个人，应另外保存“负责人—子目录—完成时间”表。现有删除式流程只会给全部保留项统一写配置中的 reviewer，不要把它误解成逐图个人签名。

成功后检查：

```text
hand_landmarker_reviews/v2-pretrain-r3/
├── negative_review_decisions.jsonl
└── review_report.json

train_pretrain_curated/v2-pretrain-r3/
├── 05_labels/
│   ├── hand_training_labels_pretrain_landmarks.jsonl
│   ├── hand_training_labels_pretrain_multitask.jsonl
│   └── hand_training_labels_pretrain_smoke.jsonl
└── qc/
    ├── curation_report.json
    └── sha256_manifest.json
```

重点查看：

- `review_report.json`：原候选数、保留确认数、删除数、decisions SHA；
- `curation_report.json`：`included_confirmed_negatives`、`multitask_records`、`negative_overlap_confirmed_hand` 和 `reason_counts`；
- `sha256_manifest.json`：multitask labels SHA、decisions SHA、图片 aggregate SHA。

“人工保留数”可能大于“最终进入 multitask 的 negative 数”，因为自动 overlap 门禁还会拒绝一部分冲突 ROI，这是正常现象。

## 13. 制作 500～1000 个 Gold ROI 的完整流程

### 13.1 这一步最终要得到什么

目标不是重新裁 500～1000 张图片，而是从现有 Train source 中选出一小批 ROI，在 CVAT 中人工确认：

- ROI 中是否存在目标手；
- 如果有手，21 点分别在哪里；
- 是 Left 还是 Right；
- 如果无法可靠标注，是否应 `ignore_for_training`。

最终每个 source 得到三件套：

```text
subset_manifest.jsonl       # 这批 ROI 的原始几何和身份
subset_gold.jsonl           # CVAT 复核后由 HLMF 05 生成的 Gold 标签
cvat_import_stats.json      # 证明 CVAT 覆盖完整且没有阻塞错误
```

07A 会用相同 `crop_id` 的 Gold 覆盖 pseudo，再聚合所有 source。不要手工把 Gold JSONL 与 pseudo JSONL 拼接。

500～1000 指**最终可用的 Gold positive**。CVAT 中可能发现无手或必须 ignore，因此建议先选 650～1200 张候选。长期更稳妥的 Gold 规模通常是 1500～3000；500～1000 是当前时间窗口下的战术最小批。

### 13.2 候选应该怎么选

建议分配：

- 50%～60%：Peak negative review 中肉眼实际有手的 teacher-abstain 候选；
- 20%～30%：Peak `POS_LOW_PALM` 或肉眼能看到 MediaPipe 21 点明显错误的 pseudo positive；
- 15%～20%：普通 Peak、Soar 或 Dragon 的清晰常见姿态，作为稳定样本，避免 finetune 只记住极端困难图。

具体覆盖：

- 明亮与暗光；
- 左右手；
- 开掌、握拳、弯指、V 手势和项目手语姿态；
- 遮挡、交叉、画面边缘、手很小、旋转大；
- `POS_RUNTIME` 与 `POS_LOW_PALM` 两种 positive 类型。

同一段视频只抽少量代表帧。不要把当前独立 inference session 加入训练。`HandFinetune0713/peak` 与这批 inference 图存在大量重叠，不能作为 Gold Train；应优先选 0714 Train source 或全新且按 session 隔离的数据。

“student–teacher 分歧大”目前没有现成筛选命令：HLMF 不读取学生预测，HLML 也没有对全 Train canonical 生成 disagreement 表的正式目标。因此当前可执行版本优先使用“负例复核中实际看见手”“`POS_LOW_PALM`”“人工看到伪标签明显错”三类，不要声称已经做了自动 student–teacher 排序。

### 13.3 候选 ID 从哪里来

最完整的跨来源索引是：

```text
/root/autodl-tmp/TrainFab/HLML-2.0/
train_pretrain_merged/05_labels/hand_train_catalog_pretrain.jsonl
```

其中包含：

- `dataset_id`：来自哪个 source；
- `source_crop_id`：该 source 内原始 ID；
- `global_crop_id`：带 namespace 的全局 ID；
- `sample_type`：四种样本类型之一；
- `quality_flags`、`selection_action`、`crop_path`。

构建 HLMF subset 时必须：

1. 先按 `dataset_id` 确定 source；
2. 再把该 source 的 `source_crop_id` 写入选择文件；
3. 回到该 source 原始 manifest/draft 按 `crop_id` 筛选。

不能从 global finalized row 反向伪造 manifest，也不能把 `peak_train_0714_dark:...` 这种 global ID 直接交给 04/05。

可以用下面的只读索引命令先导出 Peak positive 候选 CSV。它不判断点位好坏，只把人工优先查看的 `POS_LOW_PALM` 和普通 `POS_RUNTIME` 分开：

```bash
ROOT=/root/autodl-tmp/TrainFab/HLML-2.0
CATALOG="$ROOT/train_pretrain_merged/05_labels/hand_train_catalog_pretrain.jsonl"
OUT="$ROOT/hand_landmarker_gold/v2-finetune-r1/_candidate_indexes"
mkdir -p "$OUT"

python - "$CATALOG" "$OUT" <<'PY'
import csv
import json
import sys
from pathlib import Path

catalog = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
with catalog.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if not str(row.get("dataset_id", "")).startswith("peak_"):
            continue
        if not bool((row.get("hand_presence") or {}).get("present")):
            continue
        if row.get("sample_type") not in {"POS_LOW_PALM", "POS_RUNTIME"}:
            continue
        if row.get("quality_tier") == "INVALID":
            continue
        if row.get("selection_action") not in {"include", "drop_duplicate"}:
            continue
        rows.append({
            "dataset_id": row.get("dataset_id"),
            "source_crop_id": row.get("source_crop_id"),
            "global_crop_id": row.get("global_crop_id"),
            "sample_type": row.get("sample_type"),
            "palm_score": row.get("palm_score"),
            "selection_action": row.get("selection_action"),
            "quality_flags": "|".join(row.get("quality_flags") or []),
            "crop_path": row.get("crop_path"),
        })

fields = [
    "dataset_id", "source_crop_id", "global_crop_id", "sample_type",
    "palm_score", "selection_action", "quality_flags", "crop_path",
]
for sample_type, filename in (
    ("POS_LOW_PALM", "peak_pos_low_palm.csv"),
    ("POS_RUNTIME", "peak_pos_runtime.csv"),
):
    selected = sorted(
        (row for row in rows if row["sample_type"] == sample_type),
        key=lambda row: (str(row["dataset_id"]), str(row["source_crop_id"])),
    )
    with (out / filename).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    print(filename, len(selected))
PY
```

这里允许 catalog 中当前为 `drop_duplicate` 的行进入**候选 CSV**，因为 07A finetune 会在 Gold 覆盖 pseudo 后重新去重，并让 `human_gold` 优先于同簇 pseudo；它不代表旧的 `drop_duplicate` 会直接进入训练。但不要在同一原图、同一近重复簇里标多个几乎相同的 Gold：多个 Gold 彼此仍可能再次去重，白白消耗人工。最终可用 Gold 数必须以 `finalize_train_finetune_report.json` 和 finetune canonical 中 `selection_action=include` 的 `human_gold` 行数为准。

人工查看文件：

- ROI 原图：CSV 的 `crop_path`；
- 如果该 source 已生成 MediaPipe overlay：`<source>/02_roi_crops/hand_landmarks_visualization/<同名 PNG>`；
- negative teacher-abstain：第 12 节 `negative_candidates/`；
- 原图/Palm anchor：HLMF 原 source 的 `images/`、`01_palm/palm_detections.jsonl`。

从 CSV 人工选择后，按 `dataset_id` 分别把 `source_crop_id` 写入各自的 `selected_source_crop_ids.txt`。不要把整个 CSV 自动全部送入 Gold；必须控制连续帧重复、场景比例和实际标注工作量。

### 13.4 当前工具缺口

HLMF 当前已有 04/05 的 subset 参数，但没有以下工具：

- 自动候选选择器；
- student–teacher disagreement 计算器；
- `subset_manifest.jsonl` / `subset_draft.jsonl` 生成器；
- subset PNG materialize/打包命令。

也就是说，HLMF 操作手册中 04/05 命令的前提是 subset 已由外部步骤正确生成。以下给出一个**临时、可审计的一次性 materialize 方案**；它不是仓库 Make 目标。若时间允许，长期应把同样逻辑实现为正式脚本并增加测试。

### 13.5 每个 source 单独建立工作目录

`train_sources/` 已经是 r3 的冻结训练输入，不应在其中新增工作副本。Gold 工作区单独放在 `hand_landmarker_gold/<FINETUNE_ID>/<dataset_id>/`。以 `peak_train_0714_dark` 为例：

```text
/root/autodl-tmp/TrainFab/HLML-2.0/
hand_landmarker_gold/v2-finetune-r1/peak_train_0714_dark/
├── selection/
│   ├── selected_source_crop_ids.txt
│   ├── subset_manifest.jsonl
│   ├── subset_draft.jsonl
│   └── subset_build_report.json
├── cvat/
│   ├── images/*.png
│   ├── subset_autolabel.xml
│   └── subset_reviewed.xml
├── 03_reviewed/
│   └── subset_gold.jsonl
└── qc/
    ├── cvat_export_stats.json
    └── cvat_import_stats.json
```

其他 source 在 `hand_landmarker_gold/v2-finetune-r1/<dataset_id>/` 建立独立工作区。不同 source 不共用 CVAT task，因为 04/05 按 basename 和 source-local `crop_id` 匹配，跨 source 可能同名。

`cvat/images/` 是供 CVAT 上传的字节副本，不是新的训练图片源。07A 最终 canonical 的 `crop_path` 仍由该 source 的 `crop_images_dir` 定位到 `train_sources/.../02_roi_crops/images` 原件；Gold 只覆盖标签，不替换 ROI 图片。

### 13.6 先制作 `selected_source_crop_ids.txt`

一行写一个 source-local `crop_id`，不要写 `dataset_id:` 前缀。例如：

```text
720x1280_8bit_20260714191305266_0:neg0:crop
720x1280_8bit_20260714191305321_0:palm0:crop
```

空行和以 `#` 开头的说明行可以忽略。一个文件只属于一个 source。

负例删除式复核中发现的清晰手，可根据第 12.2 节 CSV 把 global ID 转回 source-local ID。pseudo positive 则直接从 catalog 的 `source_crop_id` 取值。

### 13.7 一次性生成 subset JSONL 和 CVAT 图片目录

以下命令在服务器运行，只读取完整 source 的 manifest/draft/images，在独立 Gold 工作区写 subset。示例仍使用 Peak dark：

```bash
ROOT=/root/autodl-tmp/TrainFab/HLML-2.0
SRC="$ROOT/train_sources/HandViolenceEnhanced0714/peak_dark"
WORK="$ROOT/hand_landmarker_gold/v2-finetune-r1/peak_train_0714_dark"

mkdir -p "$WORK/selection"
# 先把人工整理的 selected_source_crop_ids.txt 上传到 $WORK/selection/

python - "$SRC" "$WORK" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
ids_path = work / "selection" / "selected_source_crop_ids.txt"
manifest_path = src / "02_roi_crops" / "hand_roi_crops_manifest.jsonl"
draft_path = src / "02_roi_crops" / "hand_landmarks_autolabel_draft.jsonl"
source_images = src / "02_roi_crops" / "images"
out_manifest = work / "selection" / "subset_manifest.jsonl"
out_draft = work / "selection" / "subset_draft.jsonl"
out_images = work / "cvat" / "images"
out_report = work / "selection" / "subset_build_report.json"

for path in (ids_path, manifest_path, draft_path):
    if not path.is_file():
        raise SystemExit("missing input: " + str(path))
if not source_images.is_dir():
    raise SystemExit("missing source images: " + str(source_images))
if out_manifest.exists() or out_draft.exists() or out_report.exists():
    raise SystemExit("refusing to overwrite an existing subset artifact")
if out_images.exists() and any(out_images.iterdir()):
    raise SystemExit("refusing to write into non-empty cvat/images")

def load_jsonl(path):
    rows = []
    index = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            crop_id = str(row.get("crop_id") or "")
            if not crop_id:
                raise SystemExit(f"{path}:{line_number}: missing crop_id")
            if crop_id in index:
                raise SystemExit(f"{path}: duplicate crop_id: {crop_id}")
            rows.append(row)
            index[crop_id] = row
    return rows, index

selected = []
for raw in ids_path.read_text(encoding="utf-8-sig").splitlines():
    value = raw.strip()
    if value and not value.startswith("#"):
        selected.append(value)
if not selected:
    raise SystemExit("selection is empty")
if len(selected) != len(set(selected)):
    raise SystemExit("selected_source_crop_ids.txt contains duplicates")
selected = sorted(selected)

_, manifest_index = load_jsonl(manifest_path)
_, draft_index = load_jsonl(draft_path)
missing_manifest = sorted(set(selected) - set(manifest_index))
missing_draft = sorted(set(selected) - set(draft_index))
if missing_manifest or missing_draft:
    raise SystemExit(
        "selection coverage error; missing_manifest={} missing_draft={}".format(
            missing_manifest[:10], missing_draft[:10]
        )
    )

manifest_rows = [manifest_index[crop_id] for crop_id in selected]
draft_rows = [draft_index[crop_id] for crop_id in selected]
manifest_ids = {str(row["crop_id"]) for row in manifest_rows}
draft_ids = {str(row["crop_id"]) for row in draft_rows}
if manifest_ids != draft_ids or manifest_ids != set(selected):
    raise SystemExit("manifest/draft/selection ID sets differ")

basenames = [Path(str(row.get("crop_path") or "")).name for row in manifest_rows]
if any(not name for name in basenames) or len(basenames) != len(set(basenames)):
    raise SystemExit("empty or duplicate crop basename inside subset")

out_manifest.parent.mkdir(parents=True, exist_ok=True)
out_images.mkdir(parents=True, exist_ok=True)

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

write_jsonl(out_manifest, manifest_rows)
write_jsonl(out_draft, draft_rows)

image_hashes = {}
for row, basename in zip(manifest_rows, basenames):
    source = source_images / basename
    target = out_images / basename
    if not source.is_file():
        raise SystemExit("source crop missing: " + str(source))
    shutil.copyfile(source, target)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    target_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    if source_hash != target_hash:
        raise SystemExit("copied image hash mismatch: " + basename)
    image_hashes[basename] = source_hash

report = {
    "status": "ok",
    "source_root": str(src),
    "selected_count": len(selected),
    "manifest_count": len(manifest_rows),
    "draft_count": len(draft_rows),
    "image_count": len(image_hashes),
    "selected_source_crop_ids": selected,
    "image_sha256": image_hashes,
}
out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: report[key] for key in ("status", "selected_count", "image_count")}, indent=2))
PY
```

必须满足：selected、manifest、draft、images 数量相同；没有重复 ID/basename；复制前后每张 PNG 的 SHA 相同。

### 13.8 HLMF 04：导出 CVAT 初始标注

在服务器 **HLMF 仓库根目录**执行：

```bash
cd /root/HandLandmarksFab
conda activate anfab

ROOT=/root/autodl-tmp/TrainFab/HLML-2.0
SRC="$ROOT/train_sources/HandViolenceEnhanced0714/peak_dark"
WORK="$ROOT/hand_landmarker_gold/v2-finetune-r1/peak_train_0714_dark"
export HAND_DATA_ROOT="$WORK"

python scripts/04_export_cvat_xml.py \
  --config configs/autolabel_train.yaml \
  --manifest "$WORK/selection/subset_manifest.jsonl" \
  --draft-jsonl "$WORK/selection/subset_draft.jsonl" \
  --output-xml "$WORK/cvat/subset_autolabel.xml"
```

真实输出：

```text
$WORK/cvat/subset_autolabel.xml
$WORK/qc/cvat_export_stats.json
```

04 **不会复制图片**。`cvat_export_stats.json` 会写 `copied_images=0` 和 `copy_policy=disabled_use_roi_crops_images_directly`。真正上传 CVAT 的图片是上一步生成的：

```text
$WORK/cvat/images/*.png
```

终端打印的 `upload_images=$WORK/02_roi_crops/images` 只是配置推导提示，不是本 subset 实际图片目录。

### 13.9 CVAT 中的实际操作

每个 source 建一个独立 image task：

1. 新建 task；
2. 在 label 的 Raw 配置中粘贴 HLMF `configs/cvat_label.json`；
3. 上传 `$WORK/cvat/images/` 中的全部图片；
4. 导入 `$WORK/cvat/subset_autolabel.xml`，格式选择 **CVAT for images 1.1**；
5. 逐图检查并保存；
6. 完成后仍以 **CVAT for images 1.1** 导出为 `$WORK/cvat/subset_reviewed.xml`。

CVAT 的 skeleton 子点名称是 1～21，而模型 ID 是 0～20，对应关系如下：

| CVAT 点 | 模型 ID | 部位 |
|---:|---:|---|
| 1 | 0 | wrist，手腕 |
| 2～5 | 1～4 | 拇指 CMC、MCP、IP、指尖 |
| 6～9 | 5～8 | 食指 MCP、PIP、DIP、指尖 |
| 10～13 | 9～12 | 中指 MCP、PIP、DIP、指尖 |
| 14～17 | 13～16 | 无名指 MCP、PIP、DIP、指尖 |
| 18～21 | 17～20 | 小指 MCP、PIP、DIP、指尖 |

每张图只允许以下三种最终状态之一：

**A. 明确有目标手**

- 恰好一个 `hand_landmarks` skeleton；
- 恰好 21 个子点；
- 恰好一个 `Left` 或 `Right` tag；
- 不得保留 `no_hand`；
- 修正每个点到真实关节/指尖，而不是仅把整团平移；
- teacher-abstain 候选初始 XML 通常带 `no_hand`，必须先删除该 tag，再补 skeleton 和 handedness。

**B. 明确无手**

- 删除 skeleton；
- 删除 Left/Right；
- 添加并只保留 `no_hand`。

**C. 无法可靠标注**

- 添加 `ignore_for_training`；
- 不猜测完整 21 点；
- 典型情况包括目标手不唯一、严重遮挡到无法确定关节、ROI 中多手且无法确认 Palm anchor 指向哪只手。

若 ROI 中有两只手，只标 Palm anchor 指向的目标手。应通过 source manifest 的 `palm_det_id`、`roi_rect`，必要时结合该 source 的 `01_palm/palm_detections.jsonl` 和原图判断。若 TrainFab 交付中没有这些上游文件，应回 HLMF 原 source 查找；无法确定则 `ignore_for_training`，不能以“更大、更居中、Google 先检测到”为替代规则。

### 13.10 HLMF 05：把 reviewed XML 导回 Gold JSONL

仍在 **HLMF 仓库根目录**，并保持每个 source 使用自己的 `$WORK`：

```bash
cd /root/HandLandmarksFab
conda activate anfab

ROOT=/root/autodl-tmp/TrainFab/HLML-2.0
SRC="$ROOT/train_sources/HandViolenceEnhanced0714/peak_dark"
WORK="$ROOT/hand_landmarker_gold/v2-finetune-r1/peak_train_0714_dark"
export HAND_DATA_ROOT="$WORK"

python scripts/05_import_cvat_xml.py \
  --config configs/autolabel_train.yaml \
  --reviewed-xml "$WORK/cvat/subset_reviewed.xml" \
  --manifest "$WORK/selection/subset_manifest.jsonl" \
  --draft-jsonl "$WORK/selection/subset_draft.jsonl" \
  --output-jsonl "$WORK/03_reviewed/subset_gold.jsonl"

python -m json.tool "$WORK/qc/cvat_import_stats.json"
```

05 的 `--output-jsonl` 参数真实存在，但没有 `--output-report` 参数；报告固定写入配置的 `paths.qc_dir/cvat_import_stats.json`。因此必须让每个 source 使用独立 `$WORK`，否则多个 task 的报告会互相覆盖。

检查：

- `import_integrity.errors` 为空；
- `label_heuristics.errors` 为空；
- `coverage.manifest_images = coverage.xml_images = coverage.reviewed_rows = coverage.seen_manifest_images = N`；
- `coverage.missing_from_xml = 0`；
- 非 ignored Gold positive 都有 21 点和 Left/Right；
- 非 ignored 行没有 `cvat_import_errors`。

有问题就回 CVAT 修改并重新导出/导入；不要手改 `subset_gold.jsonl` 或伪造 QC 报告。

### 13.11 在 HLMF `finalize_train.yaml` 登记 Gold 三件套

每个完成 Gold 的 source，在原 source 条目中增加：

```yaml
gold_manifest: <相对 source root 的 subset_manifest.jsonl>
gold_labels: <相对 source root 的 subset_gold.jsonl>
gold_import_report: <相对 source root 的 cvat_import_stats.json>
```

这三个字段既可写相对该 source `root` 的路径，也可写展开后为绝对路径的表达式。由于本文把 Gold 工作区放在冻结 `train_sources/` 之外，推荐使用 `${HAND_DATA_ROOT...}/hand_landmarker_gold/...`，避免写很长且容易算错的 `../../`。Peak dark 的 source root 是 `.../HandViolenceEnhanced0714`，示例为：

```yaml
  - dataset_id: peak_train_0714_dark
    contributor: Peak
    root: ${HAND_DATA_ROOT:-../autodl-tmp/TrainFab/HLML-2.0}/train_sources/HandViolenceEnhanced0714
    manifest: peak_dark/02_roi_crops/hand_roi_crops_manifest.jsonl
    pseudo_labels: peak_dark/02_roi_crops/hand_landmarks_autolabel_draft.jsonl
    crop_images_dir: peak_dark/02_roi_crops/images
    gold_manifest: ${HAND_DATA_ROOT:-../autodl-tmp/TrainFab/HLML-2.0}/hand_landmarker_gold/v2-finetune-r1/peak_train_0714_dark/selection/subset_manifest.jsonl
    gold_labels: ${HAND_DATA_ROOT:-../autodl-tmp/TrainFab/HLML-2.0}/hand_landmarker_gold/v2-finetune-r1/peak_train_0714_dark/03_reviewed/subset_gold.jsonl
    gold_import_report: ${HAND_DATA_ROOT:-../autodl-tmp/TrainFab/HLML-2.0}/hand_landmarker_gold/v2-finetune-r1/peak_train_0714_dark/qc/cvat_import_stats.json
```

其他 source 将路径中的 `dataset_id` 和工作目录替换为各自值。正式流程始终登记三件套；虽然 07A 实现上 `gold_manifest` 和 `gold_import_report` 可选，省略它们会失去覆盖率和身份审计证据。

### 13.12 HLMF 07A：自动聚合为 finetune canonical

注意把 `HAND_DATA_ROOT` 从单个 `$WORK` 恢复为总 HLML 数据根。仍在 **HLMF 仓库根目录**执行：

```bash
cd /root/HandLandmarksFab
conda activate anfab

export HAND_DATA_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0
make finalize_train_finetune
```

等价真实 CLI：

```bash
python scripts/07A_finalize_training_labels.py \
  --config configs/finalize_train.yaml \
  --stage finetune
```

输出：

```text
/root/autodl-tmp/TrainFab/HLML-2.0/train_finetune_merged/
├── 05_labels/
│   ├── hand_train_catalog_finetune.jsonl
│   ├── hand_training_labels_finetune.jsonl
│   └── hand_training_excluded_finetune.jsonl
└── qc/
    └── finalize_train_finetune_report.json
```

07A 自动：

1. 读取所有 source 的完整 manifest 和 pseudo；
2. 对登记了 Gold 的 source，按 source-local `crop_id` 覆盖 pseudo；
3. 将 Gold 标记为 `human_gold/gold`，其余保持 `mediapipe_pseudo/pseudo`；
4. 重新分型、去重、设置采样和 loss 字段；
5. 按 `dataset_id` 生成 global namespace；
6. 原子写出最终 JSONL 和 SHA。

检查 `finalize_train_finetune_report.json`：

- `status=ok`；
- `fatal_errors=[]`；
- 每个 source 的 `gold` 数量与 CVAT task 一致；
- `counts.provenance.human_gold` 为预期数量；
- 输出 `sha256` 三项存在；
- finetune canonical 同时有 `supervision_tier=gold` 和 `pseudo`；
- 同一 global `crop_id` 只出现一行，Gold 已覆盖 pseudo。

没有任何 Gold 时，07A finetune 会以 `finetune_requires_at_least_one_human_gold_row` 拒绝发布。

07A 只负责生成训练接口，不负责决定每个 batch 中 Gold 占 30% 还是 40%；该比例由 HLML finetune sampler 决定。

## 14. Multitask 的含义与精确启动方法

### 14.1 Multitask 到底做什么

Multitask 从 r3 geometry 的 `best.weights.h5` 开始，继续使用相同 pseudo positive，并加入第 12 节人工确认的 true negative。

它同时训练三个输出：

- landmarks：仍然是主要任务；
- `hand_flag`：学会拒绝 Palm 产生但实际无手的 ROI；
- handedness：用很小的权重学习 Left/Right。

默认 batch 组成：

```text
POS_RUNTIME               72%
POS_LOW_PALM              18%
NEG_RUNTIME_CANDIDATE      8%
NEG_LOW_PALM_CANDIDATE     2%
```

即 90% positive、10% confirmed negative。学习率 `5e-5`，低于 geometry 的 `3e-4`。

它能帮助过滤降低 Palm threshold 后产生的假 ROI，但不能：

- 恢复 Palm 根本没有生成的 ROI；
- 直接修正错误的 pseudo 21 点；
- 重新训练 Palm Detector；
- 替代 Gold finetune。

### 14.2 人工与程序分别负责什么

人工负责：

1. 删除 negative review 工作区中所有有手或不确定图片；
2. 确认全部 reviewer 完成；
3. 回传正确的 retained 目录；
4. 查看 gate/inspect/训练/Val 报告；
5. 决定 multitask 是否比 geometry 更适合作为候选。

程序负责：

1. 生成带 SHA 证据的 confirmed negative；
2. 检查数量门槛和 review 字段；
3. 检查数据路径、schema、图片 SHA 和 split 泄漏；
4. 从 geometry best 初始化；
5. 按固定比例采样；
6. 自动 checkpoint、降学习率、早停和 best 选择。

### 14.3 数据门禁

在服务器 **HLML 仓库根目录**执行：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
make paths
make check-multitask-data
```

真实调用：

```bash
python -B scripts/check_multitask_data.py \
  --config configs/train_multitask.yaml
```

报告：

```text
/root/autodl-tmp/TrainFab/HLML-2.0/
hand_landmarker_runs/v2-pretrain-r3/multitask_data_gate.json
```

当前硬门槛：

- confirmed negative 总数不少于 500；
- `NEG_RUNTIME_CANDIDATE` 不少于 100；
- `NEG_LOW_PALM_CANDIDATE` 不少于 100；
- `review_method=retained_after_visual_deletion_review`；
- 每条 negative 都有 `reviewer`、`reviewed_at`、`review_method`、`review_image_sha256`。

查看报告中的：

- `status`；
- `confirmed_negative_count`；
- `confirmed_negative_by_sample_type`；
- `checks`；
- `violations`。

数量不足时仍可使用 geometry，但不可启动 multitask。不要通过把已删除图片补回来、把模糊图片当背景，或降低门槛来“通过”。

### 14.4 Inspect

运行：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
make inspect-multitask
```

它检查 multitask Train、Val、锁定 Test、图片 shape/SHA、schema 和跨 split 泄漏。Make 目标默认把 JSON 打到终端；若要保存：

```bash
python -B scripts/inspect_dataset.py \
  --config configs/train_multitask.yaml \
  --output /root/autodl-tmp/TrainFab/HLML-2.0/hand_landmarker_runs/v2-pretrain-r3/inspect_multitask.json
```

输入核心文件：

```text
train_pretrain_curated/v2-pretrain-r3/
├── 05_labels/hand_training_labels_pretrain_multitask.jsonl
└── qc/sha256_manifest.json

hand_landmarker_runs/v2-pretrain-r3/
└── geometry/checkpoints/best.weights.h5

val_merged/05_labels/hand_validation_labels.jsonl
test_merged/05_labels/hand_test_labels.jsonl
train_sources/**/02_roi_crops/images/*.png
```

### 14.5 启动 multitask

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
make pretrain-multitask
```

该目标会自动再次执行：

```text
check-multitask-data
inspect-multitask
scripts/train.py --config configs/train_multitask.yaml
```

输出：

```text
/root/autodl-tmp/TrainFab/HLML-2.0/
hand_landmarker_runs/v2-pretrain-r3/multitask/
├── checkpoints/
│   ├── best.weights.h5
│   ├── last.weights.h5
│   ├── final.weights.h5
│   └── *.state.json / *.state/
├── experiment_metadata.json
├── history.json
├── training_report.json
├── model_summary.txt
└── logs/
    ├── history.csv
    └── tensorboard/
```

成功完成后检查：

- `experiment_metadata.json.status=complete`；
- `training_report.json.status=complete`；
- `starting_state.path` 指向 r3 geometry `best.weights.h5`；
- `starting_state.mode=initial_weights`，表示新阶段使用新 optimizer；
- `checkpoint_selection.history_best_epoch/value` 与 history 一致；
- `best.weights.h5` 与 `final.weights.h5` SHA 一致；
- `val_landmark_mae` 没有因 presence 训练显著恶化。

multitask best 使用：

```text
val_multitask_score
  = val_landmark_mae
  + 0.02  × (1 - val_hand_flag_accuracy)
  + 0.005 × (1 - val_handedness_accuracy)
```

分类任务参与选 best，但 landmark 仍占主导。

### 14.6 Val、Test、端到端推理和导出

先运行 Val：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
make eval-val-multitask
```

查看：

```text
hand_landmarker_runs/v2-pretrain-r3/eval/multitask/val/
├── metrics.json
└── predictions.jsonl
```

当前 Val 全是 positive，所以 `hand_flag_accuracy=1` 不能证明模型能拒绝背景。应另建一个不参与训练的 presence dev，包含明确 positive 和 100～200 个以上明确 negative，用它联合选择 Palm threshold 与 Hand flag threshold。

只有 checkpoint 和阈值冻结后才运行：

```bash
make eval-test-multitask
```

查看：

```text
hand_landmarker_runs/v2-pretrain-r3/eval/multitask/test/
├── metrics.json
└── predictions.jsonl
```

端到端推理：

```bash
make infer-multitask
```

查看：

```text
hand_landmarker_inference/v2-pretrain-r3/multitask/
├── summary.json
├── predictions.jsonl
└── rendered/
```

正式导出：

```bash
make export-multitask
```

查看：

```text
hand_landmarker_runs/v2-pretrain-r3/export/multitask/
├── hand_landmarker_v2.onnx
├── hand_landmarker_v2.contract.json
└── model_conversion/
```

接受 multitask 的最低条件：

- geometry Val mean/P90 相对退化不超过 3%；
- PCK 下降不超过 2 个百分点；
- presence dev 上确实降低 false positive；
- 独立推理中降低 Palm threshold 后新增假 ROI 能被 `hand_flag` 拒绝；
- 固定 examples 的 landmark 没有明显恶化。

## 15. Gold finetune 的含义、现状与启动前置条件

### 15.1 当前到底能不能直接运行

不能直接运行。

HLML 底层已经具备：

- `scripts/train.py` 接受 `stage=finetune`；
- sampler 能在每个 batch 混合 Gold/pseudo；
- checkpoint 能区分 `initial_checkpoint` 和 `resume_checkpoint`；
- 数据契约和相关单元测试已经覆盖一部分 finetune 行为。

但公开操作面仍明确限定为 pretrain：

- 没有 `configs/train_finetune.yaml`；
- 没有 finetune curate/hash manifest；
- 没有 finetune inspect；
- 现有 `check_pretrain_smoke.py` 只接受 pretrain，不能冒充 finetune smoke；
- 没有 finetune Val/Test/infer/export 配置；
- 没有 finetune Make target；
- `tests/test_stage_routes.py` 当前还强制 `configs/` 只包含 9 个 pretrain 配置，并禁止出现 finetune。

所以第 13 节的 HLMF Gold 数据制作可以现在进行；HLML 训练必须等正式入口作为单独一次实现任务完成并通过测试。不存在可以直接执行的 `make finetune`。

### 15.2 Finetune 与 multitask 的区别

| 阶段 | 主要新监督 | 解决的问题 |
|---|---|---|
| geometry | 大量 pseudo positive | 学会基本 21 点几何 |
| multitask | 人工确认 negative | 让 `hand_flag` 拒绝假 ROI |
| Gold finetune | 人工修正的 21 点 positive | 纠正 teacher 漏标、点位偏差和塌缩 |

推荐最终依赖：

```text
geometry best → multitask best → Gold finetune
```

其中 pretrain→新阶段必须使用 `initial_checkpoint`：只载入模型权重，创建新 optimizer，从新阶段 epoch 0 开始。

`resume_checkpoint` 只用于**同一次 finetune 被中断后续跑**，会在 sidecar 存在时恢复 optimizer 和已完成 epoch。两者互斥。

### 15.3 Finetune canonical 的硬契约

HLMF 输出的 `hand_training_labels_finetune.jsonl` 必须满足：

- `schema_version=train_finalize_v1`；
- `training_stage=finetune`；
- 训练行 `selection_action=include`；
- 同时存在 `supervision_tier=pseudo` 和 `gold`；
- Gold 行为 `annotation_provenance=human_gold`；
- pseudo 行为 `annotation_provenance=mediapipe_pseudo`；
- `sampling_bucket=<supervision_tier>:<sample_type>`；
- positive 恰好 21 点、ID 0～20；
- negative 的 landmark 数组为空；
- 所有 sampling/loss weight 非负、有限。

CVAT XML、`evaluation_gold_v1` Val JSONL 或手工拼接文件都不能直接代替该 canonical。

### 15.4 `gold_fraction=0.30` 不等于“重复很少”

当前 sampler 强制 `training.gold_fraction` 在 0.30～0.50，并按每个 batch 精确分配。

如果 combined 数据约 6 万条、Gold 只有 500，而 `sampling.epoch_size` 默认等于全部记录数，那么每个 epoch 约抽 1.8 万次 Gold：

- 500 Gold：平均每条每 epoch 约 36 次；
- 1000 Gold：平均每条每 epoch 约 18 次。

这会严重重复，因此旧版“从 0.30 开始即可避免过度重复”的说法不成立。

建议正式 finetune 配置显式减小 `sampling.epoch_size`。例如：

```text
batch_size = 64
epoch_size = 6400       # 每 epoch 100 batch
gold_fraction = 0.30
```

每 epoch 约抽 1920 个 Gold draw：

- 500 Gold：平均约 3.84 次；
- 1000 Gold：平均约 1.92 次。

`6400` 是当前规模下的建议起点，不是仓库现成配置；仍需结合 Val、增强和早停验证。

### 15.5 sample type 的隐藏门槛

只要某个 `sample_type` 的 fraction 大于 0，Gold 和 pseudo 两层都必须至少存在一个该类型，否则 sampler fail-closed。

如果进行 landmark-first positive finetune，可使用类似：

```text
POS_RUNTIME               75%
POS_LOW_PALM              25%
NEG_RUNTIME_CANDIDATE      0%
NEG_LOW_PALM_CANDIDATE     0%
```

这要求 Gold 中同时存在 `POS_RUNTIME` 和 `POS_LOW_PALM`。因此候选选择不能全部来自单一类型。

如果想在 finetune 中继续抽 negative，则 Gold/pseudo 两层还必须各自包含启用的 negative 类型。当前 HLMF finetune canonical 中的 pseudo negative 仍是未经人工确认的 candidate，而 HLML confirmed negative 在另一份 curated multitask JSONL 中；仓库目前没有把两者安全合并为 finetune canonical 的正式工具。因此短期不应在 finetune 中启用这些 pseudo negative。

若从 multitask best 做 positive-only finetune，需要在独立 presence dev 上重新验证 `hand_flag` 是否遗忘；模型权重不会自动保证分类能力保持不变。

### 15.6 正式 finetune 入口至少要补齐什么

这是一份实现清单，不是当前命令清单：

1. 独立实验身份，例如 `HAND_FINETUNE_ID=v2-finetune-r1`；
2. `configs/train_finetune.yaml`；
3. finetune canonical 的 SHA manifest；
4. `inspect-finetune`；
5. 独立 finetune smoke 和 gate；
6. 正式 `finetune` 训练目标；
7. finetune 专用 Val/Test 配置；
8. finetune 专用 infer/export 配置；
9. 对应单元测试和 README/Makefile 文档。

不能把 finetune checkpoint 放在：

```text
hand_landmarker_runs/v2-pretrain-r3/finetune/
```

该路径同时包含 `pretrain` 和 `finetune` 阶段 token，评估/推理/导出 provenance guard 可能判冲突。应使用独立身份：

```text
hand_landmarker_runs/v2-finetune-r1/finetune/
```

finetune config 不能直接沿用 r3 的 `curation_manifest`；该 manifest 认证的是 pretrain curated JSONL，不是新 Gold+pseudo canonical。正式入口需要自己的 hash manifest。

### 15.7 正式入口完成后的预期操作顺序

以下目标名只是建议接口；在代码真正实现并通过 `make test` 前不得执行或写进提交日志：

```bash
# 建议将来提供
make inspect-finetune
make finetune-smoke
make finetune
make eval-val-finetune
make eval-test-finetune
make infer-finetune
make export-finetune
```

底层真实训练 CLI 已存在；一旦正式 `configs/train_finetune.yaml` 完成，最终 Make target 会调用类似：

```bash
python -B scripts/train.py --config configs/train_finetune.yaml
```

建议的训练配置原则：

- `stage: finetune`；
- `model.checkpoint_stage: finetune`；
- `data.labels` 指向 HLMF `hand_training_labels_finetune.jsonl`；
- `initial_checkpoint` 优先指向 Val 最优 multitask best；若只比较 landmark-first 分支，可指向 geometry best；
- `gold_fraction=0.30`；
- `sampling.epoch_size` 显式限制，例如 6400；
- 全网络低学习率，约为 geometry 的 1/10～1/30；
- 短训练、严格 early stopping；
- 先 positive-only sample fractions；
- 使用独立 finetune run 目录；
- Val 决策冻结后才运行 Test 和正式导出。

若正式入口无法在剩余时间内完成，Gold 数据仍然有价值，可以作为后续改进资产；本次交付使用经 Val 证明有效的 r3 geometry 或 multitask，不临时绕过仓库契约。

## 16. 从新原图到可训练模型的全流程总表

本节用于以后新增数据 source。当前 r3 已完成其中 16.1～16.4，无需重复执行。

### 16.1 HLMF：准备一个新的 Train source

原图要求：

- 1280×720；
- 正向，不由脚本自动旋转；
- 灰度 `.tiff`；
- 与 Val/Test/inference 按完整 session 隔离。

推荐 source 目录：

```text
<SOURCE_ROOT>/
├── images/*.tiff
├── 01_palm/
├── 02_roi_crops/
├── 03_reviewed/
├── 04_visualization/
├── 05_labels/
└── qc/
```

在服务器 **HLMF 仓库根目录**：

```bash
cd /root/HandLandmarksFab
conda activate anfab
export HAND_DATA_ROOT=<SOURCE_ROOT>

make validate_images_train
make palm_detection_train
make build_roi_train
make run_mediapipe_train
```

各步骤含义和输出：

| 命令 | 人工输入 | 程序输出 | 主要检查 |
|---|---|---|---|
| `validate_images_train` | `images/*.tiff` | `qc/image_validation_report.json` | 尺寸、可读性、灰度、方向 |
| `palm_detection_train` | 原图、Palm ONNX | `01_palm/palm_detections.jsonl`、`qc/palm_detection_stats.json` | detection/negative candidate 数量、阈值 |
| `build_roi_train` | Palm JSONL、原图 | `02_roi_crops/images/*.png`、`hand_roi_crops_manifest.jsonl`、`qc/roi_crop_stats.json` | 256×256、旋转 ROI、图片数量 |
| `run_mediapipe_train` | ROI PNG、manifest | `hand_landmarks_autolabel_draft.jsonl`、`qc/mediapipe_roi_stats.json` | positive/abstain、21 点、handedness |

如需同时生成 MediaPipe draft 的 ROI 可视化：

```bash
make run_mediapipe_train VISUALIZE_MEDIAPIPE_ROIS=1
```

输出：

```text
02_roi_crops/hand_landmarks_visualization/*.png
```

大规模 pseudo pretrain 不要求所有 Train ROI 进入 CVAT；只有选中的 Gold subset 才走第 13 节 04/05。

### 16.2 HLMF：将新 source 登记到 07A

编辑 HLMF `configs/finalize_train.yaml`，为新 source 增加唯一 `dataset_id`：

```yaml
sources:
  - dataset_id: <全局唯一且带版本的 ID>
    contributor: <贡献者>
    root: <source 所在父目录>
    manifest: <相对 root 的 hand_roi_crops_manifest.jsonl>
    pseudo_labels: <相对 root 的 hand_landmarks_autolabel_draft.jsonl>
    crop_images_dir: <相对 root 的 02_roi_crops/images>
```

不要手工给原 JSONL 加 Peak/Soar 前缀；07A 会自动生成 `dataset_id:source_crop_id` global namespace。

### 16.3 HLMF：生成 pretrain canonical

```bash
cd /root/HandLandmarksFab
conda activate anfab
export HAND_DATA_ROOT=/root/autodl-tmp/TrainFab/HLML-2.0
make finalize_train_pretrain
```

输出：

```text
train_pretrain_merged/
├── 05_labels/
│   ├── hand_train_catalog_pretrain.jsonl
│   ├── hand_training_labels_pretrain.jsonl
│   └── hand_training_excluded_pretrain.jsonl
└── qc/
    └── finalize_train_pretrain_report.json
```

检查：

- `status=ok`、`fatal_errors=[]`；
- 每个 source 的 `manifest/pseudo/included/excluded`；
- `quality_tiers`、`sample_types`、`actions`；
- `crop_image_missing` 必须为 0；
- 输出 SHA 存在；
- 新 source 的实际 included 数不是 0。

HLMF 到此完成“可交付给 HLML 的全量 pretrain canonical”。

### 16.4 HLML：以新实验 ID curate

在 HLML `Makefile` 顶部设置并提交新的 ID，例如：

```make
HAND_TRAIN_ROOT := /root/autodl-tmp/TrainFab/HLML-2.0
HAND_PRETRAIN_ID := v2-pretrain-r4
```

不得复用已经有 run/snapshot 的 ID，不通过 `overwrite` 覆盖旧实验。

在服务器 **HLML 仓库根目录**：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
make paths
make compile
make test-unit
make pretrain-curate
make test
```

`pretrain-curate` 输出：

```text
train_pretrain_curated/<ID>/
├── 05_labels/
│   ├── hand_training_labels_pretrain_landmarks.jsonl
│   ├── hand_training_labels_pretrain_multitask.jsonl
│   └── hand_training_labels_pretrain_smoke.jsonl
├── audit/
└── qc/

hand_landmarker_reviews/<ID>/negative_candidates/
```

检查 `qc/curation_report.json`：positive 数、来源构成、sample type 构成、negative review queue、排除原因。检查 `qc/sha256_manifest.json`：source label、canonical label 和图片哈希。

### 16.5 HLML：geometry 门禁与完整训练

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
make doctor
make inspect-geometry
make pretrain-geometry-smoke
make pretrain-geometry
```

含义：

1. `doctor`：Python/TensorFlow/CUDA/GPU；
2. `inspect-geometry`：Train/Val/Test schema、图片、SHA、泄漏；
3. `pretrain-geometry-smoke`：固定 128 ROI 可过拟合门禁；
4. `pretrain-geometry`：再次验证 smoke 后启动完整训练。

输入：

```text
train_pretrain_curated/<ID>/05_labels/hand_training_labels_pretrain_landmarks.jsonl
train_pretrain_curated/<ID>/qc/sha256_manifest.json
val_merged/05_labels/hand_validation_labels.jsonl
train_sources/**/images/*.png
```

输出：

```text
hand_landmarker_runs/<ID>/geometry/
```

### 16.6 HLML：geometry 评估、推理、导出

完整训练成功后先运行：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29
make eval-val-geometry
```

根据 Val 冻结 checkpoint 和策略后，才运行：

```bash
make eval-test-geometry
make infer-geometry
make export-geometry
```

不要因为 Test 指标不满意再回头调训练；这会把 Test 变成第二个 Val。

### 16.7 后续分支

```text
geometry
├── 删除式负例复核 → curate-reviewed → multitask
└── HLMF Gold subset → CVAT → 05 → finalize_train_finetune
                               └── 等 HLML 正式 finetune 入口
```

## 17. 在哪里看训练与评估结果

### 17.1 训练进行中

```text
hand_landmarker_runs/<ID>/<phase>/
├── experiment_metadata.json
├── logs/history.csv
├── logs/tensorboard/
└── checkpoints/
    ├── best.weights.h5
    ├── best.weights.h5.state.json
    ├── last.weights.h5
    └── last.weights.h5.state.json
```

- `history.csv`：每个已完成 epoch 的 train/val loss 和指标；
- TensorBoard：曲线查看；
- `best.state.json`：当前 best epoch 与 monitor；
- `last.state.json`：最近完成 epoch；
- `experiment_metadata.json.status=running` 只表示进程曾开始，最终必须看 complete。

### 17.2 训练完整成功

必须新增：

```text
history.json
training_report.json
checkpoints/final.weights.h5
```

并满足：

- `experiment_metadata.json.status=complete`；
- `training_report.json.status=complete`；
- best checkpoint selection 与 history 最优 epoch 一致；
- `final.weights.h5` 来源为 best；
- 正式使用 `best.weights.h5`，不是 last。

### 17.3 ROI Val/Test

Geometry：

```text
hand_landmarker_runs/<ID>/eval/geometry/val/metrics.json
hand_landmarker_runs/<ID>/eval/geometry/val/predictions.jsonl
hand_landmarker_runs/<ID>/eval/geometry/test/metrics.json
hand_landmarker_runs/<ID>/eval/geometry/test/predictions.jsonl
```

Multitask 将路径中的 `geometry` 换成 `multitask`。

重点指标：

- mean/median/P90/P95 pixel error；
- NME；
- PCK@0.05/0.10/0.15；
- 按 Peak/Soar/source 分组；
- Gold-relative 骨架长度、bbox 面积和 collapse 比例；
- multitask 的 presence 还必须看独立 negative dev，不能只看 positive-only Val。

### 17.4 独立整图推理

```text
hand_landmarker_inference/<ID>/<phase>/
├── summary.json
├── predictions.jsonl
└── rendered/
```

逐图先区分：

1. Palm 是否产生正确数量 ROI；
2. 已有 ROI 的 `hand_flag` 是否接受/拒绝合理；
3. 已接受 ROI 的 21 点是否准确或塌缩。

不要把 Palm 无 ROI、Hand flag 拒绝和 landmark 错误混成一个“漏检率”。

### 17.5 导出

```text
hand_landmarker_runs/<ID>/export/<phase>/
├── hand_landmarker_v2.onnx
├── hand_landmarker_v2.contract.json
└── model_conversion/
```

检查 contract：

- 模型/训练 checkpoint SHA；
- ONNX SHA；
- 输入输出名称、顺序和 shape；
- operator whitelist；
- Keras 融合前后 parity；
- Keras→ONNX parity。

## 18. 人工与程序职责总表

| 环节 | 人工负责 | 程序负责 |
|---|---|---|
| HLMF 原图准备 | 采集、session 隔离、放置 TIFF、配置 source | 格式校验 |
| Palm/ROI/pseudo | 启动命令、检查 QC | Palm、ROI crop、MediaPipe draft |
| 07A pretrain | 登记 source、检查报告 | namespace、校验、去重、分型、canonical |
| HLML curate | 设置新实验 ID、启动、检查来源分布 | 生成 geometry/smoke/multitask 索引和 review workspace |
| negative review | 删除有手/模糊图；只留明确背景 | 无 |
| 7z/网盘 | 保持字节和路径；staging 回传 | 最终 SHA fail-closed 校验 |
| curate-reviewed | 全部 reviewer 完成后启动 | decisions、SHA、curated 重建、overlap gate |
| geometry | 启动、监控、判断 Val | 采样、训练、checkpoint、早停 |
| multitask gate | 查看失败原因，不绕过 | 数量、review 证据、schema 检查 |
| multitask | 启动、比较 geometry | 从 geometry best 初始化并联合训练 |
| Gold 候选 | 选 source-local ID、避免 session 泄漏 | 当前无正式自动选择工具 |
| CVAT | 逐图确认 presence/21点/Left-Right/ignore | 04 生成初稿、05 恢复结构并校验 |
| 07A finetune | 登记 Gold 三件套、检查报告 | Gold 覆盖 pseudo、生成 combined canonical |
| HLML finetune | 当前需先实现正式入口 | 底层 trainer/sampler 已有部分能力 |
| Val/Test | 先 Val 选方案，冻结后 Test | 生成 metrics/predictions |
| infer/export | 查看 examples、板端验收 | 端到端推理、ONNX 和 contract |

最终执行检查清单：

- [ ] r3 geometry 完整成功，Val 与 r2 比较完成；
- [ ] r3 geometry trained ONNX 已导出并作为 fallback；
- [ ] negative review 只保留明确背景；
- [ ] 回传没有覆盖解压，staging SHA 预检通过；
- [ ] `pretrain-curate-reviewed` 成功；
- [ ] multitask data gate 达标；
- [ ] multitask Val landmarks 未明显退化；
- [ ] presence dev 能证明 `hand_flag` 确实拒绝 negative；
- [ ] 每个 Gold source 有独立 subset/CVAT task；
- [ ] CVAT import coverage 完整、errors 为空；
- [ ] HLMF finetune report 为 ok，Gold/pseudo 两层都存在；
- [ ] 未使用与 inference/Val/Test 同 session 的 Gold Train；
- [ ] HLML finetune 正式入口未完成前，没有执行临时 `make finetune`；
- [ ] 最终模型和阈值冻结后才运行 Test；
- [ ] 最终 ONNX contract、厂商转换和板端结果均已保存。
