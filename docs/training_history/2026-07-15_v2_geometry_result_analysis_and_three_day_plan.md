# v2 geometry 完整训练结果分析与三天优化方案

日期：2026-07-15  
对象：`v2-pretrain-r2` geometry 完整训练、Gold Val/Test、独立 Palm→Hand 推理  
结论性质：只读审计与决策建议；本次没有修改服务器文件、训练代码或配置

## 1. 结论先行

当前模型已经证明“模型定义和训练链路能正常工作”：训练可以稳定收敛，checkpoint、早停、验证、推理都能完整执行，输出无 NaN、Inf 或越界，“关键点漫天飞舞”的数值故障已经消失。

但当前 `best` 还不适合作为最终效果模型。它的主要问题不是训练轮数不够，也没有证据表明首先应扩大模型，而是以下四项问题叠加：

1. **明显的 Train–Val 泛化差距**：训练误差持续下降，而 Gold Val 在 epoch 20 后恶化；按原数据原配置继续 geometry 很可能只会更贴合伪标签。
2. **Peak 困难域覆盖不足和 teacher selection bias**：Google 在大量 Peak Gold positive 上完全不输出 Hand Landmarker 结果，这些困难姿态不会进入 positive 伪标签训练。
3. **r2 存在一项确定的数据交付错误**：整批 `peak_train_0714_dark` 没有进入训练，其中包含 18,525 个原始 positive 候选。
4. **端到端漏检主要发生在 Palm Detector，而非当前 Hand Landmarker 的 `hand_flag`**；geometry 的 `hand_flag` 已饱和为 1，既不能恢复无 ROI 的手，也不能过滤降低 Palm 阈值后出现的假 ROI。

因此，三条路线不应纯选其一。三天内推荐的主线是：

> **修复已有 Peak 暗光数据交付 → 使用新实验 ID 重建数据并训练一次新 geometry → 用 500～1000 个高价值 Gold positive 做短程 finetune。**

人工负例复核和 multitask 可并行推进，用于让 `hand_flag` 可用并支持降低 Palm 阈值；但它不是修复 landmark 塌缩的主手段。当前不建议单纯延长 r2、随机扩大同类伪标签数据或扩大模型。

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

这是当前最值得先修的确定性问题。不能在 r2 的已哈希快照内原地补文件；应正确交付 ROI 后使用新 `HAND_PRETRAIN_ID` 重新 finalize、curate、inspect 和 smoke，保证实验可复现。

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

另一个值得单独排查的上游信号是 Palm `head7`：其 ROI 的塌缩率显著高于 `head14`。这可能表示该 head 承担了更困难/更小的手，也可能包含 p0→p9 方向和 ROI 旋转不稳定；目前只能作为分层复核条件，不能仅凭相关性断言 decoder 有 bug。

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
| Advance：高价值 Gold finetune | **最高** | 直接覆盖 teacher-abstain 和坐标偏差，是 geometry 当前最需要的纠偏。 |

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
500～1000 个定向 Gold positive 短程 finetune
        ↓
最终 Test → 导出真实 best → 厂商转换与板端验收
```

## 7. 三天执行安排

### Day 0～Day 1：保底与修复数据

1. 冻结 r2 的 `best.weights.h5`，导出**实际训练 best**，完成一次厂商转换和板端冒烟，作为随时可提交的 fallback。当前 run 目录可见的是 untrained preflight ONNX，不能把它当作正式训练模型。
2. 正确交付已有 57,962 张 Peak dark ROI PNG，以新实验 ID 重跑完整数据门禁；不要改写 r2 快照。
3. 新 geometry 暂时保持 v2 架构、coordinate loss 和主要超参数不变，只改变数据，便于确认收益来自哪里。
4. 查看新 curated 后的**每来源、每 sample type 实际训练抽样数**。Peak 应按部署域提高占比，但 `POS_LOW_PALM` 的坏标签尾部也最重，复核前不要盲目把所有 LOW 样本放大。
5. 团队并行复核 negative candidates：
   - 明确无手/背景：confirmed negative；
   - 肉眼有手但 teacher abstain：不能作为 negative，应进入 Gold positive 队列；
   - 不确定：HOLD。

### Day 1～Day 2：制作高价值 Gold 与训练新 geometry

建议制作 500～1000 个 Gold ROI，而不是随机均匀标注：

- 40%～50%：Peak teacher-abstain、肉眼明确有手的 hard positive；
- 30%～40%：Peak pseudo positive 中预测塌缩、student–teacher 分歧大或点明显偏差的样本；
- 10%～30%：Soar/普通 Peak 的稳定姿态作为 anchor，防止 finetune 遗忘。

覆盖开掌、握拳、弯指、遮挡、画面边缘、左右手、亮/暗、距离、极端旋转和 ROI 偏移。视频连续帧只选少量代表帧，并按完整采集 session 隔离 split。

不要把当前独立 inference session 直接并入 Gold 训练。服务器的 `HandFinetune0713/peak` bright/dark 数据与这 192 张 inference 图存在大量精确重叠（分别 100、79 张），相邻帧还会形成近重复泄漏。优先从 0714 train source 或全新、按 session 隔离的采集数据中选 Gold。

新 geometry 完成后只使用固定 Val 比较 r2，至少分为：

- overall；
- Peak independent / Peak shared / Soar；
- teacher-success / teacher-abstain；
- LOW_PALM / RUNTIME；
- collapse / non-collapse 分位组。

### Day 2：multitask 是否进入主线

若人工负例门禁已经满足，就从新 geometry best 启动 multitask：

- confirmed negative 总数不少于 500；
- `NEG_RUNTIME` 不少于 100；
- `NEG_LOW_PALM` 不少于 100；
- 每个样本有团队复核证据。

同时建立一个不进入训练的辅助 presence dev，至少包含 100～200 个明确背景和相当数量 positive，用于报告真实 FPR/recall。现有 Val/Test 只有 positive，无法选择 Hand flag threshold。

如果负例数量或复核质量不达标，不要绕过 fail-closed gate，也不要让 multitask 阻塞 Gold finetune。最终演示若特别依赖降低 Palm 阈值，multitask 优先级提高；若最关键的是 landmarks，Gold finetune 优先。

### Day 2～Day 3：短程 Gold finetune 与冻结交付

当前公开 README、配置和 Makefile 范围仍是 v2 pretrain，没有正式 finetune YAML/Make target；底层训练器虽然已有 finetune sampler，仍需要先补齐一次可审计的正式入口和 smoke。不要在最后一天用不可复现的临时命令训练最终模型。

建议的训练原则：

- 从 Val 最优的新 geometry 或 multitask checkpoint 初始化；
- 伪标签与 Gold 混合，先从 `gold_fraction=0.30` 开始，避免几百个 Gold 被过度重复；
- 全网络低学习率短训，建议从 geometry 学习率的 1/10～1/30 量级起步；
- 严格 early stopping；
- 最多保留一个主实验和一个单变量备选，不做大规模 sweep；
- Val 决策冻结后才做最终 Test、trained ONNX export、厂商转换和板端验收。

若 Day 1 结束仍无法形成安全的 finetune 入口，则保守 fallback 是“修复 Peak dark 后的新 geometry + 合格 multitask”，不要仓促拼接未经验证的最终训练流程。

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

1. **先 Enlarge，但不是重新录制：先补回已存在却未交付的 Peak dark ROI。**
2. **再 Review，但不是删除困难场景：把坏伪标签分流为 drop、Gold correction、HOLD。**
3. **重点 Advance 到定向 Gold finetune；负例 multitask 并行，为降低 Palm 阈值和过滤假框服务。**
4. **不延长 r2、不优先扩模、不先加硬骨骼先验。**
5. **把 Palm recall 与 Hand geometry 分开验收，否则会把上游无 ROI 与下游关键点偏差混为一个问题。**

这条路线同时保留了困难场景的泛化价值，又能纠正 teacher 的系统性漏标和 landmark 偏差，并且把三天内的实验变量控制在可验证范围内。
