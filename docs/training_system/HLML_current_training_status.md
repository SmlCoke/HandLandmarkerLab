# HLML 当前训练状态与问题结论

更新时间：2026-07-19。本文件只记录当前事实和结论；最后一轮训练命令见 [HLML 下一步计划](HLML_next_step_plan.md)。

## 1. 结论先说

当前约 21 px 的误差，**不是 ONNX 导出或厂商转换造成的，也不能归咎于 MediaPipe 伪标签普遍错误**。

最主要的问题是：现有学生模型能拟合伪标签训练集，但不能把这种能力泛化到独立 Val；MediaPipe 又只给自己能检出的 ROI 提供正标签，困难姿态天然缺少监督。模型结构和逐点 Huber 回归会进一步放大“平均姿态/关键点聚团”，但现有证据不足以断言“只要扩大参数量就能解决”。

最后一轮应优先增加**部署同域、姿态多样、非连续近重复**的无损 TIFF 伪标签数据，并增加严格的阶段门槛。不要继续扫 finetune Gold/pseudo 比例，也不要在提交前贸然重写模型。

## 2. 已完成结果

固定 Val 共 1,226 个独立 Hand ROI，全部为人工 Gold positive。各阶段使用同一份 labels SHA-256 `be7074...9062f`。

| 模型 | mean px | median px | P90 px | PCK@0.10 | 关键点 spread < 0.10 |
|---|---:|---:|---:|---:|---:|
| geometry | **20.73** | 18.21 | **37.32** | 0.334 | **157** |
| multitask | 22.01 | 18.38 | 39.18 | 0.302 | 197 |
| finetune data-only，Gold=1 | 21.48 | 17.40 | 44.67 | 0.359 | 207 |
| finetune data-only，Gold=2 | 21.27 | **16.93** | 44.77 | **0.361** | 244 |
| finetune data-only，Gold=4 | 21.81 | 17.17 | 46.01 | 0.358 | 228 |
| finetune structure，Gold=1 | 21.84 | 17.27 | 45.94 | 0.349 | 12 |

说明：`spread < 0.10` 表示 21 点相对手腕的整体展开尺度过小，是“塌缩”的自动近似，不等同于所有关键点是否正确。

P0 已回答两个问题：

- Gold/pseudo Loss 倍率从 1 提到 2 或 4，mean 只变化约 0.2～0.5 px，不能解决当前问题。
- structure Loss 几乎消除了几何聚团，但 mean、P90、PCK 和 handedness 没有改善。它只是强迫骨架展开，并没有让手指落到正确位置。

因此停止现有 P0 扫参，不再跑 Gold=1.5/3/5，也不再调 structure 系数。

## 3. MediaPipe 教师实测上限

已用 HLMF 当前官方 `hand_landmarker.task`，直接对同一批 1,226 张 Gold Val crop 推理。输入是 HLML 实际读取的 256×256 灰度 ROI 转三通道，不是原图上的另一套坐标。

| 指标 | MediaPipe 教师 |
|---|---:|
| 检出 | 836 / 1,226（68.19%） |
| 已检出 ROI 的 mean px | **4.38** |
| 已检出 ROI 的 median px | **0.63** |
| 已检出 ROI 的 P90 px | **12.83** |
| 已检出点的 PCK@0.10 | **0.953** |
| 已检出 ROI 中 spread ratio < 0.5 | 5 |

分来源 mean：`peak_vali_independent_v1=7.48 px`、`peak_vals_shared_v1=4.07 px`、`soar_vals_shared_v1=3.62 px`。

这说明：

- MediaPipe 在能检出的 ROI 上非常接近现有 Gold，伪标签坐标定义没有 20 px 级系统偏差。
- MediaPipe 仍会对 390 个困难 ROI 弃权。这些姿态不会成为普通伪标签正样本，是自动标注路线的盲区。
- “抽查伪标签看起来很好”与“学生模型最终很好”不是一回事。前者只证明教师给出的目标好，不证明学生结构、训练覆盖和泛化已经足够。

## 4. 学生究竟差在哪里

### 4.1 对教师能检出的容易子集，学生仍未学到教师

| 模型 | 教师已检出 836 ROI：对 Gold mean | 教师已检出 836 ROI：对教师 mean | 教师漏检 390 ROI：对 Gold mean |
|---|---:|---:|---:|
| geometry | 17.57 px | 17.03 px | 27.51 px |
| multitask | 18.85 px | 18.01 px | 28.78 px |
| finetune Gold=1 | 17.92 px | 17.50 px | 29.11 px |
| finetune Gold=2 | 17.69 px | 17.26 px | 28.95 px |

困难样本缺标签造成了约 10 px 的额外恶化；但即使教师只错 4.38 px 的容易子集，学生仍错约 17 px。因此不能只把问题解释为“MediaPipe 漏检”。

### 4.2 训练集拟合与独立 Val 存在明显间隙

旧 geometry 使用 59,952 个伪标签正 ROI。best checkpoint 在随机 3,584 个伪标签训练 ROI 上的无增强实测为：

```text
mean = 6.87 px
median = 5.18 px
P90 = 11.71 px
spread ratio < 0.5 = 9 / 3584
```

同一个 best checkpoint 在独立 Gold Val 上为 20.73 px。训练日志也显示：best Val 位于 epoch 33；之后训练误差继续下降，Val 不再改善。主要矛盾是泛化，不是模型完全不能反向传播或不能记住标签。

### 4.3 训练域仍有差异和大量视频相关帧

抽样统计：

```text
伪标签 Train 灰度均值：0.449
Gold Val 灰度均值：      0.569
```

两者的手部中心和骨架尺度大体接近，但亮度域有明显偏移。旧训练集 59,952 个正 ROI 来自约 34,657 个 source group；同一视频、同一动作的连续帧仍可能高度相似。HLMF 会处理同帧重复 ROI，但不能把连续视频帧自动变成独立场景。

新增“数十万张”只有在人物、姿态、距离、方向、亮度和背景真正增加时才有效；大量相邻帧只是增加训练耗时和记忆机会。

## 5. 模型和板端算子是否是原因

### 5.1 不是导出/转换数值错误

当前 Keras 到 deploy graph 的 BN folding/reparameterization 误差约为 `1e-6`；ONNX parity、A1 算子审计、厂商转换和上板均已通过。PC 上的 Keras Val 本身就是约 21 px，因此板端转换不可能是这一误差的来源。

### 5.2 为板端重写的学生结构可能限制上限，但不是唯一主因

当前学生约 1.95M 参数，输入为 `1×256×256` 灰度，最终压缩成 `2×2×384` 后直接回归 42 个二维坐标。部署算子只使用 Conv/Depthwise Conv/Add/MaxPool/ReLU/Sigmoid 等受支持算子。

官方 MediaPipe landmark 子模型输入为 `224×224×3`，除 presence/handedness 外同时预测 63 个图像空间坐标和 63 个 world 坐标；其图中还使用 Fully Connected 和 Mean，并带有量化权重。两者不是同一个网络。

因此，板端限制影响的是“我们选择了怎样的学生结构”，不是“导出后把正确模型算坏了”。官方模型的三维/世界坐标多任务监督和不同 head 可能提供更强姿态先验；当前逐点二维 Huber 在信息不足时容易学成平均姿态。

不能只凭文件大小判断参数量是否不足。当前学生能把训练 ROI 降到约 6.9 px，却在 Val 上达到 20.7 px，说明直接扩宽网络很可能先加重记忆。若没有固定数据上的小规模 A/B、Val 提升和完整工具链验证，提交前不改架构。

## 6. 根因优先级

1. **训练到独立部署域的泛化不足**：证据最强，约 6.9 px 对 20.7 px。
2. **伪标签选择偏差**：教师只覆盖 68.2% Val；教师漏检姿态没有正监督。
3. **数据有效多样性不足**：连续帧多，人物/姿态/亮度的独立变化小于图片数量。
4. **学生结构与二维逐点回归的姿态先验不足**：会产生平均姿态和塌缩；structure P0 只治形状、不治位置。
5. **multitask/finetune 对 geometry 的遗忘或过拟合**：当前 geometry 的 mean/P90 反而最好；multitask best 在第 1 epoch，finetune Train 持续变好而 Val 基本不动。
6. **纯参数量不足**：有可能，但现有证据不能把它排到前面。
7. **ONNX/厂商转换错误**：现有证据已基本排除。

## 7. 数据路径注意事项

旧 `v3-pretrain-r1` 冻结标签仍记录迁移前的绝对 `crop_path`。本次诊断只在内存中重定位到 `DatesetFab/PretrainSource`，没有修改文件。旧 checkpoint 和历史结果可继续使用，但不能拿旧 JSONL 直接启动新训练。

最后一轮必须由 HLMF 重新执行 `finalize_train_pretrain`，生成指向 `PretrainSource` 真源的新 merged labels，再用新的 `HAND_PRETRAIN_ID` 重新 curate。不得覆盖 `v3-pretrain-r1`。
