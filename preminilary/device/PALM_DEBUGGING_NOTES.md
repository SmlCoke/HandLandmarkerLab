# Palm Detection 上板调试复盘

本文档记录 palm 模型从最初上板效果很差，到最终实现稳定关键点和手掌检测框定位的排查过程。重点不是记录每一次代码修改，而是总结以后接入新模型时最容易踩坑的地方。

## 1. 最终结论

最终成功的关键点有 4 个：

1. 输入必须先把原始 `720x1280` 摄像头图像顺时针旋转 90 度，再 resize 到 `224x224`。
2. resize 要接近 Python 参考脚本中的 OpenCV 默认双线性插值，而不是随便换成其他插值。
3. SSNE 转换后的输出 flatten 顺序按 `HWC` 解码才和模型语义对齐。
4. 手掌框本来已经被后处理算出来了，最后看不到主要是 OSD 空心框绘制路径不稳定，改为 4 条 solid line 后解决。

这几个问题里，前 3 个属于模型输入/输出对齐问题，第 4 个属于可视化问题。

## 2. 初始现象

最开始上板运行时，现象是：

- 没有稳定的手掌检测框。
- 关键点乱跳。
- 关键点和真实手掌位置不重合。
- 但同一个 `palm` 模型在转换前的 ONNX/Python 推理中效果很好。

这个现象说明问题不太像训练失败，而更像部署链路和 Python 参考链路不一致。优先怀疑：

- 输入图像方向不一致。
- resize 或归一化不一致。
- 输出 tensor 顺序或内存布局不一致。
- 后处理 decode 写错。
- 坐标反变换写错。
- OSD 绘制和真实结果不一致。

## 3. Bug 1：摄像头图像方向和训练方向不一致

### 现象

板端摄像头输出是 `720x1280` 的竖屏图像，但训练和 Python 推理所用图像方向与板端直接输入不同。

如果直接 resize 原始图像再送入模型，模型看到的手掌方向和训练分布不一致，检测会明显漂移。

### 正确做法

原始图像必须先顺时针旋转 90 度：

```text
720x1280 -> clockwise rotate -> 1280x720 -> resize -> 224x224
```

也就是用户描述的：

```text
第 i 行变成倒数第 i 列
```

### 修复

代码中增加：

- `rotate_clockwise = true`
- `ResizeClockwiseRotatedBilinear()`
- `MapPoint()`
- `MapBox()`

推理前做正向变换，推理后做反向坐标变换。

关键点反变换：

```cpp
original_x = rotated_y;
original_y = original_height - 1 - rotated_x;
```

检测框反变换不能只变换左上角和右下角，因为旋转会改变角点关系。当前做法是变换 4 个角点，再重新取 `min/max`。

## 4. Bug 2：resize 插值方式和 Python 参考不一致

### 现象

修正旋转后，模型输出仍然不稳定。日志显示模型确实有输出，输出 shape 也符合预期，但检测结果与手掌不重合。

### 分析

Python 参考脚本 `infer_model_gray.py` 使用的是：

```python
cv2.resize(gray, (224, 224))
```

OpenCV 默认是双线性插值。早期 C++ 侧 resize 方式和这个不完全一致，导致输入像素分布和 Python 侧不一致。

对于小模型和灰度输入，这种差异可能足以让分类分数和回归结果明显变化。

### 修复

实现接近 OpenCV half-pixel 映射的双线性 resize：

```cpp
src_x = (dst_x + 0.5) * scale_x - 0.5;
src_y = (dst_y + 0.5) * scale_y - 0.5;
```

旋转版 resize 直接从原图采样虚拟旋转后的坐标，避免额外开一张 `1280x720` 中间图。

## 5. Bug 3：输出内存布局按 NCHW 解码是错的

### 现象

日志显示 4 个输出的元素数量是合理的：

```text
output[0]: 14x14 reg, elements=3136
output[1]: 14x14 cls, elements=392
output[2]: 7x7 reg, elements=784
output[3]: 7x7 cls, elements=98
```

输出顺序和 Python 模型顺序也基本能对应：

```text
reg14, cls14, reg7, cls7
```

但是按最初假设的 NCHW 方式读取时，关键点位置仍然不对。

### 分析

Python 参考后处理里有类似逻辑：

```python
cls = cls_pred[0].transpose(1, 2, 0).reshape(-1)
reg = reg_pred[0].transpose(1, 2, 0).reshape(-1, vals_per_anchor)
```

也就是说，Python 侧 decode 最终是按每个 cell 连续读取 channel 的方式工作。

SSNE 转换后的输出 metadata 只有 width/height，没有可靠暴露 channel 维含义。因此不能只看 `width=14,height=14` 就认定数据是 NCHW。

### 修复

增加输出布局枚举：

```cpp
enum PalmOutputLayout {
    kPalmOutputLayoutNchw = 0,
    kPalmOutputLayoutHwc = 1,
};
```

当前默认：

```cpp
const PalmOutputLayout output_layout = kPalmOutputLayoutHwc;
```

索引计算：

```cpp
// HWC
index = cell_index * channel_count + channel;

// NCHW
index = channel * spatial + cell_index;
```

为了以后调试，verbose 日志中保留双布局对照：

```text
layout_compare NCHW detections=..., HWC detections=...
layout_NCHW_det[...]
layout_HWC_det[...]
```

### 结果

切到 HWC 后，关键点稳定落在手腕和中指根部，并且能跟随双手移动。这是确认输入方向、resize、输出布局基本正确的关键证据。

## 6. Bug 4：检测框不是没算出来，而是 OSD 没稳定画出来

### 现象

关键点已经非常稳定，但手掌框几乎看不到，只在极少数时刻闪现。

### 分析证据

日志中实际已经有正常的检测框：

```text
det[0]: pixel_box=(153,332,393,569)
keypoint[0]: pixel=(357,426)
keypoint[1]: pixel=(197,436)
```

这些框：

- 坐标在画面范围内。
- 宽高正常。
- 两个关键点也落在框附近或框内。

因此问题不是模型没有输出框，也不是 NMS 把框过滤掉，而是可视化层没有稳定显示框。

### 可疑点

当时 palm 可视化使用：

- `osd_device.Draw(boxes, ...)`
- `TYPE_HOLLOW`
- `box_color_ = 1`
- `box_border_ = 2`

而关键点使用：

- `DrawPoint()`
- `TYPE_SOLID`
- `point_color_ = 2`

关键点一直可见，说明 solid OSD 图元路径是可靠的。官方 `./demo` 人脸例程也能画框，说明 layer 0 本身可以用于检测框。

### 修复

把 palm 框从 `TYPE_HOLLOW` 矩形改成 4 条 solid line：

```cpp
DrawLine(x1, y1, x2, y1)
DrawLine(x2, y1, x2, y2)
DrawLine(x2, y2, x1, y2)
DrawLine(x1, y2, x1, y1)
```

同时：

- 继续使用检测 layer 0。
- 颜色改为和关键点一样的 `2`。
- 线宽从 `2` 加粗到 `6`。

### 结果

手掌检测框稳定显示，并且定位精准。至此可以确认：框的后处理结果早就正确，最后一处问题在 OSD 绘制方式。

## 7. 哪些方向没有继续深挖

这次调试过程中，没有优先去调以下参数：

- score threshold
- NMS IoU threshold
- anchor 尺寸
- max detections

原因是日志已经证明输出 tensor shape、分类分数、关键点和框坐标本身是合理的。盲目调阈值会掩盖真正的问题。

这点很重要：当 ONNX/Python 准确、板端不准时，第一优先级通常是对齐部署链路，而不是调模型阈值。

## 8. 当前仍存在的非 bug 问题

当前模型本身鲁棒性不算很高，因此检测框和关键点会有小范围快速抖动。这不是本次部署链路 bug，而是模型能力、输入噪声、推理速度和时序稳定性共同造成的。

后续如果要进一步优化体验，可以考虑：

- 对 box 和 keypoint 做时间滤波。
- 增加检测结果跟踪。
- 对低置信度结果做更平滑的保留策略。
- 用更鲁棒的数据重新训练或微调模型。

当前版本暂时没有做这些，因为核心目标是先保证上板推理链路正确。

## 9. 后续接入新模型的推荐排查顺序

以后增加新模型时，建议按这个顺序排查，别一上来就改阈值：

1. Python 参考链路
   - 明确输入尺寸、颜色格式、归一化、resize 插值、是否 letterbox、是否旋转。
   - 保存一张参考输入和参考输出。

2. 板端输入 tensor
   - 打印 dtype、format、width、height、mem_size。
   - dump 预处理后的输入 raw buffer。
   - 和 Python 输入逐像素比较。

3. 模型转换后的输出
   - 打印每个 output 的元素数量和值域。
   - 不要只相信输出顺序，最好按元素数量和语义双重确认。
   - 同时尝试 NCHW/HWC 或其他 flatten 方式。

4. 后处理
   - 先复刻 Python decode，不要顺手改公式。
   - anchor、stride、grid、exp/sigmoid、score 是否已经在模型里做过，都要确认。
   - 对照一帧 raw output，逐项比较 Python 和 C++ decode。

5. 坐标映射
   - 明确模型坐标相对于哪张图：原图、裁剪图、旋转图、resize 图还是 letterbox 图。
   - 反变换要按相反顺序做。
   - 框经过旋转时，优先变换 4 个角点再取 `min/max`。

6. 可视化
   - 如果日志坐标正确但屏幕不显示，优先怀疑 OSD。
   - 用已知固定框测试 layer、颜色、alpha、线宽。
   - 尽量复用官方 demo 已验证的绘制路径。

7. 最后再调阈值
   - 只有当输入、输出、decode、坐标、OSD 都确认正确后，再调 score/NMS。

## 10. 本项目中最有价值的调试工具

当前稳定版本不再向板端文件系统写 raw dump。调试信息默认关闭，需要时通过 `verbose=true` 打印到控制台日志。

### 10.1 历史上的首帧 dump

调试中曾经使用过首帧 raw dump，用于把板端 `224x224` 输入和 4 个输出 tensor 与 Python/ONNX 逐元素对齐。系统稳定后，这部分 C++ 文件写入逻辑已经移除，避免烧录到板端后产生无法从 PC 侧获取的临时文件。

后续新增模型时，如果确实需要 raw dump，建议临时加入受宏或显式开关保护的代码，用完后再移除，不要让文件写入路径长期默认存在。

### 10.2 输出 shape 和值域日志

日志中会打印：

- dtype
- mem_size
- inferred_elements
- min/max/mean
- sample_first/sample_center/sample_last

用途：判断输出是否为空、是否全 0、是否 dtype 读错、是否元素数量不匹配。

### 10.3 NCHW/HWC 布局排查

当前稳定版本固定使用 `HWC` 解码，不再在常规 verbose 日志里每次同时跑 NCHW/HWC 双路对照。后续新增模型时，如果怀疑输出 flatten 顺序不一致，可以临时切换 `PalmOutputLayout` 或重新加入一次性对照日志。

### 10.4 最终检测结果日志

日志中打印：

- `model_box`
- `pixel_box`
- 每个 keypoint 的 model 坐标、pixel 坐标、归一化坐标

用途：区分后处理错误和 OSD 绘制错误。

## 11. 经验总结

这次最关键的判断是：关键点稳定以后，不要继续怀疑整个模型链路。因为关键点和框来自同一套输出、同一套 anchor、同一套坐标反变换。关键点准，说明大部分链路已经对了。

后面框不显示时，日志里的 `pixel_box` 已经正常，所以问题自然转移到 OSD。这个切换很重要：调试时要根据证据及时改变怀疑对象。

一句话总结：

```text
先证明输入一致，再证明输出读对，再证明后处理一致，最后证明画出来的东西和日志一致。
```
