# SLR 手语识别系统工作流程说明

本文档说明当前 `ssne_ai_demo` 中 SLR（Sign Language Recognition，手语识别）级联系统的运行流程。当前板端程序由 palm detector 和可选 hand landmarker 两级模型组成：入口文件是 `main.cpp`，默认只运行 palm detector；启动时传入 `--enable_hand` 后，会在 palm 检测成功后串行调用 hand landmarker，并在 OSD 上绘制手掌框、palm 两个关键点和 hand 21 点骨架连线。

## 1. 当前功能概览

当前程序完成的任务：

- 从板端摄像头获取 `720x1280` 的 `SSNE_Y_8` 灰度图。
- 将原始图像顺时针旋转 90 度，再双线性 resize 到 `224x224`。
- 将 `224x224` 灰度图送入 `/app_demo/app_assets/models/palm.m1model`。
- 解码模型 4 个输出，得到最多 2 个手掌检测结果。
- 每个检测结果包含 1 个手掌框和 2 个关键点：手腕、中指根部。
- 将模型坐标反变换回原始 `720x1280` 摄像头画面坐标。
- 默认使用 OSD 绘制 palm 检测框和 palm 两个关键点。
- 如果启动时传入 `--enable_hand`，且 palm 检测有效，则根据手掌框和两个关键点生成 hand ROI，裁剪 `256x256` 8-bit 灰度 ROI，并作为 `SSNE_Y_8` 输入串行调用 `/app_demo/app_assets/models/hand.m1model`。
- 启用 hand 时，解码 hand landmarker 的 21 个二维关键点，投影回原始摄像头画面坐标，并绘制 hand 21 点骨架连线。

## 2. 主循环和刷新策略

主循环位于 `main.cpp`：

```cpp
while (!check_exit_flag()) {
    // 实际代码会在每个阶段前后记录时间戳，用于生成 FrameTiming。
    processor.GetImage(&image_tensor);

    PalmPredictTiming palm_timing;
    if (frame_index % infer_interval == 0) {
        const bool verbose_log = verbose && (frame_index % verbose_interval == 0);
        palm_model.Predict(&image_tensor, &palm_result, frame_index, verbose_log, &palm_timing);
        if (enable_hand) {
            hand_model.Predict(&image_tensor, palm_result, &hand_result);
        } else {
            hand_result.Clear();
        }
    }
    visualizer.DrawDetections(palm_result, hand_result);

    const FrameTiming timing = ...;
    perf_monitor.AddFrame(frame_index, timing);

    frame_index += 1;
}
```

当前使用命令行参数控制推理策略：每 `kInferInterval` 帧执行一次 palm 推理；启用 hand 时，同一个推理帧会继续执行 palm + hand 级联推理。其余帧跳过推理但仍使用上一次的检测结果执行绘图。默认 `kInferInterval = 1`，即每帧刷新 palm 检测结果。

每个推理帧完整执行：

1. 摄像头取图（每帧都执行）。
2. 输入预处理。
3. palm 模型推理和后处理。
4. 如果 `--enable_hand` 已启用且 palm 有有效检测，基于 palm bbox + wrist/middle 两点生成 ROI，串行执行 hand landmarker。
5. OSD 清屏并重绘（每帧都执行，非推理帧使用缓存的上一次 palm/hand 检测结果）。

其中 `[PERF][stage_ms]` 里的 `palm_total_*` 总耗时包住的是整个 `PALMDETECTOR::Predict()`，不是单独的 NPU 推理耗时。Palm 内部耗时会通过 `[PERF][palm_detail_ms]` 额外拆分为 CPU 预处理、输入装载、`ssne_inference()`、`ssne_getoutput()` 和后处理耗时。

关键问题的当前答案：

- 每隔多少帧推理一次：每 `kInferInterval` 帧推理一次。当前默认 `kInferInterval = 1`。
- 每隔多少帧绘图一次：每 1 帧绘图一次。非推理帧使用缓存的上一次 palm/hand 检测结果。
- 图像刷新频率是多少：代码没有固定 FPS，由取图、绘图和（仅在推理帧）palm 预处理、palm 推理、palm 后处理、hand ROI 预处理、hand 推理和 hand decode 的总耗时决定。
- 每隔多少帧打印详细日志：只有 `verbose=true` 时，每隔 `verbose_interval` 帧打印一次。`verbose_interval` 是最靠近 50 的 `kInferInterval` 的正整数倍。当前 `kInferInterval=1` 时，`verbose_interval=50`。
- 当前是否执行 hand landmarker：默认否；启动命令增加 `--enable_hand` 后才会加载 hand 模型并执行 hand 推理。
- 当前默认是否打印详细日志：否，`verbose = false`。
- 当前是否向板端文件系统写调试文件：否。C++ 程序不再写 `/tmp/palm_debug` 或任何 raw dump 文件。
- 当前是否打印性能统计：是。`perf_log_enabled = true` 时，每 `perf_report_interval_frames` 帧打印一次 `[PERF]` 日志。注意：非推理帧的 `palm_total_*` 和 `palm_detail_*` 阶段耗时会记录为 0。

如果后续需要调试，优先把 `verbose` 临时改成 `true`，让诊断信息走控制台日志，而不是写板端文件。

## 3. 性能测试系统

当前 `main.cpp` 内置 `PerformanceMonitor`，用于评估 SLR 级联系统的实时性、端到端延迟、阶段耗时和抖动。该模块不写文件，只向终端打印 `[PERF]` 日志；离线统计和绘图由 `scripts/analyze_perf_log.py` 完成。

详细性能测试系统介绍见：[SLR_PERFORMANCE_TESTING.md](./SLR_PERFORMANCE_TESTING.md)。

## 4. 摄像头取图

摄像头取图逻辑在 `src/pipeline_image.cpp`。

初始化时：

```cpp
format_online = SSNE_Y_8;
OnlineSetCrop(kPipeline0, 0, width, 0, height);
OnlineSetOutputImage(kPipeline0, format_online, width, height);
OpenOnlinePipeline(kPipeline0);
```

当前图像尺寸配置在 `main.cpp`：

```cpp
const std::array<int, 2> image_shape = {720, 1280};
```

含义是：

- `image_shape[0] = width = 720`
- `image_shape[1] = height = 1280`

每帧通过：

```cpp
GetImageData(img_sensor, kPipeline0, kSensor0, 0);
```

得到一帧 `ssne_tensor_t`。

## 5. 输入预处理

预处理入口是 `PALMDETECTOR::PreprocessRotateResize()`，位于 `src/palm_detector.cpp`。

当前关键配置：

```cpp
const std::array<int, 2> palm_input_shape = {224, 224};
const bool use_ai_preprocess = false;
const bool rotate_clockwise = true;
```

由于训练数据方向和板端摄像头方向不同，不能直接把 `720x1280` resize 到 `224x224`。正确流程是：

1. 读取原始 `720x1280` 灰度图。
2. 顺时针旋转 90 度，得到 `1280x720` 图像。
3. 使用接近 OpenCV 默认 `cv2.resize` 的双线性插值 resize 到 `224x224`。
4. 将 `224x224` 的 `uint8` 灰度 buffer 直接拷贝到模型输入 tensor。

当前实现是融合路径：对每个 `224x224` 输出像素，直接反算到旋转后的坐标，再映射回原始摄像头图像取 4 邻域做双线性插值。它不会生成完整的 `1280x720` 旋转中间 buffer。

相关函数：

- `ResizeClockwiseRotatedBilinear()`
- `ResizeBilinear()`

当前没有使用 SDK 的 AI preprocess pipeline。模型 normalize 参数仍会在初始化时读取并打印，但实际输入走手写预处理路径：

```cpp
load_tensor_buffer_ptr(inputs[0], manual_input_buffer.data(), manual_input_buffer.size());
```

## 6. 模型输入输出

palm 模型路径：

```text
/app_demo/app_assets/models/palm.m1model
```

palm 模型输入：

- `224x224`
- `SSNE_Y_8`
- `uint8` 灰度

palm 模型输出数量：

```cpp
static const int kPalmOutputCount = 4;
```

当前根据输出元素数量自动映射 4 个输出：

- `reg14`: `14 * 14 * 16 = 3136`
- `cls14`: `14 * 14 * 2 = 392`
- `reg7`: `7 * 7 * 16 = 784`
- `cls7`: `7 * 7 * 2 = 98`

每个网格有 2 个 anchor。每个 anchor 的回归通道数为：

```cpp
4 + 2 * 2 = 8
```

含义是：

- 4 个 box 回归量：`dx, dy, dw, dh`
- 2 个关键点，每个关键点有 `x, y`

hand landmarker 模型路径：

```text
/app_demo/app_assets/models/hand.m1model
```

hand landmarker 输入：

- `256x256`
- `SSNE_Y_8`
- `uint8` 灰度 ROI
- ROI 来自 palm 检测结果，不独立运行。

hand ROI 生成逻辑位于 `src/hand_landmarker.cpp`：

1. 使用 palm 检测框作为基础 rect。
2. 使用 palm 的两个关键点（手腕、中指根部）计算旋转角。
3. 按 `scale_x=1.8`、`scale_y=1.8`、`shift_y=-0.1` 扩展并平移 ROI。
4. 将原始摄像头灰度图按旋转 ROI 做 affine bilinear crop，得到 `256x256` 的 8-bit ROI。
5. 将 8-bit ROI 装载到 `SSNE_Y_8` tensor。

hand landmarker 输出数量：

```cpp
static const int kHandOutputCount = 3;
```

当前根据输出元素数量自动查找 42 个 landmark 值，并把其余两个标量按输出顺序作为 `hand_flag` 和 `handedness`：

- `landmarks`: `21 * 2 = 42`
- `hand_flag`: 手是否存在的置信度，当前只记录，不拦截绘制；首版优先保证 landmarks 可视化便于验证。
- `handedness`: 左右手概率，目前只记录，不参与 OSD 绘制。

landmark 输出如果数值范围大于 2，会按 `256` 归一化；否则按已经归一化的 `0..1` crop 坐标处理。之后通过 ROI 三点仿射关系投影回原始 `720x1280` 画面坐标。

## 7. 输出内存布局

当前板端 SSNE 输出按 `HWC` 方式解码：

```cpp
const PalmOutputLayout output_layout = kPalmOutputLayoutHwc;
```

代码仍保留 `PalmOutputLayout` 和 `GetOutputIndex()`，这样以后如果某个新模型或新转换工具输出布局不同，可以切换到 `kPalmOutputLayoutNchw` 验证。但常规 verbose 日志中不再每 50 帧同时跑 NCHW/HWC 双路对照，避免稳定版本做额外计算和刷冗余日志。

## 8. Anchor 和后处理

核心常量定义在 `include/common.hpp`：

```cpp
kPalmFeature14 = 14
kPalmFeature7 = 7
kPalmNumAnchorsPerCell = 2
kPalmNumKeypoints = 2
kPalmScoreThreshold = 0.50
kPalmNmsIouThreshold = 0.30
kPalmCrossHeadSuppressIou = 0.35
kPalmMaxDetections = 2
```

Anchor 尺寸：

- `14x14` head:
  - anchor 0: `0.10 x 0.10`
  - anchor 1: `0.18 x 0.18`
- `7x7` head:
  - anchor 0: `0.25 x 0.25`
  - anchor 1: `0.40 x 0.40`

box 解码：

```cpp
cx = anchor.cx + dx * anchor.w
cy = anchor.cy + dy * anchor.h
w = anchor.w * exp(dw)
h = anchor.h * exp(dh)
```

关键点解码：

```cpp
kx = anchor.cx + raw_kx * anchor.w
ky = anchor.cy + raw_ky * anchor.h
```

候选框分数低于 `0.50` 会被过滤。之后执行 NMS，并最多保留 2 个检测结果。

## 9. 坐标反变换

模型输出坐标是相对于旋转后再 resize 的 `224x224` 图像的归一化坐标。绘图需要映射回原始 `720x1280` 摄像头画面。

关键函数：

- `MapPoint()`
- `MapBox()`

在顺时针旋转 90 度时，归一化点先映射到旋转后图像：

```cpp
rotated_x = model_x * (rotated_width - 1)
rotated_y = model_y * (rotated_height - 1)
```

再反变换回原始图像：

```cpp
original_x = rotated_y
original_y = original_height - 1 - rotated_x
```

检测框通过 4 个角点分别反变换，然后重新取 `min/max` 得到原图上的轴对齐框。

## 10. OSD 绘制

绘制逻辑位于 `src/utils.cpp`。

当前每帧调用：

```cpp
visualizer.DrawDetections(palm_result, hand_result);
```

绘图步骤：

1. 清理 OSD 图层 `0..6`。
2. 绘制 palm 检测框。
3. 绘制 palm 的两个关键点。
4. 如果启动时启用了 `--enable_hand` 且存在有效 hand 结果，绘制 hand landmarker 的 21 点骨架连线。
5. flush 图层 `0..6`。

当前手掌框不用 `TYPE_HOLLOW` 矩形绘制，而是用 4 条 `DrawLine()` 绘制：

- 上边
- 右边
- 下边
- 左边

这样走实心 OSD 图元路径，实测比空心矩形稳定。

当前可视化参数：

```cpp
point_size_ = 5
point_color_ = 2
box_border_ = 6
box_color_ = 2
hand_line_thickness_ = 3
hand_line_color_ = 3
```

hand landmarker 当前不绘制 21 个点本身，只绘制连线。连线关系与 `ref/osd/visualizer.cpp` 一致：

```text
0-1-2-3-4
0-5-6-7-8
5-9-10-11-12
9-13-14-15-16
13-17-18-19-20
0-17
```

## 11. 调试输出

当前默认：

```cpp
kInferInterval = 1
enable_hand = false
verbose = false
verbose_interval = 50
```

当 `verbose=false` 时，主循环不打印每帧详细诊断日志，只保留初始化和错误日志。

启用 `--enable_hand` 后，hand landmarker 会额外打印一次轻量诊断信息，包括首个有效 ROI 的四边形、灰度统计、归一化 landmark 原始数值范围以及投影后的 landmark bbox。该日志用于判断 hand 骨架偏移或偏小发生在 ROI、模型输出还是坐标投影阶段。当前 v2 不再猜测输出单位或应用额外坐标缩放。

当临时改成 `verbose=true` 时，程序每 `verbose_interval` 帧通过控制台打印一次 palm 诊断信息。`verbose_interval` 会随 `kInferInterval` 自动调整为最靠近 50 的推理帧间隔，例如 `kInferInterval=1` 时为 50，`kInferInterval=3` 时为 51。

- 摄像头 tensor 信息。
- 手写预处理后的输入 tensor 信息。
- 模型输入 tensor 信息。
- 4 个输出 tensor 的尺寸、dtype、元素数量和值域统计。
- 输出映射结果。
- 当前使用的输出布局。
- 最终检测框和关键点坐标。

当前版本不再提供 C++ 文件 dump，不会向 `/tmp` 或其他板端目录写 raw 输入/输出。这样更适合烧录到开发板后的长期运行。

如果后续新增模型时确实需要逐元素对齐 Python/ONNX，建议临时增加受宏或显式开关保护的 dump 逻辑，用完后再移除；不要让文件写入路径长期默认存在于板端程序中。

## 12. 快速入手读代码路线

建议按下面顺序读：

1. `main.cpp`
   - 看主流程、模型路径、图像尺寸、是否旋转、`--kInferInterval`、`--enable_hand`、verbose 开关和性能统计配置。
   - 确认当前推理频率（默认 `kInferInterval=1`，即每帧执行一次 palm 推理）和 hand 是否启用（默认不启用）。

2. `src/pipeline_image.cpp`
   - 看摄像头 pipeline 如何打开。
   - 确认输出格式是 `SSNE_Y_8`，尺寸是 `720x1280`。

3. `include/common.hpp`
   - 看 palm 相关结构体和常量。
   - 重点看 palm anchor、阈值、输出布局、`PalmDetection`、`PalmResult`、`PalmPredictTiming`，以及 hand 的 `HandDetection`、`HandResult`、`HANDLANDMARKER`。

4. `src/palm_detector.cpp`
   - 先看 `Initialize()` 和 `Predict()`。
   - `Predict()` 内部已经拆分计时：`palm_preprocess`、`palm_input_load`、`palm_inference`、`palm_getoutput`、`palm_output_meta`、`palm_decode`、`palm_verbose_log`。
   - `palm_preprocess` 内部还会继续拆为 `palm_preprocess_transform` 和 `palm_preprocess_manual_load`。当前旋转和 resize 是融合实现，`palm_preprocess_transform` 不是单独旋转耗时，而是融合旋转缩放耗时。
   - 再看 `PreprocessRotateResize()`。
   - 然后看 `MapOutputs()`、`DecodeHead()`、`SelectDetections()`。
   - 最后看 `MapPoint()` 和 `MapBox()`。

5. `src/hand_landmarker.cpp`
   - 先看 `Predict()`：它只在 palm 有检测结果时工作。
   - 再看 `BuildRoiRect()`、`PreprocessRoi()`、`DecodeOutputs()`。
   - 如果 hand 骨架不准，优先检查一次性 `[HAND][debug]` 日志中的 ROI 参数、归一化 hand 输出范围和投影 bbox。

6. `src/utils.cpp`
   - 看 OSD 如何清屏、画框、画关键点。
   - 如果日志坐标正常但画面显示异常，优先检查这里。

7. `src/osd-device.cpp`
   - 看底层 OSD 图元如何转成 SDK 调用。
   - 重点关注 layer、颜色、alpha、`TYPE_HOLLOW`/`TYPE_SOLID`。

8. `ref/palm/infer_model_gray.py`
   - 作为 Python 侧参考实现。
   - 新模型接入时，优先对齐这里的预处理、输出 reshape、后处理和 NMS。

9. `ref/hand/infer_frames_with_roi.py` 和 `ref/hand/roi_utils.py`
   - 作为 hand ROI、crop、21 点解码和反投影的 Python 侧参考实现。

## 13. 修改参数时的注意事项

- 如果想调整推理频率，启动时传入 `--kInferInterval N`（默认为 1）。值越大推理越少、平均处理压力越低，但 palm/hand 检测结果刷新越慢。
- 如果想启用 hand landmarker，启动时传入 `--enable_hand`。默认不启用 hand，不加该参数时只加载和执行 palm detector。
- 如果只想减少日志，不要改推理逻辑，先改 `verbose` 或 `verbose_interval`。
- 如果现场传感器 FPS 不是 90，先改 `sensor_fps_cfg`，否则性能分数估算会偏。
- 如果后续串行接入更多模型导致处理压力上升，可以增大 `kInferInterval` 来启用更激进的跳帧策略。
- 如果更换模型，必须重新确认：
  - 输入尺寸。
  - 输入 dtype。
  - 是否需要旋转。
  - resize 插值方式。
  - 输出数量和顺序。
  - 输出内存布局。
  - anchor 配置。
  - 坐标反变换。
  - OSD 绘制路径。
