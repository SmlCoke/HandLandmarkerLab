# SLR 性能测试系统讲解

本文档说明 `ssne_ai_demo` 板端程序的性能测试系统，包括 `[PERF]` 日志的统计口径、window 定义、均值和 P95 指标含义、跳帧推理时的读数方式，以及离线分析脚本 `scripts/analyze_perf_log.py` 的使用方法。

## 1. 系统组成

性能测试链路由两部分组成：

1. 板端在线统计：`PerformanceMonitor`
   - 代码位置：`include/performance_monitor.hpp`、`src/performance_monitor.cpp`、`main.cpp`。
   - 功能：在程序运行时采集每帧各阶段耗时，并每隔固定帧数向终端打印 `[PERF]` 日志。
   - 特点：不写板端文件，只走控制台输出，适合烧录到开发板后长期运行。

2. PC 端离线分析：`scripts/analyze_perf_log.py`
   - 输入：板端保存下来的 `result.log`。
   - 输出：中文摘要、英文详细报告和 SVG 图表。
   - 用途：统计采集帧数、推理帧数、FPS、丢帧率、各阶段均值/P95 延迟，并绘制吞吐、阶段耗时、Palm 细分耗时、P95 时间线和 P95 饼图。

## 2. 关键配置

主流程中的性能相关配置位于 `main.cpp`：

```cpp
uint32_t kInferInterval = 1;
bool enable_hand = false;
const uint32_t infer_interval = kInferInterval == 0 ? 1 : kInferInterval;
const uint32_t verbose_multiple = (50 + infer_interval / 2) / infer_interval;
const uint32_t verbose_interval =
    (verbose_multiple == 0 ? 1 : verbose_multiple) * infer_interval;
const bool perf_log_enabled = true;
const double sensor_fps_cfg = 90.0;
const uint32_t perf_report_interval_frames = 120;
```

- `kInferInterval`: 跳帧推理间隔。`1` 表示每帧推理，`3` 表示每 3 帧推理一次。
- `enable_hand`: 是否启用 hand landmarker。默认不启用；传入 `--enable_hand` 后启用 palm + hand 级联。
- `sensor_fps_cfg`: 摄像头配置帧率，目前为 `90.0` FPS，用于实时性比例、丢帧率估计和以帧周期计的延迟分析。
- `perf_report_interval_frames`: 性能统计窗口长度，目前为 `120` 帧。

启动示例：

```text
./ssne_ai_demo
./ssne_ai_demo --kInferInterval 3
./ssne_ai_demo --kInferInterval=3 --enable_hand
./scripts/run.sh --kInferInterval=3 --enable_hand
```

## 3. Window 的定义

性能日志中的 `window` 指性能统计窗口（performance report window），不是滑动窗口，而是固定长度、互不重叠的分段统计区间。

当前配置：

```cpp
const uint32_t perf_report_interval_frames = 120;
```

因此：

```text
1 个 window = 120 帧
```

`PerformanceMonitor::AddFrame()` 每处理完一帧就把该帧耗时加入当前 window 的缓存。当 `total_frames % perf_report_interval_frames == 0` 时，程序对当前 window 内的样本计算统计量，打印一组 `[PERF]` 日志，然后清空当前 window 缓存，开始下一个 window。

以当前 `120` 帧窗口为例：

```text
Window 1: frame 0   到 frame 119，打印时 summary 中 frame=119
Window 2: frame 120 到 frame 239，打印时 summary 中 frame=239
Window 3: frame 240 到 frame 359，打印时 summary 中 frame=359
...
```

每次 `[PERF]` 展示只统计当前 window 内的数据，不包含之前 window 的样本。比如 `frame=239 window_frames=120` 这一组日志只统计 `frame 120..239`，不包含 `frame 0..119`。

如果程序在一个 window 尚未攒满 120 帧前退出，当前实现不会额外打印这个不完整 window。

## 4. 一个 Window 里统计了哪些数据

每一帧都会记录一个 `FrameTiming`，包含：

```cpp
get_image_ms
palm_total_ms
palm_preprocess_ms
palm_preprocess_transform_ms
palm_preprocess_manual_load_ms
palm_input_load_ms
palm_inference_ms
palm_getoutput_ms
palm_output_meta_ms
palm_decode_ms
palm_verbose_log_ms
palm_accounted_ms
hand_total_ms
draw_ms
loop_ms
process_ms
```

这些字段进入当前 window 后，会分别形成一组样本。例如一个 120 帧 window 中，`loop_ms` 有 120 个样本，`draw_ms` 有 120 个样本，`palm_total_ms` 也有 120 个样本。

需要特别注意跳帧推理：

- `GetImage()` 和 OSD 绘图每帧都执行，所以 `get_image_ms`、`draw_ms`、`loop_ms`、`process_ms` 每帧都有实际耗时；日志中对应为 `get_image_*`、`draw_*`、`loop_*`、`process_*`。
- `palm detector` 只在 `frame_index % kInferInterval == 0` 时执行。
- `hand landmarker` 只有在 `--enable_hand` 启用、当前帧是推理帧，并且 palm 有有效检测时才执行。
- 非推理帧的 `palm_total_*` 和 `palm_detail_ms` 内各 `palm_*` 字段会记录为 `0`。
- 未执行 hand 的帧中，`hand_total_*` 通常接近 `0`。

因此当 `kInferInterval > 1` 时，`palm_total_avg` 和 `hand_total_avg` 是“按全部帧摊薄后的每帧平均耗时”，不是单次推理耗时。离线脚本会额外给出 `avg * kInferInterval` 的单次推理平均耗时估计。

举例：

```text
kInferInterval = 1: 每个 120 帧 window 约 120 次 palm 推理
kInferInterval = 3: 每个 120 帧 window 约 40 次 palm 推理
```

## 5. 均值 Avg 的含义

日志中的 `*_avg` 是当前 window 内该字段所有样本的算术平均值：

```text
avg = sum(samples) / sample_count
```

例如：

```text
loop_avg = 当前 120 帧 loop_ms 的平均值
palm_total_avg = 当前 120 帧 palm_total_ms 的平均值
palm_inference_avg = 当前 120 帧 palm_inference_ms 的平均值
```

当 `kInferInterval=1` 时，`palm_total_avg` 通常接近“每次 palm detector 调用平均耗时”。

当 `kInferInterval=3` 时，`palm_total_avg` 包含约 80 帧的 0 值和约 40 帧的实际推理耗时，因此它表示摊薄到每帧后的平均处理压力。若要粗略估计单次推理平均耗时，可以看离线脚本输出的：

```text
per_infer_avg_est = per_frame_avg * kInferInterval
```

## 6. P95 的含义

P95 是第 95 百分位数（95th percentile），表示当前 window 内有大约 95% 的样本不超过该值，剩下约 5% 的样本比它更高。

它比 `max` 更稳健，也比 `avg` 更能反映偶发高延迟。实时系统中，P95 常用于回答：

```text
绝大多数帧的高尾延迟大约是多少？
```

当前 C++ 代码中的计算方式是：

1. 拷贝当前 window 内某个阶段的所有耗时样本。
2. 升序排序。
3. 取 `ceil(n * 0.95) - 1` 位置的样本作为 P95。

以 `n=120` 为例：

```text
p95_index = ceil(120 * 0.95) - 1 = 113
```

也就是排序后第 114 个样本。它不是最大值，但已经非常接近高尾延迟。

P95 不应乘以 `kInferInterval`。P95 是当前 window 内样本分布的直接统计结果，离线脚本也不会对 P95 做跳帧换算。

## 7. 每组 PERF 日志的含义

每个 window 会输出 5 类日志：

```text
[PERF][summary] ...
[PERF][stage_ms] ...
[PERF][palm_detail_ms] ...
[PERF][latency] ...
[PERF][jitter] ...
```

### 7.1 `[PERF][summary]`

常用字段：

- `frame`: 当前 window 最后一帧的 frame index。
- `total_frames`: 程序启动后累计处理的总帧数。
- `window_frames`: 当前 window 的样本帧数，正常为 `120`。
- `elapsed_s`: 程序启动到当前日志打印时的应用侧运行时间。
- `sensor_fps_cfg`: 配置的摄像头帧率。
- `app_fps_total`: 从程序启动到当前的整体平均 FPS。
- `app_fps_window`: 当前 window 内的应用处理 FPS。
- `R`: 实时性比例，计算为 `app_fps_window / sensor_fps_cfg`。
- `realtime_score_est`: 按 `floor(10 * clamp(R, 0, 1))` 估算的实时性评分。
- `drop_rate_est_pct`: 根据 `sensor_fps_cfg` 和应用侧累计处理帧数估算的丢帧率。

`app_fps_window` 的计算方式：

```text
app_fps_window = window_frames * 1000 / sum(loop_ms in current window)
```

`drop_rate_est_pct` 是应用侧估计值，不是硬件 sensor 直接上报的真实丢帧计数。若 `sensor_fps_cfg` 与现场实际 sensor FPS 不一致，该值会有偏差。

### 7.2 `[PERF][stage_ms]`

这是顶层阶段耗时，单位 ms：

- `get_image_*`: `GetImage()` 阶段耗时，包含等待图像的时间。
- `palm_total_*`: 整个 `PALMDETECTOR::Predict()` 耗时，包含 palm 预处理、输入装载、`ssne_inference()`、`ssne_getoutput()`、输出 metadata、decode/NMS 和 verbose 日志。
- `hand_total_*`: `HANDLANDMARKER::Predict()` 耗时，包含 ROI affine crop、hand 输入装载、hand 推理、输出获取、输出映射和 landmark 投影。
- `draw_*`: OSD 清屏和绘制耗时。
- `process_*`: 从 `GetImage()` 返回后到绘图完成的处理耗时。
- `loop_*`: 从开始取图到绘图完成的整帧循环耗时。
- `loop_max`: 当前 window 中 `loop_ms` 的最大值。

`loop_ms` 包含 `GetImage()` 等待时间，通常更接近应用侧一帧循环的真实墙钟耗时。`process_ms` 不包含取图等待，更适合观察取图之后的纯处理压力。

### 7.3 `[PERF][palm_detail_ms]`

这是 palm detector 内部的细分耗时，单位 ms：

- `palm_preprocess_*`: `PreprocessRotateResize()` 总耗时。
- `palm_preprocess_transform_*`: 图像变换耗时。当前是融合的顺时针旋转 + 双线性 resize，不会单独生成旋转中间图。
- `palm_preprocess_manual_load_*`: 把 `manual_input_buffer` 装载到 `manual_input` tensor 的耗时。
- `palm_input_load_*`: 把预处理输出装载到模型输入 tensor 的耗时。
- `palm_inference_*`: palm 模型的 `ssne_inference()` 调用耗时，最接近纯 palm 模型推理时间。
- `palm_getoutput_*`: palm 模型的 `ssne_getoutput()` 耗时，可能包含输出同步或拷贝。
- `palm_output_meta_*`: 获取 palm 输出 tensor metadata 并完成输出映射的耗时。
- `palm_decode_*`: palm 输出 decode、NMS、候选选择和坐标反变换耗时。
- `palm_verbose_log_*`: verbose 诊断打印耗时，默认应接近 0。
- `palm_accounted_*`: 上述细分阶段加和，可与 `palm_total_*` 对照，用于观察未细分的函数调用或计时误差。

### 7.4 `[PERF][latency]`

这是面向端到端延迟的指标：

- `sensor_period_ms`: 摄像头帧周期，当前 `90 FPS` 时约为 `11.111 ms`。
- `e2e_loop_p95_ms`: 当前 window 的 `loop_ms` P95。
- `e2e_loop_p95_T`: `e2e_loop_p95_ms / sensor_period_ms`，表示 P95 循环耗时相当于多少个 sensor 帧周期。
- `e2e_process_p95_ms`: 当前 window 的 `process_ms` P95。
- `e2e_process_p95_T`: `e2e_process_p95_ms / sensor_period_ms`。
- `latency_score_est_by_loop`: 按 `floor(11 - e2e_loop_p95_T)` 估算的延迟评分，并限制在 `0..10`。

当前没有硬件 sensor 时间戳，因此这里的端到端延迟是应用侧估算。建议分析时同时保留 `loop` 和 `process` 两个口径。

### 7.5 `[PERF][jitter]`

这是抖动指标：

- `loop_ms_avg`: 当前 window 内 `loop_ms` 均值。
- `loop_ms_p95`: 当前 window 内 `loop_ms` P95。
- `loop_jitter_p95_vs_avg_pct`: `loop_ms` 的 P95 相对均值波动比例。
- `instant_fps_avg`: 当前 window 内瞬时 FPS 的均值，单帧瞬时 FPS 按 `1000 / loop_ms` 计算。
- `instant_fps_p95`: 当前 window 内瞬时 FPS 的 P95。
- `fps_jitter_p95_vs_avg_pct`: 瞬时 FPS 的 P95 相对均值波动比例。

`loop_jitter_p95_vs_avg_pct` 的计算方式：

```text
100 * abs(loop_p95 - loop_avg) / loop_avg
```

这个值越大，说明高尾帧耗时与平均帧耗时差距越大。

## 8. 离线分析脚本

脚本位置：

```text
scripts/analyze_perf_log.py
```

命令格式：

```text
python scripts/analyze_perf_log.py --mode palm --kInferInterval 1 --log result.log
python scripts/analyze_perf_log.py --mode palm_hand --kInferInterval 3 --log result.log --out_dir perf_out
```

参数：

- `--mode palm`: 统计 palm-only 模式。
- `--mode palm_hand`: 统计 palm + hand 级联模式。
- `--kInferInterval N`: 板端运行时使用的推理间隔。
- `--log`: 日志文件路径。
- `--out_dir`: 输出目录，默认与 log 文件相同。

输出文件：

- `*_stats_summary.txt`: 中文短摘要，列出采集总帧数、推理总帧数、FPS、丢帧率、各阶段均值/P95、单次迭代总延迟等核心指标。
- `*_stats_detail.txt`: 英文详细报告，包含完整 window 表格和跨 window 汇总。
- `*_stats_throughput.svg`: FPS、实时性比例和丢帧率趋势图。
- `*_stats_stage_latency.svg`: 最新 window 顶层阶段均值/P95 柱状图。
- `*_stats_palm_detail.svg`: 最新 window palm 内部细分耗时图。
- `*_stats_iteration_pie.svg`: P95 推理迭代耗时构成饼图。
- `*_stats_iteration_pie_info.txt`: 饼图每个扇区对应的阶段、颜色、耗时和百分比。
- `*_stats_p95_timeline.svg`: P95 延迟随 window 变化的时间线。

离线脚本对推理次数的统计是估计值：根据 `window` 的 frame 范围和 `kInferInterval` 计算 palm 推理帧数量；在 `palm_hand` 模式中，hand 调用帧数按推理帧估计。实际 hand 模型调用还会受 palm 是否检测到手和 ROI 数量影响，当前日志没有逐 ROI 计数。

## 9. 饼图的统计口径

`*_stats_iteration_pie.svg` 使用 P95 口径，而不是均值口径。

当前饼图数据来自所有 `[PERF]` window 中对应阶段 P95 的正值窗口均值。例如 `Hand total` 使用所有 `hand_total_p95 > 0` 的 window 求均值；`Palm inference` 使用所有 `palm_inference_p95 > 0` 的 window 求均值。

饼图中的总和是各阶段 P95 分量相加：

```text
pie_sum = get_image_p95 + palm_preprocess_transform_p95 + palm_inference_p95 + hand_total_p95 + ...
```

这个总和适合用于 PPT 展示瓶颈构成，但它不是严格意义上“同一帧的端到端 P95”，因为不同阶段的 P95 可能来自不同帧。

图内文字只展示阶段名和百分比，精确 ms 值保存在 `*_stats_iteration_pie_info.txt` 中。

## 10. 现场解读建议

优先看这些指标：

- `app_fps_window`: 当前窗口实际处理 FPS。
- `R`: 是否接近或超过 1。`R >= 1` 表示应用处理速度达到配置 sensor FPS。
- `drop_rate_est_pct`: 应用侧估算丢帧率，注意它依赖 `sensor_fps_cfg`。
- `loop_p95`: 一帧完整循环的高尾耗时。
- `process_p95`: 取图返回后处理链路的高尾耗时。
- `palm_detail_ms palm_inference_*`: palm 纯 `ssne_inference()` 耗时。
- `palm_detail_ms palm_preprocess_transform_*`: palm 旋转 + resize 融合预处理耗时。
- `hand_total_*`: hand landmarker 总耗时，启用 hand 后通常是级联系统瓶颈之一。

如果 `avg` 很低但 `P95` 很高，说明系统存在明显高尾抖动；如果 `loop_p95` 远大于 `process_p95`，说明 `GetImage()` 等待或取图侧行为对整帧循环影响较大；如果 `process_p95` 很高，则瓶颈主要在模型、预处理、后处理或绘图链路。
