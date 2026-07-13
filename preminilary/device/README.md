# AetherSign: Sign Language Recognition System

## I. 代码简介

本段程序是基于思特威 A1 NPU 的中文手语识别系统的板端程序源代码。
除了本文件夹下的代码，还需要配合：

```
D:\Docker\project\smartsen\data\A1_SDK_SC132GS\smartsens_sdk\output\opt\m1_sdk\usr\include\smartsoc
```

下的头文件，尤其是：

- ssne_api.h
- osd_lib_api.h

主要文档：

- （古老）最开始 Palm Detector 模型上板踩坑总结出来的经验：[PALM_DEBUGGING_NOTES.md](./PALM_DEBUGGING_NOTES.md)
- SLR 手语识别系统工作流程说明：[SLR_WORKFLOW.md](./SLR_WORKFLOW.md)
- SLR 性能测试说明：[SLR_PERFORMANCE_TESTING.md](./SLR_PERFORMANCE_TESTING.md)


## II. 文件目录结构

### `./main.cpp`

- 程序入口文件，负责解析命令行参数、初始化 SDK/模型/摄像头/OSD、运行主循环、调度 palm + hand 级联推理、绘图和性能统计。

### `./include/`

- `common.hpp`: 定义 palm detector、hand landmarker、图像处理器等核心类、结果结构体和模型相关常量。
- `log.hpp`: 简单日志宏定义。
- `osd-device.hpp`: OSD 底层绘制封装的头文件，声明线框、点、线段等显示接口。
- `performance_monitor.hpp`: 性能统计系统的头文件，定义 `FrameTiming` 和 `PerformanceMonitor`。
- `utils.hpp`: 上层可视化绘制工具声明，负责把检测结果转成 OSD 绘制调用。

### `./scripts/`

- `analyze_perf_log.py`: PC 端性能日志分析与绘图脚本，读取 `result.log` 并生成 txt 报告和 SVG 图表。（该脚本仅在 PC 端使用）
- `run.sh`: 板端启动脚本，将命令行参数转发给 `ssne_ai_demo`。

### `./src/`

- `hand_landmarker.cpp`: hand landmarker 的模型加载、ROI 裁剪、推理、输出解码和关键点反投影实现。
- `osd-device.cpp`: OSD 底层设备初始化、图层管理和基础图元绘制实现。
- `palm_detector.cpp`: palm detector 的模型加载、预处理、推理、输出解码、NMS 和坐标映射实现。
- `performance_monitor.cpp`: 性能统计系统实现，负责按 window 汇总并打印 `[PERF]` 日志。
- `pipeline_image.cpp`: 摄像头在线取图 pipeline 的初始化、取图和释放实现。
- `utils.cpp`: 检测框、palm 关键点和 hand 骨架的上层绘制逻辑。



### `./CMakeLists.txt`

- CMake 构建入口，定义 `ssne_ai_demo` 可执行文件、源码列表、头文件路径、SDK 库链接和安装规则。

#### `./app_assets/`

- `colorLUT.sscl`: OSD 绘制使用的颜色查找表。
- `models/palm.m1model`: palm detector 板端模型文件。
- `models/hand.m1model`: hand landmarker 板端模型文件。

### `./cmake_config/`

- `Paths.cmake`: 配置 M1 SDK 头文件、库文件和第三方依赖路径，供 `CMakeLists.txt` 引用。


## III. 编译运行方式

1. **编译：** 将该文件整个文件夹放置在赛题二仓库的 `/smartsens_sdk/smart_software/src/app_demo/slr_system/ssne_ai_demo/` 目录下（注意 `slr_system` 是我们项目的文件夹名字，与原来的 face_detection 例程处于同一层级）。然后：
    ```bash
    cd A1_SDK_SC132GS/smartsens_sdk/
    ./scripts/a1_sc132gs_build.sh
    ```

2. **运行：** 烧录到 A1 开发板后，重启开发板即可启动我们的 slr_system 程序。也可通过：
    ```bash
    cd app_demo/
    ./ssne_ai_demo --kInferInterval <推理帧间隔> # 单 Palm Detector 模式
    ./ssne_ai_demo --kInferInterval <推理帧间隔>  --enable_hand # Palm Detector + Hand Landmarker 级联模式
    ```