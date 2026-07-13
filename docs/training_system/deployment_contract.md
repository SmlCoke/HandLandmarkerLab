# A1 板端与 ONNX 部署契约

## 1. PC 与板端输入

Keras/ONNX 对外输入固定为 `float32 (1,1,256,256)`，值域 `[0,1]`。模型内部第一步把 NCHW Permute 成 NHWC。

A1 C++ 创建的是 `256×256 SSNE_Y_8`，把 ROI 的 uint8 原始字节直接装入 tensor；C++ 没有执行 `/255`。因此 ONNX→m1model 工具链必须把输入归一化配置为与 PC 的 `uint8/255` 等价。若转换配置遗漏这一点，即使 ONNX 评估正确，板端也会严重失真。

## 2. 外部原图/板端路径的 Palm 与 ROI

本节只约束 `make infer` 处理任意外部图片以及 A1 板端的端到端运行路径。Val/Test 输入已经是 `256×256` Hand ROI，评估时不会加载、调用 Palm，也不会重新构造 ROI。

冻结 Palm 输入是 upright `1280×720` 灰度图直接 half-pixel 双线性 resize 到 `224×224`，无 letterbox。板端来自 `720×1280` 传感器时先顺时针旋转，再 resize。

Palm 输出已经是 sigmoid score，不能再次 sigmoid。后处理严格遵循板端：

- anchors：14-head `0.10/0.18`，7-head `0.25/0.40`；
- 每 anchor：`dx,dy,dw,dh,p0x,p0y,p9x,p9y`；
- score threshold `0.50`；
- 14-head intra-head NMS `0.30`；
- 7-head 按分数进入 selected-candidate suppression `0.35`；
- 最多 2 个 ROI。

注意：标注仓旧 Python decode 会先对 7-head 做 `0.30` NMS，而当前板端不会。运行时参考实现选择板端行为，并用单元测试锁定差异。

Palm 坐标映射使用 `(width-1,height-1)` 并执行 C++ `std::round`。Hand ROI：

```text
rotation = normalize(pi/2 - atan2(-(p9_y-p0_y), p9_x-p0_x))
center shift = (0, -0.1) in rotated box axes
long_side = max(clamped_bbox_width, clamped_bbox_height)
roi_size = long_side × 1.8
corners = TL, TR, BR, BL
```

ROI 以 `x/255,y/255` 的 endpoint mapping 做双线性采样，越界补 0，插值后按 C++ 规则 round 为 uint8。关键点反投影：

```text
P_image = TL + x_roi × (TR-TL) + y_roi × (BL-TL)
```

不 clamp landmark。

## 3. 输出

输出语义顺序必须固定：

1. landmarks，42 个 FLOAT32；
2. hand flag，1 个 sigmoid FLOAT32；
3. handedness，1 个 sigmoid FLOAT32。

板端按元素数识别 landmarks，但两个标量只能按输出顺序区分；交换后不会报错，只会静默颠倒语义。Hand 输出必须保留 FLOAT32：当前 C++ 对 INT8/UINT8 输出没有按 quantization scale/zero-point 反量化。

新训练的 landmark 监督与输出语义是归一化 crop 坐标，但 landmark head 是线性输出，并不会把数值 clamp 到 `[0,1]`。板端为兼容旧模型，在整手 `max_abs>2` 时会把全部坐标除以 256；一个异常点超过阈值就会触发整手缩放，因此导出/评估会报告最大绝对值、越界计数和实际采用的 scale divisor。

## 4. ONNX 导出

```bash
make export
```

导出器执行以下门禁：

- 加载版本化 Keras 结构及指定权重；
- 断言 Keras 输入/输出接口；
- TensorSpec 名固定为 `inputs`；
- 静态 batch=1，默认 opset 11；
- `onnx.checker`；
- 输出元素数、语义顺序和 shape 检查；
- A1 算子白名单及属性检查；
- zeros/ones/random 输入的 Keras↔ONNX 数值比对；
- 输出 `.contract.json`，包含权重/ONNX SHA-256、operators、属性审计、I/O、数值误差和输出范围。

属性门禁来自当前 A1 project-9 约束，不只检查算子名称：

- Conv 的 kernel、stride 与 pad 各维均不得大于 16；每个输出 kernel 的 `Kw×Kh×Cin` 不得大于 2048，且权重必须是静态 initializer；
- MaxPool 的 kernel 各维必须在 `1..8`；
- LeakyRelu 的 `alpha` 只能是 `0.1` 或 `0.01`。

`strict_a1_operators: true` 时，白名单外算子或任一属性违规都会让导出失败；失败的临时模型不会替换最终 ONNX。

目标优化图应只包含 `Conv/Add/LeakyRelu/MaxPool/Sigmoid`（允许无语义的 Identity）。Keras 源码中的 Permute 应由 tf2onnx 优化消除；若最终 ONNX 仍出现 Transpose、动态 shape、Div、Softmax 等，严格导出会失败，不应继续送入 A1 工具链。

默认 parity 探针是 1 个全零、1 个全一和 4 个固定随机种子的 `[0,1]` 输入。契约报告对每个探针、每个输出 head 分别记录最大绝对/相对误差，并同时记录 Keras 与 ONNX 的 minimum、maximum、max-abs；landmark 输出还记录按板端规则推导的 scale divisor。验收使用配置中的 `atol=1e-5`、`rtol=1e-4` 做逐元素 `allclose`，任一 case/head 失败即中止导出。这里验证的是 Keras 与 ONNX 的数值等价及合成输入下的输出健康度，不是数据集精度，也不是 `.m1model` 板端实测。

默认产物为：

```text
${HAND_DATA_ROOT}/hand_landmarker_runs/v1/export/hand_landmarker_v1.onnx
${HAND_DATA_ROOT}/hand_landmarker_runs/v1/export/hand_landmarker_v1.contract.json
```

`export.overwrite: false` 同时保护 ONNX 与 `.contract.json`；任一产物已存在都会中止。只有确认两个文件都可以替换时才启用覆盖。

ONNX 生成后仍需使用 A1 官方工具转为 `.m1model`。该厂商转换步骤不在本仓库中自动执行；转换时必须再次核对 `/255` 输入归一化、三个 FLOAT32 输出、输出顺序和板端实测精度。
