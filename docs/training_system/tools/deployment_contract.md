# A1 板端与 ONNX 部署契约

## 1. PC 与板端输入

Keras/ONNX 对外输入固定为 `float32 (1,1,256,256)`，值域 `[0,1]`。模型内部第一步把 NCHW Permute 成 NHWC。

A1 C++ 创建的是 `256×256 SSNE_Y_8`，把 ROI 的 uint8 原始字节直接装入 tensor；C++ 没有执行 `/255`。因此 ONNX→m1model 工具链必须把输入归一化配置为与 PC 的 `uint8/255` 等价。若转换配置遗漏这一点，即使 ONNX 评估正确，板端也会严重失真。

## 2. 外部原图/板端路径的 Palm 与 ROI

本节只约束 `make infer-geometry` / `make infer-multitask` 处理任意外部图片以及 A1 板端的端到端运行路径。Val/Test 输入已经是 `256×256` Hand ROI，评估时不会加载、调用 Palm，也不会重新构造 ROI。

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

landmark 监督与输出语义唯一固定为归一化 crop 坐标。landmark head 是线性输出，不会把数值 clamp 到 `[0,1]`；板端也不会对异常值做 `/256` 或其他单位猜测。评估与推理报告最大绝对值和越界坐标数，便于把模型异常直接暴露出来。两个 sigmoid head 必须输出 `[0,1]`；PC 参考实现遇到范围外结果会拒绝该输出。

## 4. ONNX 导出

```bash
make export-geometry
```

multitask 完成后使用 `make export-multitask`。目标名直接决定 checkpoint 和校准训练集子阶段。

导出器执行以下门禁：

- 加载 v2 训练图及指定权重；
- 将训练期 Conv+BN 精确折叠为单 Conv 部署图，并验证融合前后数值一致；
- 断言 Keras 输入/输出接口；
- TensorSpec 名固定为 `inputs`；
- 静态 batch=1，默认 opset 11；
- `onnx.checker`；
- 输出元素数、语义顺序和 shape 检查；
- A1 算子白名单及属性检查；
- zeros/ones/random 输入的 Keras↔ONNX 数值比对；
- INT8 量化准备检查：所有 Conv 权重必须有限且不能整张量为零，group 不得超过 128，三个输出均须对探针输入产生非零动态范围；
- 输出 `.contract.json`，把配置的 `model.checkpoint_stage` 记为 `model_checkpoint_stage`，并包含权重/ONNX SHA-256、operators、属性审计、I/O、数值误差和输出范围。

属性门禁来自当前 A1 project-9 约束，不只检查算子名称：

- Conv 的 kernel、stride 与 pad 各维均不得大于 16；每个输出 kernel 的 `Kw×Kh×Cin` 不得大于 2048，且权重必须是静态 initializer；
- MaxPool 的 kernel 各维必须在 `1..8`；

`strict_a1_operators: true` 时，白名单外算子或任一属性违规都会让导出失败；失败的临时模型不会替换最终 ONNX。

当项目内维护的清单尚未覆盖已经由 A1 官方工具链验证过的算子时，可以只对本次运行追加 `--force`：

```bash
make export-geometry EXPORT_ARGS=--force
```

该参数只绕过 A1 算子名称与属性门禁。`onnx.checker`、固定 I/O/类型/shape、静态 batch、opset 11、Keras↔ONNX 数值比对、量化准备检查、转换数据检查和覆盖保护仍然执行。全零 Conv、恒定输出或 group 超限不能用 `--force` 绕过。contract 的 `a1_operator_audit` 会保留 `unsupported` 与属性违规，并记录 `strict: true`、`forced: true`、`enforced: false`。若目标产物已存在，仍需单独、显式追加 `--overwrite`。

目标优化图应只包含 `Conv/Add/Relu/MaxPool/Sigmoid/Reshape`（允许无语义的 Identity）。`Reshape` 已确认属于当前官方白名单；`LeakyRelu` 已被当前官方转换工具拒绝。训练期 BatchNormalization 必须在导出前折叠；若最终 ONNX 仍出现 BatchNormalization、LeakyRelu、Transpose、动态 shape、Div、Softmax 等，严格导出会失败，不应继续送入 A1 工具链。

默认 parity 探针是 1 个全零、1 个全一和 4 个固定随机种子的 `[0,1]` 输入。契约报告对每个探针、每个输出 head 分别记录最大绝对/相对误差，并同时记录 Keras 与 ONNX 的 minimum、maximum、max-abs；还会聚合每个输出的动态范围和标准差。验收使用配置中的 `atol=1e-5`、`rtol=1e-4` 做逐元素 `allclose`，任一 case/head 失败即中止导出；任一输出聚合动态范围小于 `1e-6` 也会中止。这里验证的是 Keras 与 ONNX 的数值等价及合成输入下的输出健康度，不是数据集精度，也不是 `.m1model` 板端实测。

默认产物按 checkpoint 阶段隔离：

```text
${HAND_TRAIN_ROOT}/hand_landmarker_runs/<PRETRAIN_ID>/export/<phase>/hand_landmarker_v2.onnx
${HAND_TRAIN_ROOT}/hand_landmarker_runs/<PRETRAIN_ID>/export/<phase>/hand_landmarker_v2.contract.json
${HAND_TRAIN_ROOT}/hand_landmarker_runs/<PRETRAIN_ID>/export/<phase>/model_conversion/datasets.zip
```

其中 `<phase>` 为 `geometry` 或 `multitask`。配置中的 `model.checkpoint_stage: pretrain` 声明大训练阶段，phase 则由 Makefile 选择当前 pretrain 子阶段。contract 额外记录 `reparameterization_parity` 的训练/部署参数量与逐输出融合误差。

每次 `make test` 都会在训练前额外生成 disposable preflight bundle：

```text
${HAND_TRAIN_ROOT}/hand_landmarker_runs/<PRETRAIN_ID>/export/preflight/
├── hand_landmarker_v2_untrained.onnx
├── hand_landmarker_v2_untrained.contract.json
├── untrained.weights.h5
└── model_conversion/datasets.zip
```

它采用确定性的、非零的 disposable 量化探针权重，只用于提前提交 A1 官方转换工具验证结构、算子和 INT8 量化路径，不能用于推理效果或精度评估。正式训练仍使用模型定义中的稳定初始化；preflight 不会回写或改变训练配置。正式 checkpoint 和 preflight 都执行真实 ONNX `maximum_model_size_mb: 15.0` 门禁；contract 记录 `model_size_bytes/model_size_mb` 和 `quantization_readiness`。当前 group=128 是根据旧 v1 已经到达算子检查阶段的图所设置的保守兼容边界，不代表厂商公开宣称的硬件极限。

`export` 还会自动制作模型转换输入：从当前阶段 Train 以稳定 SHA-256 分层抽取 100 个校准 ROI，从 Val/Test 各抽取 25 个评测 ROI。所有文件都是经 `uint8/255` 得到的 `float32 (1,1,256,256)` NCHW Hand ROI；不会运行 Palm、读取原图或写入模型输出。严格的 `datasets/` 树、可直接交付的 `datasets.zip` 以及树外的来源 manifest/report 位于同阶段 `model_conversion/`。详细格式见[模型转换数据制作说明](../model_conversion/conversion_method.md)。

`export.overwrite: false` 同时保护 ONNX、`.contract.json` 与 `model_conversion/`；任一产物已存在都会中止。只有确认这些产物都可以替换时才启用覆盖。

单次导出可以在不改 YAML 的情况下覆盖三个路径：

```bash
python scripts/export_onnx.py --config configs/export.yaml \
  --weights-path /path/to/checkpoint.weights.h5 \
  --output-path /path/to/hand_landmarker.onnx \
  --contract-path /path/to/hand_landmarker.contract.json \
  --conversion-output-dir /path/to/model_conversion
```

确认可替换产物时再追加 `--overwrite`。这些 CLI 参数不会改写 YAML，也不会根据自定义权重路径自动改变 `model.checkpoint_stage`；应使用与权重真实来源一致的 `export_<stage>.yaml` wrapper。

ONNX 生成后仍需使用 A1 官方工具转为 `.m1model`。该厂商转换步骤不在本仓库中自动执行；转换时必须再次核对 `/255` 输入归一化、三个 FLOAT32 输出、输出顺序和板端实测精度。
