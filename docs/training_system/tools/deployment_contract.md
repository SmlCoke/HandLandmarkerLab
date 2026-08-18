# HLML 4.0 部署契约

所有注册模型 `v2`、`v3-pro`、`v3-max`、`v3-lite` 保持同一板端接口：

- 输入：静态 `[1,1,256,256]`、NCHW、单通道 `float32`、uint8 除以 255。
- 输出顺序：`landmarks[42]`、`hand_flag[1]`、`handedness[1]`。
- landmarks：ROI 归一化 xy，MediaPipe 0..20 顺序。
- ONNX：opset 11、静态 batch、A1 算子白名单和严格数值一致性验证。

`HLML_MODEL_VERSION` 控制训练、评估、推理和导出使用的结构。`v3-pro` 与未修改 v2 同构；`v3-max` 的训练多分支在导出前精确融合，部署参数量与 v2 同为 1,912,324；`v3-lite` 部署参数量为 852,832。正式 ONNX 不保留 BatchNormalization 或训练分支。

正式训练前执行：

```bash
export HLML_MODEL_VERSION=v3-pro  # 或 v3-max / v3-lite
make export-preflight HLML_STAGE=geometry
```

该入口生成明确标记为 untrained 的 ONNX、contract 和转换数据包，只供 A1 图/算子兼容测试，不代表精度。`make export HLML_STAGE=multitask|multi_finetune` 则导出正式 checkpoint，并生成 Train 100 个 calibration、Val/Test 各 25 个 evaluation，均为 `float32 (1,1,256,256)` `.npy`，打包为 `model_conversion/datasets.zip`。

`make infer` 默认运行 Eos-2.1 Palm → Hand ROI → 所选 Iris。Palm 模型部署为 `palm_detector/eos-2.1/model_opt.onnx`，默认原图目录为 `/root/autodl-tmp/DatesetFab/InferSource/0718/images`。Palm 接口仍为灰度 float32 NCHW `[1,1,224,384]`，解码两层共 840 个 Anchor 并执行一次全局 NMS；Hand ROI 保持 scale `1.8/1.8`、shift `0/-0.1` 和 `256×256`。文件夹级联与 Val/Test 固定 ROI 评估互不混用。
