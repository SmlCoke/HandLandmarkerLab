# HLML 4.0 部署契约

Hand Landmarker 保持 v2 接口：

- 输入：静态 `[1,1,256,256]`、NCHW、单通道 `float32`、uint8 除以 255。
- 输出顺序：`landmarks[42]`、`hand_flag[1]`、`handedness[1]`。
- landmarks：ROI 归一化 xy，MediaPipe 0..20 顺序。
- ONNX：opset 11、静态 batch、A1 算子白名单和严格数值一致性验证。

`make export HLML_STAGE=multitask|multi_finetune` 导出指定阶段 checkpoint，并同步生成转换工具配套数据：Train 100 个 calibration、Val/Test 各 25 个 evaluation，均为 `float32 (1,1,256,256)` `.npy`，打包为 `model_conversion/datasets.zip`。Export 不改变网络深度、loss 或 ROI 几何。

`make infer` 运行 EOS 2.0 Palm → Hand ROI → Hand。Palm 接口固定为灰度 float32 NCHW `[1,1,224,384]`，解码 `14×24` 与 `7×12` 两层共 840 个 Anchor，候选合并后执行一次全局 NMS；模型部署为 `palm_detector/eos-2.0/model_opt.onnx`。Hand ROI 仍为 scale `1.8/1.8`、shift `0/-0.1` 和 `256×256`。该部署级联与 Val/Test 固定 ROI 评估互不混用。
