# HLML 4.0 部署契约

Hand Landmarker 保持 v2 接口：

- 输入：静态 `[1,1,256,256]`、NCHW、单通道 `float32`、uint8 除以 255。
- 输出顺序：`landmarks[42]`、`hand_flag[1]`、`handedness[1]`。
- landmarks：ROI 归一化 xy，MediaPipe 0..20 顺序。
- ONNX：opset 11、静态 batch、A1 算子白名单和严格数值一致性验证。

`make export` 只导出冻结 winner/指定阶段 checkpoint，不改变网络深度、loss 或 ROI 几何。`make infer` 可在任意原图上运行 Palm → Hand ROI → Hand；该部署级联与 Val/Test 固定 ROI 评估互不混用。
