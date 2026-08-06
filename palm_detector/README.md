# Eos Palm Detector 模型目录

文件夹推理只从本目录选择 Eos 系列模型：

```text
palm_detector/
  <model_id>/
    model.py
    model_opt.onnx
```

`model_id` 是单层安全目录名，例如 `eos-1.0`。每个版本保持相同的 NCHW 灰度输入和四输出 Palm 接口；`model_opt.onnx` 按仓库忽略策略在执行环境单独部署，不提交 Git。

全局默认由 `configs/inference.yaml` 的 `palm.model_id` 设置，也可用单次高优先级覆盖：

```bash
make infer HLML_STAGE=<stage> INFER_ARGS='--palm-model-id eos-2.0'
```

旧 `preminilary/palm` 不再是模型资产或推理入口；`preminilary/device` 仅保留 A1 调度参考代码。
