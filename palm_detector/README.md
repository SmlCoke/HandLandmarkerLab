# Eos Palm Detector 模型目录

文件夹推理只从本目录选择 Eos 系列模型：

```text
palm_detector/
  <model_id>/
    model.py
    model_opt.onnx
```

当前唯一受支持的运行契约是 EOS 2.0：灰度 float32 NCHW `[1,1,224,384]`，四个 NCHW 输出对应 `14×24`、`7×12` 两层的 regression/classification，每层两组矩形 Anchor，合计 840 个 Anchor。两层候选合并后执行一次全局 NMS；EOS 1.0 的方形输入和跨 head 抑制不再兼容。

HLMF 源模型文件为 `models/palm_detector/eos-2.0/model_384x224_opt.onnx`；部署到本仓库时复制为 `palm_detector/eos-2.0/model_opt.onnx`。ONNX 按仓库忽略策略在执行环境单独部署，不提交 Git。

全局默认由 `configs/inference.yaml` 的 `palm.model_id` 设置，也可用单次高优先级覆盖：

```bash
make infer HLML_STAGE=<stage>
```

旧 `preminilary/palm` 不再是模型资产或推理入口；`preminilary/device` 仅保留 A1 调度参考代码。
