# HLML 当前状态（2026-08-19）

## Iris v3 代码状态

Iris v3 已实现并接入公共 registry、训练、固定 ROI 评估、文件夹推理和 ONNX/A1 导出入口，固定输入仍为 NCHW `[1,1,256,256]`，输出顺序与语义仍为 `landmarks[42]`、`hand_flag[1]`、`handedness[1]`。ROI 几何、loss、训练阶段和 HLMF `hlmf_dataset_v1` 合同没有改变，历史 `v2` 入口及实现保持不变。

| 版本 | 训练参数 | 融合后部署参数 | 训练/部署参数比 | 定位 |
| --- | ---: | ---: | ---: | --- |
| `v3-pro` | 1,951,756 | 1,912,324 | 1.02 | 与未修改的 v2 同构 |
| `v3-max` | 7,629,268 | 1,912,324 | 3.99 | 明显扩大训练期多分支容量，部署仍与 v2 同量级 |
| `v3-lite` | 878,272 | 852,832 | 1.03 | 缩减通道的轻量档 |
| `v2` | 1,951,756 | 1,912,324 | 1.02 | 历史入口，只做回归兼容 |

`v3-max` 的普通卷积训练期采用四个 Conv+BN 分支；Depthwise block 采用四个 3×3 Depthwise+BN 分支，并在形状允许时增加可融合的 1×1 与 identity BN 分支。所有分支在部署前精确折叠为单 Conv/Depthwise，未向 A1 图中引入分支或 BN。

## 当前数据成员与矛盾处理

当前 geometry/multitask 正样本来自四个 Eos-2.1 + HaMeR r4 发布集：`FullEnhance0801`、`FullEnhance0803`、`FullEnhance0810`、`FullEnhance0817`。固定 Val/Test 来自 `FullEnhanceVal0801`、`FullEnhanceVal0803`、`FullEnhanceVal0808`、`RainEnhanceVal0817` 的配置白名单。

后续 multitask 明确使用完整已发布负样本集 `neg-eos_2.0-hcf0813-hp0.5`。为解决 Eos-2.1 replay、Eos-2.0 负样本和可复用旧 Palm hard release 之间的 proposal variant 冲突，warehouse 对 PretrainSource、每个已发布负样本集和每个已发布 hard dataset 分别执行 variant 唯一性门禁；同一发布域内出现多个 proposal variant 仍失败。split、raw image、ROI ID、Registry、路径和图片解码门禁没有放宽。

服务器真实全量只读审计结果：

- geometry：Train 82,902、Val 14,411、Test 5,343，membership errors 为 0。
- multitask：Train 99,812，其中 positive 82,902、negative 16,910；Val 14,411、Test 5,343，membership errors 为 0。
- performer 跨 split 仍按既有 `warn` 策略报告，不影响 membership 结论。
- `hard-hands-0816-r01` 已由 HLMF 发布并通过只读审核：manifest 记录 462 条训练记录，其中 379 条 positive、83 条 CVAT `no_hand` negative，另有 38 条 ignored；462 张独立 published 图片、registry 身份、21 点/handedness 结构均一致。ID 中的 `r01` 作为该通用数据集的发布修订号使用，manifest 不含训练 snapshot/run 绑定。
- 配置后的 multi-finetune 全量不落盘预检为 Train 100,274、Val 14,411、Test 5,343，membership errors 为 0。Train 包含 17,372 条 hard/gold 侧（462 条 hard + 16,910 条真负样本）和 82,902 条 replay；462 条 hard 行全部派生为 `human_gold`，类型为 379 条 `POS_RUNTIME` 与 83 条 `NEG_RUNTIME_CANDIDATE`。

## GPU 与部署预检查

服务器为 RTX 3090 24 GB，`hand-landmarker-tf29` 环境只有在以下变量生效后才能稳定枚举并使用 GPU：

```bash
readonly CUDA_LIBRARY_DIR=/usr/local/cuda-11.2/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="$CUDA_LIBRARY_DIR:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDNN_V8_API_DISABLED=1
```

三档模型均已完成未训练权重的 `export-preflight`，该结果只说明图、融合、ONNX 和 A1 算子兼容，不代表模型精度：

| 版本 | ONNX 大小 | 随机输入融合最大绝对误差 | ONNX 最大绝对误差 | A1 算子 |
| --- | ---: | ---: | ---: | --- |
| `v3-pro` | 7.309 MiB | 5.96e-08 | 5.96e-08 | Add/Conv/MaxPool/Relu/Reshape/Sigmoid |
| `v3-max` | 7.313 MiB | 4.17e-07 | 6.56e-07 | 同左 |
| `v3-lite` | 3.289 MiB | 2.98e-08 | 5.96e-08 | 同左 |

三档均低于 15 MiB，并各自生成 Train 100、Val/Test 共 50 条的 `datasets.zip`。Eos-2.1 文件夹级联也已在默认 `InferSource/0718/images` 中各取一张图片完成 GPU smoke，三档均为 `status=ok`、失败 0；由于 Hand 权重未训练，这些 prediction 不作精度结论。

## 自动验收状态

- 82 个 Python 文件语法检查通过；HLML 完整单元测试 198 项通过。
- HLMF 上游 79 项测试、HLML warehouse 合同 16 项测试和 acceptance config check 全部通过。
- `v3-pro`、`v3-max`、`v3-lite` 三种 `HLML_MODEL_VERSION` 的公共 config check 均为 `status=ok`。
- 指向独立审计 snapshot 的 environment check 为 `ok=true`，TensorFlow 创建 RTX 3090 GPU device；默认正式 snapshot 未创建，符合本轮不写入 TrainFab 的边界。

## 本轮执行边界

本轮没有启动 multi-finetune 正式训练，也没有创建或覆盖 snapshot/checkpoint。`DatesetFab`、`TrainFab` 和已有 Pretrain/Eval 发布资产保持只读；审核与全量成员预检均在内存完成。环境依赖未增加，`requirements.txt` 与 `environment.yml` 无需更新。
