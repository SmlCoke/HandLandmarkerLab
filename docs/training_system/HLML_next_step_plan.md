# HLML 下一阶段计划

## 当前目标

在三台克隆服务器上分别训练 `v3-pro`、`v3-max`、`v3-lite`，保持相同 snapshot、数据成员、随机种子、训练阶段和超参数，只让 `HLML_MODEL_VERSION` 与 `HLML_EXPERIMENT_ID` 不同。首先完成 geometry 对照，再以各自 geometry winner 初始化 multitask；multitask 必须使用完整负样本集 `neg-eos_2.0-hcf0813-hp0.5`。

本计划不把训练前 `export-preflight` 视为精度结果；multi-finetune 只有在通用 hard dataset 已发布且全量成员审计通过后才能启动。当前 `hard-hands-0816-r01` 已满足这两个前置条件。

## 执行顺序

1. 三台服务器加载相同代码提交，设置 GPU 环境变量并执行 `make environment-check`、`make config-check`。
2. 三台分别设置 `HLML_MODEL_VERSION=v3-pro|v3-max|v3-lite` 和唯一 `HLML_EXPERIMENT_ID`；共同使用 `HLML_SNAPSHOT_ID=iris-v3-data-r1`。
3. 分别执行 `make export-preflight HLML_STAGE=geometry`，复核结构、融合、ONNX、A1 算子与转换数据包。
4. 以完全相同的数据/训练配置分别执行 geometry；每台完成固定 ROI Val 和 Eos-2.1 代表性原图 infer，再依据预先约定的 geometry 指标比较。
5. 每档仅用自己的 geometry winner 初始化 multitask。先执行全量 data audit，确认 Train 99,812（positive 82,902 + negative 16,910）、Val 14,411、Test 5,343，再启动正式 multitask。
6. 每档 multitask 完成固定 ROI Val、Eos-2.1 原图 infer 和正式 export；选择结构时同时比较精度、稳定性、推理成本和 15 MiB 部署约束。
7. 使用已发布并完成 CVAT 复核的 `hard-hands-0816-r01`；`datasets.yaml` 默认引用该 ID，并允许用 `HLML_HARD_DATASET_ID` 显式覆盖。multi-finetune 采用已通过真实 snapshot 采样预检的 `epoch_size=3000`，正式启动前复核环境变量、multitask winner 路径和 rare-cell epoch plan。
8. 冻结唯一 winner 后运行 locked Test；Test 不参与模型选择、阈值选择、困难挖掘或重训。

## 对照与验收条件

- 三档必须使用相同 Git 提交、snapshot、数据配置、seed、batch、epoch、augmentation、loss 和评估协议。
- `v3-pro` 必须与未修改 v2 同构；`v3-max` 训练/部署参数比必须保持约 3.99，部署参数量仍为 1,912,324；`v3-lite` 部署参数量为 852,832。
- 训练期多分支与部署期单分支在固定输入上的数值误差必须继续通过自动测试；正式 export 图中不得残留 BatchNormalization 或训练分支。
- multitask snapshot 必须完整包含 `neg-eos_2.0-hcf0813-hp0.5`。正负 proposal variant 只在各自发布域内判唯一；split、原图、ROI、Registry、解码与 Test 隔离门禁必须全部通过。
- 每次正式训练后都必须记录 best epoch、监控指标、固定 ROI Val、原图 infer 与部署导出结果；不能用训练前随机权重 smoke 代替。
- `hard_datasets` 只能填写 HLMF registry 中状态为 published 的真实 ID；当前默认成员为 `hard-hands-0816-r01`。其旧 Eos-2.0 proposal variant 在独立 hard 发布域内审计，不迫使 Eos-2.1 replay 降级或改写。
