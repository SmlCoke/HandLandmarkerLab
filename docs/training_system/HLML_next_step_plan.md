# HLML 下一阶段计划

## 当前目标

在三台克隆服务器上分别训练 `v3-pro`、`v3-max`、`v3-lite`，保持相同 snapshot、数据成员、随机种子、训练阶段和超参数，只让 `HLML_MODEL_VERSION` 与 `HLML_EXPERIMENT_ID` 不同。首先完成 geometry 对照，再以各自 geometry winner 初始化 multitask；multitask 必须使用完整负样本集 `neg-eos_2.0-hcf0813-hp0.5`。

本计划不把训练前 `export-preflight` 视为精度结果，也不在 hard dataset 尚未发布时启动 multi-finetune。

## 执行顺序

1. 三台服务器加载相同代码提交，设置 GPU 环境变量并执行 `make environment-check`、`make config-check`。
2. 三台分别设置 `HLML_MODEL_VERSION=v3-pro|v3-max|v3-lite` 和唯一 `HLML_EXPERIMENT_ID`；共同使用 `HLML_SNAPSHOT_ID=iris-v3-data-r1`。
3. 分别执行 `make export-preflight HLML_STAGE=geometry`，复核结构、融合、ONNX、A1 算子与转换数据包。
4. 以完全相同的数据/训练配置分别执行 geometry；每台完成固定 ROI Val 和 Eos-2.1 代表性原图 infer，再依据预先约定的 geometry 指标比较。
5. 每档仅用自己的 geometry winner 初始化 multitask。先执行全量 data audit，确认 Train 99,812（positive 82,902 + negative 16,910）、Val 14,411、Test 5,343，再启动正式 multitask。
6. 每档 multitask 完成固定 ROI Val、Eos-2.1 原图 infer 和正式 export；选择结构时同时比较精度、稳定性、推理成本和 15 MiB 部署约束。
7. winner 确定后只从 Train snapshot 挖掘困难 ROI，经 HLMF CVAT 复核并发布通用 hard dataset；把发布 ID 写入 `hard_datasets` 后，才审计并启动 multi-finetune。
8. 冻结唯一 winner 后运行 locked Test；Test 不参与模型选择、阈值选择、困难挖掘或重训。

## 对照与验收条件

- 三档必须使用相同 Git 提交、snapshot、数据配置、seed、batch、epoch、augmentation、loss 和评估协议。
- `v3-pro` 必须与未修改 v2 同构；`v3-max` 训练/部署参数比必须保持约 3.99，部署参数量仍为 1,912,324；`v3-lite` 部署参数量为 852,832。
- 训练期多分支与部署期单分支在固定输入上的数值误差必须继续通过自动测试；正式 export 图中不得残留 BatchNormalization 或训练分支。
- multitask snapshot 必须完整包含 `neg-eos_2.0-hcf0813-hp0.5`。正负 proposal variant 只在各自发布域内判唯一；split、原图、ROI、Registry、解码与 Test 隔离门禁必须全部通过。
- 每次正式训练后都必须记录 best epoch、监控指标、固定 ROI Val、原图 infer 与部署导出结果；不能用训练前随机权重 smoke 代替。
- 没有 HLMF 已发布 hard dataset 时，`hard_datasets` 必须保持空列表，multi-finetune 不得以占位 ID 启动。
