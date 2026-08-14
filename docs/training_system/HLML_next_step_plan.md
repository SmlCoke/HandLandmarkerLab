# HLML 下一阶段计划

## 当前目标

`iris-1.1-geometry-eos2-hcf0813-r1` 的 geometry、固定 ROI Val 和 EOS 2.0 原图 infer 已完成；`neg-eos_2.0-hcf0813-hp0.5` 已作为 16,910 条真负样本发布。当前目标是复用同一 snapshot/experiment ID，从 geometry winner 启动 Iris-1.1 multitask。

## 执行顺序

1. 已完成负样本发布与真实 multitask data audit：Train 91,135（positive 74,225、negative 16,910）、Val 6,937、Test 2,342，成员错误为 0。
2. 已将 multitask 采样修正为 `POS_RUNTIME=0.90`、`NEG_LOW_PALM_CANDIDATE=0.10`，不存在的 `POS_LOW_PALM` 与本轮不采用的 `NEG_RUNTIME_CANDIDATE` 均为 0。
3. 已完成 RTX 3090 一步 GPU smoke，确认从 geometry `best.weights.h5` 初始化、正负样本采样、反向传播、Val、checkpoint 和报告链路均可用。
4. 设置 `HLML_SNAPSHOT_ID=iris-1.1-geometry-eos2-hcf0813-r1`、同名 `HLML_EXPERIMENT_ID`、`HLML_NEGATIVE_DATASET_ID=neg-eos_2.0-hcf0813-hp0.5` 和 `HLML_STAGE=multitask`，执行 `make multitask`。
5. multitask 完成后使用同一组 ID 执行固定 ROI Val、EOS 2.0 代表性原图 infer 和 export；Test 继续锁定。
6. 使用 multitask winner 进行 Train-only mining，把请求交给 HLMF 做困难 positive 的人工删除式复核与发布，再进入 multi-finetune。

## 验收条件

- multitask 必须从 `runs/iris-1.1-geometry-eos2-hcf0813-r1/geometry/checkpoints/best.weights.h5` 新鲜初始化优化器，不得误用 resume。
- 正式 snapshot 必须精确包含三个 HCF0813 Train positive dataset 和已发布负样本集 `neg-eos_2.0-hcf0813-hp0.5`；空来源不进入负样本 manifest，但不能导致发布或 HLML audit 失败。
- 采样器每个完整 batch 只从实际存在的 `POS_RUNTIME` 与 `NEG_LOW_PALM_CANDIDATE` 单元取样；不得为缺失的 `POS_LOW_PALM` 静默重分配。
- Val/Test 继续使用已冻结的 Eos-2.0 source 白名单；Test 不回流到采样、阈值、checkpoint 或困难挖掘。
- multitask 完成后必须保存唯一 winner，并完成 Val、infer 和 export；export 同时交付 ONNX/A1 审计物与配套 `datasets.zip`。
- multi-finetune selection 只来自 Train mining 与人工删除式复核，读取 HLMF 独立 `published_relpath` 图片并以 `source_crop_relpath` 核对来源身份。
