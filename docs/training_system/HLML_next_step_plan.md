# HLML 下一阶段计划

## 当前目标

以 HLMF 3.0 已发布的 `FullEnhance0801` Train 和 `FullEnhanceVal0801` Val/Test 完成首个 HLML 4.0 geometry 正式训练。2026-08-07 全量数据审计和训练烟雾测试完成后，由操作者启动正式训练。

## 执行顺序

1. 为正式运行设置新的 `HLML_SNAPSHOT_ID`、`HLML_EXPERIMENT_ID`，并固定 `HLML_PRETRAIN_DATASET_ID=FullEnhance0801`、`HLML_EVAL_DATASET_ID=FullEnhanceVal0801`、`HLML_PROPOSAL_VARIANT=eos_1.0-gate`、`HLML_EVAL_PROPOSAL_VARIANT=eos-1.0`。
2. 执行 `make geometry`，观察 tqdm、Val loss、checkpoint 和 wall-time 门控。
3. geometry 结束后执行 `make val HLML_STAGE=geometry` 与代表性原图 `make infer HLML_STAGE=geometry`。
4. 在 HLMF 将人工复核的真负样本发布为 negative dataset，写入 `configs/datasets.yaml` 后开始 multitask。
5. multitask 结束后执行 Val、infer、export；确认 ONNX/A1 报告和 conversion `datasets.zip`。
6. 运行 Train-only mining，在 HLMF 删除教师错误并发布 selection。
7. 以困难 55%、replay 45% 完成 multi-finetune，再执行 Val、infer、export。
8. 根据固定 Val 冻结唯一 winner，最后执行一次 locked Test。

## 验收条件

- geometry 正式训练使用审计通过的 65,089/5,091 Train/Val ROI；Train 按实际单元组成使用 100% `pseudo/POS_RUNTIME` 严格抽样，Test 不参与选择。
- 每阶段均保存 winner 并完成 Val/infer；multitask、multi-finetune 的 export 同时交付模型与配套数据包。
- multitask 负样本仅来自 HLMF published negative dataset；multi-finetune selection 仅来自 Train mining 与人工删除式复核。
- Test 不回流到采样、阈值、checkpoint 或困难挖掘。
