# HLML 下一阶段计划

## 当前目标

`iris-1.1-geometry-eos2-hcf0813-r1` 的 geometry、multitask、两阶段固定 ROI Val、EOS 2.0 原图 infer 与 multitask export 均已完成；`neg-eos_2.0-hcf0813-hp0.5` 已作为 16,910 条真负样本发布。当前目标是在同一 snapshot 下分轮挖掘困难 ROI，经 HLMF CVAT 1.1 精修与通用 Gold 发布后，准备 multi-finetune。

## 执行顺序

1. 已完成负样本发布与真实 multitask data audit：Train 91,135（positive 74,225、negative 16,910）、Val 6,937、Test 2,342，成员错误为 0。
2. 已将 multitask 采样修正为 `POS_RUNTIME=0.90`、`NEG_LOW_PALM_CANDIDATE=0.10`，不存在的 `POS_LOW_PALM` 与本轮不采用的 `NEG_RUNTIME_CANDIDATE` 均为 0。
3. 已完成 RTX 3090 一步 GPU smoke，确认从 geometry `best.weights.h5` 初始化、正负样本采样、反向传播、Val、checkpoint 和报告链路均可用。
4. 正式 multitask 已完成 27 epoch，winner 为 epoch 14；handedness 明显改善，但 landmark 略退化，presence 对已发布真负样本仍无有效区分。
5. 设置同一 `HLML_SNAPSHOT_ID`，执行第一轮 `make mine-hard MINING_ARGS='--round-id r01 --max-rois 1000'`；困难度按 landmark 80%、presence 10%、handedness 10% 排序。
6. 在 HLMF 用独立通用 `hard_dataset_id` 执行 `hard-review`，完成 CVAT 1.1 精修后执行 `hard-import` / `hard-publish`。需要更多样本时使用 `r02 --max-rois 1500`；ledger 自动排除 r01 ROI。
7. 将一个或多个 `hard_dataset_id` 写入 `hard_datasets`；如有新录制数据，按 Eval 同款链路发布到 `GoldSource/ReviewedDatasets` 后写入 `gold_datasets`，再执行 multi-finetune audit/smoke/正式训练。
8. HaMeR 独立 variant 可用于后续 RTMPose 困难来源补充，但不得原地混入当前冻结 snapshot。先在 HLMF 完成人工复核与发布，再在 HLML 用新 snapshot ID 显式登记 dataset/source/variant 并执行 data audit；direct HaMeR 与 HaMeR TFLite rescue provenance 必须保留。

## 验收条件

- multitask winner 与当前评估结果保持只读；本次数据合同更新不擅自重训或改写 checkpoint。
- 正式 snapshot 必须精确包含三个 HCF0813 Train positive dataset 和已发布负样本集 `neg-eos_2.0-hcf0813-hp0.5`；空来源不进入负样本 manifest，但不能导致发布或 HLML audit 失败。
- 采样器每个完整 batch 只从实际存在的 `POS_RUNTIME` 与 `NEG_LOW_PALM_CANDIDATE` 单元取样；不得为缺失的 `POS_LOW_PALM` 静默重分配。
- Val/Test 继续使用已冻结的 Eos-2.0 source 白名单；Test 不回流到采样、阈值、checkpoint 或困难挖掘。
- 每个 mining round 必须提供正整数 `max_rois` 与未使用的 `round_id`；同一 snapshot 已筛 ROI 不得再筛，且不要求对整个 DatesetFab 做全局历史去重。
- multi-finetune hard dataset 只来自 Train mining 与 CVAT 1.1 人工精修，读取 HLMF 独立 `published_relpath` 图片并以 `source_crop_relpath` 核对来源身份；发布 ID 不得绑定 snapshot/run/round。
- 可选 recorded Gold 必须是新录制 train 数据，含人工确认 positive/negative，不能取自既有 PretrainSource/EValSource；其发布 ID 同样长期可复用。
- 既有 EValSource/PretrainSource manifest、完整标注链路、质量门控和负样本发布合同必须继续通过回归测试。
- HaMeR 行只改变 landmark teacher provenance，不改变 `256×256` 灰度 ROI、21 点、presence/handedness 或训练目标契约；禁止根据 teacher 类型静默改变采样权重。
