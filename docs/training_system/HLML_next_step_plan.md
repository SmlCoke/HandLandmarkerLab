# HLML 下一阶段计划

## 当前目标

`national-final-geometry-eos_1.0-gate-r1` 的 geometry 正式训练、固定 ROI Val 和 EOS 1.0 历史 infer 已完成。HLMF 已发布 near/mid EOS 2.0 Gold、`0809-soar-enhance` 等人工确认的 true-negative dataset；HLML 已兼容部分 source 发布 variant，文件夹推理也已切换到 EOS 2.0。下一目标是冻结本轮 near/mid 数据成员，选定负样本并完成 multitask 训练与评估。

## 执行顺序

1. 保留 geometry 第 68 epoch winner、Val 指标和 0718 inference 产物，不使用 Test 重新选 checkpoint。
2. 复核并选定已发布的 true-negative dataset；当前可用成员包括 `0809-soar-enhance` 与 `background-neg-0801-full`。
3. 在 `configs/datasets.yaml` 的 `stages.multitask.datasets` 与 `negative_datasets` 列表逐项配置 ID、proposal variant 和权重；固定 ROI Val 使用 `FullEnhanceVal0801/eos_2.0-rtmpose-gate` 的两条 Val source。EOS 2.0 的 far 排除继续由 HLMF 发布契约负责。
4. 设置新的 multitask `HLML_SNAPSHOT_ID`、`HLML_EXPERIMENT_ID`，执行 `make multitask`。
5. multitask 结束后执行 Val、EOS 2.0 infer、export；infer 使用 `[1,1,224,384]` Palm 与现有 ROI 几何，确认 ONNX/A1 报告和 conversion `datasets.zip`。
6. 运行 Train-only mining，在 HLMF 删除教师错误并发布一个或多个 selection；可同时加入多个 `new_datasets` 和 negative dataset。
7. 以困难 55%、replay 45% 完成 multi-finetune，再执行 Val、infer、export。
8. 根据固定 Val 冻结唯一 winner，最后执行一次 locked Test。

## 验收条件

- geometry 基线固定为第 68 epoch winner；Val landmarks mean pixel error 为 10.1930 px、PCK@0.10 为 0.8377。
- unknown handedness positive 保留 presence/landmarks 指标，只从 handedness 指标排除。
- 每阶段均保存 winner 并完成 Val/infer；multitask、multi-finetune 的 export 同时交付模型与配套数据包。
- multitask 负样本仅来自 HLMF published negative dataset；multi-finetune selection 仅来自 Train mining 与人工删除式复核。两类输入都读取 HLMF 独立 `published_relpath` 图片，selection 同时用 `source_crop_relpath` 核对来源身份。
- 多成员合并必须保持 ROI 唯一、split 隔离和同一 capture source 单 variant。
- 以新 snapshot ID 运行 data audit，确认部分 variant source 选择、manifest、Registry、0809/0813 HCF 溯源字段和 `256×256` published ROI；`FullEnhanceVal0801` 两条 s01 Test 与 s05 frozen Test 不参与调参。
- Test 不回流到采样、阈值、checkpoint 或困难挖掘。
