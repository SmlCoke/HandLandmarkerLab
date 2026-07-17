# HLML 下一步优化计划

适用时间：代码发布后的两天冲刺。

目标：人工总标注不超过 800 个 Hand ROI，在同一冻结数据快照上完成 `data_only` 与 `structure` 两个必做候选；只有结构损失有效且时间允许时，才做 `structure_roi_aug`。

## Day 1：程序选样 + 团队 Gold

程序自动完成：

1. HLMF 直接从 DatesetFab 聚合 pretrain/Val/Test，不复制 ROI。
2. HLML 建立 replay 和全量 disagreement score pool。
3. HLMF 从新录制来源确定性选择最多 200/300 个 ROI。
4. HLML 汇总历史 Gold、历史 selection、当前 CVAT、Val/Test 的五类身份，排除重复。
5. HLML 用 disagreement 自动补足冻结的 600/800 总预算。
6. HLMF 生成任务 XML、SHA 证据和每 100 张的 `cvat_job_plan.json`。

人工完成：

1. 最多 20～30 分钟录制握拳、数字 1、侧向张掌、遮挡/重叠、远近/轻微 ROI 偏移；不同人员独立 session。
2. 查看最多 40 张自动 error overlay，不制作手工表格。
3. 团队合计标注冻结的 600 或 800 张，绝不超过 800。
4. 开始前共同标 10 张校准图；每人结束后由负责人抽查约 5%。
5. 无法可靠标注的 ROI 直接 `ignore_for_training`。

当晚程序自动导入、聚合、去重、curate、gate 和 smoke；人工只确认报告为 pass。

## Day 1 晚：候选 A

```text
FINETUNE_PROFILE=data_only
gold_fraction=0.35
Gold role 权重：dragon 0.40 / negative-removed 0.15 /
                disagreement 0.30 / new-recorded 0.15
```

程序完成训练、Val、infer 和错误分析。人工只查看摘要、代表 overlay 和固定手势样例。

## Day 2：候选 B

`structure` 与候选 A 使用完全相同的 `HAND_FINETUNE_ID`，只改变：

- 20 条真实骨连接的向量 Huber；
- 预测/Gold 整体 spread 的 log-ratio Huber。

两项只对 `supervision_tier=gold`、presence=true、landmark mask 有效的样本生效；pseudo replay、negative 和 ignored 行的结构权重严格为 0。不使用固定骨长、固定方向或手势模板。

## 可选候选 C

仅当 B 的塌缩显著减少、Val 未退化且仍表现出 ROI 偏移敏感时运行 `structure_roi_aug`：旋转 ±10°、scale 0.90～1.10、translation 0.05。

## 决策门槛

程序比较：overall/per-dataset/per-landmark、配对误差、PCK、presence、handedness、infer 检出数、spread、top 改善/退化 overlay。

人工最终选择时优先级：

1. 塌缩样本明显减少；
2. 困难手势和固定 infer 样例改善；
3. Val 关键点误差/PCK 不明显退化；
4. presence 不因“拉开关键点”产生更多假阳性；
5. ONNX 转换和板端运行通过。

locked Test 只在候选冻结后运行一次，不用于反复调参。
