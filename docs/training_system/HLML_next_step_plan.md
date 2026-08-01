# HLML 下一阶段计划

## 目标

使用 HLMF 3.0 新 schema 发布的 Train、Val、Test、真负样本和困难 selection，完成首个 HLML 4.0 可复现实验。旧数据不迁移、不删除，也不混入新 snapshot。

## 执行顺序

1. 在 HLMF 发布至少一个 PretrainSource dataset、一个同时含固定 Val/Test ROI 的 EValSource dataset 和一个真负样本 dataset。
2. 在 `configs/datasets.yaml` 冻结 dataset ID、proposal variant、negative dataset ID 与权重。
3. 执行 geometry 与 multitask smoke/full 训练，在固定 Val 比较结果。
4. 运行 Train-only mining，将 request 交给 HLMF 删除明显教师错误并发布 selection。
5. 以默认困难 55%、replay 45% 完成 multi-finetune；若调整比例，replay 必须保持大于零。
6. 在固定 Val 冻结唯一 winner descriptor，再执行一次 locked Test。
7. 对 winner 完成 ONNX/A1 算子审计和数值一致性验证。

## 验收条件

- 数据审计没有 capture/raw split 泄漏或同来源多 proposal variant。
- geometry snapshot 无负样本，multitask 负样本全部来自已发布 negative dataset ID。
- mining 未读取 Val/Test，Test 未参与采样、阈值或 checkpoint 选择。
- Val/Test 输入均是 HLMF 已生成并经复核的固定 Hand ROI。
- 报告不包含 Palm 漏检、双手召回率或原图级联准确率。
- v2 模型、ROI、checkpoint、ONNX 与 A1 回归测试全部通过。
