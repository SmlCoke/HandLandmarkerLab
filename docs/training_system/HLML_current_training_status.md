# HLML 当前状态（2026-08-01）

代码已切换到 HLML 4.0 破坏性公共接口：公共配置按单一职责保留 `datasets.yaml`、`training.yaml`、`evaluation.yaml`、`inference.yaml` 与只负责 ONNX/A1 导出的 `deploy.yaml`；统一入口为 `scripts/hlml.py`，Makefile 只暴露数据审计、三阶段训练、Train-only 挖掘、固定 ROI Val、winner 冻结、locked Test、推理、导出、环境检查和测试。

已实现 HLMF manifest 零拷贝读取、SQLite/路径/解码门控、split 与 proposal variant 泄漏检查、geometry 负样本禁用、按 negative dataset 权重采样、默认 55/45 且 replay 非零的 multi-finetune、困难来源聚合、固定 ROI 分组指标以及不可覆盖的 winner/Test 锁定。

Hand Landmarker v2 网络、ROI 几何、损失、checkpoint、ONNX/A1 算子与数值契约保持不变。当前未加入辅助 head 或模型结构实验。

截至本状态文档更新时，本地完整单元测试为 176 项通过、7 项因可选 TensorFlow/ONNX 环境跳过。尚未在本文档中声明新的服务器训练 winner；实际训练结果应在完成新的 4.0 数据 snapshot 后更新。
