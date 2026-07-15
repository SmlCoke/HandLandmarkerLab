# 数据制作系统边界

本仓库只维护 Hand Landmarker 的提纯、训练、评估、推理与导出，不复制数据制作系统的操作文档或实现。

Train/Val/Test 的原图处理、Palm/MediaPipe 自动标注、ROI 生成、人工 Gold 复核及 canonical JSONL 生成，均由独立仓库 `HandLandmarkerFab` 负责。当前本机权威副本位于：

```text
D:\CICIEC\datasets\HandLandmarkerFab
```

需要重新生成 JSONL 或修改 ROI 生成策略时，请直接阅读该仓库的 `README.md` 与 `docs/`。本仓库从 `HAND_TRAIN_ROOT` 下的 `train_pretrain_merged`、`val_merged`、`test_merged` 接收其落盘产物；接口、门禁和训练步骤见 [Pretrain 数据与分阶段训练操作手册](../training_system/data_and_training.md)。

不再在此复制 HandLandmarkerFab 文档，避免两个仓库对同一流程给出不同版本的说明。
