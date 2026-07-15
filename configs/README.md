# 当前配置文件

`configs/` 只保留 v2 pretrain 当前流程所需的 9 个配置，不提供旧字段或旧版本 wrapper：

| 配置 | 唯一用途 |
|---|---|
| `curate_pretrain.yaml` | 提纯 geometry positive、生成负例可视化审查区，并在人工删除式复核后落盘 multitask 数据 |
| `train_smoke.yaml` | 固定 128 个 ROI 的 geometry 过拟合门禁 |
| `train_geometry.yaml` | 完整 pretrain geometry 训练 |
| `train_multitask.yaml` | 从 geometry best 初始化的 pretrain multitask 训练 |
| `eval_val.yaml` | 对当前 phase 的 Val canonical Hand ROI 评估 |
| `eval_test.yaml` | 对当前 phase 的锁定 Test canonical Hand ROI 评估 |
| `infer.yaml` | 对外部原图运行 Palm → ROI → Hand 推理 |
| `export.yaml` | 融合 v2、导出当前 phase ONNX 与转换数据 |
| `export_preflight.yaml` | 正式训练前导出非零探针权重 ONNX 与转换数据，验证官方工具链 |

路径根和实验 ID 由 Makefile 顶部的 `HAND_TRAIN_ROOT`、`HAND_PRETRAIN_ID` 固定。`HAND_PRETRAIN_PHASE` 只由明确的 Make 目标设置为 `geometry` 或 `multitask`，不应手工修改 YAML 来混用阶段。

`curate_pretrain.yaml` 的 `source.crop_root` 是唯一训练 ROI 根目录。Curate 产生的 JSONL 保留该目录下的直接 `crop_path`，提纯目录本身只写入索引和审计文件。

当前配置模式只接受 `data`、`hand.model_path`、`output.dir`、`export.model_path` 等现行字段。旧的 `dataset`、`model.checkpoint`、`paths`、`pipeline`、`output.directory` 与 `export.output` 不再兼容。
