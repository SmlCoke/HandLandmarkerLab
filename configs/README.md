# 当前配置文件

`configs/` 只保留 v2 端到端流程所需的 13 个配置：9 个 pretrain/通用配置和 4 个 finetune 配置；不为 eval/infer/export 复制 finetune wrapper，也不提供旧字段兼容层。

| 配置 | 唯一用途 |
|---|---|
| `curate_pretrain.yaml` | 提纯 geometry positive、生成负例可视化审查区，并在人工删除式复核后落盘 multitask 数据 |
| `train_smoke.yaml` | 固定 128 个 ROI 的 geometry 过拟合门禁 |
| `train_geometry.yaml` | 完整 pretrain geometry 训练 |
| `train_multitask.yaml` | 从 geometry best 初始化的 pretrain multitask 训练 |
| `prepare_finetune_sources.yaml` | 自动生成 b=`negative_removed_gold`、c=`disagreement_gold` 的 Gold 请求，并生成 d=`pretrain_replay` |
| `curate_finetune.yaml` | 认证 HLMF Gold aggregate 与单源 descriptor、合并 replay、去重并发布 finetune 快照 |
| `train_finetune_smoke.yaml` | 固定 256 ROI、三 head 全覆盖的 finetune 过拟合门禁 |
| `train_finetune.yaml` | 从 multitask best 初始化的正式 Gold+pseudo finetune |
| `eval_val.yaml` | 对 Make 目标显式注入的 stage/phase 执行 Val canonical ROI 评估 |
| `eval_test.yaml` | 对 Make 目标显式注入的 stage/phase 执行锁定 Test 评估 |
| `infer.yaml` | 对外部原图运行 Palm → ROI → Hand 推理 |
| `export.yaml` | 融合 v2、导出当前 phase ONNX 与转换数据 |
| `export_preflight.yaml` | 正式训练前导出非零探针权重 ONNX 与转换数据，验证官方工具链 |

路径根和实验 ID 由 Makefile 顶部的 `HAND_TRAIN_ROOT`、`HAND_PRETRAIN_ID`、`HAND_FINETUNE_ID` 固定。评估、推理、导出由明确的 Make 目标注入 `HAND_EXPERIMENT_ID`、`HAND_RUN_PHASE`、`HAND_MODEL_STAGE` 和 `HAND_TRAIN_CONFIG`；不要在 shell 中手工拼接 checkpoint 或修改 YAML 混用阶段。

| Make 目标后缀 | `HAND_EXPERIMENT_ID` | `HAND_RUN_PHASE` | `HAND_MODEL_STAGE` | 导出校准所用 `HAND_TRAIN_CONFIG` |
|---|---|---|---|---|
| `geometry` | `HAND_PRETRAIN_ID` | `geometry` | `pretrain` | `configs/train_geometry.yaml` |
| `multitask` | `HAND_PRETRAIN_ID` | `multitask` | `pretrain` | `configs/train_multitask.yaml` |
| `finetune` | `HAND_FINETUNE_ID` | `finetune` | `finetune` | `configs/train_finetune.yaml` |

例如，正式使用 `make eval-val-finetune`、`make infer-finetune`、`make export-finetune` 和 `make conversion-data-finetune`。这些目标分别复用 `eval_val.yaml`、`infer.yaml` 和 `export.yaml`，并把同一个 finetune best checkpoint 路由到对应输出目录；没有独立的 finetune eval/infer/export wrapper。

`curate_pretrain.yaml` 的 `source.crop_root` 是唯一训练 ROI 根目录。Curate 产生的 JSONL 保留该目录下的直接 `crop_path`，提纯目录本身只写入索引和审计文件。

当前配置模式只接受 `data`、`hand.model_path`、`output.dir`、`export.model_path` 等现行字段。旧的 `dataset`、`model.checkpoint`、`paths`、`pipeline`、`output.directory` 与 `export.output` 不再兼容。
