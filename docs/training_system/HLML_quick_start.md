# HLML 4.0 Quick Start

本页给出三阶段训练、固定 ROI 评估和导出的最短完整路径。详细输入契约与 YAML 参数见 `HLML_training_workflow.md`。

## 环境依赖（首次部署）

输入：仓库根目录的 `environment.yml` 和 `requirements.txt`。处理：创建原有 Python 3.8、TensorFlow 2.9 环境。输出：Conda 环境 `hand-landmarker-tf29` 及环境检查结果。

```bash
cd /path/to/HandLandmarkerLab
conda env create -f environment.yml
conda activate hand-landmarker-tf29
python -m pip check
make environment-check
```

已有环境时只执行 `conda activate hand-landmarker-tf29`。

## 0. 设置运行身份

`HAND_DATASET_ROOT` 保存 HLMF 发布的数据和 registry；`HAND_TRAIN_ROOT` 只保存 HLML 索引、训练与发布产物。

```bash
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_TRAIN_ROOT=/root/autodl-tmp/TrainFab/HLML-4.0
export HLML_SNAPSHOT_ID=v4-r1
export HLML_EXPERIMENT_ID=v4-r1
export HLML_RELEASE_ID=v4-r1
cd /path/to/HandLandmarkerLab
```

五个公共 YAML 各自只有一种职责：`datasets.yaml` 管成员选择，`training.yaml` 管三阶段训练，`evaluation.yaml` 管固定 ROI Val/Test，`inference.yaml` 管原图文件夹级联推理，`deploy.yaml` 管 ONNX/A1 模型与配套转换数据导出。

## 1. 数据成员选择（Dataset Selection）

输入：HLMF 已发布的 dataset、negative dataset、selection ID。操作：编辑 `configs/datasets.yaml` 或设置其中对应的 `HLML_*` 环境变量。输出：只确定成员关系，不复制图片。

```bash
export HLML_PRETRAIN_DATASET_ID=FullEnhance0801
export HLML_NEGATIVE_DATASET_ID=background-neg-0801
export HLML_SELECTION_ID=hard-positive-0801
export HLML_EVAL_DATASET_ID=national-eval-0801
export HLML_PROPOSAL_VARIANT=palm-v1
```

## 2. 配置解析（Config Check）

输入：五份单一职责公共 YAML 和环境变量。输出：终端 `status=ok` 及已解析的训练、评估、推理和导出配置。

```bash
make config-check
```

## 3. Geometry 阶段

输入：PretrainSource 的可靠 positive 和 geometry profile；当前 `FullEnhance0801` 的 72,226 条 Train 记录均为 `POS_RUNTIME`，配置按 100% `POS_RUNTIME` 抽样。处理：先生成零拷贝 snapshot，再训练 v2 几何。输出：`snapshots/<id>/geometry/` 和 `runs/<experiment>/geometry/checkpoints/best.weights.h5`。本阶段禁止负样本。

```bash
make geometry
make val HLML_STAGE=geometry
export HLML_INFER_INPUT=/path/to/representative/images
make infer HLML_STAGE=geometry
```

训练结束后必须执行固定 ROI Val 与代表性原图 infer。

## 4. Multitask 阶段

输入：geometry winner、Train positive 和按 ID/权重选择的已发布真负样本。处理：训练 landmarks、presence、handedness 多任务。输出：`snapshots/<id>/multitask/` 和 `runs/<experiment>/multitask/checkpoints/best.weights.h5`。

```bash
make multitask
make val HLML_STAGE=multitask
export HLML_INFER_INPUT=/path/to/representative/images
make infer HLML_STAGE=multitask
make export HLML_STAGE=multitask
```

Export 同时生成 ONNX/A1 报告和配套 conversion `datasets.zip`。

## 5. Train-only 困难来源挖掘

输入：multitask Train snapshot 和 multitask winner。输出：`mining/<snapshot_id>/source_ranking.json`、`student_predictions.jsonl` 和交给 HLMF 的 `hlmf_review_request.jsonl`。命令拒绝 Val/Test。

```bash
make mine-hard
```

在 HLMF 完成 `hard-review` / `hard-publish` 后，把发布的 `selection_id` 写回 `configs/datasets.yaml`。

## 6. Multi-finetune 阶段

输入：multitask winner、困难 selection、可选新录制 Train 数据、真负样本和 mandatory pretrain replay。默认 hard/new 55%、replay 45%，replay 不可为 0。输出：`snapshots/<id>/multi_finetune/` 和 `runs/<experiment>/multi_finetune/checkpoints/best.weights.h5`。

```bash
make multi-finetune
make val HLML_STAGE=multi_finetune
export HLML_INFER_INPUT=/path/to/representative/images
make infer HLML_STAGE=multi_finetune
make export HLML_STAGE=multi_finetune
```

三条阶段后命令使用同一 snapshot/experiment/stage。

## 7. 固定 Hand ROI Val

输入：HLMF 已复核的 Val ROI snapshot 和候选 checkpoint。处理：只运行 Hand Landmarker，可在 Val 选择 presence threshold。输出：`runs/<experiment>/eval/multi_finetune/val/{predictions.jsonl,metrics.json}`。

```bash
make val HLML_STAGE=multi_finetune
```

## 8. 冻结唯一 Winner

输入：固定 ROI Val 的 `metrics.json` 和 `best.weights.h5`。输出：不可变的 `releases/<release_id>/winner.json`。

```bash
make freeze-winner HLML_STAGE=multi_finetune HLML_RELEASE_ID="$HLML_RELEASE_ID"
```

## 9. Locked Fixed-ROI Test

输入：winner descriptor 和 HLMF 已复核 Test ROI。处理：只测锁定 checkpoint/threshold，不运行 Palm，不允许覆盖或调参。输出：`releases/<release_id>/test/{predictions.jsonl,metrics.json}`。

```bash
make locked-test HLML_STAGE=multi_finetune HLML_RELEASE_ID="$HLML_RELEASE_ID"
```

## 10. ONNX/A1 导出

输入：multitask 或 multi-finetune v2 checkpoint、当前 stage snapshot 和 export profile。处理：导出 opset 11 ONNX、审计 A1 算子/数值，并生成 Train 100、Val 25、Test 25 个 NCHW `.npy`。输出：`runs/<experiment>/export/<stage>/` 下的 ONNX、contract/report 和 `model_conversion/datasets.zip`。

```bash
make export HLML_STAGE=multi_finetune
```

## 11. 可选文件夹推理（Folder Inference）

输入：任意原图文件夹。处理：Palm → Hand ROI → Hand；这不是 Val/Test 协议。输出：`inference/<experiment>/multi_finetune/` 的 JSONL、summary 和可视化。

```bash
export HLML_INFER_INPUT=/path/to/images
make infer HLML_STAGE=multi_finetune
# 单次覆盖全局 Eos 选择：
make infer HLML_STAGE=multi_finetune INFER_ARGS='--palm-model-id eos-2.0'
```

## 12. 环境、语法、测试与验收

输入：训练环境、代码、配置和合成 warehouse。输出：终端检查结果。

```bash
make environment-check
make compile
make test
make acceptance-smoke
make help
```

Val/Test 只衡量已经提供的固定 Hand ROI；不重新运行 Palm，也不报告 Palm 漏检或原图级联性能。
