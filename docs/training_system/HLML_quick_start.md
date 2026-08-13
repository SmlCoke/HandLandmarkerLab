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

`HAND_DATASET_ROOT` 保存 HLMF 发布的数据和 registry；其中负样本与困难样本使用 HLMF 独立 published 图片副本。`HAND_TRAIN_ROOT` 只保存 HLML 索引、训练与发布产物，不复制图片。

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

输入：一个或多个 HLMF 已发布的 dataset、negative dataset、selection ID。操作：在 `configs/datasets.yaml` 的对应列表中逐项填写 ID、variant、可选 `capture_source_ids` 白名单和 `weight`。没有白名单时，同一 manifest 中没有发布所选 variant 的历史 source 会被跳过；有白名单时，每个 source 必须存在、属于目标 split 并发布所选 variant。near/mid/far 能力由 HLMF 在发布前限制。当前 Iris-1.1 已冻结三个 HCF0813 Train dataset、五条 Eos-2.0 Val source 与两条 Eos-2.0 Test source；Eos-1.0 只可单独做 legacy/stress 回放。输出：只确定成员关系；negative/selection 从 HLMF 的 `published_relpath` 读取，HLML 不复制图片。

## 2. 配置解析（Config Check）

输入：五份单一职责公共 YAML 和环境变量。输出：终端 `status=ok` 及已解析的训练、评估、推理和导出配置。

```bash
make config-check
```

## 3. Geometry 阶段

输入：`FullEnhance0801`、`FullEnhance0803`、`FullEnhance0810` 的 `eos_2.0-rtmpose-hcf0813-gate` positive 和 geometry profile，固定 Val/Test 由 `configs/datasets.yaml` 的 source 白名单选择。处理：先生成零拷贝 snapshot；TFLite rescue 的 `norm×256` 上游辅助 crop-pixel 表示会在严格核对后规范化为 HLML canonical `norm×255`，归一化训练目标不变；再按 100% `POS_RUNTIME` 训练 v2 几何。输出：`snapshots/<id>/geometry/` 和 `runs/<experiment>/geometry/checkpoints/best.weights.h5`。本阶段禁止负样本。

```bash
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_TRAIN_ROOT=/root/autodl-tmp/TrainFab/HLML-4.0
export HLML_SNAPSHOT_ID=iris-1.1-geometry-eos2-hcf0813-r1
export HLML_EXPERIMENT_ID=iris-1.1-geometry-eos2-hcf0813-r1
export HLML_STAGE=geometry
make geometry
make val HLML_STAGE=geometry
export HLML_INFER_INPUT=/path/to/representative/images
make infer HLML_STAGE=geometry
```

训练结束后必须执行固定 ROI Val 与代表性原图 infer。Val 中 unknown handedness positive 仍评估 presence/landmarks，只跳过 handedness 指标。

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

在 HLMF 完成 `hard-review` / `hard-publish` 后，把发布的 `selection_id` 写回 `configs/datasets.yaml`；HLML 会核对 `source_crop_relpath`，并读取独立的 `published_relpath` 图片。

## 6. Multi-finetune 阶段

输入：multitask winner、带独立 published 图片的困难 selection、可选新录制 Train 数据、真负样本和 mandatory pretrain replay。默认 hard/new 55%、replay 45%，replay 不可为 0。输出：`snapshots/<id>/multi_finetune/` 和 `runs/<experiment>/multi_finetune/checkpoints/best.weights.h5`。

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

输入：任意原图文件夹、HLMF EOS 2.0 ONNX 和 Hand checkpoint。先将 HLMF 模型部署为 HLML 约定文件名；处理使用 `[1,1,224,384]`、840 个矩形 Anchor 和全局 NMS，再沿用原 Hand ROI 几何运行 Hand。这不是 Val/Test 协议。输出：`inference/<experiment>/multi_finetune/` 的 JSONL、summary 和可视化。

```bash
mkdir -p palm_detector/eos-2.0
cp /path/to/HandLandmarksFab/models/palm_detector/eos-2.0/model_384x224_opt.onnx \
  palm_detector/eos-2.0/model_opt.onnx
export HLML_INFER_INPUT=/path/to/images
make infer HLML_STAGE=multi_finetune
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
