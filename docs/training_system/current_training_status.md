# HLML 当前训练与数据状态

更新日期：2026-07-17

本文是随实验推进更新的“当前状态快照”，记录当前服务器、数据集、实验和已知问题。它不规定通用操作流程；通用命令和目录约定见[端到端训练快速操作手册](end_to_end_training_quick_runbook_v1_0.md)，完整原理与数据契约见[完整训练流程](end_to_end_training_workflow_v1_0.md)。

## 1. 当前结论

1. `v2-pretrain-r3` 的 geometry 和 multitask 均已完成，模型、训练器、评估、推理和 ONNX 导出链路可以正常工作。
2. `v2-finetune-r1` 已完成 Gold 制作、数据聚合、smoke、正式训练、Val、独立 infer 和导出。
3. `v2-finetune-r2` 是复用 r1 冻结 finetune 数据、只把每个 batch 的 Gold 比例提高到 50% 的对照实验。它已经完整完成，但关键点总体精度与 r1 基本相同，不能证明“replay 太多”是当前误差和塌缩的主要原因。
4. 当前最好的固定 Val 平均像素误差仍是 geometry 的 `22.2212 px`；multitask、finetune r1 和 Gold 50% r2 都没有超过它。
5. 固定端到端 infer 中仍存在较多关键点聚成一团的预测。r1 与 r2 的检出数量完全相同，Gold 50% 没有消除塌缩。
6. 当前 Val/Test 都只有 positive Hand ROI，适合比较 landmarks 和 positive recall，但不能据此证明 `hand_flag` 对无手背景的误报率。

下一步工作以[finetune r2 后续优化计划](../training_history/2026-07-17_post_finetune_r2_optimization_plan.md)为准。

## 2. 服务器与仓库

当前活动服务器是从上一台同配置服务器完整克隆得到的实例。系统盘、数据盘、HLMF、HLML、Conda 环境和训练产物均随克隆迁移。

| 项目 | 当前值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090，24 GB |
| 系统内存 | 约 90 GB |
| HLML 仓库 | `/root/HandLandmarkerLab` |
| HLMF 仓库 | `/root/HandLandmarksFab` |
| HLML Conda | `hand-landmarker-tf29` |
| HLMF Conda | `anfab` |
| 中央数据根 | `/root/autodl-tmp/TrainFab/HLML-2.0` |
| 已核验 HLML commit | `6868cda5d988717fd416177d196bf2cb692966eb` |
| 核验时 Git 状态 | clean |

服务器地址和密码不写入版本库。更换或再次克隆服务器后，先运行 `make paths`、`make compile`、`make test-unit` 和数据 gate，不能仅凭“目录看起来一样”判断快照有效；训练 loader 仍会逐项复核 manifest 和 SHA-256。

## 3. 当前实验身份

| 角色 | ID | 状态 |
|---|---|---|
| Pretrain 基线 | `v2-pretrain-r3` | geometry、multitask、Val、infer、export 已完成 |
| Finetune 主实验 | `v2-finetune-r1` | 数据、smoke、full、Val、infer、export 已完成 |
| Finetune 对照 | `v2-finetune-r2` | Gold 50% 对照已完成 |

`v2-finetune-r2` 没有重新制作一份 Gold 或 replay 数据。它使用：

```text
train_finetune_merged/v2-finetune-r1/
```

作为只读输入，通过服务器数据盘上的实验配置把输出路由到：

```text
hand_landmarker_runs/v2-finetune-r2/
hand_landmarker_inference/v2-finetune-r2/
```

r2 实验配置和日志位于：

```text
experiments/v2-finetune-r2-gold50/
```

该目录在中央数据盘，不在 HLML Git 仓库中。r2 的导出 provenance 已显式绑定 r1 的 frozen curation manifest。

## 4. Pretrain 数据状态

### 4.1 原始聚合与 curate

`v2-pretrain-r3` 的 curate 报告记录：

| 项目 | 数量 |
|---|---:|
| 原始 canonical 记录 | 108,926 |
| 可用于 geometry 的 landmark positive | 59,952 |
| 原始负例候选 | 48,643 |
| 人工保留为“明确无手”的候选 | 1,049 |
| 进入 multitask 的 confirmed negative | 1,022 |
| overlap quarantine | 27 |
| 进入 `negative_removed` | 47,594 |
| multitask canonical 记录 | 60,974 |
| multitask resolved epoch size | 6,400 |

关键目录：

```text
train_pretrain_merged/
train_pretrain_curated/v2-pretrain-r3/
hand_landmarker_reviews/v2-pretrain-r3/
hand_landmarker_runs/v2-pretrain-r3/
```

关键报告：

```text
train_pretrain_curated/v2-pretrain-r3/qc/curation_report.json
hand_landmarker_reviews/v2-pretrain-r3/review_report.json
hand_landmarker_runs/v2-pretrain-r3/multitask_data_gate.json
```

人工负例事务已经提交，`review_report.json` 为 `status=committed`；multitask 数据 gate 为 `status=pass`。

### 4.2 Geometry 与 multitask

| 阶段 | 完成轮数 | Best epoch | 选择指标 | Best checkpoint SHA-256 |
|---|---:|---:|---:|---|
| geometry | 31 | 11 | `val_landmark_mae=0.0556243` | `5d886d3b04967486dfde03f2a377df6411e4993404618e5c1a09ab8d73fab187` |
| multitask | 46 | 31 | `val_multitask_score=0.0595793` | `6082d3e43ba11ccf8a8831247ce2b8f709dae97bca72b87dbe0becae166fee91` |

multitask ONNX 已通过厂商官方工具链转换并成功上板。用户观察到：完全张开且手心正对屏幕的手势较准确；握拳、侧向张掌和数字“1”等手势仍较差。

## 5. Finetune 数据状态

### 5.1 已有 Gold source

当前 `v2-finetune-r1` 使用三个 Gold source；新录制来源 e 尚未制作。

| Gold source | Reviewed/匹配 | 可训练 | Ignore | Positive | Negative |
|---|---:|---:|---:|---:|---:|
| `dragon_gold_0716_v1` | 5,191 matched ROI | 5,189 | 2 | 5,189 | 0 |
| `disagreement_gold` | 300 | 237 | 63 | 227 | 10 |
| `negative_removed_gold` | 300 | 260 | 40 | 243 | 17 |
| 新录制 e | 0 | 0 | 0 | 0 | 0 |

Dragon 原始输入包含 8,593 张图片和 4,500 行人工 Hand 标注；有 3,565 张图片参与唯一 Hand–Palm 匹配。Dragon 不提供 handedness，因此其 handedness mask/loss weight 为 0。

HLMF Gold aggregate：

| 项目 | 数量 |
|---|---:|
| Catalog | 5,791 |
| Included Gold | 5,686 |
| Excluded/ignored | 105 |
| Duplicate | 0 |
| Conflict | 0 |
| Gold source | 3 |

关键目录：

```text
finetune/v2-finetune-r1/sources/gold/
finetune/v2-finetune-r1/cvat/
finetune/v2-finetune-r1/hmlf_gold_merged/
```

### 5.2 HLML 最终 finetune 快照

`train_finetune_merged/v2-finetune-r1` 的结果：

| 项目 | 数量 |
|---|---:|
| Gold | 5,686 |
| Replay after Gold override | 9,971 |
| 被 Gold 替代的 replay | 29 |
| 最终训练记录 | 15,657 |
| Finetune smoke 固定记录 | 256 |

r1 正式配置的 `gold_fraction=0.35` 在 batch size 64 下实际为每 batch 22 Gold + 42 replay，即 Gold `34.375%`。r2 对照使用每 batch 32 Gold + 32 replay，即 Gold `50%`。

Gold tier 内部当前有效角色权重约为：

```text
Dragon                 63.16%
negative_removed_gold  21.05%
disagreement_gold      15.79%
```

新录制 e 缺失时，其权重不生成空数据，而是在已有 Gold 角色内重新归一化。

## 6. 固定 Val/Test 与推理输入

### 6.1 Val

Val 有 1,226 个全部为 positive 的 Gold Hand ROI：

| 数据来源 | ROI |
|---|---:|
| `peak_vali_independent_v1` | 245 |
| `peak_vals_shared_v1` | 561 |
| `soar_vals_shared_v1` | 420 |

Handedness：Left 512，Right 714。

### 6.2 Test

锁定 Test 有 985 个全部为 positive 的 Gold Hand ROI：

| 数据来源 | ROI |
|---|---:|
| `peak_test_shared_v1` | 638 |
| `soar_test_shared_v1` | 347 |

Handedness：Left 441，Right 544。当前优化仍只使用 Val；Test 应在最终方案、threshold、checkpoint 和导出配置冻结后运行一次。

### 6.3 固定端到端 infer

固定 infer 输入目录：

```text
hand_landmarker_inference/input/
```

当前包含 192 张与 Train/Val/Test 独立的原图。infer 会运行冻结 Palm Detector，再生成 rotated Hand ROI，最后运行目标 Hand Landmarker。

## 7. 当前训练结果

### 7.1 固定 Val 对比

| 模型 | Mean pixel | Median pixel | P90 pixel | PCK@0.10 | Presence recall@0.5 | Handedness accuracy |
|---|---:|---:|---:|---:|---:|---:|
| r3 geometry | 22.2212 | 19.4438 | 39.9202 | 30.40% | 100.00% | 54.57% |
| r3 multitask | 23.3062 | 20.0845 | 40.8567 | 28.70% | 100.00% | 74.96% |
| finetune r1 | 23.1402 | 20.6467 | 39.6535 | 29.42% | 94.29% | 74.23% |
| finetune r2 Gold 50% | 23.1415 | 21.2058 | 39.5696 | 30.37% | 94.86% | 74.23% |

r2 相对 r1 的 1,226 ROI 配对误差平均变化是 `+0.00136 px`，95% 区间为 `[-0.221, +0.224] px`，没有显著总体收益。

分来源变化：

```text
peak_vali_independent_v1  26.3846 → 25.7991 px
peak_vals_shared_v1       27.6194 → 27.6293 px
soar_vals_shared_v1       15.2646 → 15.5969 px
```

### 7.2 固定 infer 对比

| 指标 | r1 | r2 Gold 50% |
|---|---:|---:|
| 输入图片 | 192 | 192 |
| 有检测的图片 | 140 | 140 |
| 总检测 | 217 | 217 |
| Landmark spread 中位数 | 0.1060 | 0.1050 |
| Spread `<0.10` | 97 | 99 |
| Spread `<0.08` | 53 | 41 |

`spread` 是 21 点相对其中心的归一化均方根距离，只作为塌缩的自动筛查指标，不能单独代替人工看图。r1/r2 逐图检测数量完全相同，说明提高 Gold 比例没有改变这组输入的 Palm/Hand 接受结果。

## 8. r1/r2 训练与导出产物

### 8.1 Finetune r1

```text
hand_landmarker_runs/v2-finetune-r1/finetune/
hand_landmarker_runs/v2-finetune-r1/eval/finetune/val/
hand_landmarker_inference/v2-finetune-r1/finetune/
hand_landmarker_runs/v2-finetune-r1/export/finetune/
```

r1 完成 40 轮，best epoch=14，best checkpoint SHA-256：

```text
c6c5309f7c69ceceb1752455a561bccae2e30cb4de63a108a663332e1fd95343
```

### 8.2 Gold 50% r2

```text
hand_landmarker_runs/v2-finetune-r2/finetune_smoke/
hand_landmarker_runs/v2-finetune-r2/finetune/
hand_landmarker_runs/v2-finetune-r2/eval/finetune/val/
hand_landmarker_inference/v2-finetune-r2/finetune/
hand_landmarker_runs/v2-finetune-r2/export/finetune/
```

r2 完成 40 轮，best epoch=4，best checkpoint SHA-256：

```text
f470ba217c5df962b0a7a77156da5e6e88f3b7717cf4d1645f29024a8c327939
```

r2 ONNX：

```text
hand_landmarker_runs/v2-finetune-r2/export/finetune/hand_landmarker_v2.onnx
SHA-256: 329a0d0100326ca7bd80f3ca0a63561df7e1f1766fa8f0b5552860b2c23eb0f8
size: 7,729,272 bytes
```

导出结果：A1 严格算子审计无违规；Keras→ONNX 最大数值误差约 `2.98e-7`；转换数据包含 100 个 calibration ROI 和 50 个 evaluation ROI。

## 9. 当前已知问题

1. landmarks 的总体误差约 22～23 px，finetune 尚未超过 geometry。
2. 固定 infer 和上板观察中仍有大量塌缩骨架，尤其是握拳、侧向张掌、数字“1”等非正面张掌姿态。
3. 一部分端到端失败来自 Palm ROI 落在前臂、身体或不完整手部；另一部分在正确 canonical Val ROI 上仍有较大误差，不能全部归因于 Palm。
4. 当前 Gold 数量由 Dragon 主导，但 Dragon handedness 不可用，且其姿态/成像分布不一定覆盖 Peak/Soar 和固定 infer 的困难区域。
5. 已人工标注的 disagreement 与 negative-removed Gold 共只有 497 条可训练记录；来源 e 尚缺失。
6. Val/Test 缺少 negative，presence 只能看 positive recall，不能测 false-positive rate。
7. r2 已证明单独减少 replay 占比不能解决 landmarks 泛化与塌缩。

## 10. 状态文档更新规则

每完成一个正式实验后更新：

- 当前 commit 与 Git clean/dirty 状态；
- 实验 ID、数据快照和继承关系；
- Gold/replay/ignored/duplicate/conflict 数量；
- best epoch、checkpoint SHA、Val 指标；
- 固定 infer 的输入数、检测数和塌缩统计；
- ONNX SHA、contract、转换与上板状态；
- 新发现的问题和下一步计划链接。

不要把这些动态数字重新写回 quick runbook；quick runbook 只维护通用流程和命令。
