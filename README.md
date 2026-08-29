<div align="center">

<img src="./docs/assets/aethersign-logo-minimal.svg" alt="AetherSign minimal logo" width="116" />

<h1>HandLandmarkerLab（HLML 4.0）</h1>

**AetherSign Iris 训练、固定 ROI 评估与 A1 部署导出系统**

[![Archive](https://img.shields.io/badge/Status-Competition_Final-8B5CF6?style=flat-square)](#-i-项目定位与归档状态) [![Tag](https://img.shields.io/badge/Tag-HLML_4.0_final-0891B2?style=flat-square)](https://github.com/SmlCoke/HandLandmarkerLab/tree/HLML-4.0-final) [![Tests](https://img.shields.io/badge/Tests-198_Passed-059669?style=flat-square)](#-vii-验证与依赖) [![Input](https://img.shields.io/badge/Hand_ROI-256%C3%97256-2563EB?style=flat-square)](#411-输入输出契约)

[项目定位](#-i-项目定位与归档状态) · [复现环境](#-ii-全国总决赛阶段复现环境) · [配置与文档](#-iii-配置与文档入口) · [系统边界](#-iv-系统边界) · [常用操作](#-vi-常用操作)

</div>

---

## ✦ I. 项目定位与归档状态

### 1.1 项目定位

HandLandmarkerLab（HLML）是 AetherSign **Iris Hand Landmarker** 的下游训练仓库。它直接读取 HandLandmarkerFab（HLMF）发布的稳定 dataset manifest 和 Hand ROI，完成三阶段训练、固定 ROI Val/Test、Eos → Iris 文件夹级联推理，以及 ONNX/A1 模型和转换数据导出。

#### 1.1.1 仓库边界

本仓库不制作或人工修改训练数据，也不训练 Eos Palm Detector 或 Muse Gloss Translator。HLMF 负责原图、Palm proposal、`256×256` Hand ROI、教师标签、CVAT 复核、Registry 和数据集发布；HLML 只消费已发布身份，并把数据 snapshot、训练 run 和模型 release 分开管理。

#### 1.1.2 上下游关系

```text
HLMF 3.0 published datasets
        │
        ▼
HLML 4.0: geometry → multitask → multi-finetune
        │
        ├── fixed Hand ROI Val / locked Test
        ├── Eos → Iris folder inference
        └── ONNX + A1 conversion datasets
```

### 1.2 全国总决赛归档状态

> [!IMPORTANT]
> AetherSign 已于 2026-08-25 完成全国总决赛答辩并获得全国一等奖。本仓库针对本届比赛的使命已经完成；最终可复现代码状态由 annotated tag `HLML-4.0-final` 固定。

#### 1.2.1 Git 归档范围

tag 保存代码、五份公共配置、测试、文档和 AetherSign 小 Logo。正式 `HAND_DATASET_ROOT`、`HAND_TRAIN_ROOT`、模型 checkpoint、ONNX/m1model、转换数据包及其他大体积外部资产仍按既有策略独立保存，不包含在 Git tag 中。

#### 1.2.2 正式提交模型

AetherSign 正式提交并上板使用的 Iris 模型为：

| 产品版本 | HLML 结构/阶段 | Mean pixel error | P95 pixel error | Handedness Acc | 部署参数 | A1 延迟 |
| :-- | :-- | --: | --: | --: | --: | --: |
| **Iris-2.0-Lite** | `v3-lite` multitask | 10.43 px | 24.98 px | 89.55% | 0.85 M | ≈20 ms |
| **Iris-2.0-Max** | `v3-max` multi-finetune | **9.71 px** | **23.26 px** | **98.26%** | 1.91 M | ≈22 ms |

`v3-pro` 完成了同协议训练与评估，但不属于最终正式提交模型。完整项目背景、Eos/Iris benchmark 与板端级联结果见 [project-12.md](./project-12.md)。

---

## ◇ II. 全国总决赛阶段复现环境

### 2.1 软件环境

#### 2.1.1 基础环境

| 项目 | 配置 |
| :-- | :-- |
| 操作系统 | Ubuntu 20.04 |
| Python | 3.8 |
| TensorFlow / Keras | 2.9.0 / 2.9.0 |
| CUDA | 11.2 |
| Conda 环境 | `hand-landmarker-tf29` |

#### 2.1.2 依赖来源

`environment.yml` 和 `requirements.txt` 共同定义正式环境；`requirements-dev.txt` 只补充开发检查依赖。CUDA/cuDNN 使用服务器系统动态库，不在 Conda 中重复安装。

### 2.2 训练服务器

#### 2.2.1 计算资源

| 项目 | 配置 |
| :-- | :-- |
| GPU | NVIDIA RTX 3090，24 GB × 1 |
| CPU | 14 vCPU，Intel Xeon Gold 6330 @ 2.00 GHz |
| 内存 | 90 GB |

#### 2.2.2 存储职责

| 根目录 | 职责 |
| :-- | :-- |
| `HAND_DATASET_ROOT` | HLMF 持久数据仓、Registry 和已发布数据集 |
| `HAND_TRAIN_ROOT` | HLML 零拷贝索引 snapshot、run、评估和导出物 |

---

## 🧭 III. 配置与文档入口

### 3.1 五份公共配置

| 配置 | 单一职责 |
| :-- | :-- |
| [`configs/datasets.yaml`](./configs/datasets.yaml) | geometry、multitask、multi-finetune 的数据集 ID 与成员选择 |
| [`configs/training.yaml`](./configs/training.yaml) | 三阶段模型、loss、优化器、采样和 checkpoint |
| [`configs/evaluation.yaml`](./configs/evaluation.yaml) | 固定 Hand ROI Val、winner 冻结与 locked Test |
| [`configs/inference.yaml`](./configs/inference.yaml) | Eos → Iris 原图文件夹级联推理 |
| [`configs/deploy.yaml`](./configs/deploy.yaml) | 分支融合、ONNX/A1 约束与转换数据导出 |

### 3.2 核心入口文档

| 文档 | 作用 |
| :-- | :-- |
| [完整训练工作流](./docs/training_system/HLML_training_workflow.md) | 每个阶段的输入、命令、输出与参数原理 |
| [Quick Start](./docs/training_system/HLML_quick_start.md) | 最短端到端命令路径 |
| [当前状态](./docs/training_system/HLML_current_training_status.md) | 全国总决赛最终结果与归档边界 |
| [后续维护计划](./docs/training_system/HLML_next_step_plan.md) | 归档后的复现和维护规则 |
| [常见问题与解答](./docs/annotating_system/HLML_qa.md) | 已记录的问题与答案 |

### 3.3 专项契约

- [数据与三阶段训练契约](./docs/training_system/tools/data_and_training.md)
- [固定 ROI 评估契约](./docs/training_system/tools/evaluation.md)
- [ONNX/A1 部署契约](./docs/training_system/tools/deployment_contract.md)
- [服务器环境说明](./docs/training_system/tools/environment.md)

---

## ⬡ IV. 系统边界

### 4.1 数据与模型契约

#### 4.1.1 输入输出契约

Iris 固定输入为单通道 NCHW `float32 [1,1,256,256]`，输出顺序为 `landmarks[42]`、`hand_flag[1]`、`handedness[1]`。ROI 几何、21 点顺序、坐标语义和 A1 接口在 v2/v3 间保持不变。

#### 4.1.2 数据身份与零拷贝

训练成员只能由已发布 dataset、negative dataset、CVAT-reviewed hard dataset 和可选 recorded Gold dataset ID 选择。HLML 不手工拼接 canonical JSONL、不复制 ROI，也不把数据集绑定到训练 run ID。

### 4.2 三阶段训练边界

#### 4.2.1 Geometry

只使用可靠 positive，优先学习 21 点骨骼几何；negative 配额必须为 0。

#### 4.2.2 Multitask

从 geometry winner 初始化，加入完整审核发布的真负样本，同时训练 landmark、hand presence 和 handedness 三个输出。

#### 4.2.3 Multi-finetune

从 multitask winner 初始化，混合 CVAT-reviewed hard/Gold positive-negative 与 mandatory pretrain replay。replay 不得关闭；rare-cell 重复抽样门禁不得放宽。

### 4.3 评估边界

#### 4.3.1 固定 ROI Val/Test

Val/Test 只读取 HLMF 已生成并经 CVAT 复核的固定 Hand ROI，不运行 Palm Detector，也不从原图重建 ROI。Val 用于模型选择和阈值选择；locked Test 只能对冻结 winner 运行一次。

#### 4.3.2 不报告的指标

固定 ROI 评估不报告 Palm 漏检、双手级联召回率或原图级联性能。此类现象只能由独立文件夹推理或端侧应用 benchmark 描述，不能与 Iris ROI 精度混合。

### 4.4 推理与部署边界

#### 4.4.1 文件夹级联

`make infer` 是独立的 Eos → Hand ROI → Iris 定性/级联检查入口；困难样本挖掘只读取 Train，并由 snapshot ledger 保证同一 snapshot 内多轮不重复。

#### 4.4.2 分支融合与 A1

`v3-max` 的训练期多分支在导出前精确融合为单 Conv/Depthwise 图。正式 ONNX 继续执行数值一致性、opset 11、15 MiB 大小和 A1 算子白名单检查。

---

## ◈ V. Iris v3 模型家族

### 5.1 结构对照

| 结构 | 训练参数 | 融合后部署参数 | 训练/部署参数比 | 定位 |
| :-- | --: | --: | --: | :-- |
| `v3-pro` | 1,951,756 | 1,912,324 | 1.02 | 与未修改 v2 同构 |
| `v3-max` | 7,629,268 | 1,912,324 | 3.99 | 扩大训练期多分支容量，部署仍与 v2 同量级 |
| `v3-lite` | 878,272 | 852,832 | 1.03 | 缩减通道的轻量档 |
| `v2` | 1,951,756 | 1,912,324 | 1.02 | 历史回归入口 |

### 5.2 部署图约束

普通卷积和 Depthwise block 的训练分支会在 export 时融合；正式部署图不得残留 BatchNormalization 或训练分支。三档 export-preflight 均低于 15 MiB，并通过 Add/Conv/MaxPool/Relu/Reshape/Sigmoid 白名单检查。

---

## 🚀 VI. 常用操作

### 6.1 复现归档版本

```bash
git fetch origin --tags
git checkout HLML-4.0-final
conda env create -f environment.yml
conda activate hand-landmarker-tf29
make environment-check
make config-check
```

### 6.2 三阶段训练

```bash
make data-audit HLML_STAGE=geometry
make geometry
make multitask
make mine-hard MINING_ARGS='--round-id r01 --max-rois 1000'
make multi-finetune
```

### 6.3 评估、推理与导出

```bash
make val HLML_STAGE=multi_finetune
make freeze-winner HLML_STAGE=multi_finetune
make locked-test
make infer HLML_STAGE=multi_finetune
make export HLML_STAGE=multi_finetune
```

### 6.4 仓库检查

```bash
make compile
make test
make config-check
```

---

## 🧩 VII. 验证与依赖

### 7.1 最终自动验证

最终归档前完成 Python 语法检查、198 项 HLML 单元测试和五份公共配置解析。HLMF 发布资产、正式训练数据和模型文件不属于 Git 内单元测试夹具，复现时必须单独恢复。

### 7.2 依赖维护

本次归档未增加依赖，`requirements.txt` 与 `environment.yml` 保持全国总决赛实际环境。未来只有在明确恢复开发时才升级依赖；复现比赛结果应优先使用 tag 固定版本。

---

## 🗂 VIII. 仓库结构

```text
HandLandmarkerLab/
├── configs/                 # 五份单一职责公共配置
├── hand_landmarker/         # 数据合同、模型、训练、评估与导出实现
├── scripts/                 # hlml.py 统一 CLI 及辅助工具
├── tests/                   # 合同、路由、训练与部署回归测试
├── docs/
│   ├── assets/              # README 独立维护视觉资产
│   ├── training_system/     # 工作流、状态、计划与专项契约
│   └── training_history/    # 历史训练分析
├── models/                  # 模型结构定义与轻量仓库资产
├── palm_detector/           # 文件夹级联所需 Eos 接口/配置
├── environment.yml
├── requirements.txt
└── Makefile
```

---

<div align="center">

<img src="./docs/assets/aethersign-logo-minimal.svg" alt="AetherSign" width="52" />

<sub>AetherSign · Eos → Iris → Muse · 全国总决赛一等奖归档</sub>

</div>
