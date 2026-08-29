# HLML 当前状态（2026-08-29 最终归档）

## I. 归档结论

AetherSign 已于 2026-08-25 完成全国总决赛答辩并获得全国一等奖。HandLandmarkerLab 针对本届比赛的训练、评估与部署导出使命已经完成，不再保留进行中的训练计划。最终可复现的 Git 代码状态由 annotated tag `HLML-4.0-final` 固定。

### 1.1 Git 归档范围

tag 包含 HLML 4.0 代码、五份公共 YAML、单元测试、入口/专项文档和 README 视觉资产。以下资产不进入 Git，复现时必须从比赛归档单独恢复：

- HLMF 的正式 `HAND_DATASET_ROOT`、Registry 和已发布 Pretrain/Eval/negative/hard/Gold 数据；
- HLML 的 `HAND_TRAIN_ROOT`、snapshot、checkpoint、评估预测、ONNX 和转换数据包；
- A1 工具链生成的 m1model 及板端应用资源。

### 1.2 上下游版本

上游数据制作代码以 HLMF annotated tag `HLMF-3.0-final` 为准；本仓库以 `HLML-4.0-final` 为准。两个 tag 固定的是代码和配置，不替代外部数据、模型和服务器环境备份。

## II. 全国总决赛正式模型

正式提交并上板使用的是 Iris-2.0-Lite（multitask）与 Iris-2.0-Max（multi-finetune）；Pro 完成同协议训练和评估，但未作为正式提交模型。

| 产品版本 | HLML 结构/阶段 | Mean pixel error | P95 pixel error | Handedness Acc | 部署参数 | A1 延迟 |
| :-- | :-- | --: | --: | --: | --: | --: |
| **Iris-2.0-Lite** | `v3-lite` multitask | 10.43 px | 24.98 px | 89.55% | 852,832 | ≈20 ms |
| **Iris-2.0-Max** | `v3-max` multi-finetune | **9.71 px** | **23.26 px** | **98.26%** | 1,912,324 | ≈22 ms |
| Iris-2.0-Pro | `v3-pro` | 10.14 px | 23.77 px | 81.59% | 1,912,324 | ≈22 ms |

以上为 AetherSign 自建 402 张 Hand ROI benchmark 的最终记录。与分赛区版本 Iris-1.0 的 21.97 px mean / 55.01 px P95 相比，三档 Iris-2.0 均显著改善。完整 Eos/Iris benchmark 和板端 Fullcascade 结果见仓库根目录 `project-12.md`。

### 2.1 已知能力边界

- Iris ROI 关键点与 handedness 达到本届比赛最终使用状态，但 hand presence 分类头未学到可靠的无手区分，实际倾向输出高 presence；不能把该输出描述为已解决的拒识能力。
- 固定 ROI benchmark 只评价 Iris，不包含 Eos Palm 漏检或原图级联召回。
- A1 上 `palm_hand` / `fullcascade` 的 P95 端到端延迟约 78 ms，受单核 Cortex-A7、串行 NPU 推理和完整 Eos → Iris → Muse 链路限制。
- Eos-2.1 与最终数据/演示能力覆盖 near、mid；far 不属于比赛最终支持域。

## III. 最终系统状态

### 3.1 Iris v3 结构

| 结构 | 训练参数 | 融合后部署参数 | 训练/部署参数比 | 定位 |
| :-- | --: | --: | --: | :-- |
| `v3-pro` | 1,951,756 | 1,912,324 | 1.02 | 与未修改 v2 同构 |
| `v3-max` | 7,629,268 | 1,912,324 | 3.99 | 训练期多分支，部署期与 v2 同量级 |
| `v3-lite` | 878,272 | 852,832 | 1.03 | 最终轻量档 |

`v3-max` 的 Conv/Depthwise 多分支在导出前精确融合，正式 ONNX 不保留 BatchNormalization 或训练分支。三档均保持 NCHW `[1,1,256,256]` 输入和 `landmarks[42]`、`hand_flag[1]`、`handedness[1]` 输出契约。

### 3.2 数据成员与审计快照

- geometry：Train 82,902；Val 14,411；Test 5,343；membership errors 为 0。
- multitask：Train 99,812，其中 positive 82,902、negative 16,910；Val/Test 同上。
- multi-finetune：Train 100,274，其中 hard/gold 17,372、replay 82,902；Val/Test 同上。
- 已发布负样本：`neg-eos_2.0-hcf0813-hp0.5`。
- 已发布困难样本：`hard-hands-0816-r01`，462 条训练记录（379 positive、83 CVAT `no_hand` negative），另有 38 ignored。
- multi-finetune 最终 `epoch_size=3000`；真实 snapshot 精确采样计划中 Gold `POS_RUNTIME=1436`，平均/最大期望重复 3.789，低于 4/8 rare-cell 门禁。

PretrainSource、EValSource、negative/hard 发布集、完整标注链路、Registry、split/Test 隔离及既有质量门控在最终归档中均保持不变。

## IV. 最终验证状态

- Python 语法检查通过。
- HLML 完整单元测试 198 项通过。
- 五份公共配置解析通过。
- 三档模型的训练期/部署期融合、ONNX 数值一致性、15 MiB 大小和 A1 算子约束已有自动回归覆盖。
- 本次归档只修改 README、入口文档和独立 Logo，没有增加环境依赖；`requirements.txt` 与 `environment.yml` 不变。

## V. 后续状态

仓库进入只读归档优先状态。未来如确需复现，应从 `HLML-4.0-final` 创建独立分支并恢复匹配的 HLMF tag、数据仓和训练产物；如恢复产品开发，应使用新版本号和新 tag，不移动或覆盖本届比赛最终 tag。
