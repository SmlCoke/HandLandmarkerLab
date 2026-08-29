# HLML 归档后维护计划

## I. 当前目标

HLML 针对 2026 年全国大学生集成电路创新创业大赛全国总决赛的使命已经完成。当前没有继续训练、调参或扩展数据集的活动计划；后续工作的默认目标是保存可复现性，而不是在 `main` 上继续演进比赛版本。

## II. 复现优先级

1. 检出 HLML annotated tag `HLML-4.0-final`。
2. 检出上游 HLMF annotated tag `HLMF-3.0-final`。
3. 恢复与比赛阶段一致的 `HAND_DATASET_ROOT`、Registry、published datasets 和 `HAND_TRAIN_ROOT`。
4. 使用 Ubuntu 20.04、Python 3.8、TensorFlow/Keras 2.9、CUDA 11.2 与 `hand-landmarker-tf29` 环境。
5. 先运行 `make environment-check`、`make config-check`、`make compile` 和 `make test`，再按 workflow 复现 snapshot、评估或导出。
6. 复现正式提交模型时，优先核对 Iris-2.0-Lite multitask 与 Iris-2.0-Max multi-finetune 的 checkpoint/release 身份和导出 provenance。

## III. 归档维护规则

### 3.1 不可变对象

- 不移动、删除或重签 `HLML-4.0-final`。
- 不覆盖已发布 PretrainSource、EValSource、negative/hard/Gold 数据集。
- 不改变已冻结 Test 的成员、标签或 locked Test 结果。
- 不把模型权重、大规模数据或服务器目录补交进 Git tag。

### 3.2 文档修正

发现不影响行为的错字、失效链接或复现说明遗漏时，可以在 `main` 上追加修正文档，但不得回写 final tag。任何会改变代码、配置、数据合同、训练结果或依赖的工作都应使用新版本号和独立分支。

### 3.3 恢复开发

如比赛后重新启动 Iris 研发，应先创建新里程碑，明确新的 HLMF/HLML 版本组合、数据成员、模型产品名和验收协议。不得沿用 `final` 身份发布不同内容。

## IV. 未来研究候选（非活动计划）

- 重新设计 hand presence 的有效负样本与损失配比，解决高置信度全有手问题。
- 在不破坏 A1 算子/大小约束的前提下继续降低 ROI 关键点误差。
- 将 Eos far/extreme-pose 能力、Iris 拒识能力和端侧串行延迟作为整条链路共同优化，而不是只调单一模型。
- 若硬件或工具链允许，评估 CPU/NPU 流水化、模型结构利用率和更低延迟部署方案。

这些条目只记录已知方向，不构成对归档仓库的待办承诺。
