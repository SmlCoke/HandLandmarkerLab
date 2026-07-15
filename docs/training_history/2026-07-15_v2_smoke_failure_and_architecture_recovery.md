# 2026-07-15 v2 smoke 失败与架构恢复

## 结论

commit `9411d6f56ff3289c63759410252f4dc8b7a276d9` 的 geometry smoke **没有通过门控，不能据此启动正式 geometry pretrain**。训练流程、数据读取和梯度执行均完成，但旧 v2 数值初始化严重失稳，128 张训练样本跑满 300 epoch 后仍无法记忆。

服务器仅做了只读审计；本次代码、配置和文档修改只发生在本地仓库。

## 服务器证据

审计目录：

```text
/root/autodl-tmp/TrainFab/HLML-2.0/hand_landmarker_runs/v2-pretrain-r1/smoke
```

`smoke_gate_report.json`：

| 指标 | 实际值 | 门槛 |
|---|---:|---:|
| mean landmark MAE | 1.047229 | 0.01 |
| p90 sample MAE | 1.360035 | 0.02 |
| max sample MAE | 6.649611 | 0.05 |
| mean coordinate pixel error | 267.04 px | 2.55 px（由 0.01 换算） |

`history.json` 中 landmark MAE 从 epoch 1 的 15.6358 缓慢降到 epoch 299 的 0.93617，说明梯度存在，但优化始终处于修复异常输出尺度的阶段。best checkpoint 的顺序、无增强、全 128 张复测仍为 1.04723。

同一批真实 ROI 的额外只读探针显示：

- 旧 v2 未训练、`training=True`：landmark 范围约 `[-113,134]`，标准差 26.8；
- 旧 v2 未训练、`training=False`：landmark 达到约 `[-2.09e9,2.67e9]`；
- label 实际范围仅约 `[0.01,0.89]`；
- 训练 300 epoch 后输出仍约 `[-6.07,5.39]`。

根因是 7 个 stage 每层 8 次、总计 56 个多分支残差块连续累加随机分支；BN 初始 moving statistics 与线性 landmark head 又放大了训练态/推理态差异。继续增加 epoch 或直接启动全量训练不合理。

完整 curated geometry 本身没有发现结构错误：25,162 条全部为 positive pseudo，其中 HIGH 24,651、MEDIUM 511；`POS_RUNTIME=15,907`、`POS_LOW_PALM=9,255`；Left/Right 为 13,507/11,655。smoke 报告未发现重复图或 Train 泄漏。当前 geometry 的 75%/25% 两类采样与这些数据相容，因此此次 smoke 灾难不是由正负样本混入或左右手极端失衡造成。

但暗光有效 geometry 监督明显不足：`peak_train_0714_dark=3`、`soar_train_0714_dark=117`，合计仅占 25,162 个正样本的约 0.48%。这意味着通过新 smoke 后可以合理期待模型学习一般手部几何，但不能仅凭现有 pseudo 数据保证暗光效果。暗光 teacher abstention 必须继续 HOLD/删除，不能伪装成负样本；若要改善暗光 geometry，需要更多 teacher 成功且可视化正确的暗光 21 点，或后续人工 Gold。

## 本地恢复方案

- stage 深度改为 `[2,2,3,4,4,6,6]`，通道改为 `[24,32,64,128,192,256,384]`；
- 移除对 1x1 pointwise 没有新增感受野的重复辅助分支；
- 保留可折叠的 Conv+BN，部署时精确融合为单 Conv；
- residual/downsample 主分支末端 BN gamma 零初始化；
- landmark head 初始恒为 0.5，hand flag/handedness 初始概率也为 0.5；
- 所有训练、评估、推理和导出配置共享同一个深度表；
- ONNX 实际大小硬门禁设为 15 MiB；
- `make test` 增加训练前 untrained ONNX + 真实转换数据包预检。

候选代码通过服务器内存探针（未写服务器文件）：

| 项目 | 结果 |
|---|---:|
| training 参数 | 2,344,396 |
| deploy 参数 | 2,302,276 |
| 序列化 ONNX | 8.787 MiB |
| 融合最大误差 | 0 |
| ONNX 算子 | Conv/Add/Relu/MaxPool/Sigmoid/Identity/Reshape |

单批 32 张 ROI 的纯内存优化探针在 30 个更新步内把训练态 MAE 从 0.1201 降到 0.00826。这证明新结构能够接收有效梯度，但不替代完整 128 张、顺序推理态 smoke gate。

## 下一步硬门禁

同步新代码后，旧 smoke 与当前 Git/config provenance 不一致，不能复用。归档或删除旧的失败 smoke 目录后重新执行：

```bash
make compile
make test
make inspect-geometry-smoke
make pretrain-geometry-smoke
```

先将以下预检文件提交官方转换工具：

```text
hand_landmarker_runs/v2-pretrain-r1/export/preflight/hand_landmarker_v2_untrained.onnx
hand_landmarker_runs/v2-pretrain-r1/export/preflight/model_conversion/datasets.zip
```

只有官方结构转换成功，并且新的 `smoke_gate_report.json` 为 `status: pass`，才执行：

```bash
make pretrain-geometry
```
