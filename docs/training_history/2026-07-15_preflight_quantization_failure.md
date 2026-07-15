# 2026-07-15 preflight INT8 量化失败分析

## 结论

首次 v2 preflight 在公司工具链报出：

```text
m1_parser_main_func failed
SSQuantizesPt. 1 Failed!
exit, error_code=150
```

现有证据不支持“ONNX 文件损坏”或“opset 过高”的泛化判断。该文件已通过 `onnx.checker`、ONNX Runtime 推理和 Keras↔ONNX parity，且本身就是固定的 opset 11、IR version 6。日志明确进入 `SSQuantizesPt`，结合图审计，最可能的直接原因是 preflight 使用了训练稳定性所需的零初始化，导出后出现大量全零 Conv，三个输出对校准输入恒为 `0.5`，INT8 工具无法得到有效的量化 scale/quantization point。

这是根据工具日志与模型图作出的根因推断；厂商工具内部不可见，只有重新提交修复后的模型成功转换才能最终确认。

## 服务器真实产物证据

失败文件：

```text
/root/autodl-tmp/TrainFab/HLML-2.0/hand_landmarker_runs/v2-pretrain-r1/export/preflight/hand_landmarker_v2_untrained.onnx
```

审计结果：

| 项目 | 失败 preflight |
|---|---:|
| ONNX 大小 | 8.787 MiB |
| opset / IR | 11 / 6 |
| Conv 数 | 108 |
| 全零 Conv 权重 | 35 |
| 全零 initializer | 35 / 111 |
| 最大 grouped/depthwise group | 192 |
| zeros/ones/random 探针的三个输出 | 全部恒为 0.5 |
| 优化后 Sigmoid / Identity | 1 / 1 |

作为对照，旧 v1 ONNX 同样为 opset 11、IR 6，也包含 3 个 Reshape；它有 0 个全零 Conv，最大 group 为 128，并且官方工具能继续解析到明确的 LeakyReLU 不兼容错误。因此 Reshape、opset 11 和文件格式不是两次行为差异的合理解释。group=192 是否也会触发厂商限制无法由当前日志证明，但它超出了旧图已经走过解析阶段的范围，作为次要风险一并收敛到 128。

## 本地修复

1. `make test` 的 preflight 不再直接导出精确的训练初始权重；它复制一份 disposable 模型，把零 gamma 残差尾部和三个 head 替换为固定种子的微小非零探针权重。正式训练初始化不受影响。
2. v2 bottleneck 通道封顶为 128，使所有 depthwise/grouped Conv 的 group 不超过 128。
3. 两个 sigmoid head 的正式训练初始 kernel 改为极小的确定性非零值，初始概率仍接近 0.5；这也避免 geometry 阶段未监督 handedness 时导出精确全零 head。
4. 导出器新增不可绕过的量化门禁：拒绝缺少静态权重、非有限或全零的 Conv，拒绝 group>128，拒绝对合成探针没有动态范围的任一输出。
5. contract 新增 `quantization_readiness`，便于在提交公司工具前审计。

修复候选在服务器只读环境中以内存方式构建、融合并序列化，没有写入服务器文件：

| 项目 | 修复候选 |
|---|---:|
| training / deploy 参数量 | 1,951,756 / 1,912,324 |
| ONNX 大小 | 7.309 MiB |
| 全零 Conv 权重 | 0 |
| 最大 grouped/depthwise group | 128 |
| 算子 | Conv/Add/Relu/MaxPool/Reshape/Sigmoid |
| landmarks 聚合动态范围 | 0.005881 |
| hand_flag 聚合动态范围 | 0.000234 |
| handedness 聚合动态范围 | 0.000310 |

候选仍保持固定输入输出接口，并远低于 15 MiB 上限。

## 复验顺序

服务器完成 `git pull` 后执行：

```bash
make compile
make test
```

然后检查：

```text
hand_landmarker_runs/v2-pretrain-r1/export/preflight/hand_landmarker_v2_untrained.contract.json
```

其中应有：

```text
quantization_readiness.graph.violations = []
quantization_readiness.graph.observed_maximum_group = 128
```

并且三个 `quantization_readiness.aggregate_output_ranges[].dynamic_range` 均不小于 `1e-6`。把本次新生成的 ONNX 与同目录新生成的 `model_conversion/datasets.zip` 成对提交官方工具链。不要继续提交旧的 SHA-256 对应文件，也不要用降低 opset 或重新打包旧 ONNX 代替重新导出。

只有公司工具转换成功后，才能说明结构兼容性 preflight 通过。之后仍需重新执行 geometry smoke；preflight 不证明模型精度，也不能替代 smoke gate。
