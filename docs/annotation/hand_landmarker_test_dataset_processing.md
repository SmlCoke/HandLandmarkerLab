# Hand Landmarker 测试集处理方案

> 文档定位：定义锁定测试集的人工复核、Gold 筛选、双手 ROI、共享与冻结规则。  
> 共享方式：Peak 与 Soar 100% 使用同一份 Test；文件已带 `peak_` / `soar_` namespace。  
> 当前数据：1247 张原始 TIFF，生成 856 个 ROI；自动结果为 452 positive、404 negative。  
> 更新时间：2026-07-10。

## 1. 测试集的角色

Test 用于评价已经冻结的最终方案，不用于训练和模型选择。

Test 不允许：

- 参与第一阶段伪标签训练；
- 参与第二阶段微调；
- 选择 epoch 或 checkpoint；
- 调 presence/handedness 阈值；
- 选择数据增强、采样比例或 loss；
- 根据逐样本失败案例修改训练数据；
- 根据某个模型的失败情况删除 ROI。

推荐第一阶段只报告人工 Val 结果。第二阶段模型、阈值、后处理、量化方案全部冻结后，再运行 Test。

如果比赛汇报必须同时给出第一阶段和第二阶段 Test 结果，应在第一次解锁 Test 前预先登记：

- 两个 checkpoint；
- 所有阈值；
- 评测脚本；
- included/ignored manifest；
- 不因第一阶段 Test 结果修改第二阶段的承诺。

更安全的默认做法仍是：Test 只用于最终模型。

## 2. 人工复核前流程

继续使用：

```text
configs/autolabel_test.yaml
00_validate_images.py
01_export_palm_detections.py
02_build_hand_roi_crops.py
03_run_mediapipe_on_rois.py
04_export_cvat_xml.py
```

Test 配置必须与最终板端运行设置一致：

```text
Palm score threshold = 0.50
NMS IoU = 0.30
cross-head suppress IoU = 0.35
max detections = 2
keep_low_score_candidates_for_negatives = false
ROI scale = 1.8 × 1.8
shift_y = -0.1
output = 256 × 256
```

禁止为了减少人工量或提高测试指标而改变阈值。低分 `negative_candidates` 不属于主 Test；超过阈值但人工确认无手的 detection 是合法且重要的 hard negative。

在 CVAT 中判断双手目标前，可以用 draft 运行：

```powershell
python scripts/06_visualize_autolabels.py --config configs/autolabel_test.yaml --labels-jsonl <hand_landmarks_autolabel_draft.jsonl>
```

它生成全图 Palm/ROI overlay，只用于追溯 `palm_det_id` 对应的目标手，不得根据学生模型 Test 输出决定标注或排除。

## 3. CVAT 人工复核规则

Test 与 Val 使用完全相同的标注语义和 CVAT labels：

```text
no_hand
Left
Right
ignore_for_training
hand_landmarks
```

为在不改变 CVAT labels 的前提下保留双手/ignore 原因，Test 必须与 Val 使用同一格式的 `review_context.csv` sidecar：

```text
crop_id,palm_det_id,review_reason,context_group,target_hand_rule,reviewer,note
```

至少所有 ignored、双手和 anchor 异常样本必须填写。否则只能报告 generic ignore，无法可靠生成 multi-hand 分组或 challenge 清单。

正式复核前，标注者必须使用相同校准样例冻结 handedness、目标手与 ignore 规则。详细目标手判定和标注操作见 [验证集处理方案](hand_landmarker_val_dataset_processing.md)。本节给出 Test 必须独立满足的决策表。

| ROI 实际情况 | CVAT 操作 | 是否进入主 Test |
|---|---|---:|
| 完全没有手 | 只保留 `no_hand` | 是 |
| 一只手，21 点可可靠确定 | 一个完整 skeleton + 唯一 Left/Right | 是 |
| teacher 标错但人工可修正 | 修正所有错误点、presence 和 handedness | 是 |
| 困难光照/模糊/遮挡，但仍可可靠标全 | 完整标注，不因困难删除 | 是 |
| 实际有手但无法可靠标全 21 点 | `ignore_for_training`，绝不能 `no_hand` | 否 |
| 两只手，Palm anchor 目标明确 | 只标目标手，另一只为干扰 | 是 |
| 两只手，目标归属或点位不唯一 | `ignore_for_training` | 否 |
| 图像损坏、无法建立可信真值 | `ignore_for_training` | 否 |

### 3.1 非 ignored 正样本

- 没有 `no_hand`；
- 恰好一个 `hand_landmarks` skeleton；
- 恰好 21 点；
- 恰好一个 Left 或 Right；
- 21 点属于 Palm anchor 对应的同一只目标手；
- 坐标可由人工可靠确定。

### 3.2 非 ignored 负样本

- 恰好一个 `no_hand`；
- 没有 skeleton；
- 没有 Left/Right；
- 无论 Palm score 多高都保留为 negative。

### 3.3 Ignored 样本

- 使用 `ignore_for_training`；
- 不添加 `no_hand`；
- 不进入主 Test 指标；
- 不要求继续修正自动 skeleton，从而节省人工时间；
- 必须保留在 ignored 清单和覆盖率报告中。

07B 输出 ignored 行时必须标记 `ground_truth_valid=false`。其中未修正的自动 presence、handedness 和 landmarks 不具备 Gold 语义，不能用于训练、主评测或 challenge landmark 指标。

`ignore_for_training` 名称虽然包含 training，但在 Val/Test 中解释为“从主评测中排除”。为了保持现有 CVAT schema，本轮不新增 `ignore_for_evaluation`。

## 4. Test 中的双手 ROI

### 4.1 不应一律 ignore

当前模型每个 ROI 只输出一套 21 点，但第二只手可以作为真实干扰存在。

Test 必须逐项复用 [验证集处理方案第 5 节](hand_landmarker_val_dataset_processing.md) 的完整 anchor 判定方法，不允许为 Test 另定更宽松规则。最低步骤是：

1. 由 CVAT basename 找 manifest row；
2. 由 manifest 的 `palm_det_id` 找到同一原图的唯一 Palm detection；
3. 读取该 detection 的 bbox、p0 和 p9；
4. 检查 p0 与 p9 是否共同对应同一只手的 wrist 和 middle MCP；
5. 检查 p0→p9 方向与该手 wrist→middle-MCP 轴向是否一致；
6. 检查 bbox 是否主要覆盖同一只手的 palm core；
7. 通常只有一只手形成完整成套对应时，才能判为 anchor 目标。

与 Val 一致的唯一 fallback：如果 anchor 不对应任何手，但 crop 中只有一个能构成合理、完整目标的手实例，可以按 crop-level presence 语义标该手，并记录 `single_visible_hand_fallback`。少量第二只手边缘/残缺片段不构成第二个有效目标；如果存在两只都可成为合理目标的手，则 fallback 禁止，必须 ignore。

如果 p0 更像手 A 而 p9 更像手 B，或者 bbox 同时覆盖两掌且没有明确轴向区分，必须 ignore。现有全图 overlay 同时画多个 anchors，不能只看颜色或 Palm score猜测；必须完成 `crop_id → palm_det_id → detection` 的 ID 追溯。

只要能够根据以下信息唯一确定目标手，就保留并只标目标手：

1. 当前 `palm_det_id`；
2. 原图 Palm bbox、p0/p9；
3. rotated ROI 的中心、方向和尺度；
4. 全图 overlay；
5. 目标手完整 21 点是否可可靠区分。

Google skeleton 不是最终目标选择依据。如果 Google 标到非目标手，人工应改到 anchor 对应手。

只有以下情况才 ignore：

- 两只手都可能是当前 ROI 的合理目标；
- Palm anchor 融合两手或来源不可判断；
- 两手交叉导致关键点归属不唯一；
- 目标手的完整 21 点无法可靠确定。

有手但歧义时绝不能标 `no_hand`。

sidecar 中对通过样本记录 `context_group=multi_hand_target_clear` 和实际 `target_hand_rule`；对未通过样本记录 `multi_hand_target_ambiguous/anchor_ambiguous`。所有 Test 双手样本应由第二人快速复核目标归属。

### 4.2 是否重新录制

不建议立即废弃当前 Test。先在不查看任何学生模型 Test 结果的前提下统计：

```text
multi_hand_total
multi_hand_target_clear
multi_hand_target_ambiguous
anchor_or_roi_failure
other_ignored
eligible_positive
eligible_negative
```

然后：

- 目标清楚的双手 ROI 继续进入 Test；
- 真正歧义的 ROI ignore；
- 如果 ignore 后有效 Test 数量或关键场景覆盖明显不足，只定向补录缺口；
- 时间允许时按真正 ignored 数量近似一对一补录，使 eligible Test 尽量恢复到当前约 850～1000 ROI 的量级；
- 补录规则必须在运行学生模型 Test 前确定；
- 补录数据应作为新 Test 版本重新冻结，不允许把模型失败样本逐个替换成容易样本；
- 旧歧义双手数据保留为 `multi_hand_challenge`，单独报告。

如果 20%～30% 只是“包含第二只手”，实际 ambiguous 比例可能远低于这个数字，因此不能直接按 20%～30% 推算全部重录工作量。

歧义双手没有当前单 skeleton schema 能表达的唯一 landmark Gold，因此 `multi_hand_challenge` 默认只报告数量、覆盖率、Palm anchor ambiguity/failure rate 和定性案例。若要量化双手召回或两套 landmarks，必须另建原图级双手实例 Gold；不能把 ignored 行里未修正的 teacher skeleton 当作真值。

### 4.3 系统级意义

若最终手语应用包含双手动作，完全移除双手场景会产生虚假的乐观指标。推荐同时保留：

1. 目标唯一的双手干扰 ROI，进入主 Hand Landmarker Test；
2. 目标歧义或 Palm 融合两手的 ROI，进入 challenge/end-to-end 报告；
3. 单手和独立双 ROI 场景，保证主指标稳定可解释。

## 5. Peak 与 Soar 是否共享 Test

Test 应 100% 共享。

共享统一 Test 的价值：

- 两条独立训练路线可公平比较；
- 不会把测试集难度误认为模型差异；
- 只需精标一次；
- 可以用相同 FP32、量化和板端评测脚本；
- 最终选择模型有唯一可信依据。

训练独立性由以下内容保证，而不是靠重复制作 Test：

- 独立的训练候选筛选；
- 独立的 Train gold 选择；
- 不同 loss、增强、采样、优化器；
- 独立 checkpoint；
- 不共享 Test 逐样本错误用于改进。

推荐权限：

- 两人可以共同承担 Test 标注；
- 双手、ignore 和困难样本进行交叉复核；
- 随机约 10% 普通样本双人独立复核；
- Gold JSONL 完成后冻结 hash；
- 由一名成员或统一评测程序保管 Test 标签与逐样本结果；
- 日常只使用共享 Val 或各自 private dev；
- 最终只返回汇总 Test 指标和冻结报告。

## 6. 数据独立性与泄漏检查

Test 必须按原始录制 session 隔离：

- 同一原图的所有 ROI 属于同一 split；
- 同一视频片段、连续帧或近重复帧不能跨 Train/Val/Test；
- 检查原图 SHA256 和感知 hash；
- 检查两位成员训练集与 Test 的同名和同内容文件；
- 主动学习只能从训练池选样本；
- 不得把 Test 失败样本复制到第二阶段微调集；
- 两位成员必须使用同一个 eligible manifest，不能分别排除各自模型失败的 ROI。

如果两位成员都参与录制 Test，可以由不同成员贡献不同 session、光照和背景，但合并后仍形成同一个共享、锁定 Test。

## 7. 当前仓库导入与 07B 筛选

人工复核后：

```text
CVAT reviewed XML
  → 05_import_cvat_xml.py
  → hand_landmarks_reviewed.jsonl + cvat_import_stats.json
  → 07B_finalize_evaluation_labels.py --config configs/finalize_test.yaml --split test
  → hand_test_labels.jsonl
```

`05` 会把 CVAT 覆盖状态和冲突写入行内及 import report；07B 对非 ignored 行执行严格 Gold 完整性门禁。

### 7.1 07B Test 规则

07B 必须同时读取 reviewed JSONL 和 CVAT import diagnostics（当前为 `cvat_import_stats.json`，未来可由 05 写入每行）。仅凭 reviewed row 无法恢复 multiple skeleton、XML 缺失等具体导入问题。

1. manifest、CVAT XML 和 reviewed JSONL 一一覆盖且唯一；
2. `source=cvat_reviewed_missing_image` 或缺失 CVAT image 为 fatal；
3. ignored 行先移入独立清单，不检查其 skeleton 是否精修；
4. 非 ignored positive 必须有唯一、完整、有限、坐标一致且位于 crop 内的 21 点和 Left/Right；
5. 非 ignored negative 必须无任何 landmarks 且 handedness unknown；
6. multiple skeleton、`no_hand+skeleton`、缺失 handedness、点数错误、越界或导入 error 必须回 CVAT 修正；
7. 正式 Test included 行默认必须 `palm_valid=true`；
8. 人工确认的高 Palm score negative 必须通过，不能触发伪标签式删除；
9. 不执行训练集重复降采样或 pseudo 质量权重；
10. 任一 fatal 时不覆盖已冻结 canonical Test。

Test 需要“结构严格、人工语义权威”，不需要训练集式启发式质量筛选。

## 8. 测试集输出与冻结包

建议输出：

```text
../autodl-tmp/test_merged/05_labels/hand_test_labels.jsonl
../autodl-tmp/test_merged/05_labels/hand_test_ignored.jsonl
../autodl-tmp/test_merged/qc/finalize_test_report.json
```

冻结包记录：

```text
原始 TIFF 清单与 SHA256
ROI manifest SHA256
CVAT reviewed XML SHA256
Gold JSONL SHA256
eligible/ignored manifest SHA256
autolabel_test.yaml SHA256
Palm ONNX 与 MediaPipe task SHA256
代码 commit
presence/handedness 阈值
评测脚本版本
ignore 数量与比例
review_context.csv SHA256 与覆盖率
```

Handedness 默认阈值为 0.5；若项目决定校准其他阈值，只能在 Val 上完成并在冻结包中登记，Test 不重新搜索。

Test 输出稳定排序。Peak 与 Soar 评测前必须核对同一 canonical hash。

## 9. 正式报告指标

主 Test 报告：

- included / ignored 数量和比例；
- positive / negative / Left / Right；
- presence Precision、Recall、F1、FPR、FNR；
- landmark mean/median/P90/P95 pixel error；
- NME 与 PCK；
- per-landmark error；
- handedness accuracy 与左右手 recall；
- hard negative 指标；
- 单手、目标明确双手干扰、边缘、暗光、模糊等分组指标；
- FP32、量化仿真、A1 板端的同口径对比；
- challenge 集单独结果。

不能只报告总体 accuracy，也不能不报告 ignore 比例。

## 10. 验收清单

- [ ] Test 与 Train/Val 的录制 session 和近重复帧隔离；
- [ ] 配置与板端运行参数一致；
- [ ] 所有 ROI 均完成 CVAT 人工判定；
- [ ] 双手目标选择规则与 Val 完全一致；
- [ ] 所有双手 ROI 都完成 `crop_id → palm_det_id → detection` 追溯并写入 sidecar；
- [ ] 真正 ambiguous/anchor 融合双手已 ignore，目标明确双手已依据 p0/p9、轴向和 bbox 正确标目标手；
- [ ] 是否补录的决策在查看学生 Test 结果前完成；
- [ ] 每个非 ignored positive/negative 均满足严格 Gold schema；
- [ ] Test 没有参与 checkpoint、阈值或超参数选择；
- [ ] 两位成员使用相同 eligible manifest、Gold hash 和评测脚本；
- [ ] ignored 比例和 challenge 集没有被隐藏；
- [ ] canonical Test 已冻结且只读；
- [ ] 最终模型与阈值已在 Test 解锁前登记。
