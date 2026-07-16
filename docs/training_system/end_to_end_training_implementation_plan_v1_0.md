# HLMF + HLML 完整训练流程：程序、代码与配置更新计划

版本：1.0

日期：2026-07-16

读者：后续负责实现的 AI / 开发者

性质：内部实现计划，不是人工操作手册

对应操作规范：`docs/training_system/end_to_end_training_workflow_v1_0.md`

## 0. 本轮边界

本轮只生成计划，不修改 HLML/HLMF 代码或配置。后续实现必须按本文分阶段提交，每个阶段先补测试，再开放 Make target。

用户指定的步骤 7～11 总体合理，不需要重做流程。实现时只做以下必要的小幅技术调整：

1. `negative_reviewed` 必须成为唯一人工保留输入，不能继续扫描原 `negative_candidates`；
2. 人工删除项、自动冲突项和最终真负例必须分成三个集合；
3. Dragon 作为 HLMF 内部 external-Gold adapter，不新建一套重复的 HLMF-dragon 仓库；
4. b/c 复用已有 manifest/draft，直接进入 HLMF 04，不重跑 MediaPipe 03；
5. d replay 必须包含已确认 negative，并与 Gold 去重；
6. finetune 来源允许缺失，但已存在来源必须 fail-closed；
7. sampler 必须支持 Gold positive 与人工 `no_hand` negative；当 Gold 恰好是 positive-only 时也必须按策略安全重分配，而 pseudo replay 使用自己独立的、包含 negative 的 tier-specific 配比；
8. pretrain ID 与 finetune ID 完全分离。

## 1. 当前能力矩阵

| 能力 | 当前状态 | 主要位置 | 结论 |
|---|---|---|---|
| pretrain curate | 已有 | `hand_landmarker/pretrain_curation.py` | 可生成 positive 和完整候选工作区 |
| 删除式 review finalize | 部分已有 | 同上 | 只扫描 `negative_candidates`，不支持 sibling `negative_reviewed` |
| removed 困难池 | 无 | — | 无 manifest、无目录、无事务清理 |
| multitask gate/train | 已有 | `scripts/check_multitask_data.py`、`configs/train_multitask.yaml` | 正式可运行，但当前 rare-cell 重复率过高 |
| train teacher–student 推理 | 无 | — | Eval 只支持 Val/Test，不能代替 train mining |
| replay materialize | 无 | — | sampler 只抽 batch，不生成 finetune source |
| Dragon external Gold | 无 | HLMF | EXIF、Gold-only、unknown handedness 均需兼容 |
| HLMF Gold subset 04/05 | 部分已有 | HLMF 04/05/07A | 缺自动 selector/materializer 和 finetune strict presence |
| finetune curation | 无 | — | 无 source contract、curate、gate、hash manifest |
| finetune trainer 内核 | 已有 | `hand_landmarker/training.py`、`data.py` | 支持 stage/initial checkpoint/gold fraction |
| finetune 正式路由 | 无 | Makefile/configs | 无 config、smoke、eval、infer、export；测试当前禁止 finetune config |

## 2. 总体架构决策

### 2.1 数据所有权

- HLMF 继续拥有 ROI geometry、投影、CVAT、外部 Gold 转换和 source package schema；
- HLML 拥有 pretrain review、student inference/mining、replay、最终 finetune curation、训练和模型产物；
- `train_sources`、`eval_sources` 是已冻结只读输入；
- 任何候选图片副本都不能成为身份真相，身份必须来自 manifest/global ID 和 SHA。

### 2.2 统一 source package

最终来源分两类目录。HLMF 只在 strict import/adapter 完整成功后发布 Gold；HLML 单独发布 replay：

```text
finetune/<HAND_FINETUNE_ID>/
├── sources/
│   ├── gold/<source_id>/
│   │   ├── finetune_source.json
│   │   ├── 02_roi_crops/
│   │   │   ├── images/
│   │   │   └── hand_roi_crops_manifest.jsonl
│   │   ├── 03_reviewed/
│   │   │   ├── hand_landmarks_reviewed.jsonl  # 与manifest全覆盖；可含ignore=true
│   │   │   └── ignored.jsonl                  # 可选派生sidecar，不是07A覆盖输入
│   │   ├── audit/selection_or_reject.jsonl
│   │   └── qc/<producer>_report.json
│   └── replay/pretrain_replay/
│       ├── finetune_source.json
│       ├── 05_labels/canonical_source.jsonl
│       └── qc/replay_report.json
├── mining/                               # b/c selection request；不是final source
└── cvat/                                 # HLMF处理中间task；不是final source
```

不是每种 source 都必须拥有所有子文件；`finetune_source.json` 明确声明实际 artifact 和 SHA。

HLMF 07A 仍作为 Gold 结构和多来源 namespace 的最终上游门禁，但目标输出改为：

```text
finetune/<HAND_FINETUNE_ID>/hmlf_gold_merged/
```

它只聚合 a/b/c/e 的 Gold source。HLML `finetune-curate` 只从这份 Gold aggregate 读取训练 Gold 行，再与 d replay 合并；单源 descriptor 只用于认证来源、head policy、角色权重和 SHA，不能再次 append raw Gold。不得让 HLMF 和 HLML 同时写 `train_finetune_merged`。

### 2.3 配置唯一权威

- `configs/prepare_finetune_sources.yaml`：只定义 b/c/d 的选择预算、上限、算法和 salt；
- `configs/curate_finetune.yaml`：只定义 Gold role/source 权重、来源发现和 curation gate；
- `configs/train_finetune.yaml`：只定义 gold fraction、epoch size、tier 内 sample type fractions、loss 和训练策略。

禁止在 selection config 中复制训练 mix。`check-finetune-data` 认证 curation manifest，并验证 train config 与 manifest 的 sampling/source 元数据兼容。

### 2.4 统一身份

每条派生 ROI 至少保留：

```text
source_id
dataset_id
source_crop_id
global_crop_id
parent_dataset_id
parent_source_crop_id
parent_global_crop_id
source_sequence_id
source_frame_index
crop_path
image_sha256
```

原生 HLMF source 的 `parent_*` 可为空；b/c 派生 subset 必须填写。跨来源去重优先使用 `parent_global_crop_id`，再使用 global ID、ROI SHA 和归一化像素 SHA。

## 3. P0：事务化 `negative_reviewed` 导入

目标：安全解锁现有 multitask，优先级最高。

### 3.1 配置更新

修改 `configs/curate_pretrain.yaml`：

```yaml
review:
  candidates_subdir: negative_candidates
  reviewed_subdir: negative_reviewed
  removed_subdir: negative_removed
  quarantine_subdir: negative_quarantine
  removed_manifest_file: negative_removed_manifest.jsonl
  quarantine_manifest_file: negative_quarantine_manifest.jsonl
  cleanup_candidates_after_success: true
  retain_reviewed_evidence: true
```

绝不能通过把 `candidates_subdir` 简单改名完成。程序仍需读取完整候选 manifest，并把 reviewed 视为其严格子集。

### 3.2 代码更新

主要文件：

- `hand_landmarker/pretrain_curation.py`
- `scripts/curate_pretrain.py`
- `scripts/check_multitask_data.py`
- `tests/test_pretrain_curation.py` 或现有对应测试文件

新增/重构职责：

1. 以 `candidate_relative_path` 为唯一 review key；不得使用旧绝对 `candidate_path`；
2. 扫描 `negative_reviewed` 时只接受支持的图片扩展；
3. 拒绝符号链接、路径穿越、未知路径、重复路径、额外 archive 和修改字节；
4. 验证 `review_manifest.source_labels_sha256` 与当前 source labels 一致；
5. 对 reviewed 图片、manifest SHA、`train_sources` 原图 SHA 做三方一致性；
6. 为 reviewed 全部生成 review evidence；
7. 把有 `NEGATIVE_OVERLAPS_CONFIRMED_HAND` 或新 overlap safety 命中的项分入 quarantine；
8. admitted = reviewed - quarantine；
9. removed = expected - reviewed；
10. removed 绝不能自动设置 `hand_presence=true`；
11. 生成 removed/quarantine manifest；
12. curated multitask 只接收 admitted。

当前预期守恒：

```text
expected=48643
reviewed=1049
quarantine=27
admitted=1022
removed=47594
```

### 3.3 `negative_removed_manifest_v1`

每行至少包含：

```json
{
  "schema_version": "negative_removed_manifest_v1",
  "crop_id": "<global crop id>",
  "dataset_id": "...",
  "source_crop_id": "...",
  "sample_type": "NEG_RUNTIME_CANDIDATE",
  "candidate_relative_path": "NEG_RUNTIME_CANDIDATE/...png",
  "source_crop_path": "/root/.../train_sources/...png",
  "candidate_image_sha256": "...",
  "source_image_sha256": "...",
  "source_labels_sha256": "...",
  "partition": "negative_removed",
  "partition_reason": "absent_from_negative_reviewed",
  "review_decision_basis": "absent_from_negative_reviewed"
}
```

Quarantine 使用独立 schema/manifest，原因明确为 overlap conflict；b selector 默认只读 removed，不读 quarantine。

### 3.4 文件移动与事务

不能边扫描边删除。目标顺序：

1. 获得单进程 lock；
2. 只读验证全部输入；
3. 在 review root 同文件系统创建 `.review-finalize.<uuid>.tmp`；
4. 在临时目录写 decisions、reports、removed/quarantine manifests；
5. removed 图片优先使用同文件系统 hardlink，fallback 才 copy，并逐图核 SHA；
6. 在临时 curated 目录完成全部 label、audit 和 SHA；
7. 写 transaction report，状态 `prepared`；
8. 依次以 `os.replace` 提交新 review artifacts 和 curated snapshot；
9. 写 transaction 状态 `committed`；
10. 最后清空/删除 `negative_candidates`；
11. 清理后写 `candidate_cleanup=complete`。

若多目录无法形成单个原子 rename，transaction report 必须允许检测并恢复中间状态。任何失败都不能让 candidates 先消失。

成功后幂等复跑必须仅依赖 authenticated review/removed/quarantine manifest，不要求 candidates 仍存在。

### 3.5 P0 测试

- reviewed 是 expected 的精确子集；
- reviewed 中未知路径拒绝；
- 重命名、修改图片、额外 ZIP 拒绝；
- 符号链接和 `../` 拒绝；
- source label snapshot 改变拒绝；
- source ROI 缺失/改字节拒绝；
- removed=complement；
- quarantine 不进入 admitted；
- 分区数量守恒；
- 所有输出成功前 candidates 不清理；
- 每个故障注入点均可恢复；
- 清理后幂等复跑；
- decisions/reports/manifests SHA 写入 curation manifest；
- 当前 1049/27/1022/47594 fixture 的集成统计测试。

## 4. P0.5：Multitask rare-cell 重复率保护

当前 `epoch_size=null` 会按约 60,974 条记录抽样；按 batch=64 的实际 largest-remainder quota，Runtime negative 约有 4,764 draw，即 128 条平均约 37.2 次/epoch（按总体 8% 粗算约 38 次）。

### 4.1 配置接口

修改 `configs/train_multitask.yaml`：

```yaml
sampling:
  epoch_size: auto
  epoch_size_upper_bound: 6400
  max_average_cell_draws_per_unique_record: 4.0
  max_expected_row_draws_per_epoch: 8.0
```

### 4.2 自动解析公式

不能只用浮点 fraction 估算，也不能把平均值称为单行最大值。解析器从一个 batch 开始枚举不超过 upper bound 的 batch 整数倍，并调用 sampler 的真实整数 `batch_quota()` 计算每个候选 epoch size 的 cell draw。

对每个 cell：

```text
average_draws_i = exact_integer_cell_draws_i / unique_count_i

expected_row_draws_j = exact_integer_cell_draws_i
                       × row_weight_j / sum(cell_row_weights)
```

候选 epoch size 必须同时满足：

```text
average_draws_i <= max_average_cell_draws_per_unique_record
max(expected_row_draws_j) <= max_expected_row_draws_per_epoch
```

选择满足条件的最大 batch 整数倍。必须至少为一个 batch；若不可行则 gate 失败。这样平均保护与高权重单行保护的含义分开且可验证。

replacement 仍可开启，但报告必须记录预计重复率。

### 4.3 代码与报告

涉及：

- `hand_landmarker/data.py`
- `scripts/check_multitask_data.py`
- `hand_landmarker/training.py` 或 sequence 构造路径
- `tests/test_data_contract.py`

报告新增：

```text
unique_by_cell
configured_fraction_by_cell
draws_per_epoch_by_cell
expected_draws_per_unique_record
max_expected_row_draws
limiting_cell
limiting_record_id
limiting_record_normalized_weight
resolved_epoch_size
```

不得静默修改 sample type fraction；只解析 epoch size。当前 Runtime-negative 权重相同时应解析到约 6,400，精确 draw 为 500，平均 `500/128=3.90625`。

## 5. P1：HLMF Dragon external-Gold adapter

### 5.1 不建立平行仓库

在 HLMF 内实现，建议文件：

- `hand_autolabel/external_gold/dragon.py`
- `hand_autolabel/external_gold/__init__.py`
- `scripts/08_prepare_dragon_gold.py`
- `configs/dragon_gold.yaml`
- HLMF Makefile `prepare_dragon_gold`
- 对应 tests

复用 HLMF：

- `roi_geometry.py`
- `projection.py`
- image/QC/atomic IO
- 07A finalization

### 5.2 输入事实与固定验收数

```text
images                      8593
annotation rows             4500
unannotated images          4093
p=0 annotated images         850
raw annotated hands         5311
unique matched hand ROI     5191
usable in-ROI Gold          5189
ignored out-of-ROI          2
```

输入审计 SHA：

```text
annotations_hand.txt  A5E6DF1A6D77AB82017E0A01693F82C7A531706920073EF0450C640A615E223A
annotations_palm.txt  C88DB227B3F1EB4E9D2EC40AD3EFCAEC670528270545E6B2E3A8DF2A5BA4A661
README.md             9CDB8F3C94D8040E0510914C8F2489166820ACA9DA57701AD017E100A68D0D59
```

这些值用于当前数据版本的验收，不应写死为所有 Dragon 版本通用常量；配置可提供 expected SHA/count，变化时要求新 source ID。

### 5.3 EXIF 与图像

全部被引用 JPEG 为物理 720×1280、EXIF Orientation=6、三通道相同的灰度 RGB。

导入器必须：

1. 记录原 JPEG SHA；
2. 使用 Pillow `ImageOps.exif_transpose` 或等价且有测试的实现；
3. 验证逻辑尺寸 1280×720；
4. 验证 RGB 三通道一致或按明确灰度规则转换；
5. 不修改原 JPEG；
6. 直接从逻辑图构造 ROI PNG；
7. 报告 EXIF 值、逻辑尺寸和派生 SHA。

禁止使用当前忽略 EXIF 的 `cv2.imdecode(..., IMREAD_UNCHANGED)` 直接解释标注坐标。

### 5.4 Hand–Palm 匹配

严格实现 Dragon README：

- `p=0`：reject，不是 Gold negative；
- `p>q`：reject；
- p=1：21 点几何中心必须只落入一个 Palm bbox；
- p=2：两个中心必须分别唯一落入两个不同 bbox；
- 其他情况：reject；
- 不使用最近中心、最大 IoU 或人工猜测兜底。

建议 ID：

```text
palm_det_id = <image stem>:palmA 或 palmB
crop_id     = <palm_det_id>:crop
```

### 5.5 ROI 与 Gold 投影

- `p0` 使用 legacy point1（wrist）；
- `p9` 使用 legacy point2（middle MCP）；
- ROI 参数严格为 1.8/1.8、shift_y=-0.1、256×256；
- `landmarks_image_px` 来自逻辑 1280×720 坐标；
- `landmarks_crop_norm` 由 ROI 仿射逆变换；
- `landmarks_crop_px = norm × 255`；
- 任一点越界不得 clamp；整行进入 ignored/reject audit。

### 5.6 缺失 Palm score

当前两端 schema 强制有限数值。最低改动兼容：

```json
{
  "palm_score": 0.5,
  "palm_score_observed": false,
  "palm_score_source": "legacy_export_missing"
}
```

0.5 只是当前有效阈值下界的兼容哨兵，报告和文档不得称其为模型实测置信度。所有 Dragon 行同值，且为 Gold positive。

### 5.7 Unknown handedness

Dragon 行：

```json
"handedness": {"label": "unknown", "score": null}
```

新增 HLMF handedness policy，不再用一个布尔值同时表达“允许逐行 unknown”和“整源不监督”：

```yaml
handedness_policy: unavailable
```

枚举语义：

- `unavailable`：Dragon；所有行必须 unknown，mask 0；
- `optional_per_row`：b/c/e；每行显式 Left/Right 时 mask 1，显式 unknown 时 mask 0；
- `required`：Val/Test；positive 必须 Left/Right。

Train Gold validation 按 policy 处理。Val/Test 仍严格要求 required，不能全局放宽。

### 5.8 Gold-only source

扩展 HLMF `finalization.py` source config：

```yaml
source_mode: gold_only
enabled_stages: [finetune]
```

`gold_only` 契约：

- manifest 与 `gold_labels` 一一覆盖；Dragon 的 `gold_labels` 必须有 5,191 行，其中 5,189 行可训练、2 行 `ignore_for_training=true`；
- 不要求 pseudo；
- ignored sidecar 只能是从全量 `gold_labels` 派生的审计文件，不能代替 07A 覆盖输入；
- 仅在 finetune finalizer 启用；
- pretrain finalizer必须跳过并报告 `disabled_for_stage`。

不得伪造空 pseudo 或把 Dragon Gold 混入 pretrain。

### 5.9 Dragon 测试

- EXIF 6 转正；
- 非 EXIF6/错误尺寸拒绝或按配置显式支持；
- 两标注文件 key set/行数/坐标数量；
- p=0、p>q、p1 unique/none/ambiguous、p2 bijection 各分支；
- ROI 参数与 HLMF 02 parity；
- image→crop→image 投影 parity；
- 越界不 clamp；
- unknown handedness masks；
- gold_only 无 pseudo；
- pretrain stage 跳过 Dragon；
- 预期 3565/5191/5189/2 统计；
- 64 张 deterministic overlay 生成。

## 6. P2：b/c/d 自动候选与 replay

### 6.1 统一 HLML 入口

建议新增：

- `hand_landmarker/finetune_selection.py`
- `hand_landmarker/train_prediction.py`
- `hand_landmarker/finetune_replay.py`
- `scripts/prepare_finetune_sources.py`
- `configs/prepare_finetune_sources.yaml`
- Makefile `prepare-finetune-sources`

总命令一次完成 b、c、d。每个 selector 可独立关闭，输出一份总报告和每个 source 的报告。

P2 同时在 HLML Makefile 首次引入并导出独立 `HAND_FINETUNE_ID ?= v2-finetune-r1`；`prepare-finetune-sources` 同时读取 `HAND_PRETRAIN_ID`（输入 r3）和 `HAND_FINETUNE_ID`（输出 finetune workspace）。`make paths` 从此必须打印两个 ID。P4 只复用这一变量，不能到训练路由阶段才首次定义。

### 6.2 通用 quota 算法

输入按 `dataset_id × sample_type` 形成 cell。默认 source 权重：

```text
w_i = user_weight_i × sqrt(eligible_count_i)
```

配额用 largest remainder 取整，并受以下条件约束：

- `max_items`；
- `per_dataset_max`；
- sample type fraction；
- source/sequence diversity；
- 可用数量；
- 已被另一 selector 选中的 parent global ID。

每个报告写：

```text
input_count
eligible_count
configured_budget
configured_weights
raw_quota
bounded_quota
actual_selected
shortfall_and_redistribution
seed/salt
input/output SHA
```

同一输入 SHA、config 和 salt 必须生成相同结果。

### 6.3 b：negative_removed selector

输入：

- authenticated `negative_removed_manifest.jsonl`；
- r3 pretrain catalog；
- 原 HLMF source manifest/draft registry；
- `train_sources` 图片。

默认配置：

```yaml
negative_removed:
  enabled: true
  max_items: 300
  per_dataset_max: 100
  sample_type_fractions:
    NEG_RUNTIME_CANDIDATE: 0.60
    NEG_LOW_PALM_CANDIDATE: 0.40
  one_per_source_group_first: true
  salt: negative_removed_gold_v1
```

规则：

- quarantine 不可进入；
- 优先每个 `source_group_id` 一条，再允许同组第二条；
- Runtime 可按 Palm score 分层，但不能只取最高分连续帧；
- Low-Palm 使用稳定 hash + session diversity；
- 每条输出保留 parent IDs 和 removal evidence；
- 复制 ROI 时逐图核对 source SHA，不能重新编码图片。

### 6.4 c：teacher–student disagreement

输入：

- r3 `hand_training_labels_pretrain_landmarks.jsonl`；
- r3 curation manifest；
- r3 geometry `best.weights.h5`；
- 对应训练配置和模型版本。

新增 train prediction runner，必须批量推理 canonical ROI，不运行 Palm，不使用 Val/Test prediction 代替。

每条结果至少记录：

```text
student_landmarks_crop_norm
student_hand_flag
mean_l2
p90_nme
max_nme
teacher_palm_width
mean_nme
teacher_edge_length
student_edge_length
collapse_log_ratio
teacher_bbox_area
student_bbox_area
checkpoint_sha256
source_labels_sha256
```

建议综合分数：

```text
score = 1.00 × percentile(mean_nme)
      + 0.50 × percentile(p90_nme)
      + 0.50 × percentile(abs(collapse_log_ratio))
      + 0.00 × percentile(hand_flag_error)   # geometry default
```

掌宽固定为 teacher landmark 5→17 距离；骨架长度固定使用 MediaPipe 20 条标准连接边。所有点误差先除以 `max(掌宽, 0.05)`，所有分项原值和 percentile 都写入报告，不能只留下不可解释的总分。

r3 geometry 的 hand-flag head 未经 negative 训练，因此 geometry checkpoint 的默认 `hand_flag_error` 权重必须是 0。只有配置明确改用通过 presence 验证的 multitask checkpoint 时，才允许非零权重并记录 checkpoint stage。

默认配置：

```yaml
teacher_student:
  enabled: true
  max_items: 300
  per_dataset_max: 100
  sample_type_fractions:
    POS_RUNTIME: 0.50
    POS_LOW_PALM: 0.50
  score_weights:
    mean_nme: 1.0
    p90_nme: 0.5
    collapse_log_ratio: 0.5
    hand_flag_error: 0.0
  salt: geometry_disagreement_v1
```

先按 source/session 去近重复，再在 cell 内取高分；相同分数用稳定 hash tie-break。

### 6.5 d：pretrain replay

输入必须是完成 negative review 后的 r3 multitask canonical，而不是原始 HLMF 未复核 negative。

规则：

1. 包含全部可用 confirmed negative，除非超过显式总上限；
2. 其余名额从 geometry positive 补齐；
3. positive 使用 75/25 Runtime/Low-Palm；
4. 按 source 平方根分配；
5. 保留 parent/global ID，暂不判断未来 b/c 哪些会成为 Gold；
6. `crop_path` 继续引用只读 `train_sources`，不复制约一万张图；
7. 输出行 `training_stage=finetune`，但保留原 annotation provenance、review evidence 和 parent provenance；
8. 输出 source descriptor 和 hash manifest；
9. 真正的 Gold-over-replay 去重只在 P3 `finetune-curate` 读取已完成 HLMF Gold aggregate 后执行。

默认：

```yaml
pretrain_replay:
  enabled: true
  max_records: 10000
  include_all_confirmed_negatives: true
  positive_fractions:
    POS_RUNTIME: 0.75
    POS_LOW_PALM: 0.25
  salt: finetune_replay_v1
```

### 6.6 P2 测试

- quota 精确和不足 cell 重分配；
- source sqrt allocation；
- per-source cap；
- deterministic salt；
- sequence/group diversity；
- removed 只读 authenticated manifest；
- quarantine 排除；
- parent ID 恢复；
- train predictions batch 与逐条一致；
- checkpoint/source SHA 改变导致新 provenance；
- disagreement metric 人工小 fixture；
- collapse ratio 数值稳定、零尺度保护；
- b/c parent 重叠去重；
- replay 含 confirmed negative，不含未复核 candidate；
- replay 引用原图且 SHA 正确；
- replay parent IDs 足以供 P3 执行 Gold supersedes replay；
- max_items=0/disabled/空 cell 行为。

## 7. P2.5：HLMF subset materializer 与 strict CVAT

### 7.1 建议组件

- `hand_autolabel/finetune_subset.py`
- `scripts/09_prepare_finetune_gold.py`
- `scripts/10_batch_import_finetune_gold.py`
- `configs/finetune_gold.yaml`
- `configs/finalize_finetune.yaml`
- `configs/cvat_label.json`（增加显式 unknown handedness，若采用该 tag）
- Make targets `export_finetune_gold`、`import_finetune_gold`

两个 target 共用的稳定参数契约：

```make
HAND_DATA_ROOT       # 中央 HLML-2.0 物理根
HAND_FINETUNE_ID     # finetune 实验 ID
FINETUNE_SOURCE_ID   # 本次 Gold source ID；批量 b/c 可由 task descriptor 自动发现
FINETUNE_SOURCE_MODE # export 必填：selection_subset 或 native_existing
FINETUNE_RAW_SOURCE_ROOT # 仅 native_existing(e) 必填
```

Makefile 必须 `export HAND_DATA_ROOT HAND_FINETUNE_ID` 给 Python 子进程。`export_finetune_gold` 缺显式 mode 时 fail-closed；`import_finetune_gold` 从已认证 task descriptor 恢复 mode，若用户另传且不一致则 fail。两个 target 在必填变量缺失、source ID 冲突或目标目录已存在时都 fail-closed。

### 7.2 Materialize

materializer 支持两种显式输入模式：

- `selection_subset`：b/c；从 HLML selection request 恢复既有 manifest/draft/ROI；
- `native_existing`：e；读取一个已经跑完 HLMF 00～03 的独立 raw source root，复用其 manifest/draft/ROI，但仍进入 finetune 专用 04/strict 05。

两种模式都必须剥离 teacher handedness、写明 parent identity（原生 source 合法为 null）、把中间 task 放在中央 `finetune/<ID>/cvat/`，并且只在 strict import 完成后发布 source。`native_existing` 不得退回普通 `export_cvat_train/import_cvat_train`，否则会绕过显式 presence/handedness 契约。

`selection_subset` 对每个父 `dataset_id`：

1. 读取 HLML selection artifact；
2. 从 source registry 找到完整 manifest/draft/images；
3. 以 parent source crop ID 精确取行；
4. 比较 selection、manifest、draft、图片 SHA；
5. 生成新的 source-local ID，同时保留 parent IDs；
6. 复制或 hardlink ROI，不重新编码；
7. 生成 subset manifest/draft；
8. 调用 finetune 专用 04 逻辑生成 XML；
9. 输出独立 task descriptor 和 QC。

`native_existing` 不读取、也不伪造 HLML selection artifact：

1. 从 `FINETUNE_RAW_SOURCE_ROOT` 读取已经完成 00～03 的 manifest/draft/images；
2. 验证三者全覆盖、dataset/source ID、ROI geometry、逐图 SHA 和 03 draft SHA；
3. 为中央 Gold package 生成新的 source/dataset namespace；原生 ROI 的 `parent_*` 按第 2.4 节设为 null，原 raw source identity 写入单独 provenance 字段；
4. hardlink/copy ROI 且不重编码，生成全覆盖 manifest/draft；
5. 调用同一个 finetune 专用 04，输出 task descriptor 和 QC。

不能把裸 `negative_removed` 图片目录直接交给 03/04。也不能用只按文件名 every-N 复制的 `tools/downsample.py`。

HLML 在 P2 只把 b/c selection request 发布到 `finetune/<ID>/mining/`。HLMF materializer 和 CVAT 中间产物只写 `finetune/<ID>/cvat/`。只有 strict 05、coverage 和 source QC 全部成功后，HLMF 才以原子 rename 发布 `sources/gold/<source_id>/finetune_source.json`。半成品永远不出现在 final source discovery 根。

### 7.3 不重跑 03

- b 的原 draft 已是 `present=false`，04 生成 no_hand 初稿；
- c 的原 draft 已有 teacher 21 点；
- student prediction 只用于候选排序，不写成 CVAT Gold 初稿；
- finetune 专用 04 必须剥离 draft 中 teacher 自动填入的 Left/Right tag；teacher handedness 只能作为非监督审计字段，不能成为 Gold tag；
- 记录原 MediaPipe draft SHA，保证复现。

### 7.4 Strict import

配置新增：

```yaml
review:
  require_explicit_presence_decision: true
  require_explicit_handedness_decision: true
  handedness_policy: optional_per_row
```

非 ignore 图片必须恰好满足：

- 一个合法 21 点 skeleton；或
- 显式 `no_hand`。

二者同时出现或二者都没有均 fatal。不能沿用当前“无 skeleton/no_hand 只 warning 后变 negative”的行为。

Positive handedness：

- Left/Right 正常训练；
- 新增并要求显式 `unknown_handedness` tag；该行写 unknown 且 mask 0；
- positive 无 handedness tag或同时有多个 handedness tag均 fatal；
- 不能用 teacher handedness 填成 human Gold；
- Val/Test strict 规则不变。

### 7.5 多 task 批处理

人工只把每个 task 导出的 XML 放到 descriptor 指定的 `reviewed.xml`。batch importer 自动发现已配置 task、运行 05、聚合 source package。

task 缺失规则与 source optional 不同：一个 source 已启用且已经导出 N 个 task，则 N 个 task 必须全部回收或显式 disabled；不能静默导入半套 source。

### 7.6 HLMF Gold aggregate 输出

为 HLMF `finalize_train_finetune` 增加 `HAND_FINETUNE_ID` 路径参数，目标目录：

```text
${HAND_DATA_ROOT}/finetune/${HAND_FINETUNE_ID}/hmlf_gold_merged/
├── hmlf_gold_aggregate.json
├── 05_labels/
│   ├── hand_train_catalog_finetune.jsonl
│   ├── hand_training_labels_finetune.jsonl
│   └── hand_training_excluded_finetune.jsonl
└── qc/finalize_train_finetune_report.json
```

本次配置只登记 a/b/c/e 的 Gold 或 gold-only subset，不登记全量 pretrain pseudo。`hmlf_gold_aggregate.json` 是 aggregate 权威 descriptor，至少认证 source descriptor SHA 列表、catalog/included/excluded/report 的路径、行数和 SHA。d replay 由 HLML 单独产生，下一阶段再合并。

07A/wrapper 在发布 aggregate 前必须新增跨 Gold source 的 parent 冲突门禁，不能只依赖 source namespace 后的 `global_crop_id`：

1. 优先用非空 `parent_global_crop_id` 对齐；原生 source 没有 parent 时依次回退到 global ID、ROI 图像 SHA、归一化像素 SHA；
2. 同一身份若 presence、`ignore_for_training`、21 点或 handedness 监督不同，立即 fatal，并在临时报告中列出双方 descriptor/source/row；
3. 标签完全一致时只保留一行，按显式 source-role precedence、再按 source ID 排序确定 owner，另一行进入 `DUPLICATE_GOLD_SAME_LABEL` audit；
4. duplicate/conflict 检查在任何 included JSONL 原子发布之前完成；
5. aggregate descriptor 记录 identity 算法版本、duplicate 数和 conflict 数。

这样即使同一原 ROI 同时被 b/c 选中并获得不同的新 subset ID，也无法绕过 Gold 冲突门禁。上游 selector 仍应主动避免 b/c 重复选择，aggregate gate 是最后保险。

为避免人工逐项编辑 source YAML，新增专用 `configs/finalize_finetune.yaml` 或等价 wrapper：

```yaml
source_discovery:
  root: "${HAND_DATA_ROOT}/finetune/${HAND_FINETUNE_ID}/sources/gold"
  descriptor_name: finetune_source.json
  allowed_kinds: [external_gold, reviewed_hard_gold, disagreement_gold, new_recorded_gold]
  missing_optional: skip_and_report

cross_source_identity:
  keys: [parent_global_crop_id, global_crop_id, roi_image_sha256, normalized_pixel_sha256]
  conflicting_label: fail
  identical_label: keep_by_role_then_source_id
  role_precedence: [external_gold, reviewed_hard_gold, disagreement_gold, new_recorded_gold]

outputs:
  descriptor: "${HAND_DATA_ROOT}/finetune/${HAND_FINETUNE_ID}/hmlf_gold_merged/hmlf_gold_aggregate.json"
  labels_dir: "${HAND_DATA_ROOT}/finetune/${HAND_FINETUNE_ID}/hmlf_gold_merged/05_labels"
  qc_dir: "${HAND_DATA_ROOT}/finetune/${HAND_FINETUNE_ID}/hmlf_gold_merged/qc"
```

07A 可以直接支持 descriptor discovery，也可以由严格 wrapper 先生成只存在于临时目录的解析后 config；两种方式都必须把发现结果和 descriptor SHA 写入报告。不得要求人工维护候选 ID 或复制粘贴多份 source 条目。

HLMF Makefile 必须新增并导出：

```make
HAND_FINETUNE_ID ?= v2-finetune-r1
export HAND_FINETUNE_ID
FINALIZE_FINETUNE_CONFIG := configs/finalize_finetune.yaml

finalize_train_finetune:
	$(PYTHON) scripts/07A_finalize_training_labels.py \
	  --config $(FINALIZE_FINETUNE_CONFIG) --stage finetune
```

不能继续让该 target 读取现有 `configs/finalize_train.yaml`，否则会把全量 pretrain pseudo 和旧输出路径带进来。HLMF 路径变量统一使用 `HAND_DATA_ROOT`；`HAND_TRAIN_ROOT` 只属于 HLML。

### 7.7 HLMF 测试

- selection→manifest/draft/images 一一覆盖；
- parent identity；
- basename 冲突；
- 图片复制 SHA；
- no_hand/skeleton/ignore 四状态；
- 缺显式 presence fatal；
- duplicate skeleton fatal；
- unknown handedness Train-only；
- finetune 04 不预填 teacher handedness；
- positive handedness decision 必须显式；
- 04/05 round trip；
- task 缺失和可选 source 区分；
- batch report 原子发布。
- aggregate descriptor 覆盖 catalog/included/excluded/report/source descriptor SHA；
- 跨 Gold source 的 parent/global/hash 对齐、相同标签去重与冲突 fatal；
- `native_existing` 新录制 source 也必须走 finetune strict 04/05；
- `finalize_train_finetune` 使用专用 config，而非 pretrain source config。

## 8. P3：`finetune_source_v1` 契约

`finetune_source.json` 示例：

```json
{
  "schema_version": "finetune_source_v1",
  "source_id": "dragon_gold_0716_v1",
  "dataset_id": "dragon_gold_0716_v1",
  "source_kind": "external_gold",
  "source_mode": "gold_only",
  "producer": "hlmf_dragon_adapter",
  "producer_version": "<git sha>",
  "created_at": "<ISO-8601>",
  "parent_pretrain_id": null,
  "enabled_stages": ["finetune"],
  "supervision_tier": "gold",
  "handedness_policy": "unavailable",
  "input_sha256": {
    "annotations_hand": "...",
    "annotations_palm": "...",
    "readme": "..."
  },
  "artifacts": {
    "manifest": {"path": "02_roi_crops/hand_roi_crops_manifest.jsonl", "sha256": "..."},
    "crop_images": {"root": "02_roi_crops/images", "sha256_manifest": "qc/crop_images_sha256.jsonl", "aggregate_sha256": "..."},
    "source_images": {"sha256_manifest": "qc/source_images_sha256.jsonl", "manifest_sha256": "...", "aggregate_sha256": "...", "count": 3565},
    "gold_labels": {"path": "03_reviewed/hand_landmarks_reviewed.jsonl", "sha256": "...", "count": 5191},
    "ignored_sidecar": {"path": "03_reviewed/ignored.jsonl", "sha256": "...", "count": 2},
    "qc_report": {"path": "qc/dragon_import_report.json", "sha256": "..."}
  },
  "counts": {
    "manifest": 5191,
    "gold_labels": 5191,
    "included": 5189,
    "ignored": 2
  }
}
```

当前 Dragon source 的 `input_sha256.readme` 必须等于第 5.2 节固定 README SHA；`source_images_sha256.jsonl` 覆盖实际参与匹配/裁剪的 3,565 张原 JPEG，记录相对路径与原始字节 SHA。descriptor 直接认证该 manifest SHA 与顺序无关 aggregate SHA，不能只认证两份 TXT。

Replay 使用同一个 schema version，但采用不同的条件分支；示例：

```json
{
  "schema_version": "finetune_source_v1",
  "source_id": "pretrain_replay",
  "dataset_id": "pretrain_replay_v2_pretrain_r3",
  "source_kind": "pretrain_replay",
  "source_mode": "canonical_replay_index",
  "producer": "hlml_finetune_replay",
  "producer_version": "<git sha>",
  "created_at": "<ISO-8601>",
  "parent_pretrain_id": "v2-pretrain-r3",
  "enabled_stages": ["finetune"],
  "supervision_tier": "pseudo",
  "handedness_policy": "optional_per_row",
  "artifacts": {
    "canonical_labels": {"path": "05_labels/canonical_source.jsonl", "sha256": "...", "count": 10000},
    "parent_curation_manifest": {"path": "qc/parent_curation_manifest.json", "sha256": "..."},
    "qc_report": {"path": "qc/replay_report.json", "sha256": "..."}
  },
  "external_crop_roots": [
    {"root": "/root/autodl-tmp/TrainFab/HLML-2.0/train_sources", "read_only": true, "row_image_sha256_required": true}
  ],
  "counts": {"canonical_labels": 10000, "confirmed_negative": 1022}
}
```

生成器必须把 `external_crop_roots.root` 写成生成时解析后的绝对路径；`finetune_source.json` 不支持 `${...}` 环境插值。validator 对该绝对路径做 allowed-root/realpath 检查，并把解析值写入 gate report；数据根迁移后应重新生成 descriptor，不能在认证后的 JSON 中动态替换字符串。

通用 descriptor 校验：

- 相对路径只能位于 source root；
- 不接受 symlink；
- artifact SHA 必须存在；
- counts 与真实行数一致；
- source ID/dataset ID 全局唯一；
- `source_mode`、`handedness_policy` 与 source kind 合法；
- producer/version/created time/input SHA 可追溯；
- source kind 必须在允许集合：`external_gold`、`reviewed_hard_gold`、`disagreement_gold`、`pretrain_replay`、`new_recorded_gold`。

随后按 `source_mode/source_kind` 条件校验，不能用一套 Gold 必填字段拒绝 replay：

- Gold（`gold_only` 或 reviewed Gold）：必须无歧义映射为 HLMF 07A 的 manifest、crop_images_dir、全覆盖 `gold_labels` 和 QC；ignored sidecar 只能是派生证据；crop/source image hash manifest 与 aggregate SHA 完整；
- replay（`canonical_replay_index`）：禁止送入 HLMF 07A；必须提供 canonical labels、`supervision_tier=pseudo`、非空 `parent_pretrain_id`、已认证 parent curation manifest、只读 external crop roots 和逐行 image SHA；每个 negative 还必须带 pretrain 人工 review evidence；
- Gold descriptor 不得引用外部未认证 crop root；replay 不得伪造 `gold_labels`/HLMF manifest 来通过校验。

## 9. P3：HLML `finetune-curate`

### 9.1 新文件

- `hand_landmarker/finetune_curation.py`
- `scripts/curate_finetune.py`
- `scripts/check_finetune_sources.py`
- `configs/curate_finetune.yaml`
- Makefile `finetune-curate`、`check-finetune-sources`
- tests

P3 的 source gate 只检查数据包、身份、Gold/replay 合并、ignored 和泄漏，不承诺 training sampler 可行。最终 `check-finetune-data` 与 `inspect-finetune` 要等 P3.5 sampler 完成后在 P4 开放。

### 9.2 配置

```yaml
schema_version: 1
task: curate_finetune

gold_source_descriptor_root: "${HAND_TRAIN_ROOT}/finetune/${HAND_FINETUNE_ID}/sources/gold"
replay_source_descriptor_root: "${HAND_TRAIN_ROOT}/finetune/${HAND_FINETUNE_ID}/sources/replay"

gold_aggregate:
  descriptor: "${HAND_TRAIN_ROOT}/finetune/${HAND_FINETUNE_ID}/hmlf_gold_merged/hmlf_gold_aggregate.json"

allowed_crop_roots:
  - "${HAND_TRAIN_ROOT}/finetune/${HAND_FINETUNE_ID}"
  - "${HAND_TRAIN_ROOT}/train_sources"

sources:
  dragon_gold:
    discover_kind: external_gold
    enabled: auto
    required: false
    target_gold_weight: 0.60
  negative_removed_gold:
    discover_kind: reviewed_hard_gold
    enabled: auto
    required: false
    target_gold_weight: 0.20
  disagreement_gold:
    discover_kind: disagreement_gold
    enabled: auto
    required: false
    target_gold_weight: 0.15
  new_recorded_gold:
    discover_kind: new_recorded_gold
    enabled: auto
    required: false
    target_gold_weight: 0.05
  pretrain_replay:
    discover_kind: pretrain_replay
    descriptor_root: replay
    enabled: true
    required: true

gate:
  minimum_gold_positive: 256
  require_at_least_one_gold_source: true
  require_replay: true
  fail_on_val_test_leakage: true

output:
  dir: "${HAND_TRAIN_ROOT}/train_finetune_merged/${HAND_FINETUNE_ID}"
```

### 9.3 Optional-source 语义

- `enabled:false`：不读目录，报告 disabled；
- `enabled:auto` 且 descriptor 不存在：报告 absent_optional；
- descriptor 一旦存在：严格校验，任何错误 fatal；
- `required:true` 缺失：fatal；
- 不能以“来源可选”为由跳过一个已经存在但损坏的 source。

### 9.4 合并、去重与覆盖

按以下顺序：

1. 验证 `hmlf_gold_aggregate.json` 及其 catalog/included/excluded/report/source-descriptor SHA；
2. **只从 aggregate included 读取 Gold 训练行**；单源 descriptor 不再次提供标签行；
3. 从 aggregate catalog/excluded 中派生 `drop_ignore` 行到最终 ignored 输出，并验证 included/ignored 不重叠；
4. 从 `sources/replay` 的唯一 required replay descriptor 读取 d；
5. 检查 source/aggregate/replay identity 唯一且 crop path 位于 allowed roots；
6. 按 `parent_global_crop_id`、global ID、dataset+source crop ID、ROI SHA 做跨输入对齐；
7. Gold 与 replay 重合：保留 Gold，replay 记 `SUPERSEDED_BY_GOLD`；
8. HLMF aggregate 内两份 Gold 坐标冲突必须已经 fail；HLML 再复核冲突证据；
9. ignored 与 included identity 重合：fatal；
10. 与 Val/Test ID、source group、图片 SHA、归一化像素 SHA 做泄漏 gate；
11. 写最终 canonical/audit/report/hash manifest。

Negative evidence 必须按 supervision tier 分支，不能复用一条含糊规则：

- pseudo replay negative：必须来自 P0 admitted，带 `INCLUDE_CONFIRMED_NEGATIVE` decision，以及 reviewer、reviewed_at、review_method、review image SHA 和 parent curation manifest 认证；
- Gold `no_hand`：必须为 `supervision_tier=gold`、`annotation_provenance=human_gold`，由 strict CVAT 显式 `no_hand` 产生，并由 task/source descriptor SHA 认证；它不需要也不得伪造 pretrain decision 字段。

### 9.5 Source sampling weight

当前 sampler 可在 tier×sample type cell 内使用行级 `sampling_weight`。Curator 根据 aggregate 行的 `dataset_id` 关联单源 descriptor 的 role，按 `target_gold_weight`、实际 source/cell 数量计算行权重，使预计 Gold role draw 接近目标。

可选来源缺失时：

```text
effective_weight_i = configured_weight_i
                     / sum(configured_weight of present valid sources)
```

报告必须写配置权重、有效权重、每 cell 归一化系数和预计 draw。不能只修改行权重而不留下解释。

一个 role 的 `discover_kind` 必须允许匹配 0～N 个父 dataset descriptor。role 的有效权重先确定，再在该 role 的 N 个 source 之间按 `sqrt(included_count)` 分配；若提供显式子 source weight 则先乘该权重后归一化。不能把第二个同 kind source 判成冲突，也不能只读取第一个。

Dragon 还应在 source 内按 `source_sequence_id` 平衡，避免 30 个 clip 中最长序列支配；可把每条 row 基础权重设为该 clip 可用 ROI 数的倒数，再在 source/cell 内归一化。

### 9.6 输出

```text
train_finetune_merged/<HAND_FINETUNE_ID>/
├── 05_labels/
│   ├── hand_training_labels_finetune.jsonl
│   ├── hand_training_ignored_finetune.jsonl
│   └── hand_training_labels_finetune_smoke.jsonl
├── audit/
│   ├── source_catalog.jsonl
│   ├── selection_catalog.jsonl
│   ├── excluded_and_superseded.jsonl
│   └── finetune_smoke_selection.jsonl
└── qc/
    ├── curation_report.json
    └── sha256_manifest.json
```

manifest schema：`finetune_curation_v1`。训练 config 必须认证该 manifest，不能复用 pretrain curation manifest。

### 9.7 P3 测试

- optional missing source 通过；
- required missing source 失败；
- present broken optional source 失败；
- artifact path traversal/symlink/SHA/count；
- included/ignored 严格分离；
- unknown handedness mask；
- pseudo confirmed negative 与 Gold explicit-no-hand 两套证据分支；
- Gold supersedes replay；
- conflicting Gold fatal；
- identity/SHA/perceptual leakage；
- Val/Test overlap；
- source weight renormalization；
- Dragon clip balancing；
- output atomic/no overwrite；
- manifest authentication；
- 256-row smoke selection 的 required/optional cell、stable hash、source/sequence diversity 与 artifact SHA；
- repeat run deterministic。

## 10. P3.5：Tier-specific sampler

### 10.1 当前阻塞

`hand_landmarker/data.py.WeightedStratifiedSampler` 当前：

- 只接受统一 `sample_type_fractions`；
- 要求每个 active tier 都存在所有 fraction>0 的 sample type；
- finetune Gold 只有 positive、pseudo replay 含 negative 时会失败；
- 即使放宽存在性，当前 cross-bucket quota 也可能把 negative 分给 Gold tier。

不能只删除 `_validate_groups()` 的检查；必须重做 quota 计算语义。

### 10.2 新配置

```yaml
training:
  gold_fraction: 0.35

sampling:
  epoch_size: 12000
  quota_scope:
    supervision_tier: per_batch_half_up
    sample_type: per_epoch_largest_remainder
  batch_distribution: deterministic_balanced_deficit
  sample_type_fractions_by_tier:
    gold:
      POS_RUNTIME: 0.70
      POS_LOW_PALM: 0.20
      NEG_RUNTIME_CANDIDATE: 0.07
      NEG_LOW_PALM_CANDIDATE: 0.03
    pseudo:
      POS_RUNTIME: 0.72
      POS_LOW_PALM: 0.18
      NEG_RUNTIME_CANDIDATE: 0.06
      NEG_LOW_PALM_CANDIDATE: 0.04
  missing_cell_policy:
    gold: redistribute_within_tier
    pseudo: fail
  rare_cell_policy:
    gold: cap_fraction_then_redistribute_within_tier
    pseudo: fail
    max_average_draws_per_unique_record: 4.0
    max_expected_row_draws_per_epoch: 8.0
```

### 10.3 Quota 算法

必须新增 epoch-level tier×sample-type draw plan；仅对每个 batch 独立做 largest remainder 无法表达“整个 epoch 只抽 4 次稀有 Gold negative”。目标算法：

1. 根据 `epoch_size` 和 `batch_size` 列出全部 full/tail batch；
2. 每个 batch 只用 `gold_fraction` half-up 固定 Gold/pseudo 个数，因此 tier 数量仍逐 batch 严格守恒；
3. 汇总整个 epoch 的 Gold 总槽位和 pseudo 总槽位；
4. 在每个 tier 内处理 missing cell：`fail` 立即失败；`redistribute_within_tier` 把缺失 fraction 按其余允许 cell 比例重分并写报告；
5. 对每个 tier 的 epoch 总槽位按有效 sample-type fractions 做一次 largest remainder，得到精确整数 `epoch_draw_quota[tier,type]`；
6. 用这个整数 quota 和 cell 内真实行权重计算 average draws 与最大单行 expected draws。Gold rare cell 超限时，把该 cell 的整数 quota 降到同时满足两个上限的最大值，再将整数差额在同 tier 允许 positive cells 内 largest-remainder 重分；pseudo active cell 缺失或约束不可行时 fail；
7. 对每个 tier 生成长度等于其 epoch 总槽位的确定性 balanced-deficit type stream：每一步选择“相对于累计目标欠账最大且仍有剩余 quota”的 type，再按各 batch 的 tier 槽位切片。这样少量 negative draw 会跨 batch 均匀分散，而不是每 batch 固定为 0 或 1；
8. 在每个 cell 内按 row `sampling_weight` 和 epoch seed 抽样；
9. 校验每 batch 的 tier 总数、每个 tier×type 的 epoch 总数和全 epoch 总数全部精确守恒。

禁止将 Gold 缺失的 negative 配额转给 pseudo 后又改变 gold_fraction；tier fraction 和 tier 内 type fraction是两级独立契约。

旧字段 `require_all_tier_sample_type_cells` 在 finetune 路径中由 `missing_cell_policy` 取代；pretrain/multitask 路径继续保持旧的逐 batch 行为与严格存在性，不能被这次改动悄悄改变。

### 10.4 报告

新增：

```text
configured_fraction_by_tier
effective_fraction_by_tier
missing_cells
redistribution
epoch_draw_quota_by_tier_type
rare_cell_quota_cap
batch_tier_quota
batch_type_schedule_sha256
drawn_by_tier_type_source
unique_by_tier_type_source
expected_repetition
actual_repetition
```

### 10.5 Sampler 测试

- Gold positive-only + pseudo 四类；
- Gold no_hand negative 获得非零 draw；
- Gold 缺 POS_LOW 时重分；
- Gold rare negative cap 后回流到 Gold positive；
- 1 个 Gold negative、上限 4 时整个 epoch 精确抽 4 次并分散到不同 batch，而不是每批 0/1；
- pseudo 缺 active negative 时 fail；
- batch 64 和尾 batch；
- gold fraction 0.30/0.35/0.50；
- exact conservation；
- deterministic epoch/stream；
- source row weight；
- invalid/missing/negative/non-finite fraction；
- pretrain 旧行为回归不变。

## 11. P4：正式 finetune 训练路由

### 11.1 Makefile

复用 P2 已引入的两个 ID，并增加训练配置变量：

```make
HAND_PRETRAIN_ID := v2-pretrain-r3
HAND_FINETUNE_ID := v2-finetune-r1
export HAND_PRETRAIN_ID HAND_FINETUNE_ID

FINETUNE_CURATE_CONFIG := configs/curate_finetune.yaml
FINETUNE_CONFIG := configs/train_finetune.yaml
FINETUNE_SMOKE_CONFIG := configs/train_finetune_smoke.yaml
```

公开目标：

```text
prepare-finetune-sources
finetune-curate
check-finetune-sources
check-finetune-data
inspect-finetune
finetune-smoke
check-finetune-smoke
finetune-train
eval-val-finetune
eval-test-finetune
infer-finetune
export-finetune
conversion-data-finetune
```

不提供含义不完整的 `make finetune` 别名。

安全依赖链：

```make
finetune-smoke: check-finetune-data inspect-finetune
	$(PYTHON) -B scripts/train.py --config "$(FINETUNE_SMOKE_CONFIG)"
	$(PYTHON) -B scripts/check_finetune_smoke.py \
	  --smoke-config "$(FINETUNE_SMOKE_CONFIG)" \
	  --full-config "$(FINETUNE_CONFIG)"

check-finetune-smoke:
	$(PYTHON) -B scripts/check_finetune_smoke.py \
	  --smoke-config "$(FINETUNE_SMOKE_CONFIG)" \
	  --full-config "$(FINETUNE_CONFIG)"

finetune-train: check-finetune-data inspect-finetune check-finetune-smoke
	$(PYTHON) -B scripts/train.py --config "$(FINETUNE_CONFIG)"
```

这样首次运行 `finetune-smoke` 训练并检查；正式训练只复查已经存在的 smoke，不会尝试覆盖重跑 smoke。checker 同时绑定当前 full config；full config、checkpoint 或数据 manifest 变化后，旧 smoke 必须失败并要求新的 finetune ID/smoke run。同步更新 `make help` 和 `make paths`；`paths` 必须打印 PRETRAIN_ID、FINETUNE_ID、finetune inbox、curated root 和 run root。

### 11.2 `configs/train_finetune.yaml`

使用完整显式 config，不直接 extends multitask，以免继承 `multitask_gate` 或错误路径。

目标完整 schema 如下；实现时应从当前可加载的 geometry config 复制公共字段并通过 config-loader 单测，不能只保留训练超参数：

```yaml
schema_version: 1
task: train
stage: finetune

experiment:
  name: hand_landmarker_v2_finetune
  seed: 20260716

environment:
  python: ">=3.8,<3.9"
  tensorflow: "==2.9.0"
  cuda: "11.2"
  cudnn_major: 8
  require_gpu: true

data:
  data_root: "${HAND_TRAIN_ROOT}/train_finetune_merged/${HAND_FINETUNE_ID}"
  labels: "${HAND_TRAIN_ROOT}/train_finetune_merged/${HAND_FINETUNE_ID}/05_labels/hand_training_labels_finetune.jsonl"
  curation_manifest: "${HAND_TRAIN_ROOT}/train_finetune_merged/${HAND_FINETUNE_ID}/qc/sha256_manifest.json"
  require_curation_schema: finetune_curation_v1
  require_schema_version: train_finalize_v1
  require_training_stage: finetune
  crop_path_key: crop_path
  path_policy: canonical_crop_path_only
  allowed_crop_roots:
    - "${HAND_TRAIN_ROOT}/finetune/${HAND_FINETUNE_ID}"
    - "${HAND_TRAIN_ROOT}/train_sources"
  crop_image_roots:
    - "${HAND_TRAIN_ROOT}/finetune/${HAND_FINETUNE_ID}"
    - "${HAND_TRAIN_ROOT}/train_sources"
  image_size: [256, 256]
  channels: 1
  color_mode: grayscale
  input_layout: NCHW
  input_dtype: float32
  input_scale: 0.00392156862745098
  input_offset: 0.0
  cache: false

model:
  version: v2
  checkpoint_stage: finetune
  input_shape: [1, 256, 256]
  input_layout: NCHW
  num_iterations: [2, 2, 3, 4, 4, 6, 6]
  output_order: [landmarks, hand_flag, handedness]
  output_sizes: {landmarks: 42, hand_flag: 1, handedness: 1}

targets:
  num_landmarks: 21
  landmark_field: landmarks_crop_norm
  landmark_space: normalized_crop_xy
  landmark_order: id_0_to_20_interleaved_xy
  presence_field: hand_presence.present
  handedness_field: handedness.label
  handedness_encoding: {Left: 0, Right: 1, unknown: null}

training:
  epochs: 40
  batch_size: 64
  steps_per_epoch: null
  initial_checkpoint: "${HAND_TRAIN_ROOT}/hand_landmarker_runs/${HAND_PRETRAIN_ID}/multitask/checkpoints/best.weights.h5"
  resume_checkpoint: null
  gold_fraction: 0.35
  gradient_clip_norm: 5.0
  mixed_precision: false
  optimizer:
    name: adam
    learning_rate: 0.00001
    beta_1: 0.9
    beta_2: 0.999
    epsilon: 1.0e-7
  checkpoint: {monitor: val_landmark_mae, mode: min}
  learning_rate_schedule:
    name: reduce_on_plateau
    monitor: val_landmark_mae
    mode: min
    factor: 0.5
    patience: 3
    min_learning_rate: 1.0e-7
  early_stopping:
    enabled: true
    monitor: val_landmark_mae
    mode: min
    patience: 8
    restore_best_weights: true

sampling:
  enabled: true
  stratify_by: [supervision_tier, sample_type]
  tier_key: supervision_tier
  sample_type_key: sample_type
  bucket_key: sampling_bucket
  weight_key: sampling_weight
  epoch_size: 12000
  quota_scope:
    supervision_tier: per_batch_half_up
    sample_type: per_epoch_largest_remainder
  batch_distribution: deterministic_balanced_deficit
  sample_type_fractions_by_tier:
    gold:
      POS_RUNTIME: 0.70
      POS_LOW_PALM: 0.20
      NEG_RUNTIME_CANDIDATE: 0.07
      NEG_LOW_PALM_CANDIDATE: 0.03
    pseudo:
      POS_RUNTIME: 0.72
      POS_LOW_PALM: 0.18
      NEG_RUNTIME_CANDIDATE: 0.06
      NEG_LOW_PALM_CANDIDATE: 0.04
  missing_cell_policy:
    gold: redistribute_within_tier
    pseudo: fail
  rare_cell_policy:
    gold: cap_fraction_then_redistribute_within_tier
    pseudo: fail
    max_average_draws_per_unique_record: 4.0
    max_expected_row_draws_per_epoch: 8.0
  quota_rounding: largest_remainder
  quota_tie_break: [POS_RUNTIME, POS_LOW_PALM, NEG_RUNTIME_CANDIDATE, NEG_LOW_PALM_CANDIDATE]
  replacement: true
  honor_record_sampling_weight: true

losses:
  landmarks: {name: huber, delta: 0.05, coefficient: 20.0}
  hand_flag: {name: binary_crossentropy, from_logits: false, coefficient: 0.25}
  handedness: {name: binary_crossentropy, from_logits: false, coefficient: 0.02}
  honor_record_loss_weights: true

augmentation:
  enabled: true
  brightness_delta: 0.08
  contrast_range: [0.90, 1.10]
  gaussian_noise_stddev: 0.005
  rotation_degrees: 5.0
  scale_range: [0.97, 1.03]
  translation_fraction: 0.02
  horizontal_flip_probability: 0.0

validation:
  enabled: true
  data_root: "${HAND_TRAIN_ROOT}/eval_sources"
  labels: "${HAND_TRAIN_ROOT}/val_merged/05_labels/hand_validation_labels.jsonl"
  ignored_labels: "${HAND_TRAIN_ROOT}/val_merged/05_labels/hand_val_ignored.jsonl"
  crop_path_key: crop_path
  path_policy: canonical_crop_path_only
  allowed_crop_roots:
    - "${HAND_TRAIN_ROOT}/eval_sources"
  crop_image_roots:
    - "${HAND_TRAIN_ROOT}/eval_sources"
  batch_size: 128
  every_epochs: 1

inspection:
  compare_datasets:
    test:
      data_root: "${HAND_TRAIN_ROOT}/eval_sources"
      labels: "${HAND_TRAIN_ROOT}/test_merged/05_labels/hand_test_labels.jsonl"
      ignored_labels: "${HAND_TRAIN_ROOT}/test_merged/05_labels/hand_test_ignored.jsonl"
      crop_image_roots:
        - "${HAND_TRAIN_ROOT}/eval_sources/peak_test/02_roi_crops/images"
        - "${HAND_TRAIN_ROOT}/eval_sources/soar_test/02_roi_crops/images"
      allowed_crop_roots:
        - "${HAND_TRAIN_ROOT}/eval_sources"
      crop_path_key: crop_path
      path_policy: canonical_crop_path_only
      require_schema_version: evaluation_gold_v1
      require_split: test
      image_size: [256, 256]
      channels: 1

outputs:
  run_dir: "${HAND_TRAIN_ROOT}/hand_landmarker_runs/${HAND_FINETUNE_ID}/finetune"
  overwrite: false

runtime:
  gpu_memory_growth: true
  deterministic: false
  fail_on_nan: true
```

`_validation_dataset_config()` 与 `inspect_config()` 不能继续复制 train `data.allowed_crop_roots` 后只替换 labels。实现时必须让 validation/test 自己的 `data_root/allowed_crop_roots/crop_image_roots` 完整覆盖 train roots；训练 fallback 只索引显式 finetune/train_sources，Val/Test fallback 只索引 eval_sources，绝不递归扫描整个 HLML-2.0。所有 `crop_path` 先解析环境变量和相对路径，再用 `resolve(strict=True)`/等价真实路径做 containment 与 symlink 检查，不能只约束 fallback root。

使用 multitask best 是主路径。若 multitask Val 明显损伤 landmarks，可以建立一个明确的 geometry-init 对照 finetune ID，但不能在同一 run 中自动切换起点。

### 11.3 Loss

建议起点：

```yaml
losses:
  landmarks: {name: huber, delta: 0.05, coefficient: 20.0}
  hand_flag: {name: binary_crossentropy, from_logits: false, coefficient: 0.25}
  handedness: {name: binary_crossentropy, from_logits: false, coefficient: 0.02}
```

记录级 mask 决定实际 head：Dragon unknown handedness 自动为 0；confirmed negative landmark/handedness 为 0；pseudo/其他 Gold 按自身能力训练。

### 11.4 Finetune smoke

不能复用当前只接受 pretrain positive-only 的 `check_pretrain_smoke.py`。P3 curator 必须持久化：

```text
train_finetune_merged/<HAND_FINETUNE_ID>/05_labels/hand_training_labels_finetune_smoke.jsonl
train_finetune_merged/<HAND_FINETUNE_ID>/audit/finetune_smoke_selection.jsonl
```

主 `finetune_curation_v1` manifest 增加两份 artifact 的 path/count/SHA 与 selection config SHA。smoke 固定 256 行：80 Gold positive、最多 16 Gold explicit-no-hand（shortfall 回填 Gold positive）、96 pseudo positive、32 `NEG_RUNTIME_CANDIDATE`、32 `NEG_LOW_PALM_CANDIDATE`。在每个 cell 内按 source/sequence 去近重复后稳定 hash 抽样。Gold negative 是 optional；pseudo 两类 negative、Gold positive 和 pseudo positive 是 required。

unknown-handedness 规则必须兼容 optional source：若 snapshot 存在 positive unknown，选择器至少选入 1 行，checker 运行时验证 handedness sample weight/mask=0；若不存在，报告 `unknown_handedness_runtime_check=not_applicable`，由 canonical/unit tests 继续覆盖该 mask 逻辑。已知 Left/Right 必须各至少 1 行，否则 handedness smoke 不可用并 fail。

`configs/train_finetune_smoke.yaml` 从 full config extends，只把 `data.labels` 指向上述 persisted JSONL。允许的 smoke-only overrides 仅限：experiment name、run dir、epochs、batch/epoch size、关闭 augmentation/validation，以及把 checkpoint、early stopping、learning-rate schedule 统一改用训练期 `loss` monitor（或显式禁用 LR schedule）。关闭 validation 时绝不能继承 full 的 `val_landmark_mae` monitor。model interface/`num_iterations`、targets、losses、input contract、optimizer、initial checkpoint 和 curation manifest 不得改变；checker 要逐字段认证所有 monitor override。

`scripts/check_finetune_smoke.py --smoke-config ... --full-config ...` 必须比较并认证：

- resolved model/interface/depth、targets、loss、input contract；
- full 与 smoke 的 initial checkpoint path+SHA；
- finetune curation manifest path+SHA；
- full labels、smoke labels、selection artifact path+SHA；
- smoke training report、resolved config、git state、history 和 best checkpoint SHA；
- 所有允许 override 的逐字段 diff；出现任何未允许 diff 即 fatal。

best checkpoint 对 256 行做无增强、顺序、全覆盖推理，固定 gate：

```yaml
smoke_gate:
  expected_records: 256
  maximum_mean_landmark_mae: 0.02
  maximum_p90_landmark_mae: 0.04
  maximum_max_landmark_mae: 0.10
  maximum_hand_flag_bce: 0.08
  minimum_hand_flag_accuracy: 0.98
  maximum_handedness_bce: 0.15
  minimum_handedness_accuracy: 0.95
```

landmark 只统计 positive+landmark mask，hand flag 统计全部 positive/negative，handedness 只统计已知 Left/Right+mask。任何 NaN、漏行、required cell 或 epoch plan 中 `effective quota>0` 的 cell 从未被抽到、mask 不符或阈值失败都拒绝 full train；optional absent/redistributed Gold-negative cell 报告 `not_applicable/redistributed` 而不失败。不保留“小集过拟合或 loss 下降”这种二选一主观门禁。

新增：

- `configs/train_finetune_smoke.yaml`
- `scripts/check_finetune_smoke.py`
- Make target `finetune-smoke`
- persisted smoke selection/manifest tests

### 11.5 Eval / infer / export

新增或参数化：

- `configs/eval_val_finetune.yaml`
- `configs/eval_test_finetune.yaml`
- `configs/infer_finetune.yaml`
- `configs/export_finetune.yaml`

每个配置必须：

- `model.checkpoint_stage=finetune`；
- checkpoint 路径位于 `${HAND_FINETUNE_ID}/finetune`；
- 输出路径使用 `${HAND_FINETUNE_ID}/.../finetune`；
- provenance guard 不同时匹配 pretrain 和 finetune token；
- export calibration source 使用 `configs/train_finetune.yaml`；
- ONNX contract 记录 finetune data manifest、initial multitask SHA、best SHA。

当前通用 eval/infer/export 依赖 `HAND_PRETRAIN_PHASE`；实现时要抽象为明确的 model stage/experiment ID，或增加 finetune 专用配置。不能把 finetune checkpoint 放在：

```text
hand_landmarker_runs/v2-pretrain-r3/finetune/
```

正确路径：

```text
hand_landmarker_runs/v2-finetune-r1/finetune/
```

### 11.6 Stage route tests

当前 `tests/test_stage_routes.py` 强制 configs 只含 9 个 pretrain 文件并禁止 finetune。更新为：

- pretrain config 集合仍完整；
- finetune config 集合明确列出；
- 每个 Make target 路由正确；
- pretrain target 不能读 finetune ID；
- finetune target 同时读取 PRETRAIN_ID（起点）和 FINETUNE_ID（输出）；
- validation/test dataset builder 用各自 eval allowed roots 覆盖 train roots，并在 resolved path 上阻止越界/symlink；
- eval/export stage provenance；
- conversion data source stage 一致。

## 12. Finetune gate 详细检查

`scripts/check_finetune_data.py` 至少检查：

### 12.1 数据与来源

- finetune curation manifest schema/SHA；
- labels SHA、逐图 SHA、aggregate SHA；
- 至少一个 Gold source和 pseudo replay；
- Gold positive ≥ config minimum；
- source descriptor/producer/input SHA；
- optional/disabled/present 状态；
- included/ignored/excluded 计数守恒；
- persisted finetune smoke labels/selection 的 256 行、required/optional cell 与 manifest SHA；
- Dragon expected heads/masks；
- pseudo replay negative 的 P0 `INCLUDE_CONFIRMED_NEGATIVE` decision 与 reviewer/time/method/SHA；
- Gold negative 的 strict CVAT explicit `no_hand`、human_gold provenance 与 source/task descriptor SHA；
- 不存在未复核 negative candidate。

### 12.2 身份与泄漏

- global/source/parent ID；
- Gold/pseudo override 结果；
- conflicting Gold；
- crop path 位于允许根；
- Train vs Val/Test：ID、source group、图片 SHA、normalized-pixel SHA；
- session ID 可用时执行 session leakage。

### 12.3 Sampler 可行性

- gold fraction 0.30～0.50；
- tier-specific fractions 合法；
- missing-cell policy；
- batch quota 可行；
- epoch size 为 batch multiple 或尾 batch策略明确；
- 每 cell/source 预计 draw 与重复率；
- source weight 目标和有效值。

### 12.4 Checkpoint

- multitask best 存在；
- experiment/training report complete；
- checkpoint SHA 与 state/report 一致；
- checkpoint model version v2；
- initial/resume 互斥；
- run dir 空且不与 pretrain 冲突。

## 13. 配置与报告的人工接口

人工只能被要求修改以下少量字段：

```text
HAND_PRETRAIN_ID
HAND_FINETUNE_ID
configs/prepare_finetune_sources.yaml :: selection.<role>.enabled
configs/prepare_finetune_sources.yaml :: selection.<role>.max_items
configs/prepare_finetune_sources.yaml :: selection.<role>.per_dataset_max
configs/prepare_finetune_sources.yaml :: selection.pretrain_replay.max_records
configs/curate_finetune.yaml :: sources.<gold_role>.target_gold_weight
configs/train_finetune.yaml :: training.gold_fraction
configs/train_finetune.yaml :: sampling.epoch_size
```

所有比例、实际配额、重分配、shortfall、selected IDs 和 SHA 由程序写报告。禁止把人工写 TXT/CSV 设计成正常流程的一部分。

## 14. 实施顺序与每阶段完成标准

### P0：解锁安全 review finalize

包含：

- negative_reviewed/removed/quarantine；
- 事务和清理；
- gate；
- tests/docs。

完成标准：

- 本地 1049 张导入；
- 1022 admitted、27 quarantine、47594 removed；
- candidates 只在成功后清理；
- gate pass；
- `make test` 通过。

P0 完成后允许运行 `pretrain-curate-reviewed`，但还不能启动正式 multitask。

### P0.5：解锁安全 multitask

包含：

- 精确 batch quota 重复率分析；
- average-cell 与 max-expected-row 两级保护；
- 自动 epoch size；
- gate/training report；
- tests/docs。

完成标准：当前数据解析为约 6,400 draw/epoch，Runtime negative 精确平均约 3.90625 次；P0 与 P0.5 都完成后才能启动 multitask。

### P1：Dragon Gold

可在 P0.5 完成、multitask 训练进行时并行开发。

完成标准：

- 5191 matched；
- 5189 included、2 ignored；
- EXIF/ROI overlay 正确；
- unknown handedness mask0；
- source package/descriptor/SHA；
- gold_only + finetune-only finalizer；
- tests。

### P2：b/c/d 自动化

完成标准：

- 300 removed + 300 disagreement 预算可配置；
- 无人工 TXT/CSV；
- HLMF 04 task 自动生成；
- strict 05；
- replay ≤10,000 且含 confirmed negative；
- parent ID/SHA/selection report；
- tests。

### P3：Finetune curation

完成标准：

- source descriptor；
- optional missing / present strict；
- Gold override/replay；
- ignored 隔离；
- Val/Test leakage；
- source weights；
- finetune manifest；
- deterministic 256-row smoke labels/selection artifact；
- source/data gate；
- tests。

### P3.5：Tier-specific sampler

完成标准：

- Gold positive/no_hand 与 pseudo replay 可使用不同 type fractions；
- missing/rare Gold cell 受控重分；
- pseudo required cell fail-closed；
- exact batch quota 和 repetition report；
- pretrain sampler 回归不变；
- tests。

### P4：训练和交付路由

完成标准：

- 最终 `check-finetune-data` 和 `inspect-finetune`；
- finetune config、256-row authenticated smoke、full-config-bound smoke gate 与 train；
- independent ID/run；
- eval/infer/export/conversion；
- stage routes/provenance；
- README、最高级流程文档和简易操作手册中的命令全部对应真实 Make target；
- `make compile && make test` 通过。

## 15. 不允许的捷径

- 未更新代码就运行当前 `pretrain-curate-reviewed`；
- 将 `negative_removed` 全部直接当 positive；
- 将 27 个 overlap conflict 混入 removed；
- 清理 candidates 后不保留 removed manifest；
- 用 `tools/downsample.py` 复制裸图片作为 b/c 正式输入；
- b/c 重新跑 teacher 并覆盖原 draft；
- 把 Dragon p=0 当 Gold negative；
- 忽略 Dragon EXIF；
- clamp Dragon 越界 Gold 点；
- 推断/伪造 Dragon handedness；
- 把 Palm score 0.5 sentinel 描述成真实置信度；
- 伪造 pseudo 以绕过 Gold-only source；
- 可选 source 存在错误时静默跳过；
- 手工拼接 JSONL 或手写候选 ID；
- Gold 与 replay 冲突时保留两份；
- 用 positive-only Val presence accuracy 选择 hand flag；
- 复用 pretrain ID 作为 finetune ID；
- 用 last 代替 Val best；
- 根据 Test 反向调参。

## 16. 后续实现交接清单

每个实现任务开始前读取：

1. `docs/training_system/end_to_end_training_workflow_v1_0.md`；
2. 本文；
3. 当前 HLML Makefile/config/tests；
4. HLMF README、finalization/04/05、Dragon README；
5. 服务器当前 ID 对应的 report 和 SHA。

每个 PR/提交必须说明：

- 完成哪个 P 阶段；
- 新增/修改的数据 schema；
- 向后兼容范围；
- 新增 tests；
- 人工操作是否增加；
- 真实数据 dry-run 报告路径；
- 是否修改任何冻结 Train/Val/Test 文件。

在 P0～P4 全部完成前，最高级流程与简易操作手册中标记为 `[待实现]`/`[实现后]` 的命令不得写成当前已可执行能力；每完成一个阶段，两份文档必须同步更新状态。
