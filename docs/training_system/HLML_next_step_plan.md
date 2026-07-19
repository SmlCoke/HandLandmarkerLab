# HLML 最后一天半：Finetune 对照与交付计划

更新时间：2026-07-19。本计划建立在以下冻结状态上：pretrain 已结束；finetune 数据快照 `v3-finetune-final-r1` 已冻结；不再增加标注、不修改 Gold/replay、不重新 pretrain，也不修改模型代码和通用训练配置。接下来只在同一个数据快照上改变 Gold/pseudo 的真实 Loss 倍率和 `FINETUNE_PROFILE`，最后锁定一个模型完成 Test、导出、厂商转换和上板。

通用原理与长期操作方法见 [HLML 完整训练流程](HLML_training_workflow.md) 和 [HLML 快速手册](HLML_quick_start.md)。本文只记录当前结果、当前候选及最后一天半的具体执行顺序。

## 1. 当前冻结基线

```text
HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
HAND_TRAIN_ROOT=/root/autodl-tmp/TrainFab/HLML-3.0
HAND_PRETRAIN_ID=v3-pretrain-r1
HAND_FINETUNE_ID=v3-finetune-final-r1
baseline experiment=v3-finetune-final-dataonly-r2
baseline profile=data_only
baseline Gold/pseudo Loss multiplier=1.0/1.0
```

本次 finetune 快照共有 11,467 条记录：Gold 1,496 条、mandatory pseudo replay 9,971 条。所有候选必须复用这个数据 ID；不得重新运行 Gold selection、finetune curate 或 finalize，不得修改其 JSONL、manifest 或来源目录。候选只更换 `FINETUNE_EXPERIMENT_ID`。

服务器当前约有 57 GB 数据盘余量。已完成的 r2 实测约耗时：smoke 12 分钟、正式 40 epoch 37 分钟。因此一个新候选完成 smoke、正式训练和 Val 约需 50～55 分钟；不要并行占用同一块 3090。

服务器没有 `tmux`，但有 `screen`。长任务建议从交互式 SSH 中运行：

```bash
screen -S hlml-final
# 在 screen 中执行本文命令。
# Ctrl+A，随后按 D：脱离但不终止任务。
screen -r hlml-final
```

## 2. r2 结果判断

### 2.1 训练和导出链路

r2 正式训练完整跑完 40 epoch，`val_landmark_mae` 的 best 位于 epoch 39：

```text
train landmark_mae（最后）：0.01392
best val_landmark_mae：       0.05398
best checkpoint SHA-256：     39e52148a0b0b8f6682943e5758e8be4bc15451b5eda6dc5c9fbffe8cdd62dbb
```

训练误差持续下降而 Val 只缓慢改善，说明仍有明显泛化间隙；继续在同一配置上增加 epoch 不是当前优先项。

导出链路正常：ONNX 约 7.37 MiB、opset 11，A1 算子审计 `unsupported=[]`，Keras/ONNX 数值误差通过门槛，重参数化校验通过，转换数据为 100 条 calibration + 50 条 evaluation。导出正确不等于精度足够，但当前没有导出格式或算子问题。

### 2.2 同一 Val 上的量化比较

下表中的 geometry、multitask 和 r2 使用相同 Val labels SHA-256 `be7074...9062f`，可以直接比较：

| 模型 | mean px | median px | P90 px | P95 px | PCK@0.10 | presence recall | handedness acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry | 20.73 | 18.21 | 37.32 | 43.21 | 0.334 | 1.000 | 0.456 |
| multitask | 22.01 | 18.38 | 39.18 | 47.07 | 0.302 | 1.000 | 0.684 |
| data_only r2 | 21.48 | 17.40 | 44.67 | 49.44 | 0.359 | 0.929 | 0.875 |

结论：

- r2 相比 multitask 的 mean、median、PCK 和 handedness 有改善，符合“部分握拳不再严重塌缩”的人工观察。
- r2 的 P90/P95 反而变差；相对 geometry，r2 的 median/PCK 更好，但 mean 和长尾更差。模型改善的是一部分常见样本，没有解决困难姿态长尾。
- Val 的 1,226 条记录全是 positive，所以 presence 的 `accuracy/recall=0.929` 只说明有 87 个正 ROI 被 hand flag 拒绝；它不能衡量 false positive。不能根据该 Val 的 `fp=0` 推断背景误检为零。
- r2 在 `peak_vals_shared_v1` 上尤其困难：mean 25.79 px、P90 49.15 px、presence recall 0.850；在 `soar_vals_shared_v1` 上则为 mean 14.30 px、P90 24.39 px。后续必须看分来源结果，不能只看全局均值。

### 2.3 “塌缩”量化与独立 infer

用与 `spread_ratio_loss` 相同的 wrist-relative RMS 定义，计算“预测 spread / Gold spread”：

| 模型 | mean ratio | median ratio | ratio < 0.5 的 ROI |
|---|---:|---:|---:|
| geometry | 0.843 | 0.901 | 146 / 1226 |
| multitask | 0.769 | 0.826 | 204 / 1226 |
| data_only r2 | 0.831 | 0.921 | 215 / 1226 |

r2 的 mean/median ratio 高于 multitask，说明典型预测更展开；但 `ratio < 0.5` 的严重长尾没有减少，反而由 204 增至 215，且仍明显多于 geometry。以 `ratio < 0.6` 统计，r2 的 281 个 ROI 平均误差约 35.59 px，其他 ROI 约 17.28 px，说明塌缩确实是当前长尾误差的重要来源。

独立 infer 有 307 张非 notebook 缓存图、529 个相同 Palm ROI。r2 相对 multitask：

- 484 / 529 个 ROI 的预测骨架包围面积变大；
- 平均包围面积从 0.0445 增至 0.0731；
- 面积小于 0.05 的比例从 68.1% 降至 31.4%；
- 部分张手、握拳姿态明显展开，但数字 1、侧向张手和遮挡姿态中仍有聚团或错误展开；面积变大本身不保证关键点正确。

r2 有 5 个 Palm ROI 的 hand flag 低于 0.5。逐图确认后，这 5 个 ROI 都是 Palm Detector 错落在胸腹部的假框，r2 的拒绝是正确的；multitask/geometry 会错误接受它们。部分帧的两只真实手完全没有 Palm ROI，这是上游 Palm Detector 漏检，不能通过本轮 Hand Landmarker finetune 修复。

当前评估目录：

```text
/root/autodl-tmp/TrainFab/HLML-3.0/hand_landmarker_runs/v3-finetune-final-dataonly-r2/
/root/autodl-tmp/TrainFab/HLML-3.0/hand_landmarker_inference/v3-finetune-final-dataonly-r2/
```

## 3. 两类实验分别回答什么问题

### 3.1 Gold/pseudo Loss 倍率

本轮不改变 `training.gold_fraction=0.35`，因此每个 batch 的 Gold 抽样数量不变。只改变：

```text
FINETUNE_GOLD_LOSS_WEIGHT=<g>
FINETUNE_PSEUDO_LOSS_WEIGHT=1.0
```

它改变 Gold/pseudo 样本进入三个基础 head Loss 的有效 sample weight。基于 r2 固定的 epoch-0 采样计划，预计 Gold 权重质量如下：

| g | 名义 Gold 质量 | landmarks Gold 质量 | hand flag Gold 质量 | handedness Gold 质量 |
|---:|---:|---:|---:|---:|
| 1.0 | 35.00% | 44.64% | 43.33% | 35.80% |
| 1.5 | 44.68% | 54.74% | 53.43% | 45.55% |
| 2.0 | 51.85% | 61.72% | 60.47% | 52.72% |
| 3.0 | 61.76% | 70.75% | 69.64% | 62.59% |
| 4.0 | 68.29% | 76.33% | 75.36% | 69.05% |
| 5.0 | 72.92% | 80.13% | 79.27% | 73.60% |

这些是 epoch-0 权重质量，不是真实梯度百分比；真实梯度还由预测误差决定。`g=5` 已经让 landmarks 约 80% 的有效权重质量来自 1,496 条 Gold，存在对少量 Gold 过拟合和遗忘 replay 场景的明显风险，所以 4/5 是压力测试，不是默认推荐值。

### 3.2 `FINETUNE_PROFILE=structure`

`structure` 保持相同模型、数据、抽样、augmentation 和基础 landmark Huber，额外启用：

```text
bone_vector coefficient=5.0
spread_ratio coefficient=1.0
```

- `bone_vector` 比较 20 条真实手骨连接的预测向量与该 Gold ROI 的人工目标向量，不使用固定骨长或固定姿态模板。
- `spread_ratio` 比较预测/Gold 的 wrist-relative 整体 spread 比例，不会简单强迫所有拳头“张大”。
- 两项结构 Loss 只对 `presence=true` 且 landmarks 有效的人工 Gold 生效；pseudo、negative 和 ignored 的 structure mask 都是 0。
- 增大 Gold tier multiplier 会提高 Gold 相对 pseudo 的基础 landmark/hand flag/handedness 权重，但 structure 内部只有 Gold，统一倍率在加权平均中会抵消；因此 `Gold=2 + structure` 不是把结构系数直接翻倍。

structure 与提高 Gold 权重是两条不同轴：前者强调 Gold 内部的形状关系，后者改变 Gold 与 replay 的基础监督竞争。必须先分别测试，再决定是否组合。

## 4. 候选 ID 与推荐优先级

所有 ID 都必须是新目录：

| 优先级 | experiment ID | profile | Gold/pseudo Loss | 用途 |
|---:|---|---|---:|---|
| baseline | `v3-finetune-final-dataonly-r2` | data_only | 1.0 / 1.0 | 已完成 |
| P0 | `v3-finetune-final-dataonly-g20-r1` | data_only | 2.0 / 1.0 | 先检查适度 Gold 增强 |
| P0 | `v3-finetune-final-dataonly-g40-r1` | data_only | 4.0 / 1.0 | 快速探测高 Gold 区间是否仍获益 |
| P0 | `v3-finetune-final-structure-g10-r1` | structure | 1.0 / 1.0 | 与 r2 隔离比较 structure 本身 |
| P1 | `v3-finetune-final-dataonly-g15-r1` | data_only | 1.5 / 1.0 | 若 g2 偏强或用于完整曲线 |
| P1 | `v3-finetune-final-dataonly-g30-r1` | data_only | 3.0 / 1.0 | 填补 g2/g4 中间点 |
| P1 | `v3-finetune-final-dataonly-g50-r1` | data_only | 5.0 / 1.0 | 仅作高权重压力测试 |
| P2 | `v3-finetune-final-structure-gbest-r1` | structure | best / 1.0 | structure 有效时才与最佳 g 组合 |

一天半内的推荐顺序：先完成三个 P0，约 2.5～3 小时；GPU 继续运行 P1 时，人工并行写说明文档/PPT。只有 structure-g10 确实减少 Val 长尾或塌缩，才运行 P2。不要进入 `structure_roi_aug`，因为它同时改变 ROI augmentation，会引入第三个变量并增加解释成本。

## 5. 每个候选的标准运行函数

每次登录：

```bash
cd /root/HandLandmarkerLab
conda activate hand-landmarker-tf29

export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_TRAIN_ROOT=/root/autodl-tmp/TrainFab/HLML-3.0
export HAND_PRETRAIN_ID=v3-pretrain-r1
export HAND_FINETUNE_ID=v3-finetune-final-r1
```

在当前 Bash 中定义函数：

```bash
run_finetune_candidate() {
  local experiment_id="$1"
  local profile="$2"
  local gold_weight="$3"

  echo "===== ${experiment_id}: smoke ====="
  make finetune-smoke \
    HAND_FINETUNE_ID=v3-finetune-final-r1 \
    FINETUNE_EXPERIMENT_ID="${experiment_id}" \
    FINETUNE_PROFILE="${profile}" \
    FINETUNE_GOLD_LOSS_WEIGHT="${gold_weight}" \
    FINETUNE_PSEUDO_LOSS_WEIGHT=1.0 || return 1

  echo "===== ${experiment_id}: formal train ====="
  make finetune-train \
    HAND_FINETUNE_ID=v3-finetune-final-r1 \
    FINETUNE_EXPERIMENT_ID="${experiment_id}" \
    FINETUNE_PROFILE="${profile}" \
    FINETUNE_GOLD_LOSS_WEIGHT="${gold_weight}" \
    FINETUNE_PSEUDO_LOSS_WEIGHT=1.0 || return 1

  echo "===== ${experiment_id}: Val ====="
  make eval-val-finetune \
    HAND_FINETUNE_ID=v3-finetune-final-r1 \
    FINETUNE_EXPERIMENT_ID="${experiment_id}" \
    FINETUNE_PROFILE="${profile}" \
    FINETUNE_GOLD_LOSS_WEIGHT="${gold_weight}" \
    FINETUNE_PSEUDO_LOSS_WEIGHT=1.0
}
```

函数在任一步失败后停止该候选，不会错误进入后续训练。不要给已有 ID 增加 `overwrite`，不要删旧目录后重跑；若一个候选中断且无法按 checkpoint 规则恢复，换 `-r2` 新 ID。

### 5.1 第一轮：三个 P0

```bash
run_finetune_candidate v3-finetune-final-dataonly-g20-r1 data_only 2.0
run_finetune_candidate v3-finetune-final-dataonly-g40-r1 data_only 4.0
run_finetune_candidate v3-finetune-final-structure-g10-r1 structure 1.0
```

### 5.2 第二轮：补齐 Gold 曲线

若时间允许，继续：

```bash
run_finetune_candidate v3-finetune-final-dataonly-g15-r1 data_only 1.5
run_finetune_candidate v3-finetune-final-dataonly-g30-r1 data_only 3.0
run_finetune_candidate v3-finetune-final-dataonly-g50-r1 data_only 5.0
```

若时间不足，按以下规则裁剪：

- g2 已比 r2 退化：优先跑 g1.5，取消 g3/g5；
- g2 改善、g4 退化：优先跑 g3，g5 取消；
- g4 仍明显改善：再跑 g3 和 g5，确认峰值是否位于 3～5；
- structure-g10 已明显退化：取消 structure-gbest；
- 任一候选训练出现 NaN、smoke gate 失败或 resolved config 不符：停止该候选，不放宽门控。

## 6. 每个候选训练后程序自动检查什么

每个实验目录必须存在：

```text
$HAND_TRAIN_ROOT/hand_landmarker_runs/<experiment-id>/
├── finetune_data_gate.json
├── finetune_smoke/smoke_gate_report.json
├── finetune/checkpoints/best.weights.h5
├── finetune/experiment_metadata.json
├── finetune/logs/history.csv
├── finetune/training_report.json
└── eval/finetune/val/{metrics.json,predictions.jsonl}
```

先确认真实倍率/profile 没传错：

```bash
export EXPERIMENT_ID=<experiment-id>

python - <<'PY'
import json, os
root = "/root/autodl-tmp/TrainFab/HLML-3.0/hand_landmarker_runs"
experiment_id = os.environ["EXPERIMENT_ID"]
meta = json.load(open(f"{root}/{experiment_id}/finetune/experiment_metadata.json"))
cfg = meta["resolved_config"]
print("profile:", cfg["resolved_profile"])
print("tier weights:", cfg["losses"]["supervision_tier_weights"])
print("structure:", cfg["losses"]["bone_vector"], cfg["losses"]["spread_ratio"])
PY
```

再查看该实验的数据门控：

```bash
python - <<'PY'
import json, os
root = "/root/autodl-tmp/TrainFab/HLML-3.0/hand_landmarker_runs"
experiment_id = os.environ["EXPERIMENT_ID"]
d = json.load(open(f"{root}/{experiment_id}/finetune_data_gate.json"))
print(json.dumps(d["sampling"]["loss_weighting"], indent=2))
PY
```

重点确认 `configured_supervision_tier_weights` 与命令一致，并记录 `epoch0_effective_head_weight_mass_fraction`。不要用图片数量或 `gold_fraction` 代替真实 Loss 权重审计。

## 7. Val 快筛与配对比较

所有候选先只做 Val，不要立刻为每个候选跑 307 张 infer 或 export。对每个已完成 Val 的候选运行：

```bash
make compare-finetune-runs \
  BASELINE_FINETUNE_ID=v3-finetune-final-dataonly-r2 \
  CANDIDATE_FINETUNE_ID=<candidate-experiment-id>
```

结果：

```text
$HAND_TRAIN_ROOT/hand_landmarker_runs/<candidate-experiment-id>/analysis/
└── compare_vs_v3-finetune-final-dataonly-r2/
    ├── summary.json
    ├── paired_comparison.json
    ├── per_roi_metrics.jsonl
    └── overlays/
```

即使 candidate 尚未运行 infer，这个命令也会完成 Val 配对分析，inference 字段显示 `missing` 是正常的。它会报告 candidate 相对 r2 的 improved/regressed ROI 数、mean error delta、按 dataset 的误差、`predicted_spread < 0.10` 塌缩数和最多 40 张困难 overlay。

批量打印核心指标：

```bash
python - <<'PY'
import json
root = "/root/autodl-tmp/TrainFab/HLML-3.0/hand_landmarker_runs"
ids = [
    "v3-finetune-final-dataonly-r2",
    "v3-finetune-final-dataonly-g15-r1",
    "v3-finetune-final-dataonly-g20-r1",
    "v3-finetune-final-dataonly-g30-r1",
    "v3-finetune-final-dataonly-g40-r1",
    "v3-finetune-final-dataonly-g50-r1",
    "v3-finetune-final-structure-g10-r1",
]
print("id mean median p90 p95 pck@.10 presence handedness")
for experiment_id in ids:
    path = f"{root}/{experiment_id}/eval/finetune/val/metrics.json"
    try:
        d = json.load(open(path))["metrics"]
    except FileNotFoundError:
        continue
    lm, pr, hd = d["landmarks"], d["presence"], d["handedness"]
    print(experiment_id,
          round(lm["mean_pixel_error"], 3),
          round(lm["median_pixel_error"], 3),
          round(lm["p90_pixel_error"], 3),
          round(lm["p95_pixel_error"], 3),
          round(lm["pck"]["0.10"], 4),
          round(pr["recall"], 4),
          round(hd["accuracy"], 4))
PY
```

### 7.1 快筛标准

r2 基准为 mean 21.48、P90 44.67、PCK@0.10 0.359、presence recall 0.929、handedness 0.875。按以下顺序筛选：

1. 首先排除 smoke/训练不完整、非有限输出、越界输出或数据审计不一致的候选。
2. 主要目标是降低 P90/P95 和 collapse count，同时 mean 不比 r2 恶化超过约 0.5 px。
3. 若长尾相近，选择 mean/median 更低、PCK@0.10/0.15 更高者。
4. presence recall 不应明显低于 0.929；handedness 不应为换取很小的 landmark 改善而大幅下降。
5. 查看 `peak_vals_shared_v1`，因为它是 r2 当前最明显的长尾来源；只改善 `soar` 而继续恶化 Peak 不能算总体胜出。
6. 配对比较优先于只看均值：改善 ROI 应多于退化 ROI，且 overlay 中不能通过把本来正确的拳头错误“撑开”来换取平均 spread 增大。

## 8. 只对前两名运行独立 infer

从 data_only 选一名、structure 选一名；若 structure 明显失败，则选两名 data_only。对 finalist 运行：

```bash
make infer-finetune \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_EXPERIMENT_ID=<candidate-experiment-id> \
  FINETUNE_PROFILE=<data_only-or-structure> \
  FINETUNE_GOLD_LOSS_WEIGHT=<candidate-gold-weight> \
  FINETUNE_PSEUDO_LOSS_WEIGHT=1.0

make compare-finetune-runs \
  BASELINE_FINETUNE_ID=v3-finetune-final-dataonly-r2 \
  CANDIDATE_FINETUNE_ID=<candidate-experiment-id> \
  ANALYSIS_OVERWRITE=1
```

查看：

```text
$HAND_TRAIN_ROOT/hand_landmarker_inference/<candidate-experiment-id>/finetune/predictions.jsonl
$HAND_TRAIN_ROOT/hand_landmarker_inference/<candidate-experiment-id>/finetune/rendered/
$HAND_TRAIN_ROOT/hand_landmarker_runs/<candidate-experiment-id>/analysis/compare_vs_v3-finetune-final-dataonly-r2/summary.json
```

人工只看固定的小集合，不浏览全部 307 张：

```text
720x1280_8bit_20260718172421789_0.tiff       # 双手弯曲/握拳
720x1280_8bit_20260718172426720_420.tiff     # 双手向下张开，r2 比 multitask 明显好
720x1280_8bit_20260718172427419_476.tiff     # 双手张开，观察手指分离
720x1280_8bit_20260718172430502_742.tiff     # 数字 1，r2 仍有严重聚团
720x1280_8bit_20260718172602071_87.tiff      # 侧向张手，r2 单手仍会塌缩
720x1280_8bit_20260718172639415_644.tiff     # 复杂张手，观察局部手指拓扑
720x1280_8bit_20260718172638699_581.tiff     # torso 假 Palm ROI，hand flag 应拒绝
720x1280_8bit_20260718172638783_588.tiff     # torso 假 Palm ROI，hand flag 应拒绝
720x1280_8bit_20260718172638865_595.tiff     # torso 假 Palm ROI，hand flag 应拒绝
720x1280_8bit_20260718172639032_609.tiff     # 一真一假 Palm ROI，只应接受真手
720x1280_8bit_20260718172639115_616.tiff     # 一真一假 Palm ROI，只应接受真手
```

人工记录四项即可：真实 Hand ROI 接受数、torso 假 ROI 拒绝情况、塌缩手数、明显拓扑错误手数。不要因 Palm Detector 没产生 ROI 而给 Hand Landmarker 判负。

## 9. 可选的 structure + 最佳 Gold 组合

只有同时满足以下条件才运行组合实验：

- structure-g10 相比 r2 的 P90/P95 或 collapse count 有明确改善；
- 最佳 data_only Gold 倍率相比 g1 也有明确改善；
- 距离停止实验的时间点仍超过 2 小时。

假设最佳倍率为 2.0：

```bash
run_finetune_candidate v3-finetune-final-structure-gbest-r1 structure 2.0
```

若 structure-g10 没有收益，停止结构路线，不调整 bone/spread coefficient，不尝试 `structure_roi_aug`。如果 g4/g5 只降低训练 Loss 而 Val/配对 overlay 退化，说明 Gold 过权重，直接回到 g1.5～g3 区间。

## 10. 锁定 winner 后的唯一最终链路

最迟在提交前 8 小时停止新实验，保留时间给 Test、导出、厂商转换、上板、PPT 和故障缓冲。设置 winner 的真实参数：

```bash
export WINNER_ID=<winner-experiment-id>
export WINNER_PROFILE=<data_only-or-structure>
export WINNER_GOLD_WEIGHT=<gold-weight>
```

如果 winner 尚未 infer，先运行一次 infer。然后只对 winner 运行 locked Test 和导出：

```bash
make eval-test-finetune \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_EXPERIMENT_ID="$WINNER_ID" \
  FINETUNE_PROFILE="$WINNER_PROFILE" \
  FINETUNE_GOLD_LOSS_WEIGHT="$WINNER_GOLD_WEIGHT" \
  FINETUNE_PSEUDO_LOSS_WEIGHT=1.0

make export-finetune \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_EXPERIMENT_ID="$WINNER_ID" \
  FINETUNE_PROFILE="$WINNER_PROFILE" \
  FINETUNE_GOLD_LOSS_WEIGHT="$WINNER_GOLD_WEIGHT" \
  FINETUNE_PSEUDO_LOSS_WEIGHT=1.0

make conversion-data-finetune \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_EXPERIMENT_ID="$WINNER_ID" \
  FINETUNE_PROFILE="$WINNER_PROFILE" \
  FINETUNE_GOLD_LOSS_WEIGHT="$WINNER_GOLD_WEIGHT" \
  FINETUNE_PSEUDO_LOSS_WEIGHT=1.0
```

结果：

```text
$HAND_TRAIN_ROOT/hand_landmarker_runs/$WINNER_ID/eval/finetune/test/
$HAND_TRAIN_ROOT/hand_landmarker_runs/$WINNER_ID/export/finetune/
```

随后完成厂商工具链转换和固定 TIFF 上板回归。若 winner 在工具链或板端出现意外，r2 已有通过 HLML export 门控的 ONNX，可作为回退候选；不要在最后数小时重新改数据或模型代码。

## 11. 人工与程序分工

程序自动完成：

- 冻结数据和 SHA 校验；
- Gold/pseudo 真实 Loss 倍率绑定与审计；
- smoke、正式训练、best checkpoint；
- Val/Test 指标、配对 ROI 统计和至多 40 张困难 overlay；
- infer、ONNX parity、算子审计和转换数据包。

人工只完成：

- 在 `screen` 中按优先级启动候选；
- 每个候选确认 profile/倍率和 `status=complete`；
- 比较汇总表与少量固定 infer 图；
- 在截止时间前锁定 winner；
- 厂商转换、上板、文档和答辩 PPT。

本阶段不再进行：新增标注、Gold 来源调整、replay 重采样、pretrain、模型扩容、Loss 代码修改、结构系数搜索、ROI augmentation 搜索或全量人工浏览推理图。
