# Hand Landmarker 数据集制作操作手册

> 适用范围：训练集、共享/独立验证集和共享测试集的自动标注、CVAT 复核、07A/07B 冻结以及训练接口。  
> 环境：所有命令均在仓库根目录执行，并使用 README 指定的 `anfab` 环境。  
> namespace：Train 和 Val/Test 都由 finalizer 按 source 的唯一 `dataset_id` 自动添加；原始文件名允许重复。

## 1. 最终数据组织

| 数据集 | 原始数据 | 是否完整人工复核 | 最终处理 | 用途 |
|---|---|---:|---|---|
| Train pseudo | Peak 约 2.2 万 ROI + Soar 约 4.6 万 ROI | 否 | 07A `pretrain` | 第一阶段教师—学生伪标签训练 |
| Train Gold 子集 | 从训练候选中挑选的数百到数千 ROI | 是 | 07A `finetune`，Gold 覆盖 pseudo | 第二阶段精调 |
| Val shared | Peak `vals_data` + Soar `vals_data` | 是 | 两份 shared 与当前路线 independent Val 一起交给 07B | 两条训练路线的共同比较基准 |
| Val independent | `vali_data` | 是 | 与 shared Val 一起交给 07B | 保留训练路线独立性 |
| Test shared | `test_data` | 是 | 07B `test` | 两人 100% 共享的最终冻结评测 |

`vals_data` 和 `vali_data` 的“80%/40%”只描述采集计划，不作为脚本采样参数。07B 按实际输入和实际 included ROI 合并，并在 `finalize_val_report.json` 中报告每个 source/partition 的真实数量。

推荐目录：

```text
../autodl-tmp/
├─ peak_train_data/          # Peak 的原始训练来源及 00-06 中间产物
├─ soar_train_data/          # Soar 的原始训练来源及 00-06 中间产物
├─ train_pretrain_merged/    # 07A pretrain 最终合并输出
├─ train_finetune_merged/    # 07A finetune 最终合并输出
├─ vals_data/                # 共享 Val
├─ vali_data/                # 当前路线独立 Val
├─ val_merged/               # 07B 生成的最终 Val
├─ test_data/                # 100% 共享 Test
└─ test_merged/              # 07B 生成的最终 Test
```

每个尚未合并的数据目录内部保持：

```text
images/
01_palm/
02_roi_crops/
03_reviewed/
04_visualization/
05_labels/
qc/
```

## 2. 环境与执行原则

激活环境：

```powershell
conda activate anfab
```

确认当前目录：

```powershell
Get-Location
```

应位于 `HandLandmarkerFab` 仓库根目录。

必须遵守：

1. Train、Val、Test 的 ROI 参数、Palm threshold、NMS 必须与板端一致。
2. Val/Test 不导出低分 `negative_candidates`。
3. 每个 source 的 `dataset_id` 必须全局唯一并包含 owner/数据用途，例如 `peak_vals_shared_v1`；不要依赖原始文件名前缀隔离。
4. 不得把 Val/Test ROI 加入训练集。
5. Test 在模型、阈值、量化方案冻结前不得用于调参。

## 3. 配置文件职责

| 配置 | 作用 |
|---|---|
| `configs/autolabel_train.yaml` | 单个 Train 来源运行 00–06 |
| `configs/finalize_train.yaml` | 07A 合并多个训练来源、自动 namespace、分型和降采样 |
| `configs/autolabel_val.yaml` | shared Val，即 `vals_data`，运行 00–06 |
| `configs/autolabel_vali.yaml` | independent Val，即 `vali_data`，运行 00–06 |
| `configs/finalize_val.yaml` | 07B 合并 Peak shared + Soar shared + 当前路线 independent Val |
| `configs/autolabel_test.yaml` | shared Test 运行 00–06 |
| `configs/finalize_test.yaml` | 07B 汇总并冻结最终 Test |

### 3.1 修改自动标注路径

例如 shared Val：

```yaml
dataset:
  id: vals_shared_v1
  split: val
  partition: shared

paths:
  images_dir: ../autodl-tmp/vals_data/images
  palm_outputs_dir: ../autodl-tmp/vals_data/01_palm
  roi_crops_dir: ../autodl-tmp/vals_data/02_roi_crops
  reviewed_dir: ../autodl-tmp/vals_data/03_reviewed
  labels_dir: ../autodl-tmp/vals_data/05_labels
  qc_dir: ../autodl-tmp/vals_data/qc
```

如果服务器目录不同，只修改 `paths`；不要为了减少标注量修改 Palm、NMS 或 ROI 参数。

## 4. Train 第一阶段：生成 pseudo 训练集

### 4.1 每个训练来源分别运行 00–03

Peak 数据：

```powershell
make validate_images_train
make palm_detection_train
make build_roi_train
make run_mediapipe_train
```

Soar 数据必须使用指向 Soar 目录的独立自动标注配置运行相同四个脚本。可以复制 `configs/autolabel_train.yaml` 为个人配置，只修改 `paths`，不要修改模型和 ROI 参数。

第一阶段需要的核心输入是：

```text
02_roi_crops/hand_roi_crops_manifest.jsonl
02_roi_crops/hand_landmarks_autolabel_draft.jsonl
02_roi_crops/images/*.png
```

不需要对全部训练集执行 CVAT 复核。

### 4.2 配置多个 Train 来源

编辑 `configs/finalize_train.yaml`：

```yaml
sources:
  - dataset_id: peak_train_v1
    contributor: Peak
    root: .
    autolabel_config: configs/autolabel_train.yaml
    manifest: ../autodl-tmp/peak_train_data/02_roi_crops/hand_roi_crops_manifest.jsonl
    pseudo_labels: ../autodl-tmp/peak_train_data/02_roi_crops/hand_landmarks_autolabel_draft.jsonl
    crop_images_dir: ../autodl-tmp/peak_train_data/02_roi_crops/images

  - dataset_id: soar_train_v1
    contributor: Soar
    root: .
    autolabel_config: configs/autolabel_train_soar.yaml
    manifest: ../autodl-tmp/soar_train_data/02_roi_crops/hand_roi_crops_manifest.jsonl
    pseudo_labels: ../autodl-tmp/soar_train_data/02_roi_crops/hand_landmarks_autolabel_draft.jsonl
    crop_images_dir: ../autodl-tmp/soar_train_data/02_roi_crops/images
```

要求：

- 每个 `dataset_id` 全局唯一且版本固定；
- 两个 `- dataset_id` 必须在 YAML 中保持同一级缩进；其余 source 字段比 `-` 多缩进两个空格；
- Train 原文件可以同名，07A 会生成 `dataset_id:source_crop_id` 形式的 `global_crop_id`；
- 不要手工把 `peak_` / `soar_` 再加到 Train JSONL，避免双重 namespace；
- `autolabel_config` 必须与该来源实际运行 00–03 时使用的配置一致。
- 如果数据搬迁后 manifest 内的 `crop_path` 已过期，必须设置 `crop_images_dir`；07A 会按 basename 定位真实图片，并把旧值保存在 `source_crop_path` 中。

### 4.3 生成 pretrain 清单

```powershell
make finalize_train_pretrain
```

主要输出：

```text
../autodl-tmp/train_pretrain_merged/05_labels/hand_train_catalog_pretrain.jsonl
../autodl-tmp/train_pretrain_merged/05_labels/hand_training_labels_pretrain.jsonl
../autodl-tmp/train_pretrain_merged/05_labels/hand_training_excluded_pretrain.jsonl
../autodl-tmp/train_pretrain_merged/qc/finalize_train_pretrain_report.json
```

训练 loader 读取 `hand_training_labels_pretrain.jsonl`。07A 已完成：

- 多来源 namespace；
- manifest/label/图片结构门禁；
- 四种 `sample_type` 分类；
- pseudo 质量 flags；
- 同原图正样本聚类；
- 负样本去重和限额；
- sampling bucket、sampling weight 和各 head loss weight 输出。

有效 head loss 权重：

```text
head_loss_weight
× supervision_loss_weight
× 对应的 head_quality_weight
```

`sampling_weight` 只用于 sampler，不乘入 loss。

## 5. Train 第二阶段：制作 Gold 子集并 finetune

### 5.1 选择人工复核候选

优先选择：

- 不同手型、距离、旋转、光照和背景；
- `POS_LOW_PALM`；
- `NEG_RUNTIME_CANDIDATE`；
- teacher/student 分歧或 `quality_flags` 较多的样本；
- 少量普通高质量样本作为稳定基线。

同一原图或连续视频帧不要大量重复进入 Gold。

### 5.2 导出并完成 CVAT 复核

对选中的 subset manifest/draft 运行：

```powershell
python scripts/04_export_cvat_xml.py `
  --config <该来源的 autolabel 配置> `
  --manifest <subset_manifest.jsonl> `
  --draft-jsonl <subset_draft.jsonl> `
  --output-xml <subset_autolabel.xml>
```

上传：

1. subset 对应的 ROI 图片；
2. `subset_autolabel.xml`；
3. 保持现有 labels：`no_hand`、`Left`、`Right`、`ignore_for_training`、`hand_landmarks`。

人工规则：

- 有手：只保留一个目标手 skeleton，修正完整 21 点，并标 Left/Right；
- 无手：删除 skeleton，添加 `no_hand`；
- 双手但 Palm anchor 唯一：只标 anchor 指向的目标手；
- 双手目标不唯一、点归属不可靠：添加 `ignore_for_training`；
- 不要用“更大、居中或 Google 首先检测到”代替 Palm anchor。

导出 reviewed XML 后运行 05：

```powershell
python scripts/05_import_cvat_xml.py `
  --config <该来源的 autolabel 配置> `
  --reviewed-xml <subset_reviewed.xml> `
  --manifest <subset_manifest.jsonl> `
  --draft-jsonl <subset_draft.jsonl> `
  --output-jsonl <subset_gold.jsonl>
```

检查对应 `qc/cvat_import_stats.json`，必须解决所有非 ignored import errors。

### 5.3 把 Gold 接入 07A

在对应 source 下增加：

```yaml
gold_manifest: ../path/to/subset_manifest.jsonl
gold_labels: ../path/to/subset_gold.jsonl
gold_import_report: ../path/to/cvat_import_stats.json
```

然后运行：

```powershell
make finalize_train_finetune
```

07A 会先按原始 `crop_id` 用 Gold 覆盖 pseudo，再做分类和去重。训练 loader 改读：

```text
../autodl-tmp/train_finetune_merged/05_labels/hand_training_labels_finetune.jsonl
```

## 6. shared Val：制作 `vals_data`

Peak 和 Soar 各自完成自己的 `vals_data` 自动标注与人工复核。这里的“shared”表示两人的最终模型都使用两份 shared source 合并后的共同部分，不表示两人必须在各自机器上保存相同的绝对路径。

### 6.1 自动标注

确认 `configs/autolabel_val.yaml` 指向 `../autodl-tmp/vals_data`，然后运行：

```powershell
make validate_images_vals
make palm_detection_vals
make build_roi_vals
make run_mediapipe_vals
make export_cvat_vals
```

### 6.2 CVAT 复核与导入

上传：

```text
../autodl-tmp/vals_data/02_roi_crops/images/
../autodl-tmp/vals_data/02_roi_crops/cvat_autolabel.xml
```

完整人工复核全部 ROI，导出至：

```text
../autodl-tmp/vals_data/03_reviewed/cvat_reviewed.xml
```

然后运行：

```powershell
make import_cvat_vals
make visualize_vals
```

不要在这一步运行 07B；必须等 independent Val 也完成后统一合并。

### 6.3 每人需要交付的 shared Val 复核包

Peak 和 Soar 分别交付一个完整 source 包。07B 最少需要：

```text
02_roi_crops/
├─ images/                                  # 与 manifest 一一对应的 ROI 图片
└─ hand_roi_crops_manifest.jsonl
03_reviewed/
├─ hand_landmarks_reviewed.jsonl            # import_cvat_vals 的主要结果
├─ cvat_reviewed.xml                        # 07B 不直接读取，但建议保留审计
└─ review_context.csv                       # 可选；存在时 ignored 行必须有原因
qc/
└─ cvat_import_stats.json                   # import_cvat_vals 的完整性报告，强制
```

不要只传 `hand_landmarks_reviewed.jsonl`。没有 manifest、ROI 图片和 import report，07B 无法验证覆盖率、坐标投影和 CVAT 冲突。

建议在各自机器上打包整个必要目录，例如：

```bash
cd /path/to/vals_data
tar -czf peak_vals_reviewed.tar.gz \
  02_roi_crops/images \
  02_roi_crops/hand_roi_crops_manifest.jsonl \
  03_reviewed/hand_landmarks_reviewed.jsonl \
  03_reviewed/cvat_reviewed.xml \
  qc/cvat_import_stats.json
```

Soar 将文件名改为 `soar_vals_reviewed.tar.gz`。如果使用了 `review_context.csv`，也把它加入压缩包。

### 6.4 在统一服务器上放置两个 shared source

推荐目录：

```text
../autodl-tmp/eval_sources/
├─ peak_vals/
│  ├─ 02_roi_crops/
│  ├─ 03_reviewed/
│  └─ qc/
└─ soar_vals/
   ├─ 02_roi_crops/
   ├─ 03_reviewed/
   └─ qc/
```

示例：

```bash
mkdir -p ../autodl-tmp/eval_sources/peak_vals
mkdir -p ../autodl-tmp/eval_sources/soar_vals
tar -xzf peak_vals_reviewed.tar.gz -C ../autodl-tmp/eval_sources/peak_vals
tar -xzf soar_vals_reviewed.tar.gz -C ../autodl-tmp/eval_sources/soar_vals
```

图片移动后，manifest 中的 `crop_path` 可能仍指向原机器路径，这是允许的。不要手工批量修改 JSONL；在 `finalize_val.yaml` 为每个 source 设置真实的 `crop_images_dir`，07B 会按 basename 重定位，并把旧值保存在 `source_crop_path`。

### 6.5 两份 shared Val 是否需要手工合并

不需要，也不允许手工使用 `copy`、文本拼接或脚本直接合并两份 manifest/reviewed JSONL。正确做法是把它们作为两个独立 source 登记到 `configs/finalize_val.yaml`，由 07B 合并。

存在两种情况：

1. **只是名字相同、实际图片不同**：允许直接作为两个 source。07B 使用不同 `dataset_id` 生成 namespace，不会冲突。
2. **实际是同一张图片被重复复核/复制**：这不是两份评测样本，不能重复计入。应先比较标注并在 CVAT 中仲裁，形成一份权威 reviewed package；配置中只登记一次。namespace 只能解决名称冲突，不能把同一张物理图片合理地变成两个样本。

具体例子：

```text
可直接合并（local ID 相同但图片不同）：
  Peak source dataset_id = peak_vals_shared_v1
  Peak local crop_id      = frame0001:palm0:crop
  Soar source dataset_id  = soar_vals_shared_v1
  Soar local crop_id      = frame0001:palm0:crop

最终 global crop_id：
  peak_vals_shared_v1:frame0001:palm0:crop
  soar_vals_shared_v1:frame0001:palm0:crop
```

如果两行实际指向同一张 ROI 图片，即使 global ID 不同也不能重复计入；应由两人查看该图片并仲裁出一份 21 点、presence 和 handedness 真值。

## 7. independent Val：制作 `vali_data`

确认 `configs/autolabel_vali.yaml` 指向当前路线自己的 `vali_data`：

```powershell
make validate_images_vali
make palm_detection_vali
make build_roi_vali
make run_mediapipe_vali
make export_cvat_vali
```

在 CVAT 完整复核后，将 XML 放到：

```text
../autodl-tmp/vali_data/03_reviewed/cvat_reviewed.xml
```

再运行：

```powershell
make import_cvat_vali
make visualize_vali
```

Peak 和 Soar 各自把自己的 independent Val 也整理为完整 source 包：

```text
../autodl-tmp/eval_sources/peak_vali/
../autodl-tmp/eval_sources/soar_vali/
```

两条训练路线的最终 Val 组成应为：

```text
Peak 路线：peak_vals + soar_vals + peak_vali
Soar 路线：peak_vals + soar_vals + soar_vali
```

因此，Peak 不应把 `soar_vali` 加入自己的 `finalize_val.yaml`；Soar 也不应把 `peak_vali` 加入自己的配置。这样 shared 部分完全相同，而 independent 部分保持路线独立。

## 8. 合并并冻结最终 Val

### 8.1 人工需要做什么

1. 收集 Peak/Soar 两份 shared Val 完整复核包；
2. 收集当前路线自己的 independent Val 完整复核包；
3. 分别解压到独立目录，不要把图片平铺进同一个 `images/`；
4. 确认每个目录都有 manifest、reviewed JSONL、import report 和 images；
5. 为三个 source 配置不同 `dataset_id`；local `crop_id` 或文件名可以相同；
6. 确认没有把同一张实际图片复制到多个 source；
7. 编辑 `configs/finalize_val.yaml` 登记三个 source；
8. 运行一次 `make finalize_val`。不需要分别对两个 shared source 运行额外的合并脚本。

### 8.2 Peak 路线配置示例

`configs/finalize_val.yaml` 的关键结构：

```yaml
dataset:
  id: peak_route_merged_val_v1
  split: val

evaluation:
  require_palm_valid: true

sources:
  - source_id: peak_vals_shared
    dataset_id: peak_vals_shared_v1
    owner: Peak
    partition: shared
    root: ../autodl-tmp/eval_sources/peak_vals
    autolabel_config: configs/autolabel_val.yaml
    crop_images_dir: 02_roi_crops/images
    manifest: 02_roi_crops/hand_roi_crops_manifest.jsonl
    reviewed: 03_reviewed/hand_landmarks_reviewed.jsonl
    import_report: qc/cvat_import_stats.json

  - source_id: soar_vals_shared
    dataset_id: soar_vals_shared_v1
    owner: Soar
    partition: shared
    root: ../autodl-tmp/eval_sources/soar_vals
    autolabel_config: configs/autolabel_val.yaml
    crop_images_dir: 02_roi_crops/images
    manifest: 02_roi_crops/hand_roi_crops_manifest.jsonl
    reviewed: 03_reviewed/hand_landmarks_reviewed.jsonl
    import_report: qc/cvat_import_stats.json

  - source_id: peak_vali_independent
    dataset_id: peak_vali_independent_v1
    owner: Peak
    partition: independent
    root: ../autodl-tmp/eval_sources/peak_vali
    autolabel_config: configs/autolabel_vali.yaml
    crop_images_dir: 02_roi_crops/images
    manifest: 02_roi_crops/hand_roi_crops_manifest.jsonl
    reviewed: 03_reviewed/hand_landmarks_reviewed.jsonl
    import_report: qc/cvat_import_stats.json

outputs:
  labels_dir: ../autodl-tmp/val_merged/05_labels
  qc_dir: ../autodl-tmp/val_merged/qc
```

Soar 路线复制该配置，只把：

```yaml
dataset:
  id: soar_route_merged_val_v1
```

以及第三个 source 改为：

```yaml
  - source_id: soar_vali_independent
    dataset_id: soar_vali_independent_v1
    owner: Soar
    partition: independent
    root: ../autodl-tmp/eval_sources/soar_vali
    autolabel_config: configs/autolabel_vali.yaml
    crop_images_dir: 02_roi_crops/images
    manifest: 02_roi_crops/hand_roi_crops_manifest.jsonl
    reviewed: 03_reviewed/hand_landmarks_reviewed.jsonl
    import_report: qc/cvat_import_stats.json
```

### 8.3 运行最终合并

运行：

```powershell
make finalize_val
```

07B 会：

1. 分别检查 Peak shared、Soar shared、当前路线 independent 的 manifest、reviewed、图片和 XML import report；
2. 检查 source 内 local ID/basename 唯一，并检查所有 source 的 `dataset_id` 唯一；
3. 为 crop、Palm、hand 和原图分组 ID 生成全局 namespace；跨 source local ID/basename 允许相同；
4. 分离 ignored；
5. 对 included Gold 执行严格 21 点、handedness、坐标和图片校验；
6. 合并并原子写入最终 Val；
7. 按 source 和 partition 报告实际 included 数量。

最终文件：

```text
../autodl-tmp/val_merged/05_labels/hand_validation_labels.jsonl
../autodl-tmp/val_merged/05_labels/hand_val_ignored.jsonl
../autodl-tmp/val_merged/qc/finalize_val_report.json
```

### 8.4 图片路径变化的完整例子

假设 Peak 在自己机器生成 manifest 时记录：

```json
{
  "crop_id": "session01_frame0001:palm0:crop",
  "crop_path": "/home/peak/vals_data/02_roi_crops/images/session01_frame0001_palm0_crop.png"
}
```

上传统一服务器后，图片实际位于：

```text
/root/autodl-tmp/eval_sources/peak_vals/02_roi_crops/images/session01_frame0001_palm0_crop.png
```

配置：

```yaml
root: ../autodl-tmp/eval_sources/peak_vals
crop_images_dir: 02_roi_crops/images
```

07B 不修改原 manifest，而是在最终 `hand_validation_labels.jsonl` 中写入：

```json
{
  "source_crop_path": "/home/peak/vals_data/02_roi_crops/images/session01_frame0001_palm0_crop.png",
  "crop_path": "/root/autodl-tmp/eval_sources/peak_vals/02_roi_crops/images/session01_frame0001_palm0_crop.png",
  "source_crop_id": "session01_frame0001:palm0:crop",
  "global_crop_id": "peak_vals_shared_v1:session01_frame0001:palm0:crop",
  "crop_id": "peak_vals_shared_v1:session01_frame0001:palm0:crop",
  "evaluation_source_id": "peak_vals_shared",
  "evaluation_owner": "Peak",
  "evaluation_partition": "shared"
}
```

训练或评测 loader 使用 `crop_path`；`source_crop_path` 仅用于追溯原始记录。

## 9. 合并并冻结 Peak/Soar 100% 共享 Test

Peak 和 Soar 各自完成自己的 `test_data` 自动标注、CVAT 完整人工复核和 `import_cvat_test`。两份数据最终共同组成一份、两条路线 100% 相同的 Test。

每人在自己的数据上运行：

```powershell
make validate_images_test
make palm_detection_test
make build_roi_test
make run_mediapipe_test
make export_cvat_test
```

在 CVAT 完整复核后，将 XML 放到各自 `test_data/03_reviewed/cvat_reviewed.xml`，然后运行：

```text
../autodl-tmp/test_data/03_reviewed/cvat_reviewed.xml
```

运行：

```powershell
make import_cvat_test
make visualize_test
```

### 9.1 上传两个 Test source

使用与 Val 完全相同的完整复核包规则，放置为：

```text
../autodl-tmp/eval_sources/peak_test/
├─ 02_roi_crops/images/
├─ 02_roi_crops/hand_roi_crops_manifest.jsonl
├─ 03_reviewed/hand_landmarks_reviewed.jsonl
└─ qc/cvat_import_stats.json

../autodl-tmp/eval_sources/soar_test/
├─ 02_roi_crops/images/
├─ 02_roi_crops/hand_roi_crops_manifest.jsonl
├─ 03_reviewed/hand_landmarks_reviewed.jsonl
└─ qc/cvat_import_stats.json
```

不手工拼接 JSONL，不把两个 images 目录平铺合并。路径变化通过各 source 的 `crop_images_dir` 解决。

同样需要区分：

- 如果 Peak/Soar Test 是互补录制数据，即使文件名/local ID 相同，也可通过不同 `dataset_id` 安全合并；
- 如果两人重复复核了完全相同的 Test 图片，先人工仲裁成一份权威结果，只登记一次，绝不能用 namespace 把同一图片计算两次。

例如两个 source 的 local `crop_id` 都是 `test_0001:palm0:crop`，但实际来自不同录制图片，07B 会分别生成 `peak_test_shared_v1:test_0001:palm0:crop` 和 `soar_test_shared_v1:test_0001:palm0:crop`。如果两个包实际复制的是同一张图片，则仍必须先仲裁并只保留一个 source 中的权威行。

### 9.2 Test 聚合配置

```yaml
dataset:
  id: peak_soar_shared_test_v1
  split: test

evaluation:
  require_palm_valid: true

sources:
  - source_id: peak_test_shared
    dataset_id: peak_test_shared_v1
    owner: Peak
    partition: shared
    root: ../autodl-tmp/eval_sources/peak_test
    autolabel_config: configs/autolabel_test.yaml
    crop_images_dir: 02_roi_crops/images
    manifest: 02_roi_crops/hand_roi_crops_manifest.jsonl
    reviewed: 03_reviewed/hand_landmarks_reviewed.jsonl
    import_report: qc/cvat_import_stats.json

  - source_id: soar_test_shared
    dataset_id: soar_test_shared_v1
    owner: Soar
    partition: shared
    root: ../autodl-tmp/eval_sources/soar_test
    autolabel_config: configs/autolabel_test.yaml
    crop_images_dir: 02_roi_crops/images
    manifest: 02_roi_crops/hand_roi_crops_manifest.jsonl
    reviewed: 03_reviewed/hand_landmarks_reviewed.jsonl
    import_report: qc/cvat_import_stats.json

outputs:
  labels_dir: ../autodl-tmp/test_merged/05_labels
  qc_dir: ../autodl-tmp/test_merged/qc
```

### 9.3 运行最终 Test 合并

两份 source 都到位后只运行一次：

```powershell
make finalize_test
```

07B 会检查两个 source 的唯一 `dataset_id`、source 内重复、图片路径、人工覆盖和 Gold 结构，然后生成全局 namespace 并原子写出一份共同 Test。它不会根据 owner 做不同筛选。

最终文件：

```text
../autodl-tmp/test_merged/05_labels/hand_test_labels.jsonl
../autodl-tmp/test_merged/05_labels/hand_test_ignored.jsonl
../autodl-tmp/test_merged/qc/finalize_test_report.json
```

两人必须使用完全相同的 Test canonical JSONL 和 SHA-256。Test 只在最终 checkpoint、阈值和量化方案冻结后运行。

## 10. 07B 常见失败及处理

| 错误 | 含义 | 处理 |
|---|---|---|
| `manifest_file_missing` | source 路径配置错误或 02 未完成 | 检查 `root` 和 manifest 路径 |
| `reviewed_file_missing` | 05 尚未完成 | 先运行对应 `import_cvat_*` |
| `crop_images_dir_missing` | 统一服务器上的真实 images 目录配置错误 | 修正该 source 的 `root` / `crop_images_dir`，不要改 JSONL 规避 |
| `missing_cvat_import_report` | 缺少 05 的 QC 报告 | 重新运行 05，不要手工伪造 |
| `duplicate_evaluation_dataset_id` | 两个 source 使用了相同 namespace | 为每个 source 设置唯一 `dataset_id` |
| `duplicate_crop_basename_within_source` | 同一个 source 内 basename 重复 | 修正该 source 自身的数据；跨 source 同名不受影响 |
| `cross_source_same_physical_crop_path` | 两个 source 实际指向同一个物理文件 | 检查是否误把同一 source 登记两次；仲裁后只保留一次 |
| `manifest_without_reviewed` | CVAT/05 覆盖不完整 | 补齐 CVAT XML 后重新导入 |
| `invalid_reviewed_rows` | 非 ignored Gold 有结构或语义冲突 | 根据 report 中 crop ID 回 CVAT 修正 |
| `evaluation_requires_palm_valid` | Eval 混入低分 negative candidate | 检查 Val/Test 配置是否错误开启候选导出 |

07B 出现 fatal 时不会覆盖上一版 canonical JSONL。不要通过删除困难 ROI、降低校验标准或修改 Palm threshold 来“修复”报告。

## 11. 最终执行清单

Train：

- [x] 每个来源 00–03 完成；
- [x] `finalize_train.yaml` 中每个 `dataset_id` 唯一；
- [x] `make finalize_train_pretrain` 成功；
- [x] 人工 Gold subset 完成 CVAT 和 05；
- [x] Gold 三个路径写入对应 source；
- [x] `make finalize_train_finetune` 成功。

Val：

- [x] Peak/Soar 两份 `vals_data` 均完成 00–05 和完整人工复核；
- [x] 当前路线自己的 `vali_data` 完成 00–05 和完整人工复核；
- [x] 三个 source 分目录上传，均包含 images、manifest、reviewed 和 import report；
- [x] 三个 source 的 `dataset_id` 唯一；不依赖原始文件名前缀；
- [x] `make finalize_val` 成功；
- [x] 查看 shared/independent 实际数量、ignored 比例和 SHA-256。

Test：

- [x] Peak/Soar 两份 `test_data` 均完成 00–05 和完整人工复核；
- [x] 两个 Test source 分目录上传且没有重复计入同一 ROI；
- [x] 最终模型与阈值已冻结；
- [x] `make finalize_test` 成功；
- [x] Peak/Soar 使用相同 canonical 文件和 SHA-256。
