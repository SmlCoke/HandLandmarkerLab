# HLML 下一阶段计划

## 当前目标

`national-final-geometry-eos_1.0-gate-r1` 的旧 geometry、固定 ROI Val 和 EOS 1.0 历史 infer 已完成。当前目标是以 Eos-2.0/HCF0813 发布域启动 Iris-1.1 geometry；本轮成员已在 `configs/datasets.yaml` 冻结，Eos-1.0 仅保留为独立 legacy/stress 回放。

## 执行顺序

1. 已刷新并冻结 HLMF dataset manifest，确认三个 Train dataset 的 HCF0813 variant 与五条 Val、两条 Test 白名单均已发布；EOS 2.0 的 far 排除继续由 HLMF 发布契约负责。
2. 双仓库测试、真实全量 data audit 与临时 CPU 1-step geometry smoke 已通过，临时产物已删除；Test 只验证输入契约，不用于模型选择或调参。
3. 在 AutoDL 重启实例/容器或请求平台刷新 NVIDIA UVM；运行 `make environment-check`，要求 TensorFlow 枚举到一张 physical GPU 且不再出现 `/dev/nvidia-uvm` EIO / `cuInit=999`。
4. 设置 `HLML_SNAPSHOT_ID=iris-1.1-geometry-eos2-hcf0813-r1`、同名 `HLML_EXPERIMENT_ID`，执行 `make geometry`。
5. geometry 结束后执行固定 ROI Val 与 EOS 2.0 代表性原图 infer；Test 仍保持锁定，直到 winner 冻结后才运行。
6. 后续再复核并选定 published true-negative dataset，进入 multitask、Train-only mining 与 multi-finetune。

## 验收条件

- 新 geometry 只使用 `FullEnhance0801/0803/0810:eos_2.0-rtmpose-hcf0813-gate` 的 Train positive；不混入 Eos-1.0 ROI 或负样本。
- Val 精确为三条 HCF0813 Eos-2.0 source 加两条旧 Eos-2.0 Gold source；Test 精确为两条 Eos-2.0 s01 Test source。每个白名单 ID 必须命中正确 split 和 variant。
- unknown handedness positive 保留 presence/landmarks 指标，只从 handedness 指标排除。
- 每阶段均保存 winner 并完成 Val/infer；multitask、multi-finetune 的 export 同时交付模型与配套数据包。
- multitask 负样本仅来自 HLMF published negative dataset；multi-finetune selection 仅来自 Train mining 与人工删除式复核。两类输入都读取 HLMF 独立 `published_relpath` 图片，selection 同时用 `source_crop_relpath` 核对来源身份。
- 多成员合并必须保持 ROI 唯一、split 隔离和同一 capture source 单 variant。
- 以新 snapshot ID 运行 data audit，确认 source 白名单、manifest、Registry、0809/0813 HCF 溯源字段、`256×256` published ROI，以及 TFLite rescue `norm×256` 上游表示到 canonical `norm×255` 辅助字段的严格规范化；`FullEnhanceVal0801` 两条 s01 Test 不参与调参，`RTMPose-Finetune-Test-0812` 不参与本轮。
- Test 不回流到采样、阈值、checkpoint 或困难挖掘。
