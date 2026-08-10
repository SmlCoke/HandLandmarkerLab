# HLML 当前状态（2026-08-10）

## 代码与流程

HLML 4.0 已对齐最新版 HLMF 3.0 manifest 接口和三阶段 v2 训练契约。PretrainSource/EValSource 原位读取来源 ROI；HLMF published negative/selection 读取其独立图片副本；HLML 的 `HAND_TRAIN_ROOT` 仍只写索引和训练产物。训练默认使用 `tqdm` epoch/batch 进度条；multitask 与 multi-finetune 的 `make export` 同时生成 ONNX/A1 审计物和符合官方转换工具要求的 `datasets.zip`。

文件夹推理默认 Palm 已更新为 EOS 2.0：模型部署在 `palm_detector/eos-2.0/model_opt.onnx`，输入为 `[1,1,224,384]`，解码 `14×24` 与 `7×12` 两层共 840 个 Anchor，并在合并候选后执行 score 0.25、IoU 0.10 的一次全局 NMS。Hand ROI 继续使用 scale `1.8/1.8`、shift `0/-0.1` 和 `256×256` 输出。

三个训练阶段结束后均要求执行固定 ROI Val 与代表性原图 infer；multitask、multi-finetune 还要求执行 export。

## 服务器环境与数据

服务器 `hand-landmarker-tf29` 的 TensorFlow/Keras 2.9.0、CUDA 11.2、cuDNN 8、ONNX 栈和 RTX 3090 GPU 可用；依赖更新后安装固定版本 `tqdm`。

2026-08-07 对正式分流组合运行 geometry 全量数据审计：`FullEnhance0801/eos_1.0-gate` Train 65,089，`FullEnhanceVal0801/eos-1.0` Val 5,091、Test 2,816；manifest、Registry、路径和 `256×256` 灰度 ROI 解码均无错误，Train 全部为 `pseudo/POS_RUNTIME`。按现有 `performer_cross_split: warn` 策略报告 peak/soar 跨 split 警告。

正式 experiment `national-final-geometry-eos_1.0-gate-r1` 已完成 70 个 epoch；`val_landmark_mae` 最优 checkpoint 位于第 68 epoch，值为 0.0253235，`final.weights.h5` 已绑定该 winner。

固定 ROI `make val` 已完成 5,091 条：landmarks mean pixel error 10.1930 px、mean NME 0.06043、PCK@0.05/0.10/0.15 为 0.5914/0.8377/0.9227，collapse 为 0。22 个 unknown-handedness positive 继续计入 presence/landmarks，只从 handedness 排除；handedness 已知标签准确率为 0.4369。

`InferSource/0718/images` 的 `make infer` 已完成 307 张原图，Palm `eos-1.0` 产生 529 个检测并全部进入 Hand Landmarker，失败 0；输出位于 `HAND_TRAIN_ROOT/inference/national-final-geometry-eos_1.0-gate-r1/geometry`。

warehouse 的 `datasets`、`negative_datasets`、`new_datasets`、`selections` 与 `evaluation.val/test` 均支持多个成员 ID；每个 dataset 类成员可独立选择 proposal variant，每个成员有独立权重。

2026-08-10 完成最新版 HLMF/HLML 接口复核：真实 `FullEnhanceVal0808/eos_1.0-gate_r2` 共 1,989 条 Val 记录，包含 RTMPose、人工修正、双头 HCF teacher ID；真实 `0809-soar-enhance` 共 281 条 published negative。两者均完成图片解码和 canonical 严格校验，错误为 0；18 条警告均为 HLMF 契约允许的截断 ROI 边界外关键点。合成 selection 还验证了 `source_crop_relpath`/`published_relpath`、源 ROI 删除后独立副本继续可读，以及 MediaPipe TFLite 几何补救字段无损保留。HLMF 54 项、HLML 184 项单元测试全部通过。

同日针对 HLMF 后续 EOS 2.0 与 HCF 0809 更新再次复核：HLMF 对 HLML 发布的数据外层契约仍为 `hlmf_dataset_v1 + 256×256` 灰度 Hand ROI，HCF 变更只体现在可选教师溯源字段，训练读取器无需改变；HLML 原有文件夹推理链路则与 EOS 2.0 不兼容，现已同步矩形前处理、Anchor、阈值与全局 NMS。服务器当前还没有 `proposal_variant=eos-2.0` 的正式 dataset manifest，因此接口验收使用当前 HLMF 契约生成的临时发布集，并在验收后删除；后续正式发布 EOS 2.0 数据时仍需用新的 snapshot ID 执行一次 `make data-audit`。更新后的 HLMF 66 项、HLML 188 项测试全部通过；同一张真实 TIFF 经同一 EOS 2.0 ONNX 推理后，两仓解码结果逐字段完全一致（1 个检测）。

当前服务器已有 `0809-soar-enhance`、`background-neg-0801-full` 等 published negative dataset，multitask 的 HLMF 数据前置条件已经具备；尚无 published hard-positive selection，因此 multi-finetune 仍需先完成困难样本复核与发布。
