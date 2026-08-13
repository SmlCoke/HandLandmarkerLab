# HLML 当前状态（2026-08-13）

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

同日针对 HLMF 后续 EOS 2.0 与 HCF 0809 更新再次复核：HLMF 对 HLML 发布的数据外层契约仍为 `hlmf_dataset_v1 + 256×256` 灰度 Hand ROI，HCF 变更只体现在可选教师溯源字段；HLML 文件夹推理已同步矩形前处理、Anchor、阈值与全局 NMS。当时服务器尚无正式 EOS 2.0 manifest，接口验收使用合成发布集；HLMF 66 项、HLML 188 项测试通过，同一张真实 TIFF 的两仓 EOS 2.0 解码结果逐字段一致。

2026-08-13 HLMF 已发布正式 EOS 2.0 Gold：`FullEnhanceVal0801/eos_2.0-rtmpose-gate` 含 Val 3,453、Test 2,342 条，`RTMPose-Finetune-Test-0812/rtmpose-finetune-test` 另含冻结 Test 396 条；全部来源均为 near/mid。前者与同一 manifest 中 6 个仅发布旧 variant 的历史 source 共存，暴露出 HLML 旧读取器会把“所选 variant 缺失”误判为错误。当前读取器已改为只消费实际发布所选 variant 的 source，并在目标 split 完全无匹配时继续失败。连接长度门控只改变 HLMF Train published/ignored 分流，新增 manifest 汇总字段不改变 HLML 行接口。HCF 默认已更新到 0813，模型结构未变；HLML 对 teacher ID 不做版本硬编码，未来 0813 行与现有 0809 Gold 均可无损读取。上述 6,191 条真实 Gold 已全部通过图片解码、Registry 路径与 canonical 校验，错误为 0；HLMF 72 项、HLML 189 项完整测试通过。

本轮 Iris-1.1 geometry 冻结成员为 `FullEnhance0801`、`FullEnhance0803`、`FullEnhance0810` 的 `eos_2.0-rtmpose-hcf0813-gate` Train，以及 `FullEnhanceVal0801` 内五条 Eos-2.0 Val source、两条 Eos-2.0 Test source。Val 同时消费 HCF0813 新 variant 与历史 `eos_2.0-rtmpose-gate` Gold，因此 HLML 新增可选 `capture_source_ids` 精确白名单；白名单中的 source 必须存在于 manifest、匹配 split 且发布指定 variant。`hlmf_dataset_v1`、256×256 单通道 ROI、21 点及 presence/handedness/provenance 字段未发生结构性变化。真实训练加载首次发现 3,955 条 `mediapipe_tflite_rescue_v1` 使用 HLMF 当前 `crop_px=norm×256` 连续 crop extent 表示，而 HLML canonical 使用 `norm×255` 像素索引表示；读取器现会严格识别两种上游约定、拒绝其他不一致，并只在内部 snapshot 规范化辅助 `crop_px`，不改变 `landmarks_crop_norm` 训练目标。Eos-1.0 旧 Palm 只覆盖手掌，ROI 域与 Eos-2.0 完整手掌加手指 Anchor 不同，不进入 Iris-1.1 主 Val/Test，只可作为独立 legacy/stress 指标；`FullEnhanceVal0803` 与 `RTMPose-Finetune-Test-0812` 不参与本轮。

服务器真实全量 audit 得到 Train 74,225、Val 6,937、Test 2,342，invalid 与成员关系错误均为 0；Train 由 92 条 near/mid source 构成，全部为 `pseudo/POS_RUNTIME` 且 HCF teacher 为 0813。一次临时 CPU 1-step geometry + 1-step Val smoke 已完成，训练器、模型、采样、checkpoint 与报告链路均通过；HLMF 72 项、HLML 191 项测试通过，smoke snapshot/run/config 随后删除。当前服务器硬件可由 `nvidia-smi` 看到 RTX 3090，但 `/dev/nvidia-uvm` 打开返回 EIO、`libcuda.cuInit()` 返回 999，TensorFlow 因而找不到 physical GPU；这是容器/宿主 NVIDIA UVM 运行态故障，不是仓库或 Conda 依赖版本问题。正式 GPU geometry 必须先在 AutoDL 重启实例/容器或由平台刷新 UVM，并以 `make environment-check` 确认 GPU 通过。

当前服务器已有 `0809-soar-enhance`、`background-neg-0801-full` 等 published negative dataset，multitask 的 HLMF 数据前置条件已经具备；尚无 published hard-positive selection，因此 multi-finetune 仍需先完成困难样本复核与发布。
