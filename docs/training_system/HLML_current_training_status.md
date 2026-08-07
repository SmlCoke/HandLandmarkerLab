# HLML 当前状态（2026-08-07）

## 代码与流程

HLML 4.0 已保持 HLMF manifest 零拷贝接口和三阶段 v2 训练契约。训练默认使用 `tqdm` epoch/batch 进度条；multitask 与 multi-finetune 的 `make export` 同时生成 ONNX/A1 审计物和符合官方转换工具要求的 `datasets.zip`。

文件夹推理的 Palm 模型已迁移到 `palm_detector/<model_id>/model_opt.onnx`。全局模型由 `configs/inference.yaml` 的 `palm.model_id` 选择，`make infer ... INFER_ARGS='--palm-model-id <id>'` 可单次覆盖。

三个训练阶段结束后均要求执行固定 ROI Val 与代表性原图 infer；multitask、multi-finetune 还要求执行 export。

## 服务器环境与数据

服务器 `hand-landmarker-tf29` 的 TensorFlow/Keras 2.9.0、CUDA 11.2、cuDNN 8、ONNX 栈和 RTX 3090 GPU 可用；依赖更新后安装固定版本 `tqdm`。

2026-08-07 对正式分流组合运行 geometry 全量数据审计：`FullEnhance0801/eos_1.0-gate` Train 65,089，`FullEnhanceVal0801/eos-1.0` Val 5,091、Test 2,816；manifest、Registry、路径和 `256×256` 灰度 ROI 解码均无错误，Train 全部为 `pseudo/POS_RUNTIME`。按现有 `performer_cross_split: warn` 策略报告 peak/soar 跨 split 警告。

正式 experiment `national-final-geometry-eos_1.0-gate-r1` 已完成 70 个 epoch；`val_landmark_mae` 最优 checkpoint 位于第 68 epoch，值为 0.0253235，`final.weights.h5` 已绑定该 winner。

固定 ROI `make val` 已完成 5,091 条：landmarks mean pixel error 10.1930 px、mean NME 0.06043、PCK@0.05/0.10/0.15 为 0.5914/0.8377/0.9227，collapse 为 0。22 个 unknown-handedness positive 继续计入 presence/landmarks，只从 handedness 排除；handedness 已知标签准确率为 0.4369。

`InferSource/0718/images` 的 `make infer` 已完成 307 张原图，Palm `eos-1.0` 产生 529 个检测并全部进入 Hand Landmarker，失败 0；输出位于 `HAND_TRAIN_ROOT/inference/national-final-geometry-eos_1.0-gate-r1/geometry`。

warehouse 的 `datasets`、`negative_datasets`、`new_datasets`、`selections` 与 `evaluation.val/test` 均支持多个成员 ID；每个 dataset 类成员可独立选择 proposal variant，每个成员有独立权重。

当前服务器没有已发布 negative dataset 或 hard-positive selection，因此 geometry 已完成，multitask 与 multi-finetune 仍需先完成对应 HLMF 发布物。
