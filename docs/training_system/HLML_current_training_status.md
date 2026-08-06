# HLML 当前状态（2026-08-07）

## 代码与流程

HLML 4.0 已保持 HLMF manifest 零拷贝接口和三阶段 v2 训练契约。训练默认使用 `tqdm` epoch/batch 进度条；multitask 与 multi-finetune 的 `make export` 同时生成 ONNX/A1 审计物和符合官方转换工具要求的 `datasets.zip`。

文件夹推理的 Palm 模型已迁移到 `palm_detector/<model_id>/model_opt.onnx`。全局模型由 `configs/inference.yaml` 的 `palm.model_id` 选择，`make infer ... INFER_ARGS='--palm-model-id <id>'` 可单次覆盖。

三个训练阶段结束后均要求执行固定 ROI Val 与代表性原图 infer；multitask、multi-finetune 还要求执行 export。

## 服务器环境与数据

服务器 `hand-landmarker-tf29` 的 TensorFlow/Keras 2.9.0、CUDA 11.2、cuDNN 8、ONNX 栈和 RTX 3090 GPU 可用；依赖更新后安装固定版本 `tqdm`。

2026-08-07 对正式分流组合运行 geometry 全量数据审计：`FullEnhance0801/eos_1.0-gate` Train 65,089，`FullEnhanceVal0801/eos-1.0` Val 5,091、Test 2,816；manifest、Registry、路径和 `256×256` 灰度 ROI 解码均无错误，Train 全部为 `pseudo/POS_RUNTIME`。按现有 `performer_cross_split: warn` 策略报告 peak/soar 跨 split 警告。RTX 3090 上的 1 epoch/2 steps smoke 已完成，并验证 tqdm、Val 和 checkpoint 链路。

当前服务器没有已发布 negative dataset 或 hard-positive selection，因此 geometry 已具备正式训练条件；multitask 与 multi-finetune 仍需先完成对应 HLMF 发布物。
