# 配置文件说明

- `train_smoke.yaml`：用于快速验证训练管线是否正常
- `export_preflight.yaml`：用于 pretrain 阶段前的导出测试，验证算子是否合规
- `train_geometry.yaml`：pretrain: geometry 阶段的训练配置
- `curate_pretrain.yaml`：训练集提纯以及 review 的配置
- `train_multitask.yaml`：pretrain: multitask 阶段的训练配置
- `eval_validation.yaml`: 用于验证阶段的评估配置
- `eval_test.yaml`: 用于测试阶段的评估配置
- `infer.yaml`: 用于推理的配置