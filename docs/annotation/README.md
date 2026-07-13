# 训练接口文档：数据集标注工程

本文件夹下的文件介绍了我们项目是如何进行数据标注的。

如果需要查看更多关于数据标注的信息，可以查看本机如下目录的 Git 仓库：`D:\CICIEC\datasets\HandLandmarkerFab`

这里仅作简要介绍：

- README_OF_HangLandmarksFab: 对应仓库的 README.md 文件，内容完全相同
- hand_landmarker_training_workflow: Hand Landmarker 模型的两阶段训练流程介绍，由于本项目时间紧张，没有足够时间来给数量庞大的训练集做精细的人工复核，因此只能采用这种"pretrain-finetuning"的训练流程进行训练
- dataset_preparation_workflow: 数据集（包含训练集/验证集/测试集）的准备流程与操作步骤手册
- hand_landmarker_train_dataset_processing: 训练集的详细准备流程
- hand_landmarker_val_dataset_processing: 验证集的详细准备流程
- hand_landmarker_test_dataset_processing: 测试集的详细准备流程
- example/: 下面有三个最终训练/验证/测试集标注文件示例
