# Hand Landmarker Training System Building

本仓库将用于构建 Mediapipe Hand Landmarker 的训练系统。

项目背景见：project-9.md

我们本次的任务是构建用于训练我们自己的 Hand Landmarker 模型的训练系统，而不是重新训练 Palm Detector 模型。

## 关键信息

1. Hand Landmarker 模型的定义：`preminilary\hand\model.py`。在后续训练过程中，我们允许对模型内部结构进行修改，但是不允许修改模型的输入输出接口。
2. A1 板端的调度程序：`preminilary\device`， 该程序将会在 A1 板端运行，负责调度 Palm Detector 模型和 Hand Landmarker 模型的运行。我们的 Hand Landmarker 模型的输入输出接口、推理所用参数（例如 ROI 尺寸等）以及推理流程必须完全与板端调度程序保持一致，否则将无法在 A1 板端运行或者影响板端程序的推理精度。
3. 我们已经准备好了数据集标注系统，并且已经标注好了数据集，详细信息见：`docs\annotation`，当然，你也可以通过阅读这个路径下的仓库获取更详细的信息：`D:\CICIEC\datasets\HandLandmarkerFab`
4. 模型两阶段训练流程：`docs\annotation\dataset_preparation_workflow.md`

## 任务要求

接下来，请你帮我们构建好这个 Hand Landmarker 模型的训练系统，至少要包含如下功能：

1. 数据集的读取与处理
2. 模型训练（注意模型不是在本机训练的，而是在服务器上。此外，两阶段训练的脚本都要写好）
3. 模型推理验证（至少应该包含：(1) 基于标准验证/测试集人工标注结果的程序化自动验证，以及 (2) 输入图片文件夹路径（不一定是测试集或者验证集），运行模型推理并且导出带有推理结果图片，由人工复核，注意，无论是哪一种，都需要 Palm Detector 作为前置）
4. 模型的 `.onnx` 格式导出
5. 其余你认为必要的其他功能

## 注意

1. 训练环境和数据集都不在本机，而在服务器上，服务器为：
    
    ```
    ssh -p 19182 root@connect.nmb2.seetacloud.com
    fsUm9Cli1kIj
    ```

    你可以直接通过 ssh 进入服务器，在 `autodl-tmp` 下查看我已经整理好的数据集。
2. 服务器配置：
   
    ```
    镜像 TensorFlow 2.9.0, Python 3.8(ubuntu20.04), CUDA 11.2
    GPU: RTX 3090(24GB) * 1
    CPU: 14 vCPU Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
    内存: 90GB
    硬盘: 系统盘:30 GB 数据盘:免费:50GB SSD 付费:0GB
    数据盘可扩容以获得更多的存储空间
    ```

    最好使用服务器上已经有的 TensorFlow 进行训练系统的构建，如果不得不使用 PyTorch，请给出理由。
3. 训练系统的构建必须基于：
    1. Hand Landmarker 模型的定义：`preminilary\hand\model.py`
    2. Hand Landmarker 数据集标注系统：`D:\CICIEC\datasets\HandLandmarkerFab`
    3. 板端调度程序： `preminilary\device`
    否则会影响数据集的读取、模型训练以及上板工作
4. 除了训练系统的构建，你还需要给出合适的 conda 环境以及依赖创建方法，本机和服务器暂时没有你想要的环境，因此本次任务你暂时无需自己运行验证，而是需要先给出 conda 环境以及依赖的创建方法，由我来进行创建。注意，你不允许自行在本机或者服务器上安装任何依赖。
5. 用 Makefile 以及 yaml 配置文件封装好训练系统的构建、模型训练、模型验证等功能，方便后续使用。(本机和服务器端都有 make 工具)
6. 训练系统除了脚本，还需要写好 README.md 文档，详细介绍训练系统的使用方法、注意事项以及相关信息。如果文档内容太多，不建议直接写入 README.md，而是建议把重要文档存放进入 docs/ 的子目录下，并在 README.md 中给出链接。
7. 后续我可能会准备修改 Hand Landmarker 模型的内部结构，所以你创建的程序不要直接使用 `preminilary\hand\model.py`，最好创建一个独立的目录单独存放模型的定义脚本（并将这个`preminilary\hand\model.py`作为第一版放进去），便于后续存放新的模型定义脚本。


