# 初赛冻结版本的相关材料

本文件夹存放初赛冻结版本的相关材料：

- device/: A1 设备端调度程序代码，包含本项目的所有调度程序代码，即封装了 Palm Detector + Hand Landmarker 的预处理、推理以及后处理调度程序，重点关注：
    - "device\src\palm_detector.cpp"
    - "device\src\hand_landmarker.cpp"
    
    其余文件选择性阅读。
    
- palm/: palm detector 模型的定义脚本，以及 onnx 文件
- hand/: hand 模型的定义脚本