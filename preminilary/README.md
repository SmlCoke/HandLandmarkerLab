# Palm 与 A1 参考材料

本文件夹只保留仍参与当前端到端契约核验的 A1 设备端参考材料：

- device/: A1 设备端调度程序代码，包含本项目的所有调度程序代码，即封装了 Palm Detector + Hand Landmarker 的预处理、推理以及后处理调度程序，重点关注：
    - "device\src\palm_detector.cpp"
    - "device\src\hand_landmarker.cpp"
    
    其余文件选择性阅读。
    
- Palm Detector 的当前模型定义和本地 ONNX 资产统一位于仓库根目录 `palm_detector/<model_id>/`；
- 当前 Hand 模型只在仓库根目录的 `models/hand_landmarker/v2.py` 定义。旧版 LeakyReLU Hand 定义已删除，不能再作为训练或导出入口。
