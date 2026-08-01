# 固定 Hand ROI 评估契约

Val/Test 的输入是 HLMF 已生成并经 CVAT 决策的 `256×256` Hand ROI。评估入口不加载 Palm Detector、不读取原图、不重建 ROI，也不把 Palm 未生成 ROI 的原图计入分母。

输出包括 mean/median/P90/P95 像素误差、PCK、collapse、presence、handedness，并按 dataset、capture source、label origin、annotation style、距离和光照分组。

Val 可调 presence threshold 并选择唯一 winner。冻结后 Test：

- 只接受 `winner.json` 中的 checkpoint；
- 使用 Val 冻结 threshold；
- 禁止 threshold sweep 和覆盖结果；
- 不能反馈给 mining、训练成员、采样权重或 checkpoint 选择。

当前评估范围不是 Palm 检测或原图级联，因此不报告 Palm 漏检率、部分双手召回率或级联准确率。
