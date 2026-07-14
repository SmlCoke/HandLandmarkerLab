# 模型转化：.m1model 模型生成

## 1. 准备 onnx 模型

请提前准备好 .onnx 文件，确保模型文件为 .onnx 格式，且结构符合 A1 的输入输出要求（如输入张量名称、维度等）。

目前只支持 onnx 模型格式，其他框架的模型请先自行转化为 onnx 模型。

得到 onnx 模型后，将 onnx 模型上传至公司 AI 助手，转化成功后即可得到 .m1model 模型。AI 助手内部调用的模型转化方法对用于不可见。

## 2. 自定义数据集

在进行模型转化时，上传一些校准的数据集，有助于校验模型转化的精度损失。

请严格按照本章节流程操作，确保数据格式、目录结构和数量满足后续模型校准（calibration）与评测（evaluation）流程的要求。

### 2.1 数据格式要求

- 存储格式：.npy
- 保存方法：必须使用 np.save(), 禁止使用 np.tofile() 或者其他二进制写入方式
- Tensor 排布：统一为：NxCxHxW，建议 N = 1
- 数据类型：常见为 Float32，与模型输入保持一致

### 2.2 预处理代码示例

```python
import os
import cv2
import numpy as np

# =========================
# 配置
# =========================
INPUT_DIR = "data"
OUTPUT_DIR = "calibrate_datasets"
TARGET_SIZE = (256, 256)

# 归一化方式：
# 1: [0, 1]
# 2: [-1, 1]  （Blaze / MediaPipe 常用）
NORMALIZE_MODE = "0_1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 归一化函数
# =========================
def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if NORMALIZE_MODE == "0_1":
        img = img / 255.0
    elif NORMALIZE_MODE == "minus1_1":
        img = img / 127.5 - 1.0
    else:
        raise ValueError("Unknown NORMALIZE_MODE")
    return img


# =========================
# 主处理逻辑
# =========================
for name in os.listdir(INPUT_DIR):
    if not name.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(INPUT_DIR, name)
    print(f"Processing: {img_path}")

    # 1. 读取图片（OpenCV 默认 BGR）
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  [WARN] Failed to read {img_path}")
        continue

    # 2. BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 3. Resize 到 (256, 256)
    img_rgb = cv2.resize(img_rgb, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

    # 4. 像素归一化
    img_norm = normalize(img_rgb)
    img_norm = img_norm.transpose(2, 0, 1)  # (3, 256, 256)
    img_norm = np.expand_dims(img_norm, axis=0)  # (1, 3, 256, 256)


    # 5. 保存为 .npy
    out_name = os.path.splitext(name)[0] + ".npy"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    np.save(out_path, img_norm)

    print(f"  Saved: {out_path}, shape={img_norm.shape}, dtype={img_norm.dtype}")

print("Done.")
```

### 2.3 数据集目录结构规范

不管是想上传校准集还是评测集，最终输入给模型的必须根据数据集的文件树，将目录打包为 datasets.zip 文件。 具体要求如下：

- 数据集必须以文件夹形式组织；
- 文件名无特殊要求；
- 文件夹名称和层级结构必须严格一致；
- 所有 .npy 文件应直接位于对应子目录下，不允许再嵌套子文件夹；
- 无多余文件（如 .jpg、.png、.DS_Store 等）。

标准目录结构：

```
datasets/ 
├── calibrate_datasets/ 
│   ├── img_0001.npy 
│   ├── img_0002.npy 
│   ├── img_0003.npy 
│   └── ... 
└── evaluate_datasets/     
   ├── img_0001.npy    
   ├── img_0002.npy    
   ├── img_0003.npy    
   └── ...
```

数量要求（强制）

| 子目录名 | 样本数量 |
| --- | --- |
| calibrate_datasets | 至少 20 |
| evaluate_datasets	 | 至少 10 |

注意：数量不足会导致校准或评测流程失败，在打包前请务必检查。

## 3. 当前训练系统的自动生成方式

本仓库已把上述输入数据制作集成到 Hand Landmarker 的 ONNX `export` 步骤。默认执行：

```bash
make export
```

会在 ONNX 通过接口、算子和数值一致性校验后，从当前阶段的 canonical 数据中只读抽样并同时生成：

- `calibrate_datasets`：从当前阶段 Train 抽取 100 个样本；
- `evaluate_datasets`：从 Val、Test 各抽取 25 个样本，共 50 个；
- 每个文件均为可直接送入 Hand Landmarker 的 `float32 (1,1,256,256)` NCHW tensor，像素值为灰度 `uint8/255`；
- 不运行 Palm Detector、不读取原图、不重新裁切 ROI，也不保存模型输出。

抽样不是运行时随机抽样，而是先按配置字段分层，再按 canonical record ID 的稳定 SHA-256 排序，因而源数据与配置不变时结果可复现。校准只使用 Train；Val/Test 仅进入转换工具的评测输入集。源 JSONL 和源 ROI 始终只读，生成文件与 Train/Val/Test 目录完全隔离。

默认产物位于：

```text
${HAND_DATA_ROOT}/hand_landmarker_runs/v1/export/<stage>/model_conversion/
├── datasets/
│   ├── calibrate_datasets/
│   │   └── img_*.npy
│   └── evaluate_datasets/
│       └── img_*.npy
├── datasets.zip
├── datasets_manifest.json
└── datasets_report.json
```

`datasets/` 内严格只有两级规定目录和 `.npy` 文件；来源追踪清单与报告放在它的外部。`datasets.zip` 内也只包含以 `datasets/` 为根的规定树，可直接交给转换流程。`manifest` 记录源配置、record ID、源 ROI 与 NPY 的 SHA-256，不包含模型输出。

若只想重建数据包而不运行 TensorFlow/ONNX 导出，可执行：

```bash
make conversion-datasets
# 或显式阶段：
make conversion-datasets-pretrain
make conversion-datasets-finetune
```

通用命令默认是 `pretrain`，只读取 pretrain Train、Val、Test，不要求 finetune 数据存在。finetune 命令才会读取 finetune Train。目标目录已存在时命令会失败；确认可替换后显式传入 `CONVERSION_ARGS=--overwrite`，或在 export 时使用 `EXPORT_ARGS=--overwrite`。
