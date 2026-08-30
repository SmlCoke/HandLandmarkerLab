# HLML 4.0 Final 下载指南

`HLML-4.0-final` 是 AetherSign 全国总决赛阶段 HandLandmarkerLab 的最终可复现代码归档。克隆仓库和切换 tag 后，还需从该 tag 对应的 [GitHub Release](https://github.com/SmlCoke/HandLandmarkerLab/releases/tag/HLML-4.0-final) 下载所需运行资产。

> Git tag 仅保存代码、配置、测试、文档和轻量视觉资产；正式数据集、训练目录、checkpoint、ONNX/m1model、转换数据包和服务器环境仍需从比赛归档独立恢复。

## 1. 获取归档代码

```bash
git clone https://github.com/SmlCoke/HandLandmarkerLab.git
cd HandLandmarkerLab
git fetch origin --tags
git checkout HLML-4.0-final
```

## 2. 下载 Release assets

在 Release 页面下载以下文件：

| 文件 | 是否为主流程必需项 | 用途 |
| --- | --- | --- |
| `palm_detector-HLML-4.0-final.zip` | 是（使用 `make infer` 或 Eos → Iris 文件夹级联时） | Eos Palm Detector 运行资产 |
| `Iris-1.1-HLML-4.0-final.zip` | 否 | 训练产物，可供测试 Iris-1.1 模型 |
| `Iris-2.0-HLML-4.0-final.zip` | 否 | 训练产物，可供测试 Iris-2.0 系列模型 |
| `HLML-4.0-final-assets-SHA256SUMS.txt` | 建议 | 三个压缩包的 SHA256 校验值 |

`Iris-1.1` 和 `Iris-2.0` 均是已训练模型的发布资产，**不参与 HLML 的训练、评估、推理和导出主流程**；按需下载后用于模型测试即可。不要修改 Git 中 `assets/` 的其他内容。

## 3. 安装 Palm Detector 运行资产

解压 `palm_detector-HLML-4.0-final.zip` 后会得到完整的 `palm_detector/` 目录。它是 Release 提供的运行时替代目录，必须**替换直接克隆仓库所得的 `palm_detector/` 文件夹**：

```bash
# 在仓库根目录执行；先自行备份当前 palm_detector/（如有本地改动）
rm -rf palm_detector
unzip palm_detector-HLML-4.0-final.zip
```

Windows PowerShell 可使用：

```powershell
# 在仓库根目录执行；先自行备份当前 palm_detector/（如有本地改动）
Remove-Item -LiteralPath .\palm_detector -Recurse
Expand-Archive -LiteralPath .\palm_detector-HLML-4.0-final.zip -DestinationPath .
```

解压后的目录应为 `./palm_detector/`，而不是额外嵌套的一层目录。完成后按 README 的“常用操作”配置环境，并执行相应命令。

## 4. 校验下载文件

下载三个 `.zip` 文件和校验清单后，在同一目录执行：

```bash
sha256sum -c HLML-4.0-final-assets-SHA256SUMS.txt
```

Windows PowerShell：

```powershell
Get-FileHash .\palm_detector-HLML-4.0-final.zip -Algorithm SHA256
Get-FileHash .\Iris-1.1-HLML-4.0-final.zip -Algorithm SHA256
Get-FileHash .\Iris-2.0-HLML-4.0-final.zip -Algorithm SHA256
```

将输出的哈希值与 `HLML-4.0-final-assets-SHA256SUMS.txt` 中对应条目比较。校验通过后再解压使用。
