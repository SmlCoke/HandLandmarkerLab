# HLML 4.0 Final download guide

`HLML-4.0-final` is the final reproducible archive of HandLandmarkerLab for the AetherSign National Finals. After cloning the repository and checking out the tag, download the required runtime assets from the matching [GitHub Release](https://github.com/SmlCoke/HandLandmarkerLab/releases/tag/HLML-4.0-final).

> The Git tag contains code, configuration, tests, documentation, and lightweight visual assets only. Restore the production datasets, training roots, checkpoints, ONNX/m1model files, conversion packages, and server environment separately from the competition archive.

## 1. Get the archived source

```bash
git clone https://github.com/SmlCoke/HandLandmarkerLab.git
cd HandLandmarkerLab
git fetch origin --tags
git checkout HLML-4.0-final
```

## 2. Download Release assets

Download these files from the Release page:

| File | Required by the main flow | Purpose |
| --- | --- | --- |
| `palm_detector-HLML-4.0-final.zip` | Yes, for `make infer` and Eos → Iris folder-cascade inference | Eos Palm Detector runtime assets |
| `Iris-1.1-HLML-4.0-final.zip` | No | Trained Iris-1.1 models for testing |
| `Iris-2.0-HLML-4.0-final.zip` | No | Trained Iris-2.0 model family for testing |
| `HLML-4.0-final-assets-SHA256SUMS.txt` | Recommended | SHA256 checksums for the three archives |

`Iris-1.1` and `Iris-2.0` are trained-model release artifacts. They **do not participate in the HLML training, evaluation, inference, or export main flow**. Download them only when they are needed for model testing, and do not modify any other content under Git's `assets/` directory.

## 3. Install Palm Detector runtime assets

Extracting `palm_detector-HLML-4.0-final.zip` creates a complete `palm_detector/` directory. This Release directory must **replace the `palm_detector/` directory obtained from a direct repository clone**:

```bash
# Run from the repository root. Back up local palm_detector/ changes first, if any.
rm -rf palm_detector
unzip palm_detector-HLML-4.0-final.zip
```

Windows PowerShell:

```powershell
# Run from the repository root. Back up local palm_detector/ changes first, if any.
Remove-Item -LiteralPath .\palm_detector -Recurse
Expand-Archive -LiteralPath .\palm_detector-HLML-4.0-final.zip -DestinationPath .
```

The resulting path must be `./palm_detector/`, with no extra enclosing directory. Then follow the “Common operations” section in the root README to configure the environment and run the desired command.

## 4. Verify downloaded archives

Place the three `.zip` archives and the checksum manifest in the same directory, then run:

```bash
sha256sum -c HLML-4.0-final-assets-SHA256SUMS.txt
```

Windows PowerShell:

```powershell
Get-FileHash .\palm_detector-HLML-4.0-final.zip -Algorithm SHA256
Get-FileHash .\Iris-1.1-HLML-4.0-final.zip -Algorithm SHA256
Get-FileHash .\Iris-2.0-HLML-4.0-final.zip -Algorithm SHA256
```

Compare the output with the corresponding entry in `HLML-4.0-final-assets-SHA256SUMS.txt`. Extract and use an archive only after its checksum matches.
