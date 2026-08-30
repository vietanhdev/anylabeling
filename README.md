<p align="center">
  <img alt="AnyLabeling" style="width: 128px; height: 128px; height: auto;" src="https://github.com/user-attachments/assets/847e47e6-acf0-4f96-9ed9-5485ab405ae0"/>
  <h1 align="center">🌟 AnyLabeling 🌟</h1>
  <p align="center">Effortless data labeling with AI support from <b>YOLO</b> and <b>Segment Anything</b>!</p>
  <p align="center"><b>AnyLabeling = LabelImg + Labelme + Improved UI + Auto-labeling</b></p>
</p>

![](https://user-images.githubusercontent.com/18329471/234640541-a6a65fbc-d7a5-4ec3-9b65-55305b01a7aa.png)

[![PyPI](https://img.shields.io/pypi/v/anylabeling)](https://pypi.org/project/anylabeling)
[![license](https://img.shields.io/github/license/vietanhdev/anylabeling.svg)](https://github.com/vietanhdev/anylabeling/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/vietanhdev/anylabeling.svg)](https://github.com/vietanhdev/anylabeling/issues)
[![Pypi Downloads](https://pepy.tech/badge/anylabeling)](https://pypi.org/project/anylabeling/)
[![Documentation](https://img.shields.io/badge/Read-Documentation-green)](https://anylabeling.nrl.ai/)
[![Follow](https://img.shields.io/badge/+Follow-vietanhdev-blue)](https://twitter.com/vietanhdev)

[![AnyLearning — open-source, offline data labeling and local model training](https://raw.githubusercontent.com/vietanhdev/anylabeling/a128499e1b5808dc9457a409af9514be840eff8f/assets/anylearning-oss-banner.webp)](https://github.com/nrl-ai/anylearning-oss)


<a href="https://youtu.be/5qVJiYNX5Kk">
  <img alt="AnyLabeling" src="https://raw.githubusercontent.com/vietanhdev/anylabeling/master/assets/screenshot.png"/>
</a>

**Auto Labeling with Segment Anything**

<a href="https://youtu.be/5qVJiYNX5Kk">
  <img style="width: 800px; margin-left: auto; margin-right: auto; display: block;" alt="AnyLabeling-SegmentAnything" src="https://user-images.githubusercontent.com/18329471/236625792-07f01838-3f69-48b0-a12e-30bad27bd921.gif"/>
</a>


- **Youtube Demo:** [https://www.youtube.com/watch?v=5qVJiYNX5Kk](https://www.youtube.com/watch?v=5qVJiYNX5Kk)
- **Documentation:** [https://anylabeling.nrl.ai](https://anylabeling.nrl.ai)
- **Download:** [https://anylabeling.nrl.ai/download](https://anylabeling.nrl.ai/download)

## Features

- [x] Image annotation for polygon, rectangle, circle, line and point.
- [x] Auto-labeling with **YOLOv8** (object detection).
- [x] Auto-labeling with **Segment Anything** family:
  - **SAM** (ViT-B / ViT-L / ViT-H) and **MobileSAM**
  - **SAM 2** and **SAM 2.1** (Hiera-Tiny / Small / Base+ / Large)
  - **SAM 3** (ViT-H) — open-vocabulary segmentation with text prompts
- [x] Text detection, recognition and KIE (Key Information Extraction) labeling.
- [x] Hardware acceleration with CUDA, CoreML, DirectML, OpenVINO, and vendor NPU providers.
- [x] Multiple languages available: English, Vietnamese, Chinese.

### Supported Models

| Model | Prompt Types | Notes |
|-------|-------------|-------|
| SAM ViT-B / ViT-L / ViT-H | Point, Rectangle | Original Segment Anything |
| MobileSAM | Point, Rectangle | Lightweight SAM |
| SAM 2 Hiera-Tiny / Small / Base+ / Large | Point, Rectangle | Meta SAM 2 |
| SAM 2.1 Hiera-Tiny / Small / Base+ / Large | Point, Rectangle | Improved SAM 2 |
| SAM 3 ViT-H | **Text**, Point, Rectangle | Open-vocabulary; text drives detection |
| YOLOv8n / s / m / l / x | — | Object detection & auto-labeling |

Required model weights are downloaded automatically on first use.

## Latest Release

[AnyLabeling v0.4.42](https://github.com/vietanhdev/anylabeling/releases/tag/v0.4.42) is the current stable release. It includes cross-platform accelerator selection, packaged CUDA/CoreML support, stability fixes for the file dialog and canvas, 16-bit TIFF editing, SAM 3 frozen-build support, and corrected Linux/macOS packaging.

All six v0.4.42 CPU and accelerated artifacts were checksum-verified and launch-tested on Linux, Windows, and Apple Silicon macOS. Avoid the superseded v0.4.40 macOS and Linux artifacts.

Use the [Download page](https://anylabeling.nrl.ai/download) for direct platform links, or see [all GitHub releases](https://github.com/vietanhdev/anylabeling/releases).

## Install and Run

### 1. Download and run executable

- Download the latest build from the [Download page](https://anylabeling.nrl.ai/download) or [GitHub Releases](https://github.com/vietanhdev/anylabeling/releases/latest).

| Platform | CPU | Accelerated |
| --- | --- | --- |
| Linux x64 | `AnyLabeling-Linux-CPU-x64` | `AnyLabeling-Linux-GPU-x64` (NVIDIA CUDA) |
| Windows x64 | `AnyLabeling-Windows-CPU-x64.exe` | `AnyLabeling-Windows-GPU-x64.exe` (NVIDIA CUDA) |
| Apple Silicon macOS | `AnyLabeling-macOS-CPU.zip` | `AnyLabeling-macOS-GPU.zip` (CoreML) |

For macOS, preserve the archive's symlinks while extracting it. See the [macOS folder mode instructions](docs/macos_folder_mode.md).

### 2. Install from PyPI

- Requirements: Python 3.11+. Recommended: Python 3.12.
- Recommended: [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html).

- Create environment:

```bash
conda create -n anylabeling python=3.12
conda activate anylabeling
```

- **(For macOS only)** Install PyQt6 using Conda:

```bash
conda install -c conda-forge pyqt=6
```

- Install anylabeling:

```bash
pip install anylabeling
```

For NVIDIA CUDA inference on Linux or Windows, use the GPU distribution in a
fresh environment:

```bash
pip install anylabeling-gpu
```

Apple Silicon users can enable both ONNX Runtime CoreML and native CoreML SAM2
models with:

```bash
pip install "anylabeling[macos]"
export ANYLABELING_DEVICE=COREML
```

AnyLabeling automatically selects CUDA for Linux/Windows GPU builds and CoreML
for the macOS GPU build, with CPU fallback for unsupported model operations.
Advanced ONNX Runtime packages can be selected with `ANYLABELING_DEVICE`;
supported values include
`CUDA`, `COREML`, `DIRECTML`, `ROCM`, `MIGRAPHX`, `OPENVINO`, `TENSORRT`,
`CANN`, `QNN`, `VITISAI`, and `WEBGPU`. NPU aliases include `NPU`,
`INTEL_NPU`, `QUALCOMM_NPU`, `AMD_NPU`, and `ASCEND_NPU`. On Windows
PowerShell, set the override with `$env:ANYLABELING_DEVICE = "DIRECTML"`.

The GPU distribution includes pip-managed CUDA 12 and cuDNN runtime libraries,
so a compatible NVIDIA driver is sufficient; a system CUDA toolkit is not
required.

NPU execution requires the matching vendor ONNX Runtime package in a fresh,
dedicated environment. For example, Intel Core Ultra systems use
`onnxruntime-openvino` with `ANYLABELING_DEVICE=INTEL_NPU`; Qualcomm Snapdragon
Windows ARM64 systems use `onnxruntime-qnn` with
`ANYLABELING_DEVICE=QUALCOMM_NPU`. Replace the default `onnxruntime` package,
because ONNX Runtime requires only one variant in an environment. Qualcomm HTP
models generally need QDQ quantization, and support still depends on the
operator coverage of the selected model.

See the [Hardware Acceleration guide](https://anylabeling.nrl.ai/docs/gpu) for isolated DirectML, OpenVINO, CUDA, CoreML, and NPU environment setup.

- Start labeling:

```bash
anylabeling
```

## Documentation

**Website:** [https://anylabeling.nrl.ai](https://anylabeling.nrl.ai)/

### Applications

| **Object Detection** | **Recognition** | **Facial Landmark Detection** | **2D Pose Estimation** |
| :---: | :---: | :---: | :---: |
| <img src='https://user-images.githubusercontent.com/72010077/273488633-fc31da5c-dfdd-434e-b5d0-874892807d95.png' height="126px" width="180px"> |  <img src='https://user-images.githubusercontent.com/72010077/277396071-79daec2c-6b0a-4d42-97cf-69fd098b3400.png' height="126px" width="180px"> |  <img src='https://user-images.githubusercontent.com/61035602/206095684-72f42233-c9c7-4bd8-9195-e34859bd08bf.jpg' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206100220-ab01d347-9ff9-4f17-9718-290ec14d4205.gif' height="126px" width="180px"> |
|  **2D Lane Detection** | **OCR** | **Medical Imaging** | **Instance Segmentation** |
| <img src='https://user-images.githubusercontent.com/72010077/273764641-65f456ed-27ce-4077-8fce-b30db093b988.jpg' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273421210-30d20e08-3b72-4f4d-8976-05b564e13d87.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273764318-e8b6a197-e733-478e-a210-e4386bafa1e4.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206095831-cc439557-1a23-4a99-b6b0-b6f2e97e8c57.jpg' height="126px" width="180px"> |
|  **Image Tagging** | **Rotation** | **And more!** |
| <img src='https://user-images.githubusercontent.com/72010077/277670825-8797ac7e-e593-45ea-be6a-65c3af17b12b.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/277395955-aab54ea0-88f5-41af-ab0a-f4158a673f5e.png' height="126px" width="180px"> | Your applications here! |
## Development

- Install the project and development tools in a dedicated environment:

```bash
python -m pip install -e ".[dev]"
```

- Recompile translations and Qt resources when they change:

```bash
python scripts/compile_languages.py
```

- Run app:

```bash
python anylabeling/app.py
```

## Build executable

- Install PyInstaller:

```bash
python -m pip install -e ".[dev]"
python -m pip install pyinstaller
```

- Build:

```bash
bash scripts/build_executable.sh
```

- Check the outputs in: `dist/`.

## Contribution

If you want to contribute to **AnyLabeling**, please read [Contribution Guidelines](https://anylabeling.nrl.ai/docs/contribution).

## Star history

[![Star History Chart](https://www.vietanh.dev/api/stars/vietanhdev/anylabeling.svg)](https://github.com/vietanhdev/anylabeling/stargazers)

## References

- Labeling UI built with ideas and components from [LabelImg](https://github.com/heartexlabs/labelImg), [LabelMe](https://github.com/wkentaro/labelme).
- Auto-labeling with [Segment Anything](https://segment-anything.com/) (SAM, SAM 2, SAM 2.1, SAM 3), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM).
- Auto-labeling with [YOLOv8](https://github.com/ultralytics/ultralytics).
- Icons from FlatIcon: [DinosoftLabs](https://www.flaticon.com/free-icons/sun "sun icons"), [Freepik](https://www.flaticon.com/free-icons/moon "moon icons"), [Vectoricons](https://www.flaticon.com/free-icons/system "system icons"), [HideMaru](https://www.flaticon.com/free-icons/ungroup "ungroup icons").
