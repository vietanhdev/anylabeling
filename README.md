<p align="center">
  <img alt="AnyLabeling" style="width: 128px; max-width: 100%; height: auto;" src="https://user-images.githubusercontent.com/18329471/232250539-2b15b9ee-5593-41d0-ba22-e0442f314cce.png"/>
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
[![Follow](https://img.shields.io/badge/+Follow-vietanhdev-blue)]([[https://anylabeling.nrl.ai/](https://twitter.com/vietanhdev)](https://twitter.com/vietanhdev))

> +⭐ Follow [vietanhdev](https://twitter.com/vietanhdev) for project updates.

<a href="https://youtu.be/5qVJiYNX5Kk">
  <img alt="AnyLabeling" src="https://raw.githubusercontent.com/vietanhdev/anylabeling/master/assets/screenshot.png"/>
</a>

**Auto Labeling with Segment Anything**

<a href="https://youtu.be/5qVJiYNX5Kk">
  <img style="width: 800px; margin-left: auto; margin-right: auto; display: block;" alt="AnyLabeling-SegmentAnything" src="https://user-images.githubusercontent.com/18329471/236625792-07f01838-3f69-48b0-a12e-30bad27bd921.gif"/>
</a>


- **Youtube Demo:** [https://www.youtube.com/watch?v=5qVJiYNX5Kk](https://www.youtube.com/watch?v=5qVJiYNX5Kk)
- **Documentation:** [https://anylabeling.nrl.ai](https://anylabeling.nrl.ai)

**Features:**

- [x] Image annotation for polygon, rectangle, circle, line and point.
- [x] Auto-labeling YOLOv8, Segment Anything (SAM, SAM2).
- [x] Text detection, recognition and KIE (Key Information Extraction) labeling.
- [x] Multiple languages availables: English, Vietnamese, Chinese.

## Install and Run

### 1. Download and run executable

- Download and run newest version from [Releases](https://github.com/vietanhdev/anylabeling/releases).
- For MacOS:
  - After installing, go to Applications folder
  - Right click on the app and select Open
  - From the second time, you can open the app normally using Launchpad

### Install from Pypi

- Requirements: Python 3.10+. Recommended: Python 3.12.
- Recommended: [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html).

- Create environment:

```bash
conda create -n anylabeling python=3.12
conda activate anylabeling
```

- **(For macOS only)** Install PyQt5 using Conda:

```bash
conda install -c conda-forge pyqt==5.15.9
```

- Install anylabeling:

```bash
pip install anylabeling # or pip install anylabeling-gpu for GPU support
```

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

- Install packages:

```bash
pip install -r requirements-dev.txt
# or pip install -r requirements-macos-dev.txt for MacOS
```

- Generate resources:

```bash
pyrcc5 -o anylabeling/resources/resources.py anylabeling/resources/resources.qrc
```

- Run app:

```bash
python anylabeling/app.py
```

## Build executable

- Install PyInstaller:

```bash
pip install -r requirements-dev.txt
```

- Build:

```bash
bash build_executable.sh
```

- Check the outputs in: `dist/`.

## Contribution

If you want to contribute to **AnyLabeling**, please read [Contribution Guidelines](https://anylabeling.nrl.ai/docs/contribution).

## Star history

[![Star History Chart](https://api.star-history.com/svg?repos=vietanhdev/anylabeling&type=Date)](https://star-history.com/#vietanhdev/anylabeling&Date)

## References

- Labeling UI built with ideas and components from [LabelImg](https://github.com/heartexlabs/labelImg), [LabelMe](https://github.com/wkentaro/labelme).
- Auto-labeling with [Segment Anything Models](https://segment-anything.com/), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM).
- Auto-labeling with [YOLOv8](https://github.com/ultralytics/ultralytics).
