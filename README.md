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
- [x] Auto-labeling with YOLOv5 and Segment Anything.
- [x] Text detection, recognition and KIE (Key Information Extraction) labeling.
- [x] Multiple languages availables: English, Vietnamese, Chinese.

## I. Install and run

### 1. Download and run executable

- Download and run newest version from [Releases](https://github.com/vietanhdev/anylabeling/releases).
- For MacOS:
  - After installing, go to Applications folder
  - Right click on the app and select Open
  - From the second time, you can open the app normally using Launchpad

### 2. Install from Pypi

- Requirements: Python >= 3.8, <= 3.10.
- Recommended: [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html).

- Create environment:

```bash
conda create -n anylabeling python=3.8
conda activate anylabeling
```

- **(For macOS only)** Install PyQt5 using Conda:

```bash
conda install -c conda-forge pyqt==5.15.7
```

- Install anylabeling:

```bash
pip install anylabeling # or pip install anylabeling-gpu for GPU support
```

- Start labeling:

```bash
anylabeling
```

## II. Development

- Generate resources:

```bash
pyrcc5 -o anylabeling/resources/resources.py anylabeling/resources/resources.qrc
```

- Run app:

```bash
python anylabeling/app.py
```

## III. Build executable

- Install PyInstaller:

```bash
pip install -r requirements-dev.txt
```

- Build:

```bash
bash build_executable.sh
```

- Check the outputs in: `dist/`.

## IV. Contribution

If you want to contribute to **AnyLabeling**, please read [Contribution Guidelines](https://anylabeling.nrl.ai/docs/contribution).

## V. Star history

[![Star History Chart](https://api.star-history.com/svg?repos=vietanhdev/anylabeling&type=Date)](https://star-history.com/#vietanhdev/anylabeling&Date)

## VI. References

- Labeling UI built with ideas and components from [LabelImg](https://github.com/heartexlabs/labelImg), [LabelMe](https://github.com/wkentaro/labelme).
- Auto-labeling with [Segment Anything Models](https://segment-anything.com/), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM).
- Auto-labeling with [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv8](https://github.com/ultralytics/ultralytics).
