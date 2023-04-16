<p align="center">
  <img alt="AnyLabeling" style="width: 128px; max-width: 100%; height: auto;" src="https://user-images.githubusercontent.com/18329471/232250539-2b15b9ee-5593-41d0-ba22-e0442f314cce.png"/>
  <h1 align="center">🌟 AnyLabeling 🌟</h1>
  <p align="center">Effortless data labeling with AI support from <b>YOLO</b> and <b>Segment Anything</b>!<p>
  <p align="center"><b>AnyLabeling = LabelImg + Labelme + Improved UI + Auto-labeling</b><p>
</p>

![](https://i.imgur.com/waxVImv.png)

[![PyPI](https://img.shields.io/pypi/v/anylabeling)](https://pypi.org/project/anylabeling)
[![license](https://img.shields.io/github/license/vietanhdev/anylabeling.svg)](https://github.com/vietanhdev/anylabeling/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/vietanhdev/anylabeling.svg)](https://github.com/vietanhdev/anylabeling/issues)
[![Pypi Downloads](https://pepy.tech/badge/anylabeling)](https://pypi.org/project/anylabeling/)

<a href="https://www.youtube.com/watch?v=5iQSGL7ebXE">
  <img alt="AnyLabeling" src="https://user-images.githubusercontent.com/18329471/232266520-1f52cd45-0776-479e-8a3c-4fef144c6b73.png"/>
</a>

**Youtube Demo:** [https://www.youtube.com/watch?v=5iQSGL7ebXE](https://www.youtube.com/watch?v=5iQSGL7ebXE)

## I. Install and run

- Requirements: Python >= 3.8
- Recommended: Miniconda/Anaconda <https://docs.conda.io/en/latest/miniconda.html>

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
pip install anylabeling
```

- Run app:

```bash
anylabeling
```

Or

```bash
python -m anylabeling.app
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

## IV. References

- Labeling UI built with ideas and components from [LabelImg](https://github.com/heartexlabs/labelImg), [labelme](https://github.com/wkentaro/labelme).
- Icons: Flat Icons
