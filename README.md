<p align="center">
  <h1 align="center">🌟 AnyLabeling 🌟</h1>
  <p align="center">Effortless data labeling with AI support from <b>YOLO</b> and <b>Segment Anything</b>!<p>
  <p align="center"><b>AnyLabeling = LabelImg + Labelme + Improved UI + Auto-labeling</b><p>
</p>

![](https://i.imgur.com/waxVImv.png)

[![PyPI](https://img.shields.io/pypi/v/anylabeling)](https://pypi.org/project/anylabeling)
[![license](https://img.shields.io/github/license/vietanhdev/anylabeling.svg)](https://github.com/vietanhdev/anylabeling/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/vietanhdev/anylabeling.svg)](https://github.com/vietanhdev/anylabeling/issues)
[![Pypi Downloads](https://pepy.tech/badge/anylabeling)](https://pypi.org/project/anylabeling/)

[![AnyLabeling 0.0.2](https://img.youtube.com/vi/5iQSGL7ebXE/0.jpg)](https://www.youtube.com/watch?v=5iQSGL7ebXE)


## I. Install and run

- Requirements: Python >= 3.8
- Recommended: Miniconda/Anaconda <https://docs.conda.io/en/latest/miniconda.html>

- Create environment:

```
conda create -n anylabeling python=3.8
conda activate anylabeling
```

- **(For macOS only)** Install PyQt5 using Conda:

```
conda install -c conda-forge pyqt==5.15.7
```

- Install anylabeling:

```
pip install anylabeling
```

- Run app:

```
anylabeling
```

Or

```
python -m anylabeling.app
```

## II. Development

- Generate resources:

```
pyrcc5 -o anylabeling/resources/resources.py anylabeling/resources/resources.qrc
```

- Run app:

```
python anylabeling/app.py
```

## III. References

- Labeling UI built with ideas and components from [LabelImg](https://github.com/heartexlabs/labelImg), [labelme](https://github.com/wkentaro/labelme).
- Icons: Flat Icons
