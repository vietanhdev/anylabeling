# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

AnyLabeling is a desktop image-annotation app built on PyQt6, with an
auto-labeling backend that runs ONNX models (YOLOv5/v8, SAM1/MobileSAM,
SAM2, SAM3) and a CoreML path for SAM2 on macOS. PyPI ships two parallel
packages from the same source tree: `anylabeling` (CPU, default) and
`anylabeling-gpu` (Linux/Windows, swaps in `onnxruntime-gpu`).

## Common commands

```bash
# Run the app from source (no install needed for dev)
python anylabeling/app.py

# Run the installed CLI
anylabeling

# Editable install for development (CPU)
pip install -e ".[dev]"
# GPU dev:    pip install -e ".[gpu,dev]"
# macOS dev:  pip install -e ".[macos,dev]"  # plus conda install -c conda-forge pyqt=6

# Lint + format (ruff config is in pyproject.toml)
ruff check .
ruff format .

# Run all tests
python -m unittest discover -s tests -v

# Run one test file
python -m unittest tests.test_label_colormap -v

# Run one test method
python -m unittest tests.test_label_colormap.TestLabelColormapMutability.test_copy_is_always_writable

# Build a wheel + sdist (CPU). For GPU, sed __preferred_device__ to "GPU" first.
python -m build --sdist --wheel --outdir dist/ .

# Build a standalone executable
bash build_executable.sh   # delegates to PyInstaller via anylabeling.spec
```

App-level CLI flags: `--reset-config`, `--logger-level {debug,info,warning,error,fatal}`,
`--config <path>`, `--output / -O / -o`, `--nodata`, `--autosave`, `--nosortlabels`,
`--flags`, plus a positional `filename` (image or label file). Default user
config lives at `~/.anylabelingrc`.

## High-level architecture

### Entry point and UI tree

`anylabeling/app.py` sets `MKL/NUMEXPR/OMP_NUM_THREADS=1` (workaround for a
macOS-M1 bus error in `np.linalg.solve`) before any heavy imports, then
constructs a `QApplication` and a `MainWindow`. The UI tree is intentionally
shallow:

```
MainWindow                          (anylabeling/views/mainwindow.py)
└── LabelingWrapper                 (anylabeling/views/labeling/label_wrapper.py)
    └── LabelingWidget              (anylabeling/views/labeling/label_widget.py, ~3.2k LOC)
        ├── Canvas                  (anylabeling/views/labeling/widgets/canvas.py)
        ├── AutoLabelingWidget      (drives ModelManager from the UI side)
        ├── LabelDialog / Brightness / FileDialogPreview / ZoomWidget …
        └── ExportDialog
```

`LabelingWidget` is the "god widget" — it owns the file list, the canvas, the
toolbars, the shape list, the label list, file I/O, undo/redo, and most
keybindings. When in doubt, that file is where things live.

### Auto-labeling pipeline

```
anylabeling/services/auto_labeling/
├── registry.py          # @ModelRegistry.register("yolov8") decorator → singleton dict
├── model.py             # abstract Model(QObject); predict_shapes() returns AutoLabelingResult
├── model_manager.py     # ModelManager(QObject): loads models.yaml, downloads weights,
│                        # dispatches predict_shapes_threading()
├── types.py             # AutoLabelingResult, AutoLabelingMode (point/rectangle, ADD/REMOVE)
├── lru_cache.py         # image-embedding cache for SAM-family models
├── segment_anything.py  # variant detector — picks SAM1/SAM2/SAM3 from ONNX inputs/config
├── sam_onnx.py          # SAM1 / MobileSAM ONNX runner
├── sam2_onnx.py         # SAM2 ONNX runner
├── sam3_onnx.py         # SAM3 ONNX runner (text + geometric prompts)
├── sam2_coreml.py       # macOS CoreML path for SAM2.1
└── yolov5.py / yolov8.py
```

Two registry-relevant facts:

- Concrete models register themselves via `@ModelRegistry.register("type-name")`
  at import time. `anylabeling/services/auto_labeling/__init__.py` imports
  every module so the side-effects fire — adding a new model means importing
  it here too.
- `models.yaml` (`anylabeling/configs/auto_labeling/models.yaml`) is the
  catalog the UI reads. Each entry has `name`, `display_name`, `type`
  (matches a registry key), `download_url`, plus model-specific fields like
  `encoder_model_path`, `decoder_model_path`, `input_size`. New model = add
  an entry here *and* a registered class.

Weights live under `~/anylabeling_data/models/<name>/` after first download.

### CPU / GPU / macOS packaging

Static metadata is in `pyproject.toml`. `setup.py` is a small shim that
reads `__preferred_device__` from `anylabeling/app_info.py` and, when set
to `"GPU"` on non-Darwin, overrides the package name to `anylabeling-gpu`
and swaps `onnxruntime` for `onnxruntime-gpu`. The publish workflows
(`.github/workflows/python-publish-{cpu,gpu}.yml`) `sed` that constant
just before building, so both wheels come out of the same source tree.

`pyproject.toml` excludes `PyQt6` on Darwin
(`PyQt6>=...; platform_system != 'Darwin'`). macOS users install PyQt
through conda. The macOS extra is `[macos]` (currently `coremltools==8.3.0`).

### Qt resources and translations

- `anylabeling/resources/resources.qrc` (XML) compiles to `resources.py`.
- `anylabeling/resources/translations/{en_US,vi_VN,zh_CN}.{ts,qm}`.
- `scripts/generate_languages.py` extracts translatable strings into `.ts`
  files and runs `pyuic6` on `.ui` files.
- `scripts/compile_languages.py` calls `lrelease` to produce `.qm` files,
  and then `pyrcc5` to rebuild `resources.py`.

Note: the project migrated from PyQt5 to PyQt6 (commit `9735fe8`), but
`scripts/compile_languages.py` still calls `pyrcc5`. PyQt6 does not ship a
`pyrcc6`; one common workaround is to keep `pyrcc5` from a PyQt5-tools
sideload, or vendor the resource bytes. `generate_languages.py` already
references `pyrcc6`. Treat this script pair as inconsistent and fix
deliberately when touching it.

### Tests

`tests/` is plain `unittest`. Notable files:

- `tests/test_label_colormap.py` — regression test for issue #227
  (`imgviz.label_colormap()` returns read-only on imgviz>=2.0; the call
  site needs `.copy()`).
- `tests/test_real_inference.py` — end-to-end ONNX inference for
  SAM1/SAM2/SAM3/YOLOv8. Each class skips itself if its model files are
  not under `~/anylabeling_data/models/`. The SAM3 text-prompt tests look
  for `../samexporter/images/truck.jpg` (sibling-repo path) and silently
  fall back to `sample_images/evan-foley-...jpg` (no truck), which makes
  three SAM3 tests fail — see step 3 of the playbook below.

## Pre-publish local experiments

Run these **before tagging a release** (`vX.Y.Z`). The CI matrix in
`.github/workflows/tests.yml` already gates publish on every tag push, but
running locally first is faster and catches obvious dep-resolution
failures before burning CI minutes.

### 1. Fresh-venv install with latest deps

The point of a *fresh* venv is to let pip resolve every dependency to the
newest version compatible with `pyproject.toml` — this is what end users
get on `pip install anylabeling[-gpu]`, and it is exactly the path that
produced the `imgviz>=2.0` read-only crash in #227.

```bash
python -m venv /tmp/anylabeling-check
/tmp/anylabeling-check/bin/pip install --upgrade pip
/tmp/anylabeling-check/bin/pip install .
```

Watch for: any wheel that fails to build, any dep that pip cannot resolve.

### 2. Run the full unittest suite

```bash
/tmp/anylabeling-check/bin/python -m unittest discover -s tests -v
```

Expected: all tests pass; `test_real_inference` cases skip cleanly when
model files are not on disk — that is fine. Step 3 below covers running
those tests with real models.

### 3. (Recommended) Real-model inference

`tests/test_real_inference.py` exercises ONNX inference end-to-end for
SAM1 / SAM2 / SAM3 / YOLOv8. Each test class skips itself when its model
files are missing, so download whichever you can validate on the local
machine. Models live under `~/anylabeling_data/models/`.

```bash
mkdir -p ~/anylabeling_data/models && cd ~/anylabeling_data/models

# YOLOv8n   (~13 MB)
curl -sL -o /tmp/yolov8n.zip https://github.com/vietanhdev/anylabeling-assets/releases/download/v0.4.0/yolov8n-r20230415.zip
mkdir -p yolov8n-r20230415 && unzip -q -o /tmp/yolov8n.zip -d yolov8n-r20230415

# MobileSAM (~37 MB)
curl -sL -o /tmp/msam.zip https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/mobile_sam_20230629.zip
mkdir -p mobile_sam_20230629 && unzip -q -o /tmp/msam.zip -d mobile_sam_20230629

# SAM2 hiera-tiny (~155 MB)
curl -sL -o /tmp/sam2.zip https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/main/sam2_hiera_tiny.zip
mkdir -p sam2_hiera_tiny_20240803 && unzip -q -o /tmp/sam2.zip -d sam2_hiera_tiny_20240803

# SAM3 ViT-H (~3.4 GB — only needed when SAM3 code paths changed)
curl -sL -o /tmp/sam3.zip https://huggingface.co/vietanhdev/segment-anything-3-onnx-models/resolve/main/sam3_vit_h.zip
mkdir -p sam3_vit_h_20260220 && unzip -q -o /tmp/sam3.zip -d sam3_vit_h_20260220
```

The SAM3 text-prompt tests need a truck image at the sibling-repo path:

```bash
mkdir -p ../samexporter/images
curl -sL -o ../samexporter/images/truck.jpg \
  https://raw.githubusercontent.com/vietanhdev/samexporter/main/images/truck.jpg
```

Then re-run the inference tests:

```bash
/tmp/anylabeling-check/bin/python -m unittest tests.test_real_inference -v
```

Source of truth for model URLs is
`anylabeling/configs/auto_labeling/models.yaml`.

### 4. Smoke-test the import chain that users hit at startup

This is the *exact* path that crashed in #227. If it imports clean against
freshly resolved deps, the package will at least start.

```bash
QT_QPA_PLATFORM=offscreen /tmp/anylabeling-check/bin/python -c "
from anylabeling.views.labeling import label_widget
from anylabeling import app
print('startup imports OK')
"
```

### 5. Repeat against every supported Python (3.11, 3.12, 3.13)

PyPI ships one wheel that has to work on every Python listed in
`pyproject.toml` classifiers. Use `uv` to spin them up quickly:

```bash
uv python install 3.11 3.12 3.13
for v in 3.11 3.12 3.13; do
  PY=$(uv python find $v)
  VENV=/tmp/al-py${v//./}
  rm -rf $VENV && $PY -m venv $VENV
  $VENV/bin/pip install --upgrade pip --quiet
  $VENV/bin/pip install . --quiet
  $VENV/bin/python -m unittest discover -s tests
done
```

### 6. Then push and let CI confirm cross-platform

The matrix in `.github/workflows/tests.yml` runs steps 1, 2, 4 on
Ubuntu + Windows + macOS × Python 3.11/3.12/3.13. The publish workflows
(`python-publish-cpu.yml`, `python-publish-gpu.yml`, `release.yml`) all
declare `needs: test`, so a red matrix blocks the PyPI upload and the
GitHub release binary builds. Step 3 (real-model inference) is *not*
automated in CI because the SAM3 model alone is 3.4 GB — run it locally
when touching ONNX inference, model loading, or preprocessing code.

## Why this gate exists

`anylabeling-gpu==0.4.30` shipped to PyPI broken because no automated test
ran `pip install .` against current dep floors before publish. The fix in
`label_widget.py:45` (call `.copy()` on `imgviz.label_colormap()`) had a
regression test in `tests/test_label_colormap.py`, but nothing executed it
on the publish path. The workflows in `.github/workflows/` now do.

When adding a new dependency or raising a floor, **assume it can break
import-time code paths** — read-only numpy arrays, removed deprecated
APIs, changed default dtypes — and rely on the steps above to catch it.
