# AnyLabeling agent guide

This file describes the current repository architecture and the checks an
automated coding agent must perform. Keep it aligned with `pyproject.toml`,
`anylabeling/app_info.py`, and `.github/workflows/` when those files change.

## Project summary

AnyLabeling is a Python 3.11+ desktop image-annotation application built with
PyQt6. Its auto-labeling service runs YOLOv5/v8, SAM/MobileSAM, SAM 2/2.1, and
SAM 3 models through ONNX Runtime; SAM 2 also has a native CoreML path on
macOS. The repository produces:

- `anylabeling`, the default CPU PyPI distribution;
- `anylabeling-gpu`, the Linux/Windows CUDA PyPI distribution; and
- six standalone CPU/accelerated binaries for Linux, Windows, and Apple
  Silicon macOS.

The application entry point is `anylabeling.app:main`. The version and default
build device are defined in `anylabeling/app_info.py`.

## Non-negotiable working rules

- Use a dedicated AnyLabeling virtual or Conda environment. Do not install
  project dependencies into a shared machine or user environment.
- Preserve unrelated changes in the working tree.
- Fix bugs with a focused regression test. Reproduce the failure first when
  practical, then verify both the focused test and the full suite.
- Set `QT_QPA_PLATFORM=offscreen` for automated Qt tests and smoke tests.
- Use PyQt6 APIs. PySide6 is a development-only resource compiler dependency,
  not the runtime UI toolkit.
- Keep UI operations on the Qt main thread. Background model workers must
  always report errors and release their worker/thread references.
- Do not grow `label_widget.py` for standalone functionality that belongs in a
  focused widget, service, or utility.
- Treat label files and model files as user data: close handles
  deterministically, preserve unknown JSON fields, and avoid destructive
  migration behavior.
- Never claim accelerator support from provider selection alone. Validate the
  provider in an isolated environment and run real inference on matching
  hardware when the runtime path changes.

## Architecture map

The main UI ownership chain is:

```text
anylabeling/app.py
└── views/mainwindow.py: MainWindow
    └── views/labeling/label_wrapper.py: LabelingWrapper
        └── views/labeling/label_widget.py: LabelingWidget
            ├── widgets/canvas.py: shapes, selection, grouping, undo state
            ├── widgets/auto_labeling/auto_labeling.py: model-facing UI
            ├── label_file.py: AnyLabeling JSON serialization
            └── dialogs and supporting widgets
```

`LabelingWidget` owns file navigation, canvas state, labels, actions, and most
save/dirty-state behavior. Canvas mutations that affect persisted annotations
must store an undo snapshot and emit the signal that marks the document dirty.
Grouping and ungrouping are examples covered by `tests/test_canvas_grouping.py`.

The auto-labeling path is:

```text
services/auto_labeling/
├── registry.py          model type registry
├── model.py             base QObject and image loading
├── model_manager.py     downloads, lifecycle, threaded prediction
├── runtime.py           ONNX provider selection/session creation
├── types.py             AutoLabelingResult and prompt modes
├── segment_anything.py  SAM family detection/dispatch
├── sam_onnx.py          SAM 1 and MobileSAM
├── sam2_onnx.py         SAM 2 ONNX
├── sam2_coreml.py       SAM 2 native CoreML
├── sam3_onnx.py         SAM 3
└── yolov5.py/yolov8.py  detection models
```

Concrete model classes register at import time with
`@ModelRegistry.register("type")`. When adding a model, import its module from
`services/auto_labeling/__init__.py` and add a matching entry to
`configs/auto_labeling/models.yaml`. Downloaded weights live in
`~/anylabeling_data/models/<model-name>/`.

Qt resources are declared in `anylabeling/resources/resources.qrc` and
compiled into `resources.py`. Translations are under
`anylabeling/resources/translations/`. After changing icons, `.qrc`, or `.ts`
files, run `python scripts/compile_languages.py`; do not hand-edit generated
resource output.

## Accelerator and package behavior

`services/auto_labeling/runtime.py` is the single provider-selection layer.
`ANYLABELING_DEVICE` overrides the build default. Supported names include
`CPU`, `GPU`, `AUTO`, `CUDA`, `COREML`, `DIRECTML`, `ROCM`, `MIGRAPHX`,
`OPENVINO`, `TENSORRT`, `CANN`, `QNN`, `VITISAI`, `WEBGPU`, and the documented
NPU aliases. Selected accelerators retain CPU fallback where available.

Provider packages must be isolated because ONNX Runtime variants conflict:

- CPU: `onnxruntime`;
- NVIDIA: `onnxruntime-gpu[cuda,cudnn]` (currently `<1.27` for CUDA 12 driver
  compatibility);
- Intel: `onnxruntime-openvino`;
- Windows DirectML: `onnxruntime-directml`;
- other NPUs: the appropriate vendor runtime.

The GPU publish workflow rewrites authoritative PEP 621 metadata in
`pyproject.toml` to produce `anylabeling-gpu` and replace the CPU runtime
dependency. There is no `setup.py`; do not reintroduce packaging logic there.
macOS intentionally excludes pip-installed PyQt6 and uses a separate
`[macos]` extra for CoreML.

Use `scripts/check_accelerator.py` to inspect provider availability and
selection. A provider appearing in `get_available_providers()` does not prove
that its native libraries, device, model operators, or inference path work.

## Dedicated development environment

Create one environment per runtime variant. For the normal CPU development
path on Linux or Windows:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]" "ruff==0.15.2"
```

On Windows, use `.venv\\Scripts\\python.exe`. On macOS, use a dedicated Conda
environment, install `pyqt=6` from conda-forge, then install `.[macos,dev]`.
Never install CPU and accelerator ONNX Runtime wheels into the same test
environment.

Run the application from the environment with either:

```bash
.venv/bin/python anylabeling/app.py
.venv/bin/anylabeling
```

## Required validation

The baseline for every code change is:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -v
.venv/bin/ruff check anylabeling --exclude anylabeling/resources/resources.py
.venv/bin/ruff format --check anylabeling --exclude anylabeling/resources/resources.py
```

Run a focused module during iteration, for example:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest tests.test_label_file -v
```

The suite is standard-library `unittest`. Tests must be deterministic, clean up
temporary files and Qt objects, and avoid network access unless they are
explicit optional integration tests. `tests/test_real_inference.py` skips
models not present under `~/anylabeling_data/models/`; skipped inference tests
are not evidence that a changed model path works.

Add proportional manual validation after automated tests:

- UI or canvas change: launch the app, perform the exact interaction, save,
  reopen, and exercise undo/redo where relevant.
- File I/O change: repeatedly save, overwrite, reload, rename/delete, and run
  with `ResourceWarning` promoted to an error on Windows when handles matter.
- Model lifecycle change: test successful load, malformed config, missing or
  corrupt download, cancellation/retry, and a subsequent valid load.
- Provider change: use a fresh environment, verify the selected provider, run
  representative real inference, and confirm CPU fallback.
- Packaging change: build from a clean checkout, inspect the archive/wheel,
  install it into a second empty environment, and smoke-test the installed
  entry point.

CI runs `.github/workflows/tests.yml` on Ubuntu, Windows, and macOS with Python
3.11, 3.12, and 3.13. All nine jobs must pass. For operating-system-specific
bugs, also test on the affected physical OS when access is available.

## Build and release

Build Python distributions and validate their metadata with:

```bash
.venv/bin/python -m build --sdist --wheel --outdir dist/ .
.venv/bin/python -m twine check dist/*
```

Build a local standalone binary with:

```bash
.venv/bin/python -m pip install pyinstaller
bash scripts/build_executable.sh
```

Tags matching `v*.*.*` start three gated workflows: CPU PyPI publishing, GPU
PyPI publishing, and GitHub Release binary builds. Before tagging:

1. update `__version__` in `anylabeling/app_info.py`;
2. run the full fresh-environment test and lint checks;
3. build and install the wheel in a clean environment;
4. run real model/accelerator checks for affected inference paths; and
5. ensure the working tree and release notes describe exactly what will ship.

After tagging, wait for every workflow. Confirm both PyPI projects, all six
named release assets, archive integrity, file sizes/checksums, and native-runner
launch smoke tests. The expected assets are:

```text
AnyLabeling-Linux-CPU-x64
AnyLabeling-Linux-GPU-x64
AnyLabeling-Windows-CPU-x64.exe
AnyLabeling-Windows-GPU-x64.exe
AnyLabeling-macOS-CPU.zip
AnyLabeling-macOS-GPU.zip
```

Never move a tag after publishing. Correct a bad release with a new patch
version. Keep the README's “Latest Release” section and the documentation
download page in sync with the latest stable tag.

## Current regression invariants

- Use scoped Qt enum members (for example,
  `QFileDialog.FileMode.ExistingFile`) so PyQt6 dialog subclasses do not depend
  on inherited enum aliases.
- Every `io_open()` caller must leave its file closed after the context exits;
  Windows must be able to overwrite and delete a label immediately afterward.
- Group/ungroup mutations must persist, mark the document dirty, and remain
  undoable.
- Invalid images in SAM preload must be skipped without killing the preload
  worker.
- Model download/load failures must clean up worker state so another model can
  be selected without restarting the app.
- Provider choice must be deterministic and append CPU fallback when
  available; requested unavailable providers must fail over cleanly.

## Change hygiene

Keep changes small and explain user-visible behavior in the PR. Reference the
issue, include the reproduction and regression test, report automated and
manual validation, and call out OS or hardware limitations honestly. Do not
mix generated files, caches, local environments, model weights, or build
outputs into commits.
