"""Regenerate translation .ts files from source and recompile resources.

PyQt6 dropped `pyrcc` entirely (the Qt Project removed the standalone
resource compiler in Qt 6). This script uses PyQt6's `pyuic6` and
`pylupdate6` for UI / translation extraction, and PySide6's `pyside6-rcc`
for resource compilation, rewriting `PySide6` → `PyQt6` in the output so
the generated module imports `PyQt6.QtCore`. Both PyQt6 and PySide6-Essentials
are declared in the `[dev]` extras in pyproject.toml.

Run from the repo root:
    python scripts/generate_languages.py
"""
import glob
import os
import shutil
import subprocess
import sys

SUPPORTED_LANGUAGES = ["en_US", "vi_VN", "zh_CN"]
TRANSLATIONS_DIR = "anylabeling/resources/translations"
QRC_PATH = "anylabeling/resources/resources.qrc"
RC_PATH = "anylabeling/resources/resources.py"

# Look up tools first next to the active interpreter (venv bin), then $PATH.
_VENV_BIN = os.path.dirname(sys.executable)


def _resolve(cmd):
    candidate = os.path.join(_VENV_BIN, cmd)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    found = shutil.which(cmd)
    if found:
        return found
    sys.exit(
        f"error: '{cmd}' not found in {_VENV_BIN} or on PATH. "
        "Install dev tools with `pip install -e \".[dev]\"`."
    )


def _run(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    pyuic = _resolve("pyuic6")
    pylupdate = _resolve("pylupdate6")
    lrelease = _resolve("pyside6-lrelease")
    rcc = _resolve("pyside6-rcc")

    py_files = glob.glob(os.path.join("**", "*.py"), recursive=True)
    ui_files = glob.glob(os.path.join("**", "*.ui"), recursive=True)

    for ui_file in ui_files:
        py_file = os.path.splitext(ui_file)[0] + "_ui.py"
        _run([pyuic, "-x", ui_file, "-o", py_file])

    for lang in SUPPORTED_LANGUAGES:
        ts_path = f"{TRANSLATIONS_DIR}/{lang}.ts"
        _run([pylupdate, *py_files, "-ts", ts_path])
        _run([lrelease, ts_path])

    _run([rcc, QRC_PATH, "-o", RC_PATH])

    with open(RC_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("from PySide6 import", "from PyQt6 import")
    content = content.replace("import PySide6", "import PyQt6")
    with open(RC_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Rewrote PySide6 → PyQt6 imports in {RC_PATH}")


if __name__ == "__main__":
    main()
