"""Compile translations and rebuild Qt resources.

PyQt6 dropped `pyrcc` entirely (the Qt Project removed the standalone
resource compiler in Qt 6). The well-known workaround is to invoke
PySide6's `pyside6-rcc` and rewrite the import line so the output
imports `PyQt6.QtCore` instead of `PySide6.QtCore`. PySide6-Essentials
is declared in the `[dev]` extras in pyproject.toml.

Run from the repo root:
    python scripts/compile_languages.py
"""
import os
import shutil
import subprocess
import sys

SUPPORTED_LANGUAGES = ["en_US", "vi_VN", "zh_CN"]
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
    lrelease = _resolve("pyside6-lrelease")
    rcc = _resolve("pyside6-rcc")

    for lang in SUPPORTED_LANGUAGES:
        _run([lrelease, f"anylabeling/resources/translations/{lang}.ts"])

    _run([rcc, QRC_PATH, "-o", RC_PATH])

    # Rewrite PySide6 imports to PyQt6 so the rest of the app can use it.
    with open(RC_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("from PySide6 import", "from PyQt6 import")
    content = content.replace("import PySide6", "import PyQt6")
    with open(RC_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Rewrote PySide6 → PyQt6 imports in {RC_PATH}")


if __name__ == "__main__":
    main()
