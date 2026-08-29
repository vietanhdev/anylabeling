# -*- mode: python -*-
# vim: ft=python

import os
import sys
from importlib.util import find_spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

sys.setrecursionlimit(5000)  # required on Windows

# Collect all ONNX Runtime provider libraries.  This includes CUDA/TensorRT shared
# libraries on accelerator builds, not only the two DLLs needed by a CPU build.
_ort_binaries = collect_dynamic_libs('onnxruntime')
_nvidia_spec = find_spec('nvidia')
if _nvidia_spec is not None and _nvidia_spec.submodule_search_locations:
    for _root_name in _nvidia_spec.submodule_search_locations:
        _root = Path(_root_name)
        for _source in _root.rglob('*'):
            _name = _source.name.lower()
            _is_runtime_library = (
                _name.endswith(('.dll', '.dylib', '.so')) or '.so.' in _name
            )
            _is_optional_duplicate = (
                '.alt.' in _name
                or _name.startswith(('libnvblas.', 'nvblas'))
                or _name.startswith(('libcufftw.', 'cufftw'))
            )
            if (
                _source.is_file()
                and _is_runtime_library
                and not _is_optional_duplicate
            ):
                _destination = _source.parent.relative_to(_root.parent)
                _ort_binaries.append((str(_source), _destination.as_posix()))

# Windows also needs the core ONNX Runtime DLLs at the bundle root because ORT's
# internal LoadLibrary calls do not use Python's AddDllDirectory search path.
try:
    import onnxruntime as _ort
    _ort_capi = os.path.join(os.path.dirname(_ort.__file__), 'capi')
    _ort_dlls = [
        os.path.join(_ort_capi, f)
        for f in os.listdir(_ort_capi)
        if f.endswith('.dll')
    ]
    # Place DLLs in both locations:
    #   onnxruntime/capi/ — matches package structure, found via DLL_LOAD_DIR
    #   .  (root _MEIPASS)  — found via PyInstaller's SetDllDirectory(_MEIPASS)
    _ort_binaries += [(dll, '.') for dll in _ort_dlls]
except Exception:
    pass

_osam_datas = collect_data_files('osam')

a = Analysis(
    ['anylabeling/app.py'],
    pathex=['anylabeling'],
    binaries=_ort_binaries,
    datas=[
       ('anylabeling/configs/auto_labeling/*.yaml', 'anylabeling/configs/auto_labeling'),
       ('anylabeling/configs/*.yaml', 'anylabeling/configs'),
       ('anylabeling/views/labeling/widgets/auto_labeling/auto_labeling.ui', 'anylabeling/views/labeling/widgets/auto_labeling')
    ] + _osam_datas,
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=['rthooks/rthook_onnxruntime.py'],
    excludes=[],
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='anylabeling',
    debug=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    icon='anylabeling/resources/images/icon.icns',
)
app = BUNDLE(
    exe,
    name='AnyLabeling.app',
    icon='anylabeling/resources/images/icon.icns',
    bundle_identifier=None,
    info_plist={'NSHighResolutionCapable': 'True'},
)
