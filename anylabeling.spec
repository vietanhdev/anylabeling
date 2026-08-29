# -*- mode: python -*-
# vim: ft=python

import os
import sys
from importlib.util import find_spec

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

sys.setrecursionlimit(5000)  # required on Windows

# Collect all ONNX Runtime provider libraries.  This includes CUDA/TensorRT shared
# libraries on accelerator builds, not only the two DLLs needed by a CPU build.
_ort_binaries = collect_dynamic_libs('onnxruntime')
_nvidia_runtime_packages = (
    'nvidia.cublas',
    'nvidia.cuda_nvrtc',
    'nvidia.cuda_runtime',
    'nvidia.cudnn',
    'nvidia.cufft',
    'nvidia.curand',
    'nvidia.nvjitlink',
)
for _package in _nvidia_runtime_packages:
    if find_spec(_package) is not None:
        _ort_binaries += collect_dynamic_libs(_package)

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
