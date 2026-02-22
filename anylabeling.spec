# -*- mode: python -*-
# vim: ft=python

import os
import sys

sys.setrecursionlimit(5000)  # required on Windows

# Collect onnxruntime native DLLs (onnxruntime.dll, onnxruntime_providers_shared.dll).
# PyInstaller resolves imports but does NOT automatically bundle the DLLs that sit
# next to the .pyd extension inside the onnxruntime/capi package directory.
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
    _ort_binaries = (
        [(dll, 'onnxruntime/capi') for dll in _ort_dlls]
        + [(dll, '.') for dll in _ort_dlls]
    )
except Exception:
    _ort_binaries = []

a = Analysis(
    ['anylabeling/app.py'],
    pathex=['anylabeling'],
    binaries=_ort_binaries,
    datas=[
       ('anylabeling/configs/auto_labeling/*.yaml', 'anylabeling/configs/auto_labeling'),
       ('anylabeling/configs/*.yaml', 'anylabeling/configs'),
       ('anylabeling/views/labeling/widgets/auto_labeling/auto_labeling.ui', 'anylabeling/views/labeling/widgets/auto_labeling')
    ],
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
