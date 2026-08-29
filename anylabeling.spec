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

# Conda's Python 3.12 pyexpat extension uses APIs from the libexpat shipped in
# that environment. PyInstaller otherwise treats libexpat as a system library,
# so the frozen app can load an older host copy and fail with an undefined
# XML_SetAllocTrackerActivationThreshold symbol.
if sys.platform.startswith('linux'):
    _conda_expat = Path(sys.prefix) / 'lib' / 'libexpat.so.1'
    if _conda_expat.is_file():
        _ort_binaries.append((str(_conda_expat), '.'))

_nvidia_spec = find_spec('nvidia')
if _nvidia_spec is not None and _nvidia_spec.submodule_search_locations:
    for _root_name in _nvidia_spec.submodule_search_locations:
        _root = Path(_root_name)
        for _source in _root.rglob('*'):
            _name = _source.name.lower()
            _package = _source.relative_to(_root).parts[0]
            _is_runtime_library = (
                _name.endswith(('.dll', '.dylib', '.so')) or '.so.' in _name
            )
            # NVRTC/JitLink are compiler tooling and are not linked by ORT's
            # CUDA/cuDNN inference libraries. Wrapper and alternate binaries
            # are also unnecessary and can push release assets over 2 GiB.
            _is_optional_duplicate = (
                _package in {'cuda_nvrtc', 'nvjitlink'}
                or '.alt.' in _name
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

# Ensure Windows core DLLs are present under onnxruntime/capi. The runtime hook
# preloads them from there by absolute path, so a duplicate bundle-root copy is
# unnecessary and would add hundreds of megabytes to GPU executables.
try:
    import onnxruntime as _ort
    _ort_capi = os.path.join(os.path.dirname(_ort.__file__), 'capi')
    _ort_dlls = [
        os.path.join(_ort_capi, f)
        for f in os.listdir(_ort_capi)
        if f.endswith('.dll')
    ]
    _existing_binaries = {str(Path(source).resolve()) for source, _ in _ort_binaries}
    _ort_binaries += [
        (dll, 'onnxruntime/capi')
        for dll in _ort_dlls
        if str(Path(dll).resolve()) not in _existing_binaries
    ]
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
