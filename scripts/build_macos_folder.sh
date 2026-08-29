#!/bin/bash

# Script to build AnyLabeling in folder mode for macOS
# This creates a directory-based application instead of a bundled .app

# Set CPU or GPU mode
if [ "$1" == "GPU" ]; then
    sed -i'' -e 's/\_\_preferred_device\_\_[ ]*=[ ]*\"[A-Za-z0-9]*\"/__preferred_device__ = "GPU"/g' anylabeling/app_info.py
    SUFFIX="-GPU"
else
    sed -i'' -e 's/\_\_preferred_device\_\_[ ]*=[ ]*\"[A-Za-z0-9]*\"/__preferred_device__ = "CPU"/g' anylabeling/app_info.py
    SUFFIX=""
fi

# Create temporary PyInstaller spec for folder mode
cat > anylabeling_folder.spec << EOL
# -*- mode: python -*-
# vim: ft=python

import sys
from importlib.util import find_spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

sys.setrecursionlimit(5000)  # required on Windows

_osam_datas = collect_data_files('osam')
_ort_binaries = collect_dynamic_libs('onnxruntime')
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
    runtime_hooks=[],
    excludes=[],
)
pyz = PYZ(a.pure, a.zipped_data)

# Create a directory structure instead of a bundled .app
exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,  # This is the key difference - exclude binaries
    name='anylabeling',
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon='anylabeling/resources/images/icon.icns',
)

# Bundle binaries in a separate folder
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='AnyLabeling-Folder${SUFFIX}',
)
EOL

# Install PyInstaller if not already installed
pip install pyinstaller

# Run PyInstaller with the folder mode spec
pyinstaller --noconfirm anylabeling_folder.spec

# Cleanup
rm anylabeling_folder.spec

# Print success message
echo "Build completed. Application folder is located at ./dist/AnyLabeling-Folder${SUFFIX}/"

# Make the script executable
chmod +x scripts/build_macos_folder.sh
