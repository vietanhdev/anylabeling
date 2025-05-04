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

sys.setrecursionlimit(5000)  # required on Windows

a = Analysis(
    ['anylabeling/app.py'],
    pathex=['anylabeling'],
    binaries=[],
    datas=[
       ('anylabeling/configs/auto_labeling/*.yaml', 'anylabeling/configs/auto_labeling'),
       ('anylabeling/configs/*.yaml', 'anylabeling/configs'),
       ('anylabeling/views/labeling/widgets/auto_labeling/auto_labeling.ui', 'anylabeling/views/labeling/widgets/auto_labeling')
    ],
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
