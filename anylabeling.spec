# -*- mode: python -*-
# vim: ft=python

import sys


sys.setrecursionlimit(5000)  # required on Windows


a = Analysis(
    ['anylabeling/app.py'],
    pathex=['anylabeling'],
    binaries=[],
    datas=[
       ('anylabeling/configs/anylabeling_config.yaml', 'anylabeling/views/labeling/labelme/config'),
       ('anylabeling/views/labeling/labelme/icons/*', 'anylabeling/views/labeling/labelme/icons'),
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
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
    #icon='anylabeling/icons/icon.ico',
)
app = BUNDLE(
    exe,
    name='anylabeling.app',
    #icon='anylabeling/icons/icon.icns',
    bundle_identifier=None,
    info_plist={'NSHighResolutionCapable': 'True'},
)