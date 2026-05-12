# -*- mode: python ; coding: utf-8 -*-
import os
import glob
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

pil_datas, pil_binaries, pil_hidden = collect_all('PIL')

_CONDA_BIN = r'D:\conda_envs\torch\Library\bin'
_PIL_DLL_NAMES = [
    'libjpeg*.dll', 'openjp2*.dll', 'libtiff*.dll', 'libpng*.dll',
    'zlib*.dll', 'lcms2*.dll', 'libwebp*.dll', 'libwebpdecoder*.dll',
    'libwebpdemux*.dll', 'libwebpmux*.dll', 'freetype*.dll',
]
extra_binaries = []
for pattern in _PIL_DLL_NAMES:
    for dll_path in glob.glob(os.path.join(_CONDA_BIN, pattern)):
        extra_binaries.append((dll_path, '.'))

a = Analysis(
    ['apps/rheed_monitor/setup_wizard.py'],
    pathex=['.'],
    binaries=extra_binaries + pil_binaries,
    datas=[
        ('apps/rheed_monitor/config.yaml', 'apps/rheed_monitor'),
    ] + pil_datas,
    hiddenimports=[
        'apps.rheed_monitor.capture.hikrobot',
        'apps.rheed_monitor.capture.file_source',
        'apps.rheed_monitor.detection.spot_detector',
        'PyQt5.QtCore', 'PyQt5.QtWidgets', 'PyQt5.QtGui',
        'yaml', 'winreg',
        'PIL', 'PIL.Image', 'PIL._imaging',
    ] + pil_hidden,
    hookspath=[],
    runtime_hooks=['rthook_pil.py'],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='RheedSetup',
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
