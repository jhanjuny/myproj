# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

# PIL(Pillow 12+ pip wheel) — DLL 정적 내장, 별도 수집 불필요
# collect_all로 pyd + py 모두 수집
pil_datas,  pil_binaries,  pil_hidden  = collect_all('PIL')
mpl_datas,  mpl_binaries,  mpl_hidden  = collect_all('matplotlib')

a = Analysis(
    ['apps/rheed_monitor/main.py'],
    pathex=['.'],
    binaries=pil_binaries + mpl_binaries,
    datas=[
        ('apps/rheed_monitor/config.yaml', 'apps/rheed_monitor'),
    ] + pil_datas + mpl_datas,
    hiddenimports=[
        'apps.rheed_monitor.capture.hikrobot',
        'apps.rheed_monitor.capture.file_source',
        'apps.rheed_monitor.detection.spot_detector',
        'apps.rheed_monitor.gui.main_window',
        'apps.rheed_monitor.storage.session',
        'PyQt5.QtCore', 'PyQt5.QtWidgets', 'PyQt5.QtGui',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.figure',
        'cv2', 'yaml', 'winreg',
        'PIL', 'PIL.Image', 'PIL._imaging',
    ] + pil_hidden + mpl_hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='RheedMonitor',
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
