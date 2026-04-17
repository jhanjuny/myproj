# -*- mode: python ; coding: utf-8 -*-
a = Analysis(
    ['apps/rheed_monitor/main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('apps/rheed_monitor/config.yaml', 'apps/rheed_monitor'),
    ],
    hiddenimports=[
        'apps.rheed_monitor.capture.hikrobot',
        'apps.rheed_monitor.detection.spot_detector',
        'apps.rheed_monitor.gui.main_window',
        'apps.rheed_monitor.storage.session',
        'PyQt5.QtCore',
        'PyQt5.QtWidgets',
        'PyQt5.QtGui',
        'matplotlib.backends.backend_qt5agg',
        'cv2',
        'yaml',
    ],
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
    upx=True,
    console=False,
)
