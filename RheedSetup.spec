# -*- mode: python ; coding: utf-8 -*-
a = Analysis(
    ['apps/rheed_monitor/setup_wizard.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('apps/rheed_monitor/config.yaml', 'apps/rheed_monitor'),
    ],
    hiddenimports=[
        'apps.rheed_monitor.capture.hikrobot',
        'apps.rheed_monitor.capture.file_source',
        'apps.rheed_monitor.detection.spot_detector',
        'PyQt5.QtCore', 'PyQt5.QtWidgets', 'PyQt5.QtGui',
        'yaml', 'winreg',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='RheedSetup',
    debug=False,
    strip=False,
    upx=True,
    console=False,
)
