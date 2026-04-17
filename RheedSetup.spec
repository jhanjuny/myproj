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
        'PyQt5.QtCore', 'PyQt5.QtWidgets', 'PyQt5.QtGui',
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
    name='RheedSetup',
    debug=False,
    strip=False,
    upx=True,
    console=False,
)
