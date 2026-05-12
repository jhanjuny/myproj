# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

pil_datas, pil_binaries, pil_hidden = collect_all('PIL')

a = Analysis(
    ['apps/rheed_monitor/setup_wizard.py'],
    pathex=['.'],
    binaries=pil_binaries + collect_dynamic_libs('PIL'),
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
