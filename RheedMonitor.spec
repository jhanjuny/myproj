# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

# PIL(Pillow) 전체 수집 - _imaging DLL 포함
pil_datas, pil_binaries, pil_hidden = collect_all('PIL')
# matplotlib 전체 수집 - backends, fonts 포함
mpl_datas, mpl_binaries, mpl_hidden = collect_all('matplotlib')

a = Analysis(
    ['apps/rheed_monitor/main.py'],
    pathex=['.'],
    binaries=pil_binaries + mpl_binaries + collect_dynamic_libs('PIL'),
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
    upx=True,
    console=False,
)
