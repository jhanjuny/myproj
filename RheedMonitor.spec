# -*- mode: python ; coding: utf-8 -*-
import os
import glob
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

# ── PIL(Pillow) / matplotlib 전체 수집 ──────────────────────────────────────
pil_datas,  pil_binaries,  pil_hidden  = collect_all('PIL')
mpl_datas,  mpl_binaries,  mpl_hidden  = collect_all('matplotlib')

# ── conda Library\bin 에서 Pillow 의존 DLL 명시적 번들 ──────────────────────
# conda 환경 경로 (빌드 머신 고정)
_CONDA_BIN = r'D:\conda_envs\torch\Library\bin'

_PIL_DLL_NAMES = [
    'libjpeg*.dll',
    'openjp2*.dll',
    'libtiff*.dll',
    'libpng*.dll',
    'zlib*.dll',
    'lcms2*.dll',
    'libwebp*.dll',
    'libwebpdecoder*.dll',
    'libwebpdemux*.dll',
    'libwebpmux*.dll',
    'freetype*.dll',
]

extra_binaries = []
for pattern in _PIL_DLL_NAMES:
    for dll_path in glob.glob(os.path.join(_CONDA_BIN, pattern)):
        extra_binaries.append((dll_path, '.'))   # '.' = EXE와 같은 디렉토리

# ────────────────────────────────────────────────────────────────────────────
a = Analysis(
    ['apps/rheed_monitor/main.py'],
    pathex=['.'],
    binaries=extra_binaries + pil_binaries + mpl_binaries,
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
    runtime_hooks=['rthook_pil.py'],   # DLL PATH 패치 훅
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='RheedMonitor',
    debug=False,
    strip=False,
    upx=False,   # UPX가 DLL 압축 시 로드 실패 유발 가능 → 비활성화
    console=False,
)
