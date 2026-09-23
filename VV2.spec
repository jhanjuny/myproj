# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

# VV1.spec과 동일한 교훈: collect_all은 이 "torch" conda 환경에 같이 깔린
# torch/sympy/pandas 등 무관한 무거운 패키지까지 정적 스캔으로 딸려오게 하므로
# (VV1 최초 빌드에서 exe가 2.7GB까지 부풀었던 원인) matplotlib에도 collect_all을
# 쓰되 excludes로 명시적으로 차단한다. RheedMonitor.spec이 이미 matplotlib
# collect_all 패턴으로 정상 빌드된 전례가 있어(107MB) 검증된 접근.
#
# 주의: PIL은 matplotlib(Agg 백엔드의 PNG 저장 등)이 실제로 필요로 하는 진짜
# 의존성이다 — VV1.spec의 EXCLUDES를 그대로 복붙하면서 PIL까지 같이 빼버려서
# "No module named 'PIL'"로 즉시 크래시한 실제 배포 버그가 있었다. RheedMonitor.spec
# 처럼 PIL도 collect_all로 명시적으로 포함해야 한다.
mpl_datas, mpl_binaries, mpl_hidden = collect_all('matplotlib')
pil_datas, pil_binaries, pil_hidden = collect_all('PIL')

EXCLUDES = [
    'torch', 'torchvision', 'torchaudio',
    'sympy', 'pandas',
    'IPython', 'notebook', 'jupyter', 'jupyter_client', 'jupyter_core',
    'tensorflow',
    # matplotlib의 다른 GUI 백엔드는 불필요(Agg만 사용) — Qt5Agg는 이미 PyQt5로
    # 임포트된 PyQt5 자체와는 별개로 matplotlib이 또 끌어올 수 있는 wx/tk 계열만 배제
    'tkinter', 'wx',
]

a = Analysis(
    ['apps/VV2/main.py'],
    pathex=['.'],
    binaries=mpl_binaries + pil_binaries,
    datas=[
        ('apps/VV2/config.yaml', 'apps/VV2'),
    ] + mpl_datas + pil_datas,
    hiddenimports=[
        'apps.VV2.capture.file_source',
        'apps.VV2.detection.spot_detector',
        'apps.VV2.analysis.wavelength',
        'apps.VV2.analysis.materials',
        'apps.VV2.analysis.streak_profile',
        'apps.VV2.analysis.spectra',
        'apps.VV2.analysis.report',
        'apps.VV2.gui.main_window',
        'apps.VV2.worker',
        'apps.VV2.storage.export',
        'PyQt5.QtCore', 'PyQt5.QtWidgets', 'PyQt5.QtGui',
        'cv2', 'yaml',
        'scipy.sparse', 'scipy.sparse.linalg', 'scipy.sparse.csgraph',
        'scipy.ndimage', 'scipy.signal',
        # scipy.signal -> scipy.stats -> _qmc/_sobol(컴파일 모듈)이 내부적으로 쓰는데
        # PyInstaller 정적분석이 못 찾아 ModuleNotFoundError 발생 — 명시적으로 포함
        'importlib.resources', 'importlib.metadata',
        'matplotlib.backends.backend_agg',
        'PIL', 'PIL.Image', 'PIL._imaging',
    ] + mpl_hidden + pil_hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='VV2',
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
