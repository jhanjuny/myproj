# -*- mode: python ; coding: utf-8 -*-
#
# 이 conda 환경("torch")에는 PyTorch/matplotlib/sympy 등 VV1과 무관한 무거운
# 패키지가 같이 설치돼 있습니다. collect_all('numpy')/collect_all('scipy')를
# 쓰면 PyInstaller가 그 패키지들의 테스트 서브모듈까지 정적 스캔하다가
# torch/torchvision까지 딸려 들어와 exe가 2.7GB로 부풀었던 문제가 있었습니다
# (VV1은 numpy/scipy/opencv-python/PyQt5/pyyaml만 필요, torch 불필요).
# → collect_all 제거 + PyInstaller 기본 훅에 맡기고, torch류는 명시적으로 exclude.

EXCLUDES = [
    'torch', 'torchvision', 'torchaudio',
    'matplotlib', 'sympy', 'pandas',
    'IPython', 'notebook', 'jupyter', 'jupyter_client', 'jupyter_core',
    'tensorflow', 'PIL',
]

a = Analysis(
    ['apps/VV1/main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('apps/VV1/config.yaml', 'apps/VV1'),
    ],
    hiddenimports=[
        'apps.VV1.capture.file_source',
        'apps.VV1.detection.spot_detector',
        'apps.VV1.analysis.wavelength',
        'apps.VV1.analysis.materials',
        'apps.VV1.analysis.streak_profile',
        'apps.VV1.analysis.spectra',
        'apps.VV1.gui.main_window',
        'apps.VV1.worker',
        'apps.VV1.storage.export',
        'PyQt5.QtCore', 'PyQt5.QtWidgets', 'PyQt5.QtGui',
        'cv2', 'yaml',
        'scipy.sparse', 'scipy.sparse.linalg', 'scipy.sparse.csgraph',
        'scipy.ndimage', 'scipy.signal',
        # scipy.signal -> scipy.stats -> _qmc/_sobol(컴파일 모듈)이 내부적으로 쓰는데
        # PyInstaller 정적분석이 못 찾아 ModuleNotFoundError 발생 — 명시적으로 포함
        'importlib.resources', 'importlib.metadata',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='VV1',
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
