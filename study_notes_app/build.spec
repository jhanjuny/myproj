# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for StudyNotes AI.

Usage:
    pyinstaller build.spec

Output: dist/StudyNotesAI/StudyNotesAI.exe  (one-folder build)
"""

import sys
from pathlib import Path

block_cipher = None

# ── Collect hidden imports ─────────────────────────────────────────────────────
hidden_imports = [
    # PyQt5 / WebEngine
    "PyQt5.QtWebEngineWidgets",
    "PyQt5.QtWebEngineCore",
    "PyQt5.QtWebChannel",
    "PyQt5.QtNetwork",
    "PyQt5.sip",

    # Anthropic SDK
    "anthropic",
    "anthropic._client",
    "anthropic.resources",
    "httpx",
    "httpcore",

    # File processors
    "fitz",           # PyMuPDF
    "docx",
    "pptx",
    "easyocr",
    "whisper",
    "ffmpeg",

    # Image
    "PIL",
    "PIL.Image",
    "numpy",
    "cv2",

    # DB / stdlib
    "sqlite3",
    "json",
    "pathlib",
]

# ── Data files to bundle ───────────────────────────────────────────────────────
datas = []

# Resources directory (icons etc.)
resources_path = Path("resources")
if resources_path.exists():
    datas.append((str(resources_path), "resources"))

# PyMuPDF needs its own data
try:
    import fitz
    fitz_dir = Path(fitz.__file__).parent
    datas.append((str(fitz_dir), "fitz"))
except ImportError:
    pass

# EasyOCR models (large – downloaded to %USERPROFILE%/.EasyOCR at runtime)
# We do NOT bundle them; they download on first OCR use.

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "scipy", "pandas", "jupyter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StudyNotesAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="StudyNotesAI",
)
