# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['apps\\vacuum_monitor\\main.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=['mss', 'apps.vacuum_monitor.capture.sources', 'apps.vacuum_monitor.wizard.ui_cv2', 'apps.vacuum_monitor.readout.ocr_tesseract', 'apps.vacuum_monitor.alerts.email_smtp'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='VacuumMonitor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
