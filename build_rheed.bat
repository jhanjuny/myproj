@echo off
echo [1/4] Installing dependencies...
D:\conda_envs\torch\python.exe -m pip install PyQt5 matplotlib pyyaml fpdf2 pyinstaller -q
echo [2/4] Generating manual PDF...
D:\conda_envs\torch\python.exe -m apps.rheed_monitor.make_manual
echo [3/4] Building RheedMonitor.exe...
D:\conda_envs\torch\python.exe -m PyInstaller RheedMonitor.spec --noconfirm
echo [4/4] Building RheedSetup.exe...
D:\conda_envs\torch\python.exe -m PyInstaller RheedSetup.spec --noconfirm
echo.
echo Build complete!
echo   RheedMonitor.exe -> dist\RheedMonitor.exe
echo   RheedSetup.exe   -> dist\RheedSetup.exe
echo   Manual           -> RHEED_Monitor_Manual.pdf
pause
