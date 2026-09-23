@echo off
echo [1/2] Installing dependencies...
C:\Users\user_ESC\miniconda3\envs\torch\python.exe -m pip install PyQt5 numpy scipy opencv-python matplotlib pyyaml pyinstaller -q
echo [2/2] Building VV2.exe...
C:\Users\user_ESC\miniconda3\envs\torch\python.exe -m PyInstaller VV2.spec --noconfirm
echo.
echo Build complete!
echo   VV2.exe -^> dist\VV2.exe
pause
