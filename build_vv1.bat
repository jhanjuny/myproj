@echo off
echo [1/2] Installing dependencies...
C:\Users\user_ESC\miniconda3\envs\torch\python.exe -m pip install PyQt5 numpy scipy opencv-python pyyaml pyinstaller -q
echo [2/2] Building VV1.exe...
C:\Users\user_ESC\miniconda3\envs\torch\python.exe -m PyInstaller VV1.spec --noconfirm
echo.
echo Build complete!
echo   VV1.exe -^> dist\VV1.exe
pause
