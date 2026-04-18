@echo off
setlocal

echo ============================================================
echo   StudyNotes AI — Build Script
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python을 찾을 수 없습니다. Python 3.10+ 이 필요합니다.
    pause & exit /b 1
)

:: Install dependencies
echo [1/4] 패키지 설치 중...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] 패키지 설치 실패
    pause & exit /b 1
)

:: Install PyInstaller
echo [2/4] PyInstaller 확인 중...
pip install pyinstaller>=6.0.0

:: Clean previous build
echo [3/4] 이전 빌드 정리 중...
if exist dist\StudyNotesAI rmdir /s /q dist\StudyNotesAI
if exist build rmdir /s /q build

:: Build
echo [4/4] 빌드 중 (수 분이 소요될 수 있습니다)...
pyinstaller build.spec --noconfirm

if errorlevel 1 (
    echo.
    echo [ERROR] 빌드 실패. 위 오류 메시지를 확인하세요.
    pause & exit /b 1
)

echo.
echo ============================================================
echo   빌드 완료!
echo   실행 파일: dist\StudyNotesAI\StudyNotesAI.exe
echo ============================================================
pause
