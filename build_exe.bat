:: build_exe.bat
@echo off
echo Building Vacuum Monitor...

:: 기존 빌드 폴더 삭제 (깨끗한 빌드를 위해)
rmdir /s /q build
rmdir /s /q dist

:: PyInstaller 실행
:: --onefile: 파일 하나로 뭉침
:: --name: 실행 파일 이름 설정
:: --add-data: 설정 파일 템플릿을 exe 내부에 포함 (형식: "원본경로;저장경로")
:: --hidden-import: 혹시나 누락될 수 있는 모듈 강제 포함
:: --icon: (선택) 아이콘 파일이 있다면 경로 지정 (없으면 이 줄 삭제)

pyinstaller --noconfirm --onefile --console --name "VacuumMonitor" ^
    --add-data "apps/vacuum_monitor/config/config.yaml;apps/vacuum_monitor/config" ^
    --hidden-import "apps.vacuum_monitor.capture.sources" ^
    --hidden-import "apps.vacuum_monitor.wizard.ui_cv2" ^
    apps/vacuum_monitor/main.py

echo.
echo Build Complete! Look in the 'dist' folder.
pause