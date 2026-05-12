# rthook_pil.py — PyInstaller runtime hook
# PyInstaller가 압축 해제한 _MEIPASS 디렉토리를 DLL 탐색 경로에 추가.
# conda Library\bin의 DLL들(libjpeg, openjp2 등)을 _MEIPASS에 번들했을 때
# 실행 시 해당 DLL을 찾을 수 있도록 PATH를 패치합니다.
import os
import sys

if getattr(sys, 'frozen', False):
    # 번들 루트를 PATH 최앞에 삽입
    meipass = sys._MEIPASS
    current_path = os.environ.get('PATH', '')
    if meipass not in current_path:
        os.environ['PATH'] = meipass + os.pathsep + current_path

    # Windows 10 1803+ : os.add_dll_directory() 로도 등록
    try:
        os.add_dll_directory(meipass)
    except (AttributeError, OSError):
        pass
