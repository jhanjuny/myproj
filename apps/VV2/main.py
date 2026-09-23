# apps/VV2/main.py
"""
VV2 실행 진입점.

    D:\\conda_envs\\torch\\python.exe -m apps.VV2.main

apps/rheed_monitor와 완전히 독립된 오프라인 정량분석 도구입니다 (실시간
카메라/미리보기 없음). 이미 녹화된 동영상 파일을 열어 정량분석을 실행합니다.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from PyQt5.QtWidgets import QApplication
from apps.VV2.gui.main_window import MainWindow


def _resolve_config() -> Path:
    """
    EXE로 실행 시: PyInstaller onefile 엔트리스크립트는 패키지 경로가 보존되지
    않고 MEIPASS 루트에 풀리므로 __file__ 기준 상대경로가 깨진다 — apps/rheed_monitor
    /main.py의 _resolve_config()와 동일한 패턴으로 sys._MEIPASS를 직접 사용한다.
    EXE 옆에 VV2_config.yaml이 있으면 그걸 우선(재빌드 없이 설정 수정 가능).
    """
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        runtime = exe_dir / "VV2_config.yaml"
        if runtime.exists():
            return runtime
        return Path(sys._MEIPASS) / "apps/VV2/config.yaml"
    return REPO_ROOT / "apps/VV2/config.yaml"


def main() -> None:
    cfg_path = _resolve_config()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(cfg=cfg)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
