# apps/VV1/main.py
"""
VV1 실행 진입점.

    D:\\conda_envs\\torch\\python.exe -m apps.VV1.main

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
from apps.VV1.gui.main_window import MainWindow


def main() -> None:
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(cfg=cfg)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
