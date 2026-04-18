"""
StudyNotes AI — entry point.
"""
import sys
import os
import traceback

# Fix high-DPI before QApplication
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("StudyNotes AI")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("StudyNotes")

    # ── Late import so the splash (if any) can show first ─────────────────────
    try:
        from src.gui.main_window import MainWindow
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())

    except ImportError as e:
        _fatal_error(
            "필수 패키지 누락",
            f"다음 패키지가 설치되지 않았습니다:\n{e}\n\n"
            "터미널에서 실행하세요:\n"
            "  pip install -r requirements.txt",
        )
    except Exception as e:
        _fatal_error("시작 오류", f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")


def _fatal_error(title: str, msg: str):
    app = QApplication.instance() or QApplication(sys.argv)
    QMessageBox.critical(None, title, msg)
    sys.exit(1)


if __name__ == "__main__":
    main()
