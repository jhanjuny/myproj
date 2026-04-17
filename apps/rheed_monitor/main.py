
# apps/rheed_monitor/main.py
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from PyQt5.QtWidgets import QApplication
from apps.rheed_monitor.gui.main_window import MainWindow


def _resolve_config() -> Path:
    """EXE로 실행 시 EXE 옆 rheed_config.yaml 우선, 없으면 기본 config."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        runtime = exe_dir / "rheed_config.yaml"
        if runtime.exists():
            return runtime
        # 기본값을 EXE 옆에 복사
        bundled = Path(sys._MEIPASS) / "apps/rheed_monitor/config.yaml"
        if bundled.exists():
            import shutil
            shutil.copy(bundled, runtime)
            return runtime
        return bundled
    # 개발 환경
    return REPO_ROOT / "apps/rheed_monitor/config.yaml"


def main() -> None:
    cfg_path = _resolve_config()
    if not cfg_path.exists():
        print(f"[Error] config not found: {cfg_path}")
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    output_dir = (Path(sys.executable).parent if getattr(sys, "frozen", False)
                  else REPO_ROOT) / "outputs" / "rheed"
    output_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(cfg=cfg, base_output_dir=output_dir)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
