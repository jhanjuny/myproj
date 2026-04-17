
# apps/rheed_monitor/main.py
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import yaml
from PyQt5.QtWidgets import QApplication
from apps.rheed_monitor.gui.main_window import MainWindow


def main() -> None:
    ap = argparse.ArgumentParser(description="RHEED Monitor")
    ap.add_argument("--config", default="apps/rheed_monitor/config.yaml",
                    help="설정 파일 경로")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path

    if not cfg_path.exists():
        print(f"[Error] config not found: {cfg_path}")
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    output_dir = repo_root / "outputs" / "rheed"
    output_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(cfg=cfg, base_output_dir=output_dir)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
