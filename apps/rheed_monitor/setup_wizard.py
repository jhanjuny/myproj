
# apps/rheed_monitor/setup_wizard.py
"""
RHEED Monitor 설정 마법사
config.yaml을 GUI로 편집하고 저장합니다.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QLabel, QMessageBox, QPushButton,
    QSpinBox, QTextEdit, QVBoxLayout, QHBoxLayout, QWidget,
)


def _cfg_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "rheed_config.yaml"
    return REPO_ROOT / "apps/rheed_monitor/config.yaml"


class SetupWizard(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RHEED Monitor — 설정")
        self.setMinimumWidth(420)
        self._cfg_path = _cfg_path()
        cfg = yaml.safe_load(self._cfg_path.read_text(encoding="utf-8")) if self._cfg_path.exists() else {}

        layout = QVBoxLayout(self)

        # ── 카메라 설정 ──
        cam_box = QGroupBox("카메라 (HIKROBOT GigE)")
        cam_form = QFormLayout(cam_box)

        self._device_index = QSpinBox(); self._device_index.setRange(0, 9)
        self._device_index.setValue(cfg.get("camera", {}).get("device_index", 0))
        cam_form.addRow("카메라 인덱스 (0=첫 번째):", self._device_index)

        self._exposure = QDoubleSpinBox()
        self._exposure.setRange(100, 1_000_000); self._exposure.setSingleStep(1000)
        self._exposure.setSuffix(" µs"); self._exposure.setDecimals(0)
        self._exposure.setValue(cfg.get("camera", {}).get("exposure_us", 10000))
        cam_form.addRow("노출 시간:", self._exposure)

        self._gain = QDoubleSpinBox()
        self._gain.setRange(0, 24); self._gain.setSingleStep(0.5)
        self._gain.setSuffix(" dB")
        self._gain.setValue(cfg.get("camera", {}).get("gain_db", 0.0))
        cam_form.addRow("게인:", self._gain)

        layout.addWidget(cam_box)

        # ── 스팟 검출 설정 ──
        det_box = QGroupBox("스팟 검출")
        det_form = QFormLayout(det_box)

        self._thresh = QDoubleSpinBox()
        self._thresh.setRange(0.1, 0.95); self._thresh.setSingleStep(0.05)
        self._thresh.setDecimals(2)
        self._thresh.setValue(cfg.get("detector", {}).get("threshold_fraction", 0.5))
        det_form.addRow("임계값 비율 (max의 몇 배):", self._thresh)

        self._min_bright = QDoubleSpinBox()
        self._min_bright.setRange(5, 200); self._min_bright.setSingleStep(5)
        self._min_bright.setValue(cfg.get("detector", {}).get("min_brightness", 20))
        det_form.addRow("최소 밝기 (이 이하=스팟 없음):", self._min_bright)

        self._broad_thresh = QSpinBox()
        self._broad_thresh.setRange(10, 100000); self._broad_thresh.setSingleStep(100)
        self._broad_thresh.setSuffix(" px²")
        self._broad_thresh.setValue(cfg.get("detector", {}).get("broad_area_threshold", 500))
        det_form.addRow("broad 판정 면적:", self._broad_thresh)

        layout.addWidget(det_box)

        # ── 녹화 설정 ──
        rec_box = QGroupBox("녹화 / 스크린샷")
        rec_form = QFormLayout(rec_box)

        self._ss_interval = QDoubleSpinBox()
        self._ss_interval.setRange(1, 3600); self._ss_interval.setSingleStep(5)
        self._ss_interval.setSuffix(" 초")
        self._ss_interval.setValue(cfg.get("screenshot_interval_sec", 30))
        rec_form.addRow("스크린샷 주기:", self._ss_interval)

        self._fps = QDoubleSpinBox()
        self._fps.setRange(1, 60); self._fps.setSingleStep(1)
        self._fps.setSuffix(" fps")
        self._fps.setValue(cfg.get("video_fps", 15))
        rec_form.addRow("녹화 FPS:", self._fps)

        layout.addWidget(rec_box)

        # ── 카메라 감지 테스트 ──
        self._detect_btn = QPushButton("카메라 감지 테스트")
        self._detect_btn.clicked.connect(self._test_camera)
        layout.addWidget(self._detect_btn)

        self._detect_result = QLabel("")
        self._detect_result.setWordWrap(True)
        layout.addWidget(self._detect_result)

        # ── 저장 버튼 ──
        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _test_camera(self):
        try:
            from apps.rheed_monitor.capture.hikrobot import list_devices, mvs_available
            if not mvs_available():
                self._detect_result.setText("MVS SDK 없음. MVS를 설치하세요.")
                return
            devs = list_devices()
            self._detect_result.setText("\n".join(devs) if devs else "카메라 없음")
        except Exception as e:
            self._detect_result.setText(f"오류: {e}")

    def _save(self):
        cfg = {
            "camera": {
                "device_index": self._device_index.value(),
                "exposure_us": self._exposure.value(),
                "gain_db": self._gain.value(),
            },
            "detector": {
                "threshold_fraction": self._thresh.value(),
                "min_brightness": self._min_bright.value(),
                "min_area": 4,
                "broad_area_threshold": self._broad_thresh.value(),
                "max_spots": 10,
                "blur_ksize": 5,
            },
            "screenshot_interval_sec": self._ss_interval.value(),
            "video_fps": self._fps.value(),
            "video_codec": "mp4v",
        }
        self._cfg_path.parent.mkdir(parents=True, exist_ok=True)
        self._cfg_path.write_text(
            yaml.dump(cfg, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )
        QMessageBox.information(self, "저장 완료", f"설정이 저장되었습니다.\n{self._cfg_path}")
        self.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    dlg = SetupWizard()
    dlg.exec_()


if __name__ == "__main__":
    main()
