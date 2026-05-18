
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
    QFileDialog, QFormLayout, QGroupBox, QLabel, QMessageBox,
    QPushButton, QSpinBox, QLineEdit, QTextEdit, QVBoxLayout,
    QHBoxLayout, QWidget,
)


def _cfg_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "rheed_config.yaml"
    return REPO_ROOT / "apps/rheed_monitor/config.yaml"


class SetupWizard(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RHEED Monitor — 설정")
        self.setMinimumWidth(480)
        self._cfg_path = _cfg_path()
        cfg = yaml.safe_load(self._cfg_path.read_text(encoding="utf-8")) if self._cfg_path.exists() else {}

        layout = QVBoxLayout(self)

        # ── MVS SDK 경로 ──────────────────────────────────────────────────
        mvs_box = QGroupBox("HIKROBOT MVS SDK 경로 (자동 탐색 실패 시 수동 지정)")
        mvs_layout = QVBoxLayout(mvs_box)
        mvs_layout.addWidget(QLabel(
            "MVS 설치 후 자동으로 탐색합니다.\n"
            "찾지 못할 경우 MvCameraControl_class.py 가 있는 폴더를 직접 지정하세요."
        ))

        path_row = QHBoxLayout()
        self._mvs_path_edit = QLineEdit()
        self._mvs_path_edit.setPlaceholderText("예) C:\\Program Files\\MVS\\Development\\Samples\\Python\\MvImport")
        self._mvs_path_edit.setText(cfg.get("mvs_sdk_path", ""))
        path_row.addWidget(self._mvs_path_edit, stretch=1)

        browse_btn = QPushButton("찾아보기...")
        browse_btn.clicked.connect(self._browse_mvs)
        path_row.addWidget(browse_btn)
        mvs_layout.addLayout(path_row)

        # 자동 탐색 결과 표시
        self._mvs_status = QLabel("탐색 결과: 미실행")
        self._mvs_status.setWordWrap(True)
        mvs_layout.addWidget(self._mvs_status)

        test_btn = QPushButton("MVS 탐색 / 카메라 감지 테스트")
        test_btn.clicked.connect(self._test_camera)
        mvs_layout.addWidget(test_btn)

        layout.addWidget(mvs_box)

        # ── 카메라 설정 ──────────────────────────────────────────────────
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

        # ── 스팟 검출 설정 ───────────────────────────────────────────────
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

        # ── ROI 설정 ─────────────────────────────────────────────────────
        roi_box = QGroupBox("ROI 설정 (RHEED 진동 측정)")
        roi_form = QFormLayout(roi_box)

        self._roi_size = QSpinBox()
        self._roi_size.setRange(5, 200); self._roi_size.setSingleStep(5)
        self._roi_size.setSuffix(" px")
        self._roi_size.setValue(cfg.get("roi_size_px", 40))
        roi_form.addRow("ROI 반변 크기 (중심±px):", self._roi_size)
        roi_form.addRow(QLabel(
            "비디오 화면을 클릭하면 ROI 중심 설정, 스팟 자동 추적\n"
            "ROI 내 녹색 채널 평균 강도 → RHEED 진동 신호"
        ))
        layout.addWidget(roi_box)

        # ── 녹화 설정 ────────────────────────────────────────────────────
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

        # ── 저장 버튼 ────────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _browse_mvs(self):
        folder = QFileDialog.getExistingDirectory(
            self, "MvCameraControl_class.py 가 있는 폴더 선택"
        )
        if folder:
            self._mvs_path_edit.setText(folder)

    def _test_camera(self):
        # 수동 경로가 있으면 먼저 적용
        manual = self._mvs_path_edit.text().strip()
        if manual:
            from apps.rheed_monitor.capture.hikrobot import set_mvs_path, mvs_error_message
            ok = set_mvs_path(manual)
            if ok:
                self._mvs_status.setText(f"✅ MVS SDK 로드 성공:\n{manual}")
            else:
                # 실패 시 상세 에러 표시 (어떤 DLL이 없는지 확인용)
                err = mvs_error_message()
                self._mvs_status.setText(
                    f"❌ SDK 로드 실패 (경로는 맞음, DLL 문제)\n"
                    f"경로: {manual}\n\n"
                    f"오류 내용:\n{err[:800]}"
                )
                return

        try:
            from apps.rheed_monitor.capture.hikrobot import (
                list_devices, mvs_available, mvs_error_message, _mvs_path_found
            )
            if not mvs_available():
                self._mvs_status.setText(
                    "❌ MVS SDK 없음\n" + mvs_error_message()
                )
                return
            self._mvs_status.setText(f"✅ MVS SDK 위치:\n{_mvs_path_found}")
            devs = list_devices()
            result = "\n".join(devs) if devs else "카메라 없음"
            QMessageBox.information(self, "카메라 감지", result)
        except Exception as e:
            self._mvs_status.setText(f"오류: {e}")

    def _save(self):
        mvs_path = self._mvs_path_edit.text().strip()
        cfg = {
            "mvs_sdk_path": mvs_path,
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
            "roi_size_px": self._roi_size.value(),
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
