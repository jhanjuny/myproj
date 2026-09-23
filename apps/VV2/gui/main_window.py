# apps/VV2/gui/main_window.py
"""
VV2 — 오프라인 RHEED 정량분석 (배치 전용, 라이브 미리보기 없음)

동영상 파일을 열어 ROI를 지정하고 캘리브레이션을 입력한 뒤
"정량분석 실행 & 압축저장" 한 번으로: ROI 추적 → 격자상수 d(t) →
apparent coherence length(t) → (정규화 후) FFT 성장률을 계산해 zip으로 저장합니다.
apps/rheed_monitor의 실시간 파이프라인과는 완전히 독립된 도구입니다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QCursor
from PyQt5.QtWidgets import (
    QComboBox, QCheckBox, QDoubleSpinBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QProgressBar, QSizePolicy, QSpinBox, QStatusBar, QTextEdit,
    QVBoxLayout, QWidget,
)

import cv2

from apps.VV2.capture.file_source import FileVideoSource
from apps.VV2.detection.spot_detector import roi_rect
from apps.VV2.analysis.streak_profile import (
    strip_sum_profile, subtract_background, find_three_peaks, solve_scale_from_standard,
)
from apps.VV2.analysis.materials import MATERIALS, find_material_key, expected_d
from apps.VV2.worker import BatchAnalysisWorker


class ClickableLabel(QLabel):
    """미리보기 프레임 클릭 → ROI 중심 수동 지정."""

    def __init__(self, on_click, parent=None):
        super().__init__(parent)
        self._frame_size: Optional[Tuple[int, int]] = None
        self._on_click = on_click
        self.setCursor(QCursor(Qt.CrossCursor))

    def set_frame_size(self, w: int, h: int) -> None:
        self._frame_size = (w, h)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._frame_size:
            fw, fh = self._frame_size
            lw, lh = self.width(), self.height()
            scale = min(lw / fw, lh / fh)
            disp_w, disp_h = int(fw * scale), int(fh * scale)
            off_x, off_y = (lw - disp_w) // 2, (lh - disp_h) // 2
            px, py = event.x() - off_x, event.y() - off_y
            if 0 <= px <= disp_w and 0 <= py <= disp_h:
                self._on_click(px * fw / disp_w, py * fh / disp_h)
        super().mousePressEvent(event)


class MainWindow(QMainWindow):

    def __init__(self, cfg: dict):
        super().__init__()
        self.setWindowTitle("VV2 — RHEED 오프라인 정량분석")
        self.resize(1100, 760)
        self._cfg = cfg

        self._video_path: Optional[Path] = None
        self._spot_csv_path: Optional[Path] = None
        self._preview_frame = None
        self._roi_center: Optional[Tuple[float, float]] = None
        self._worker: Optional[BatchAnalysisWorker] = None

        self._build_ui()
        self._status("동영상 파일을 열어주세요.")

    # ──────────────────────────────── UI 구성 ──────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ── 좌측: 미리보기 ──
        left = QVBoxLayout()
        root.addLayout(left, stretch=3)

        btn_open = QPushButton("📂 동영상 파일 열기")
        btn_open.clicked.connect(self._on_open_file)
        left.addWidget(btn_open)

        self._preview = ClickableLabel(self._on_preview_click)
        self._preview.setText("동영상을 열면 첫 프레임이 표시됩니다.\n(클릭 → ROI 중심 수동 지정)")
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setStyleSheet("background:#111; color:#666; font-size:13px;")
        self._preview.setMinimumSize(560, 420)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left.addWidget(self._preview, stretch=1)

        self._lbl_csv = QLabel("spot_data.csv: (동영상 미선택)")
        left.addWidget(self._lbl_csv)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        left.addWidget(self._progress)

        self._result_box = QTextEdit()
        self._result_box.setReadOnly(True)
        self._result_box.setMaximumHeight(150)
        self._result_box.setPlaceholderText("분석 결과 요약이 여기에 표시됩니다.")
        left.addWidget(self._result_box)

        # ── 우측: 설정 패널 ──
        right = QVBoxLayout()
        root.addLayout(right, stretch=2)

        roi_box = QGroupBox("ROI")
        roi_l = QVBoxLayout(roi_box)
        row = QHBoxLayout()
        row.addWidget(QLabel("크기 (±px):"))
        self._spin_roi = QSpinBox()
        self._spin_roi.setRange(10, 400)
        self._spin_roi.setValue(int(self._cfg["roi"]["size_px"]))
        row.addWidget(self._spin_roi)
        roi_l.addLayout(row)
        self._chk_use_csv = QCheckBox("spot_data.csv 궤적 사용 (있으면 자동 체크)")
        self._chk_use_csv.setEnabled(False)
        roi_l.addWidget(self._chk_use_csv)
        right.addWidget(roi_box)

        calib_box = QGroupBox("캘리브레이션")
        calib_l = QVBoxLayout(calib_box)
        calib = self._cfg["calibration"]

        def _spin_row(label, minv, maxv, val, decimals=3, suffix=""):
            r = QHBoxLayout()
            r.addWidget(QLabel(label))
            s = QDoubleSpinBox()
            s.setRange(minv, maxv)
            s.setDecimals(decimals)
            s.setValue(val)
            if suffix:
                s.setSuffix(suffix)
            r.addWidget(s)
            calib_l.addLayout(r)
            return s

        self._spin_kv = _spin_row("가속전압", 0.1, 100.0, float(calib["accel_voltage_kv"]), 2, " kV")
        self._spin_lcm = _spin_row("카메라거리(L)", 1.0, 100.0, float(calib["camera_length_cm"]), 2, " cm")
        self._spin_scale = _spin_row("픽셀스케일", 0.0, 1.0, float(calib["scale_cm_px"]), 6, " cm/px")
        self._spin_layer = _spin_row("층두께(1ML)", 0.1, 20.0, float(calib["layer_thickness_angstrom"]), 3, " Å")

        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("기준물질:"))
        self._combo_material = QComboBox()
        self._combo_material.addItem("(직접 스케일 입력)")
        self._combo_material.addItems(sorted(MATERIALS.keys()))
        ref_row.addWidget(self._combo_material)
        self._combo_zone = QComboBox()
        self._combo_zone.addItems(["100", "110", "111", "001", "010"])
        ref_row.addWidget(self._combo_zone)
        calib_l.addLayout(ref_row)

        btn_calib = QPushButton("현재 미리보기 프레임의 dx로 스케일 계산")
        btn_calib.clicked.connect(self._on_calibrate_from_preview)
        calib_l.addWidget(btn_calib)
        right.addWidget(calib_box)

        bg_box = QGroupBox("배경제거 / FFT 옵션")
        bg_l = QVBoxLayout(bg_box)
        row = QHBoxLayout()
        row.addWidget(QLabel("배경제거:"))
        self._combo_bg = QComboBox()
        self._combo_bg.addItems(["als", "rolling_ball", "polynomial", "moving_min", "none"])
        self._combo_bg.setCurrentText(self._cfg["background"]["method"])
        row.addWidget(self._combo_bg)
        bg_l.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("디트렌드:"))
        self._combo_detrend = QComboBox()
        self._combo_detrend.addItems(["ema", "linear", "mean"])
        self._combo_detrend.setCurrentText(self._cfg["spectra"]["detrend"])
        row.addWidget(self._combo_detrend)
        bg_l.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("강도 정규화:"))
        self._combo_normalize = QComboBox()
        self._combo_normalize.addItems(["max", "zscore", "none"])
        self._combo_normalize.setCurrentText("max" if self._cfg["spectra"]["normalize"] else "none")
        row.addWidget(self._combo_normalize)
        bg_l.addLayout(row)

        self._chk_frame_norm = QCheckBox("프레임 min-max 정규화 (노출 편차 보정)")
        self._chk_frame_norm.setChecked(bool(self._cfg["spectra"]["frame_normalize"]))
        bg_l.addWidget(self._chk_frame_norm)
        right.addWidget(bg_box)

        self._btn_run = QPushButton("🔬 정량분석 실행 & 압축저장")
        self._btn_run.setFixedHeight(42)
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._on_run)
        right.addWidget(self._btn_run)
        right.addStretch(1)

        self.setStatusBar(QStatusBar())

    # ──────────────────────────────── 파일 열기 ────────────────────────────

    @pyqtSlot()
    def _on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "동영상 파일 선택", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if not path:
            return
        self._video_path = Path(path)
        self._roi_center = None

        try:
            src = FileVideoSource(self._video_path)
            frame = src.read()
            if frame is None:
                raise RuntimeError("첫 프레임을 읽을 수 없습니다.")
            self._preview_frame = frame
            h, w = frame.shape[:2]
            self._preview.set_frame_size(w, h)
            self._show_preview(frame)
            src.release()
        except Exception as exc:
            QMessageBox.critical(self, "파일 오류", str(exc))
            return

        candidate = self._video_path.parent / "spot_data.csv"
        if candidate.exists():
            self._spot_csv_path = candidate
            self._lbl_csv.setText(f"spot_data.csv: 발견됨 ({candidate})")
            self._chk_use_csv.setEnabled(True)
            self._chk_use_csv.setChecked(True)
        else:
            self._spot_csv_path = None
            self._lbl_csv.setText("spot_data.csv: 없음 — 미리보기를 클릭해 ROI를 지정하세요.")
            self._chk_use_csv.setEnabled(False)
            self._chk_use_csv.setChecked(False)

        self._btn_run.setEnabled(True)
        self._status(f"열림: {self._video_path.name}")

    def _show_preview(self, frame) -> None:
        overlay = frame.copy()
        if self._roi_center is not None:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = roi_rect(self._roi_center, self._spin_roi.value(), (h, w))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self._preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._preview.setPixmap(pix)

    @pyqtSlot(float, float)
    def _on_preview_click(self, fx: float, fy: float) -> None:
        self._roi_center = (fx, fy)
        if self._preview_frame is not None:
            self._show_preview(self._preview_frame)
        self._status(f"ROI 수동 지정: ({fx:.1f}, {fy:.1f})")

    # ──────────────────────────────── 캘리브레이션 ─────────────────────────

    @pyqtSlot()
    def _on_calibrate_from_preview(self) -> None:
        if self._preview_frame is None or self._roi_center is None:
            QMessageBox.warning(self, "캘리브레이션 불가", "먼저 동영상을 열고 미리보기에서 ROI를 클릭하세요.")
            return
        material = self._combo_material.currentText()
        key = find_material_key(material)
        if key is None:
            QMessageBox.warning(self, "기준물질 미선택", "기준물질을 선택해야 스케일을 역산할 수 있습니다.")
            return
        zone = self._combo_zone.currentText()
        d_known = expected_d(key, zone)
        if d_known is None:
            QMessageBox.warning(self, "존축 미지원", f"{key}의 [{zone}] 방향 기대 d-spacing이 없습니다.")
            return

        h, w = self._preview_frame.shape[:2]
        rect = roi_rect(self._roi_center, self._spin_roi.value(), (h, w))
        profile = strip_sum_profile(self._preview_frame, rect)
        _bl, subtracted = subtract_background(profile, self._combo_bg.currentText(), {})
        peaks = find_three_peaks(subtracted)
        if peaks.avg_dx is None:
            QMessageBox.warning(self, "피크 검출 실패", "이 프레임에서 3개 피크를 찾지 못했습니다. ROI 위치/크기를 조정하세요.")
            return

        scale = solve_scale_from_standard(peaks.avg_dx, self._spin_lcm.value(), self._spin_kv.value(), d_known)
        if scale is None:
            QMessageBox.warning(self, "계산 실패", "스케일을 계산할 수 없습니다.")
            return
        self._spin_scale.setValue(scale)
        self._status(f"스케일 계산됨: {scale:.6g} cm/px (dx={peaks.avg_dx:.2f}px, 기준 {key}[{zone}]={d_known:.4f}Å)")

    # ──────────────────────────────── 실행 ─────────────────────────────────

    @pyqtSlot()
    def _on_run(self) -> None:
        if self._video_path is None:
            return
        out_dir, _ = QFileDialog.getSaveFileName(
            self, "결과 저장 폴더 선택 (아무 이름이나 입력 후 취소해도 폴더만 사용됨)",
            str(self._video_path.parent), "",
        )
        out_dir_path = Path(out_dir).parent if out_dir else self._video_path.parent

        detector_cfg = dict(self._cfg["detector"])
        self._worker = BatchAnalysisWorker(
            video_path=str(self._video_path),
            out_dir=str(out_dir_path),
            roi_size_px=self._spin_roi.value(),
            track_alpha=float(self._cfg["roi"]["track_alpha"]),
            detector_cfg=detector_cfg,
            bg_method=self._combo_bg.currentText(),
            bg_params={},
            v_kev=self._spin_kv.value(),
            l_cm=self._spin_lcm.value(),
            scale_cm_px=self._spin_scale.value(),
            layer_thickness_angstrom=self._spin_layer.value(),
            fft_detrend=self._combo_detrend.currentText(),
            fft_alpha=float(self._cfg["spectra"]["ema_alpha"]),
            fft_normalize=self._combo_normalize.currentText(),
            frame_normalize=self._chk_frame_norm.isChecked(),
            manual_roi_center=self._roi_center,
            spot_csv_path=str(self._spot_csv_path) if self._spot_csv_path else None,
            use_spot_csv=self._chk_use_csv.isChecked(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)

        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status("정량분석 실행 중...")
        self._worker.start()

    @pyqtSlot(int, int)
    def _on_progress(self, current: int, total: int) -> None:
        if total > 0:
            self._progress.setMaximum(total)
            self._progress.setValue(current)
        self._status(f"분석 중... {current}/{total}", temporary=True)

    @pyqtSlot(dict)
    def _on_finished(self, summary: dict) -> None:
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        gr = summary.get("growth_rate") or {}
        report_png = summary.get("report_png")
        lines = [
            f"ROI 소스: {summary.get('roi_source')}",
            f"분석 프레임: {summary.get('n_frames_with_roi')}/{summary.get('n_frames_total')}",
            f"평균 격자상수 d: {summary.get('d_angstrom_mean')} Å  (n={summary.get('d_angstrom_n')})",
            f"평균 coherence length: {summary.get('coherence_angstrom_mean')} Å",
            f"성장률: f0={gr.get('f0')} Hz(ML/s), Å/s={gr.get('growth_angstrom_s')}",
            f"요약 이미지: {report_png or '생성 실패 — ' + str(summary.get('report_png_error'))}",
            f"저장: {summary.get('zip_path')}",
        ]
        self._result_box.setPlainText("\n".join(lines))
        self._status("분석 완료 및 압축 저장됨.")

        if report_png:
            try:
                os.startfile(report_png)  # 결과를 바로 눈으로 확인할 수 있도록 기본 뷰어로 열기
            except Exception:
                pass

        QMessageBox.information(self, "완료", f"분석 완료.\n저장: {summary.get('zip_path')}")

    @pyqtSlot(str)
    def _on_failed(self, message: str) -> None:
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._status("분석 실패.")
        QMessageBox.critical(self, "분석 실패", message)

    # ──────────────────────────────── 유틸 ─────────────────────────────────

    def _status(self, msg: str, temporary: bool = False) -> None:
        if temporary:
            self.statusBar().showMessage(msg, 2000)
        else:
            self.statusBar().showMessage(msg)
