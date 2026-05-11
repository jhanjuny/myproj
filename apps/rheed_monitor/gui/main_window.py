# apps/rheed_monitor/gui/main_window.py
"""
RHEED Monitor — 메인 윈도우

■ 측정 방식
  - 녹색 채널 ROI 적분 강도 (mean_intensity) → RHEED 진동(oscillation) 신호
  - SpotDetector로 스팟 centroid 추적 → ROI가 스팟을 자동으로 따라감
  - 사용자가 비디오 클릭 → ROI 중심 수동 지정

■ 그래프 구성
  - 상단 (큰): ROI 강도 vs 시간 (진동 신호 — 핵심)
  - 중단: X 위치 vs 시간
  - 하단: Y 위치 vs 시간

■ 입력 소스
  - 카메라 모드: HIKROBOT GigE 카메라 실시간 스트림
  - 파일 모드:  로컬 동영상 파일 (.mp4/.avi) 오프라인 분석
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QCursor
from PyQt5.QtWidgets import (
    QAction, QCheckBox, QDoubleSpinBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QSizePolicy, QSlider, QSpinBox, QStatusBar, QVBoxLayout, QWidget,
    QToolBar,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from apps.rheed_monitor.detection.spot_detector import SpotDetector, SpotResult, RoiData
from apps.rheed_monitor.storage.session import Session


# ═══════════════════════════════════════════════════════════════════════════════
# 클릭 가능한 비디오 레이블
# ═══════════════════════════════════════════════════════════════════════════════

class ClickableVideoLabel(QLabel):
    """
    비디오 프레임을 표시하고 클릭 좌표를 frame 좌표계로 변환해 신호를 방출합니다.
    클릭하면 ROI 중심이 해당 위치로 이동합니다.
    """
    clicked_frame = pyqtSignal(float, float)   # (frame_x, frame_y)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_size: Optional[Tuple[int, int]] = None  # (w, h)
        self.setCursor(QCursor(Qt.CrossCursor))

    def set_frame_size(self, w: int, h: int) -> None:
        self._frame_size = (w, h)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._frame_size:
            fw, fh = self._frame_size
            lw, lh = self.width(), self.height()
            # KeepAspectRatio 스케일 계산
            scale = min(lw / fw, lh / fh)
            disp_w = int(fw * scale)
            disp_h = int(fh * scale)
            off_x = (lw - disp_w) // 2
            off_y = (lh - disp_h) // 2
            px = event.x() - off_x
            py = event.y() - off_y
            if 0 <= px <= disp_w and 0 <= py <= disp_h:
                fx = px * fw / disp_w
                fy = py * fh / disp_h
                self.clicked_frame.emit(fx, fy)
        super().mousePressEvent(event)


# ═══════════════════════════════════════════════════════════════════════════════
# 카메라 스레드
# ═══════════════════════════════════════════════════════════════════════════════

class CameraThread(QThread):
    """카메라 폴링 스레드 — frame_ready Signal 방출."""
    frame_ready = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self._camera = camera
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            try:
                frame = self._camera.read()
                if frame is not None:
                    self.frame_ready.emit(frame)
            except Exception as exc:
                self.error.emit(str(exc))
                break
            self.msleep(10)

    def stop(self):
        self._running = False
        self.wait(3000)


# ═══════════════════════════════════════════════════════════════════════════════
# 파일 비디오 스레드
# ═══════════════════════════════════════════════════════════════════════════════

class FileVideoThread(QThread):
    """
    동영상 파일을 원래 FPS에 맞춰 재생하는 스레드.
    frame_ready 신호는 CameraThread와 동일 — MainWindow는 구분하지 않아도 됩니다.
    """
    frame_ready = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int, int)   # (current_frame, total_frames)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, source, speed: float = 1.0, parent=None):
        super().__init__(parent)
        self._source = source
        self._speed = max(0.1, speed)
        self._running = False
        self._paused = False

    def run(self):
        self._running = True
        interval = 1.0 / (self._source.fps * self._speed)
        while self._running:
            if self._paused:
                self.msleep(50)
                continue
            t0 = time.perf_counter()
            frame = self._source.read()
            if frame is None:
                self.finished.emit()
                break
            self.frame_ready.emit(frame)
            self.progress.emit(self._source.current_frame, self._source.frame_count)
            elapsed = time.perf_counter() - t0
            wait_ms = max(0, int((interval - elapsed) * 1000))
            self.msleep(wait_ms)

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.1, speed)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._running = False
        self.wait(3000)


# ═══════════════════════════════════════════════════════════════════════════════
# 실시간 그래프 (ROI 강도 + XY 위치)
# ═══════════════════════════════════════════════════════════════════════════════

class RheedGraph(FigureCanvas):
    """
    상단: ROI 강도 vs 시간 (RHEED 진동 신호 — 크게)
    중단: X 위치 vs 시간
    하단: Y 위치 vs 시간
    """
    MAXPOINTS = 1000

    def __init__(self, parent=None):
        fig = Figure(figsize=(4, 6), tight_layout=True)
        fig.patch.set_facecolor("#1a1a1a")
        super().__init__(fig)
        self.setParent(parent)

        # 비율: 강도=2, X=1, Y=1
        gs = fig.add_gridspec(4, 1, hspace=0.45)
        self._ax_i = fig.add_subplot(gs[0:2, 0])   # 강도 (2칸)
        self._ax_x = fig.add_subplot(gs[2, 0])      # X
        self._ax_y = fig.add_subplot(gs[3, 0])      # Y

        _STYLE = dict(facecolor="#252525")
        for ax, ylabel, color in [
            (self._ax_i, "ROI Intensity", "#00e5ff"),
            (self._ax_x, "X (px)",        "#00ff88"),
            (self._ax_y, "Y (px)",        "#ffaa00"),
        ]:
            ax.set(**_STYLE)
            ax.set_ylabel(ylabel, color="white", fontsize=8)
            ax.tick_params(colors="white", labelsize=7)
            for sp in ax.spines.values():
                sp.set_color("#444")

        self._ax_i.set_title("RHEED Oscillation", color="#00e5ff",
                              fontsize=9, pad=3)
        self._ax_y.set_xlabel("Time (s)", color="white", fontsize=8)

        # 데이터 큐
        self._t:  Deque[float]         = deque(maxlen=self.MAXPOINTS)
        self._ii: Deque[Optional[float]] = deque(maxlen=self.MAXPOINTS)
        self._xs: Deque[Optional[float]] = deque(maxlen=self.MAXPOINTS)
        self._ys: Deque[Optional[float]] = deque(maxlen=self.MAXPOINTS)
        self._t0 = time.time()

        # 라인
        self._line_i, = self._ax_i.plot([], [], color="#00e5ff", lw=1)
        self._line_x, = self._ax_x.plot([], [], color="#00ff88", lw=1)
        self._line_y, = self._ax_y.plot([], [], color="#ffaa00", lw=1)

    def push(
        self,
        roi: Optional[RoiData],
        spots: List[SpotResult],
        t: Optional[float] = None,
    ) -> None:
        """
        새 데이터 포인트 추가.
        roi=None 이면 그래프에 갭(None)을 삽입합니다.
        """
        ts = t if t is not None else (time.time() - self._t0)
        self._t.append(ts)

        if roi is not None:
            self._ii.append(roi.mean_intensity)
        else:
            self._ii.append(None)

        if spots:
            s = spots[0]
            self._xs.append(s.x)
            self._ys.append(s.y)
        else:
            self._xs.append(None)
            self._ys.append(None)

        self._redraw()

    def _redraw(self) -> None:
        ts = list(self._t)
        for ax, line, vals, color in [
            (self._ax_i, self._line_i, self._ii, "#00e5ff"),
            (self._ax_x, self._line_x, self._xs, "#00ff88"),
            (self._ax_y, self._line_y, self._ys, "#ffaa00"),
        ]:
            # 갭 처리: None이면 새 세그먼트 시작
            seg_t, seg_v = [], []
            for tv, v in zip(ts, vals):
                if v is None:
                    if seg_t:
                        ax.plot(seg_t, seg_v, color=color, lw=1, alpha=0.9)
                    seg_t, seg_v = [], []
                else:
                    seg_t.append(tv)
                    seg_v.append(v)
            if seg_t:
                line.set_data(seg_t, seg_v)
            else:
                line.set_data([], [])
            ax.relim()
            ax.autoscale_view()
        self.draw_idle()

    def reset(self, t0: Optional[float] = None) -> None:
        self._t.clear(); self._ii.clear(); self._xs.clear(); self._ys.clear()
        self._t0 = t0 if t0 is not None else time.time()
        for line in (self._line_i, self._line_x, self._line_y):
            line.set_data([], [])
        self.draw_idle()


# ═══════════════════════════════════════════════════════════════════════════════
# 메인 윈도우
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self, cfg: dict, base_output_dir: Path):
        super().__init__()
        self.setWindowTitle("RHEED Monitor  v2")
        self.resize(1400, 860)

        self._cfg = cfg
        self._base_output_dir = base_output_dir
        self._session: Optional[Session] = None
        self._camera = None
        self._file_source = None
        self._cam_thread: Optional[CameraThread] = None
        self._file_thread: Optional[FileVideoThread] = None
        self._detector = SpotDetector(**cfg.get("detector", {}))
        self._recording = False

        # ROI 상태
        self._roi_center: Optional[Tuple[float, float]] = None
        self._roi_size: int = int(cfg.get("roi_size_px", 40))
        self._roi_locked = False   # True = 자동 추적 끄고 클릭한 위치 고정

        self._screenshot_timer = QTimer(self)
        self._screenshot_timer.timeout.connect(self._on_screenshot_timer)
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_spots: List[SpotResult] = []
        self._latest_roi: Optional[RoiData] = None

        # 파일 모드 시간 추적
        self._file_t0: float = 0.0

        self._build_ui()
        self._status("카메라 연결 또는 동영상 파일을 열어 시작하세요.")

    # ──────────────────────────────── UI 구성 ──────────────────────────────

    def _build_ui(self) -> None:
        # ── 툴바 ──
        tb = QToolBar("파일")
        self.addToolBar(tb)
        act_open = QAction("📂 동영상 파일 열기", self)
        act_open.triggered.connect(self._on_open_file)
        tb.addAction(act_open)
        act_cam = QAction("📷 카메라 연결", self)
        act_cam.triggered.connect(self._on_connect_camera)
        tb.addAction(act_cam)

        # ── 중앙 위젯 ──
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 2, 6, 6)

        # ── 비디오 + 그래프 ──
        top = QHBoxLayout()
        root.addLayout(top, stretch=1)

        # 비디오 레이블
        self._video_label = ClickableVideoLabel()
        self._video_label.setText("카메라 또는 파일을 연결하세요\n(클릭 → ROI 위치 설정)")
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setStyleSheet("background:#111; color:#666; font-size:14px;")
        self._video_label.setMinimumSize(640, 480)
        self._video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video_label.clicked_frame.connect(self._on_video_click)
        top.addWidget(self._video_label, stretch=3)

        # 그래프
        self._graph = RheedGraph()
        self._graph.setMinimumWidth(300)
        top.addWidget(self._graph, stretch=2)

        # ── 파일 재생 컨트롤 (처음엔 숨김) ──
        self._file_ctrl = QWidget()
        file_layout = QHBoxLayout(self._file_ctrl)
        file_layout.setContentsMargins(0, 0, 0, 0)

        self._btn_play_pause = QPushButton("⏸ 일시정지")
        self._btn_play_pause.setFixedHeight(32)
        self._btn_play_pause.clicked.connect(self._on_play_pause)
        file_layout.addWidget(self._btn_play_pause)

        file_layout.addWidget(QLabel("속도:"))
        self._spin_speed = QDoubleSpinBox()
        self._spin_speed.setRange(0.1, 10.0)
        self._spin_speed.setSingleStep(0.5)
        self._spin_speed.setValue(1.0)
        self._spin_speed.setSuffix("x")
        self._spin_speed.valueChanged.connect(self._on_speed_changed)
        file_layout.addWidget(self._spin_speed)

        self._slider_pos = QSlider(Qt.Horizontal)
        self._slider_pos.setRange(0, 1000)
        self._slider_pos.sliderMoved.connect(self._on_slider_moved)
        file_layout.addWidget(self._slider_pos, stretch=1)

        self._lbl_pos = QLabel("00:00 / 00:00")
        file_layout.addWidget(self._lbl_pos)

        root.addWidget(self._file_ctrl)
        self._file_ctrl.setVisible(False)

        # ── 컨트롤 패널 ──
        ctrl_box = QGroupBox("컨트롤")
        ctrl_layout = QHBoxLayout(ctrl_box)
        root.addWidget(ctrl_box)

        self._btn_record = QPushButton("⏺ 녹화 시작")
        self._btn_record.setFixedHeight(38)
        self._btn_record.setEnabled(False)
        self._btn_record.clicked.connect(self._on_toggle_record)
        ctrl_layout.addWidget(self._btn_record)

        # ROI 그룹
        roi_box = QGroupBox("ROI 설정")
        roi_layout = QHBoxLayout(roi_box)
        roi_layout.addWidget(QLabel("크기 (±px):"))
        self._spin_roi = QSpinBox()
        self._spin_roi.setRange(5, 200)
        self._spin_roi.setValue(self._roi_size)
        self._spin_roi.valueChanged.connect(self._on_roi_size_changed)
        roi_layout.addWidget(self._spin_roi)

        self._chk_roi_lock = QCheckBox("위치 고정")
        self._chk_roi_lock.setToolTip("체크: 클릭한 위치 고정 / 미체크: 스팟 자동 추적")
        self._chk_roi_lock.stateChanged.connect(self._on_roi_lock_changed)
        roi_layout.addWidget(self._chk_roi_lock)

        self._btn_roi_clear = QPushButton("ROI 초기화")
        self._btn_roi_clear.clicked.connect(self._on_roi_clear)
        roi_layout.addWidget(self._btn_roi_clear)
        ctrl_layout.addWidget(roi_box)

        # 스크린샷 주기
        ss_box = QGroupBox("스크린샷 주기 (초)")
        ss_layout = QVBoxLayout(ss_box)
        self._spin_interval = QDoubleSpinBox()
        self._spin_interval.setRange(1.0, 3600.0)
        self._spin_interval.setValue(float(self._cfg.get("screenshot_interval_sec", 30)))
        self._spin_interval.setSingleStep(1.0)
        self._spin_interval.valueChanged.connect(self._on_interval_changed)
        ss_layout.addWidget(self._spin_interval)
        ctrl_layout.addWidget(ss_box)

        self._btn_snap = QPushButton("📷 지금 스크린샷")
        self._btn_snap.setEnabled(False)
        self._btn_snap.clicked.connect(self._on_manual_snap)
        ctrl_layout.addWidget(self._btn_snap)

        self._btn_end = QPushButton("💾 세션 종료 & 저장")
        self._btn_end.setFixedHeight(38)
        self._btn_end.setEnabled(False)
        self._btn_end.clicked.connect(self._on_end_session)
        ctrl_layout.addWidget(self._btn_end)

        self.setStatusBar(QStatusBar())

    # ──────────────────────────────── 슬롯: 파일 모드 ──────────────────────

    @pyqtSlot()
    def _on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "동영상 파일 선택", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if not path:
            return
        self._stop_all_threads()

        try:
            from apps.rheed_monitor.capture.file_source import FileVideoSource
            self._file_source = FileVideoSource(path)
        except Exception as exc:
            QMessageBox.critical(self, "파일 오류", str(exc))
            return

        w, h = self._file_source.frame_size
        self._video_label.set_frame_size(w, h)
        total = self._file_source.frame_count
        self._slider_pos.setRange(0, max(1, total - 1))

        self._file_thread = FileVideoThread(self._file_source, speed=self._spin_speed.value())
        self._file_thread.frame_ready.connect(self._on_frame)
        self._file_thread.progress.connect(self._on_file_progress)
        self._file_thread.finished.connect(self._on_file_finished)
        self._file_thread.error.connect(lambda m: self._status(f"오류: {m}"))
        self._file_thread.start()

        self._file_ctrl.setVisible(True)
        self._btn_play_pause.setText("⏸ 일시정지")
        self._btn_record.setEnabled(True)
        self._btn_snap.setEnabled(True)
        self._roi_center = None
        self._graph.reset()
        self._file_t0 = time.time()
        self._status(f"파일: {Path(path).name}  ({w}×{h}, {self._file_source.fps:.1f}fps)")

    @pyqtSlot()
    def _on_play_pause(self) -> None:
        if self._file_thread is None:
            return
        if self._file_thread._paused:
            self._file_thread.resume()
            self._btn_play_pause.setText("⏸ 일시정지")
        else:
            self._file_thread.pause()
            self._btn_play_pause.setText("▶ 재생")

    @pyqtSlot(float)
    def _on_speed_changed(self, val: float) -> None:
        if self._file_thread:
            self._file_thread.set_speed(val)

    @pyqtSlot(int)
    def _on_slider_moved(self, pos: int) -> None:
        if self._file_source and self._file_thread:
            was_paused = self._file_thread._paused
            self._file_thread.pause()
            self._file_source.seek(pos)
            if not was_paused:
                self._file_thread.resume()

    @pyqtSlot(int, int)
    def _on_file_progress(self, current: int, total: int) -> None:
        if total > 0:
            self._slider_pos.blockSignals(True)
            self._slider_pos.setValue(current)
            self._slider_pos.blockSignals(False)
            fps = self._file_source.fps if self._file_source else 30.0
            cur_s = current / fps
            tot_s = total / fps
            self._lbl_pos.setText(
                f"{int(cur_s//60):02d}:{int(cur_s%60):02d} / "
                f"{int(tot_s//60):02d}:{int(tot_s%60):02d}"
            )

    @pyqtSlot()
    def _on_file_finished(self) -> None:
        self._status("파일 재생 완료.")
        self._btn_play_pause.setText("▶ 재생")

    # ──────────────────────────────── 슬롯: 카메라 모드 ───────────────────

    @pyqtSlot()
    def _on_connect_camera(self) -> None:
        from apps.rheed_monitor.capture.hikrobot import (
            HikrobotCamera, mvs_available, list_devices, mvs_error_message
        )
        if not mvs_available():
            QMessageBox.critical(
                self, "MVS SDK 없음",
                mvs_error_message() or
                "HIKROBOT MVS SDK를 찾을 수 없습니다.\n"
                "MVS를 설치하거나 RheedSetup에서 경로를 수동 지정하세요."
            )
            return

        devs = list_devices()
        if not devs or "찾을 수 없습니다" in devs[0]:
            QMessageBox.warning(self, "카메라 없음", "\n".join(devs))
            return

        self._stop_all_threads()
        try:
            cam_cfg = self._cfg.get("camera", {})
            self._camera = HikrobotCamera(
                device_index=cam_cfg.get("device_index", 0),
                exposure_us=cam_cfg.get("exposure_us", 10000.0),
                gain_db=cam_cfg.get("gain_db", 0.0),
            )
        except Exception as exc:
            QMessageBox.critical(self, "연결 실패", str(exc))
            return

        self._cam_thread = CameraThread(self._camera)
        self._cam_thread.frame_ready.connect(self._on_frame)
        self._cam_thread.error.connect(lambda m: self._status(f"카메라 오류: {m}"))
        self._cam_thread.start()
        self._file_ctrl.setVisible(False)
        self._btn_record.setEnabled(True)
        self._btn_snap.setEnabled(True)
        self._roi_center = None
        self._graph.reset()
        self._status(f"카메라 연결됨 — {devs[0]}")

    # ──────────────────────────────── 슬롯: ROI ────────────────────────────

    @pyqtSlot(float, float)
    def _on_video_click(self, fx: float, fy: float) -> None:
        """비디오 클릭 → ROI 중심 설정."""
        self._roi_center = (fx, fy)
        self._status(f"ROI 위치 설정: ({fx:.1f}, {fy:.1f})")

    @pyqtSlot(int)
    def _on_roi_size_changed(self, val: int) -> None:
        self._roi_size = val

    @pyqtSlot(int)
    def _on_roi_lock_changed(self, state: int) -> None:
        self._roi_locked = (state == Qt.Checked)

    @pyqtSlot()
    def _on_roi_clear(self) -> None:
        self._roi_center = None
        self._status("ROI 초기화. 스팟 클릭으로 위치를 지정하세요.")

    # ──────────────────────────────── 슬롯: 녹화 / 세션 ──────────────────

    @pyqtSlot()
    def _on_toggle_record(self) -> None:
        if not self._recording:
            self._start_session()
        else:
            self._stop_recording()

    def _start_session(self) -> None:
        self._session = Session(
            self._base_output_dir,
            video_fps=float(self._cfg.get("video_fps", 15)),
            video_codec=self._cfg.get("video_codec", "mp4v"),
        )
        self._recording = True
        self._graph.reset()
        interval_ms = int(self._spin_interval.value() * 1000)
        self._screenshot_timer.start(interval_ms)
        self._btn_record.setText("⏹ 녹화 중지")
        self._btn_end.setEnabled(True)
        self._spin_interval.setEnabled(False)
        self._status(f"녹화 시작: {self._session.run_dir}")

    def _stop_recording(self) -> None:
        self._screenshot_timer.stop()
        self._recording = False
        self._btn_record.setText("⏺ 녹화 시작")
        self._spin_interval.setEnabled(True)
        if self._session:
            self._session.stop_video()
        self._status("녹화 중지")

    @pyqtSlot()
    def _on_end_session(self) -> None:
        if not self._session:
            return
        reply = QMessageBox.question(
            self, "세션 종료", "세션을 종료하고 압축 저장합니까?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._stop_recording()
        zip_path = self._session.close()
        self._session = None
        self._btn_end.setEnabled(False)
        QMessageBox.information(self, "저장 완료", f"저장 완료:\n{zip_path}")
        self._status(f"세션 저장: {zip_path}")

    @pyqtSlot()
    def _on_screenshot_timer(self) -> None:
        if self._latest_frame is not None and self._session:
            frame_ov = self._make_overlay(self._latest_frame,
                                          self._latest_spots, self._latest_roi)
            path = self._session.save_screenshot(frame_ov)
            self._status(f"스크린샷: {path.name}", temporary=True)

    @pyqtSlot()
    def _on_manual_snap(self) -> None:
        if self._latest_frame is None:
            return
        frame_ov = self._make_overlay(self._latest_frame,
                                      self._latest_spots, self._latest_roi)
        if self._session:
            path = self._session.save_screenshot(frame_ov)
        else:
            from datetime import datetime
            tmp = self._base_output_dir / "snapshots"
            tmp.mkdir(parents=True, exist_ok=True)
            path = tmp / f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(path), frame_ov)
        self._status(f"스크린샷: {path.name}", temporary=True)

    @pyqtSlot(float)
    def _on_interval_changed(self, value: float) -> None:
        if self._screenshot_timer.isActive():
            self._screenshot_timer.setInterval(int(value * 1000))

    # ──────────────────────────────── 핵심: 프레임 처리 ──────────────────

    @pyqtSlot(np.ndarray)
    def _on_frame(self, frame: np.ndarray) -> None:
        self._latest_frame = frame
        h, w = frame.shape[:2]
        self._video_label.set_frame_size(w, h)

        # 1. 스팟 검출 (녹색 채널 기반)
        spots = self._detector.detect(frame)
        self._latest_spots = spots

        # 2. ROI 중심 업데이트
        #    - 자동 추적 모드: 스팟 centroid로 ROI 이동
        #    - 위치 고정 모드: 클릭한 위치 유지
        if spots and not self._roi_locked:
            best = spots[0]
            if self._roi_center is None:
                # 첫 스팟 감지 → ROI 자동 초기화
                self._roi_center = (best.x, best.y)
            else:
                # 점진적 추적 (드리프트 대응): 스팟 위치로 70% 이동
                cx, cy = self._roi_center
                alpha = 0.3  # 낮을수록 천천히 따라감
                self._roi_center = (
                    cx + alpha * (best.x - cx),
                    cy + alpha * (best.y - cy),
                )

        # 3. ROI 강도 추출
        roi: Optional[RoiData] = None
        if self._roi_center is not None:
            roi = self._detector.extract_roi_intensity(
                frame, self._roi_center[0], self._roi_center[1], self._roi_size
            )
        self._latest_roi = roi

        # 4. 시간 계산 (파일 모드: 파일 내 시각, 카메라 모드: 경과 시각)
        if self._file_source is not None:
            t = self._file_source.current_time_sec
        else:
            t = None  # graph.push가 내부 t0 기준으로 계산

        # 5. 그래프 업데이트
        self._graph.push(roi, spots, t=t)

        # 6. 세션 기록
        if self._recording and self._session:
            self._session.write_frame(frame)
            self._session.record_spots(spots, roi=roi)

        # 7. 디스플레이 오버레이
        display = self._make_overlay(frame, spots, roi)
        self._show_frame(display)

        # 8. 상태바
        parts = []
        if spots:
            s = spots[0]
            tag = "broad" if s.is_broad else "dot"
            parts.append(f"스팟({s.x:.1f},{s.y:.1f})[{tag}]")
        if roi is not None:
            parts.append(f"ROI강도={roi.mean_intensity:.1f}")
        self._status(" | ".join(parts) if parts else "스팟 없음", temporary=True)

    def _make_overlay(
        self,
        frame: np.ndarray,
        spots: List[SpotResult],
        roi: Optional[RoiData],
    ) -> np.ndarray:
        """스팟 마커 + ROI 박스 오버레이 생성."""
        out = self._detector.draw_spots(frame, spots)
        if roi is not None:
            label = f"ROI I={roi.mean_intensity:.1f}"
            out = self._detector.draw_roi(out, roi, label=label)
        return out

    # ──────────────────────────────── 표시 ────────────────────────────────

    def _show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self._video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._video_label.setPixmap(pix)

    def _status(self, msg: str, temporary: bool = False) -> None:
        if temporary:
            self.statusBar().showMessage(msg, 2500)
        else:
            self.statusBar().showMessage(msg)

    # ──────────────────────────────── 종료 ────────────────────────────────

    def _stop_all_threads(self) -> None:
        if self._cam_thread:
            self._cam_thread.stop()
            self._cam_thread = None
        if self._file_thread:
            self._file_thread.stop()
            self._file_thread = None
        if self._camera:
            self._camera.release()
            self._camera = None
        if self._file_source:
            self._file_source.release()
            self._file_source = None

    def closeEvent(self, event) -> None:
        if self._recording:
            reply = QMessageBox.question(
                self, "종료 확인", "저장하지 않고 종료합니까?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        self._screenshot_timer.stop()
        self._stop_all_threads()
        if self._session:
            self._session.close()
        event.accept()
