
# apps/rheed_monitor/gui/main_window.py
from __future__ import annotations
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSizePolicy,
    QStatusBar, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from apps.rheed_monitor.detection.spot_detector import SpotDetector, SpotResult
from apps.rheed_monitor.storage.session import Session


class CameraThread(QThread):
    """카메라 폴링 스레드 -> frame_ready Signal 방출."""
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


class RheedGraph(FigureCanvas):
    """X / Y / Brightness 실시간 그래프 (matplotlib embedded)."""
    MAXPOINTS = 500

    def __init__(self, parent=None):
        fig = Figure(figsize=(4, 5), tight_layout=True)
        fig.patch.set_facecolor("#1e1e1e")
        super().__init__(fig)
        self.setParent(parent)

        self._ax_x = fig.add_subplot(3, 1, 1)
        self._ax_y = fig.add_subplot(3, 1, 2)
        self._ax_b = fig.add_subplot(3, 1, 3)

        for ax, ylabel in [
            (self._ax_x, "X (px)"),
            (self._ax_y, "Y (px)"),
            (self._ax_b, "Brightness"),
        ]:
            ax.set_facecolor("#2d2d2d")
            ax.set_ylabel(ylabel, color="white", fontsize=8)
            ax.tick_params(colors="white", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#555")
        self._ax_b.set_xlabel("Time (s)", color="white", fontsize=8)

        self._t: Deque[float] = deque(maxlen=self.MAXPOINTS)
        self._xs: Deque[Optional[float]] = deque(maxlen=self.MAXPOINTS)
        self._ys: Deque[Optional[float]] = deque(maxlen=self.MAXPOINTS)
        self._bs: Deque[Optional[float]] = deque(maxlen=self.MAXPOINTS)
        self._t0 = time.time()

        self._line_x, = self._ax_x.plot([], [], color="#00ff88", lw=1)
        self._line_y, = self._ax_y.plot([], [], color="#ffaa00", lw=1)
        self._line_b, = self._ax_b.plot([], [], color="#44aaff", lw=1)

    def push(self, spots: List[SpotResult]) -> None:
        t = time.time() - self._t0
        self._t.append(t)
        if spots:
            s = spots[0]
            self._xs.append(s.x)
            self._ys.append(s.y)
            self._bs.append(s.brightness)
        else:
            self._xs.append(None)
            self._ys.append(None)
            self._bs.append(None)
        self._redraw()

    def _redraw(self) -> None:
        ts = list(self._t)
        for ax, line, vals, color in [
            (self._ax_x, self._line_x, self._xs, "#00ff88"),
            (self._ax_y, self._line_y, self._ys, "#ffaa00"),
            (self._ax_b, self._line_b, self._bs, "#44aaff"),
        ]:
            seg_t, seg_v = [], []
            for tv, v in zip(ts, vals):
                if v is None:
                    if seg_t:
                        ax.plot(seg_t, seg_v, color=color, lw=1)
                    seg_t, seg_v = [], []
                else:
                    seg_t.append(tv)
                    seg_v.append(v)
            if seg_t:
                line.set_data(seg_t, seg_v)
            ax.relim()
            ax.autoscale_view()
        self.draw_idle()

    def reset(self) -> None:
        self._t.clear(); self._xs.clear(); self._ys.clear(); self._bs.clear()
        self._t0 = time.time()
        for line in (self._line_x, self._line_y, self._line_b):
            line.set_data([], [])
        self.draw_idle()


class MainWindow(QMainWindow):

    def __init__(self, cfg: dict, base_output_dir: Path):
        super().__init__()
        self.setWindowTitle("RHEED Monitor")
        self.resize(1280, 780)
        self._cfg = cfg
        self._base_output_dir = base_output_dir
        self._session: Optional[Session] = None
        self._camera = None
        self._cam_thread: Optional[CameraThread] = None
        self._detector = SpotDetector(**cfg.get("detector", {}))
        self._recording = False
        self._screenshot_timer = QTimer(self)
        self._screenshot_timer.timeout.connect(self._on_screenshot_timer)
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_spots: List[SpotResult] = []
        self._build_ui()
        self._status("카메라 연결 버튼을 눌러 시작하세요.")

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        top = QHBoxLayout()
        root.addLayout(top, stretch=1)

        self._video_label = QLabel("카메라 미연결")
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setStyleSheet("background:#111; color:#888; font-size:16px;")
        self._video_label.setMinimumSize(640, 480)
        self._video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        top.addWidget(self._video_label, stretch=3)

        self._graph = RheedGraph()
        self._graph.setMinimumWidth(320)
        top.addWidget(self._graph, stretch=2)

        ctrl_box = QGroupBox("컨트롤")
        ctrl_box.setMaximumHeight(130)
        ctrl_layout = QHBoxLayout(ctrl_box)
        root.addWidget(ctrl_box)

        self._btn_connect = QPushButton("카메라 연결")
        self._btn_connect.setFixedHeight(40)
        self._btn_connect.clicked.connect(self._on_connect)
        ctrl_layout.addWidget(self._btn_connect)

        self._btn_record = QPushButton("녹화 시작")
        self._btn_record.setFixedHeight(40)
        self._btn_record.setEnabled(False)
        self._btn_record.clicked.connect(self._on_toggle_record)
        ctrl_layout.addWidget(self._btn_record)

        ss_group = QGroupBox("스크린샷 주기 (초)")
        ss_layout = QVBoxLayout(ss_group)
        self._spin_interval = QDoubleSpinBox()
        self._spin_interval.setRange(1.0, 3600.0)
        self._spin_interval.setValue(float(self._cfg.get("screenshot_interval_sec", 30)))
        self._spin_interval.setSingleStep(1.0)
        self._spin_interval.valueChanged.connect(self._on_interval_changed)
        ss_layout.addWidget(self._spin_interval)
        ctrl_layout.addWidget(ss_group)

        self._chk_warn = QCheckBox("스팟 없을 때 경고")
        self._chk_warn.setChecked(True)
        ctrl_layout.addWidget(self._chk_warn)

        self._btn_snap = QPushButton("지금 스크린샷")
        self._btn_snap.setEnabled(False)
        self._btn_snap.clicked.connect(self._on_manual_snap)
        ctrl_layout.addWidget(self._btn_snap)

        self._btn_end = QPushButton("세션 종료 & 저장")
        self._btn_end.setFixedHeight(40)
        self._btn_end.setEnabled(False)
        self._btn_end.clicked.connect(self._on_end_session)
        ctrl_layout.addWidget(self._btn_end)

        self.setStatusBar(QStatusBar())

    @pyqtSlot()
    def _on_connect(self) -> None:
        from apps.rheed_monitor.capture.hikrobot import (
            HikrobotCamera, mvs_available, list_devices
        )
        if not mvs_available():
            QMessageBox.critical(self, "MVS 없음",
                "HIKROBOT MVS SDK를 찾을 수 없습니다.\n"
                "MVS를 설치한 후 재시작하세요.\n"
                "https://www.hikrobotics.com")
            return
        devs = list_devices()
        if not devs or "찾을 수 없습니다" in devs[0]:
            QMessageBox.warning(self, "카메라 없음", "\n".join(devs))
            return
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
        self._cam_thread.error.connect(lambda msg: self._status(f"카메라 오류: {msg}"))
        self._cam_thread.start()
        self._btn_connect.setEnabled(False)
        self._btn_record.setEnabled(True)
        self._btn_snap.setEnabled(True)
        self._status(f"카메라 연결됨 — {devs[0]}")

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
        self._btn_record.setText("녹화 중지")
        self._btn_end.setEnabled(True)
        self._spin_interval.setEnabled(False)
        self._status(f"녹화 시작: {self._session.run_dir}")

    def _stop_recording(self) -> None:
        self._screenshot_timer.stop()
        self._recording = False
        self._btn_record.setText("녹화 시작")
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
            frame_ov = self._detector.draw_spots(self._latest_frame, self._latest_spots)
            path = self._session.save_screenshot(frame_ov)
            self._status(f"스크린샷: {path.name}", temporary=True)

    @pyqtSlot()
    def _on_manual_snap(self) -> None:
        if self._latest_frame is None:
            return
        frame_ov = self._detector.draw_spots(self._latest_frame, self._latest_spots)
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

    @pyqtSlot(np.ndarray)
    def _on_frame(self, frame: np.ndarray) -> None:
        self._latest_frame = frame
        spots = self._detector.detect(frame)
        self._latest_spots = spots
        display = self._detector.draw_spots(frame, spots)
        if self._recording and self._session:
            self._session.write_frame(frame)
            self._session.record_spots(spots)
        self._graph.push(spots)
        self._show_frame(display)
        if spots:
            s = spots[0]
            tag = "broad" if s.is_broad else "dot"
            self._status(
                f"스팟 {len(spots)}개 | ({s.x:.1f}, {s.y:.1f}) "
                f"밝기={s.brightness:.1f} [{tag}]",
                temporary=True,
            )
        else:
            self._status("NO SPOT", temporary=True)

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
            self.statusBar().showMessage(msg, 2000)
        else:
            self.statusBar().showMessage(msg)

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
        if self._cam_thread:
            self._cam_thread.stop()
        if self._camera:
            self._camera.release()
        if self._session:
            self._session.close()
        event.accept()
