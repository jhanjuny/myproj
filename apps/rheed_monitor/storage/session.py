# apps/rheed_monitor/storage/session.py
"""
세션 관리: 폴더 생성 / 영상 녹화 / 스크린샷 저장 / 압축 아카이브
"""

from __future__ import annotations

import csv
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ── 세션 폴더 ────────────────────────────────────────────────────────────────

class Session:
    """
    박막 성장 1회 = Session 1개.

    출력 구조:
        outputs/rheed/run_YYYYMMDD_HHMMSS/
            video.mp4
            screenshots/  ← 설정 주기마다 저장되는 PNG
            spot_data.csv ← ts, x, y, brightness, area, is_broad, spot_count
            session.log
        run_YYYYMMDD_HHMMSS.zip  ← 세션 종료 시 자동 생성
    """

    def __init__(self, base_dir: Path, video_fps: float = 15.0,
                 video_codec: str = "mp4v"):
        self.start_time = datetime.now()
        name = f"run_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = base_dir / name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "screenshots").mkdir(exist_ok=True)

        self._fps = video_fps
        self._codec = video_codec
        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_size: Optional[Tuple[int, int]] = None

        # CSV
        self._csv_path = self.run_dir / "spot_data.csv"
        with self._csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["timestamp", "spot_count", "x", "y", "brightness", "area", "is_broad"]
            )

        # log
        self._log_path = self.run_dir / "session.log"
        self.log(f"Session started: {self.start_time.isoformat()}")

    # ── 로그 ──────────────────────────────────────────────────────────────────
    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ── 영상 ──────────────────────────────────────────────────────────────────
    def write_frame(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*self._codec)
            path = str(self.run_dir / "video.mp4")
            self._writer = cv2.VideoWriter(path, fourcc, self._fps, (w, h))
            self._frame_size = (w, h)
            self.log(f"Video recording started: {path}")

        # 크기가 달라졌을 때 안전 처리
        if (w, h) != self._frame_size:
            frame = cv2.resize(frame, self._frame_size)

        self._writer.write(frame)

    def stop_video(self) -> None:
        if self._writer:
            self._writer.release()
            self._writer = None
            self.log("Video recording stopped.")

    # ── 스크린샷 ──────────────────────────────────────────────────────────────
    def save_screenshot(self, frame: np.ndarray) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = self.run_dir / "screenshots" / f"{ts}.png"
        cv2.imwrite(str(path), frame)
        return path

    # ── CSV 기록 ──────────────────────────────────────────────────────────────
    def record_spots(self, spots) -> None:
        """spots: List[SpotResult] (없으면 빈 리스트)"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with self._csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not spots:
                w.writerow([ts, 0, "", "", "", "", ""])
            else:
                # 주 스팟(가장 밝은 것)만 기록 (여러 개면 첫 행에 count 표시)
                for i, s in enumerate(spots):
                    w.writerow([
                        ts if i == 0 else "",
                        len(spots) if i == 0 else "",
                        f"{s.x:.2f}", f"{s.y:.2f}",
                        f"{s.brightness:.2f}", s.area,
                        "broad" if s.is_broad else "dot",
                    ])

    # ── 세션 종료 + 압축 ──────────────────────────────────────────────────────
    def close(self) -> Path:
        """영상 종료 후 zip 압축. zip 경로 반환."""
        self.stop_video()
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        self.log(f"Session ended. Duration: {duration:.1f}s")

        zip_path = self.run_dir.parent / f"{self.run_dir.name}.zip"
        self.log(f"Compressing → {zip_path} ...")

        with zipfile.ZipFile(zip_path, "w") as zf:
            for fpath in sorted(self.run_dir.rglob("*")):
                if not fpath.is_file():
                    continue
                arcname = fpath.relative_to(self.run_dir.parent)
                # 영상은 이미 압축된 형식 → ZIP_STORED
                compress = (zipfile.ZIP_STORED if fpath.suffix in (".mp4", ".avi")
                            else zipfile.ZIP_DEFLATED)
                zf.write(fpath, arcname, compress_type=compress)

        self.log(f"Archive saved: {zip_path}  ({zip_path.stat().st_size/1024/1024:.1f} MB)")
        return zip_path
