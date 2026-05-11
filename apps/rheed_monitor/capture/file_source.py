# apps/rheed_monitor/capture/file_source.py
"""
동영상 파일 소스 — cv2.VideoCapture 래퍼

오프라인 분석용: 녹화된 .mp4, .avi 등을 프레임 단위로 읽습니다.
실시간 카메라(HikrobotCamera)와 동일한 .read() 인터페이스를 제공합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class FileVideoSource:
    """
    동영상 파일 소스.

    Parameters
    ----------
    path       : 동영상 파일 경로 (.mp4, .avi 등)
    loop       : True이면 파일 끝에서 처음으로 되감기
    """

    def __init__(self, path: str | Path, loop: bool = False):
        self._path = str(path)
        self._loop = loop
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"동영상 파일을 열 수 없습니다: {self._path}")

    # ── 프레임 읽기 ────────────────────────────────────────────────────────
    def read(self) -> Optional[np.ndarray]:
        """다음 프레임 반환. 파일 끝이면 None (loop=False) 또는 첫 프레임부터 재시작 (loop=True)."""
        ret, frame = self._cap.read()
        if not ret:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    return None
            else:
                return None
        return frame

    def seek(self, frame_index: int) -> None:
        """특정 프레임으로 이동."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))

    def seek_time(self, time_sec: float) -> None:
        """특정 시각(초)으로 이동."""
        self._cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)

    # ── 메타데이터 ─────────────────────────────────────────────────────────
    @property
    def fps(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration_sec(self) -> float:
        fps = self.fps
        return self.frame_count / fps if fps > 0 else 0.0

    @property
    def current_frame(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    @property
    def current_time_sec(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        """(width, height)"""
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def is_finished(self) -> bool:
        return self.current_frame >= self.frame_count

    # ── 해제 ──────────────────────────────────────────────────────────────
    def release(self) -> None:
        if self._cap.isOpened():
            self._cap.release()

    def __del__(self) -> None:
        self.release()

    def __repr__(self) -> str:
        w, h = self.frame_size
        return (
            f"FileVideoSource('{Path(self._path).name}', "
            f"{w}x{h}, {self.fps:.1f}fps, {self.frame_count}f, "
            f"{self.duration_sec:.1f}s)"
        )
