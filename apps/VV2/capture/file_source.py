# apps/VV2/capture/file_source.py
"""
동영상 파일 소스 — cv2.VideoCapture 래퍼 (VV2 전용, apps/rheed_monitor와 독립)

VV2은 apps/rheed_monitor 라이브 파이프라인과 완전히 분리된 오프라인 도구이므로
비디오 리더도 별도로 둡니다 (동일 로직을 의도적으로 중복 — 상호 의존 없음).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class FileVideoSource:
    """동영상 파일 소스. 배치 분석은 순차 읽기만 필요하므로 seek는 재분석 시에만 사용."""

    def __init__(self, path: str | Path):
        self._path = str(path)
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"동영상 파일을 열 수 없습니다: {self._path}")

    def read(self) -> Optional[np.ndarray]:
        ret, frame = self._cap.read()
        return frame if ret else None

    def read_frame_at(self, frame_index: int) -> Optional[np.ndarray]:
        """미리보기용 — 특정 프레임 하나만 읽고 위치를 복원하지 않음(호출 후 재오픈 권장)."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        ret, frame = self._cap.read()
        return frame if ret else None

    @property
    def fps(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def frame_size(self) -> Tuple[int, int]:
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def release(self) -> None:
        if self._cap.isOpened():
            self._cap.release()

    def __del__(self) -> None:
        self.release()
