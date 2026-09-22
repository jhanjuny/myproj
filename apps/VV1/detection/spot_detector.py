# apps/VV1/detection/spot_detector.py
"""
RHEED 스팟 검출기 + ROI EMA 추적기 (VV1 배치 분석용)

apps/rheed_monitor/detection/spot_detector.py의 검출 알고리즘(녹색 채널 임계값 +
connectedComponents centroid)과 rheed_monitor GUI의 EMA ROI 추적 로직
(main_window._on_frame)을 VV1이 라이브 앱에 의존하지 않고도 배치 루프에서 그대로
쓸 수 있도록 독립 모듈로 재구성한 것입니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SpotResult:
    x: float
    y: float
    brightness: float
    area: int
    is_broad: bool


def normalize_frame(gray: np.ndarray) -> np.ndarray:
    """
    프레임(그레이/그린채널)을 0-255 min-max로 정규화.

    카메라 노출/게인이 세션마다 달라도 threshold_fraction 기반 검출과 이후
    배경제거 파라미터가 비슷하게 동작하도록 맞춰주는 선택적 전처리입니다.
    """
    lo = float(gray.min())
    hi = float(gray.max())
    if hi - lo < 1e-6:
        return np.zeros_like(gray, dtype=np.uint8)
    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


class SpotDetector:
    """apps/rheed_monitor의 SpotDetector와 동일한 검출 알고리즘."""

    def __init__(
        self,
        threshold_fraction: float = 0.5,
        min_brightness: float = 20.0,
        min_area: int = 4,
        max_area: int = 0,
        broad_area_threshold: int = 500,
        max_spots: int = 10,
        blur_ksize: int = 5,
    ):
        self.threshold_fraction = threshold_fraction
        self.min_brightness = min_brightness
        self.min_area = min_area
        self._max_area = max_area
        self.broad_area_threshold = broad_area_threshold
        self.max_spots = max_spots
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1

    def detect(self, frame: np.ndarray, frame_normalize: bool = False) -> List[SpotResult]:
        green = frame[:, :, 1].copy() if frame.ndim == 3 else frame.copy()
        if frame_normalize:
            green = normalize_frame(green)

        blurred = cv2.GaussianBlur(green, (self.blur_ksize, self.blur_ksize), 0)
        max_area = self._max_area or int(green.shape[0] * green.shape[1] * 0.6)
        max_val = float(blurred.max())
        if max_val < self.min_brightness:
            return []

        thresh_val = max_val * self.threshold_fraction
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        results: List[SpotResult] = []
        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area or area > max_area:
                continue
            cx, cy = float(centroids[i][0]), float(centroids[i][1])
            mask = (labels == i).astype(np.uint8)
            brightness = float(cv2.mean(blurred, mask=mask)[0])
            results.append(SpotResult(
                x=cx, y=cy, brightness=brightness, area=area,
                is_broad=area >= self.broad_area_threshold,
            ))

        results.sort(key=lambda r: r.brightness, reverse=True)
        return results[: self.max_spots]


class RoiTracker:
    """
    EMA 기반 ROI 중심 추적기 — rheed_monitor MainWindow._on_frame의
    "점진적 추적(alpha=0.3)" 로직과 동일. 고정 ROI(수동 지정/CSV 궤적)를
    쓸 경우엔 사용하지 않아도 됩니다.
    """

    def __init__(self, alpha: float = 0.3, initial: Optional[Tuple[float, float]] = None):
        self.alpha = alpha
        self._center: Optional[Tuple[float, float]] = initial

    @property
    def center(self) -> Optional[Tuple[float, float]]:
        return self._center

    def update(self, spots: List[SpotResult]) -> Optional[Tuple[float, float]]:
        if not spots:
            return self._center
        best = spots[0]
        if self._center is None:
            self._center = (best.x, best.y)
        else:
            cx, cy = self._center
            self._center = (
                cx + self.alpha * (best.x - cx),
                cy + self.alpha * (best.y - cy),
            )
        return self._center


def roi_rect(center: Tuple[float, float], size_px: int, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """center(px) ± size_px 사각형을 프레임 경계로 클리핑해 (x1,y1,x2,y2) 반환."""
    h, w = frame_shape
    cx, cy = center
    x1 = max(0, int(round(cx - size_px)))
    y1 = max(0, int(round(cy - size_px)))
    x2 = min(w, int(round(cx + size_px)))
    y2 = min(h, int(round(cy + size_px)))
    return x1, y1, x2, y2


def roi_mean_intensity(frame: np.ndarray, rect: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        return 0.0
    green = frame[y1:y2, x1:x2, 1] if frame.ndim == 3 else frame[y1:y2, x1:x2]
    return float(green.mean())
