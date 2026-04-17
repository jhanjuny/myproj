# apps/rheed_monitor/detection/spot_detector.py
"""
RHEED 스팟 검출기

임계값 전략:
  - max_val * threshold_fraction 방식 사용 (percentile 방식은 dot처럼
    매우 작은 스팟이면 대부분 픽셀이 0이라 percentile 값도 0이 되는 문제)
  - dot / broad 모두 centroid 로 처리 (broad 는 퍼진 분포의 중앙점)
  - no spot: max 밝기 < min_brightness 이면 빈 리스트 반환
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class SpotResult:
    x: float           # centroid x (pixels)
    y: float           # centroid y (pixels)
    brightness: float  # 검출 영역 내 평균 밝기 (0–255)
    area: int          # 영역 넓이 (pixels)
    is_broad: bool     # True = 퍼진 패턴, False = dot


class SpotDetector:
    """
    Parameters
    ----------
    threshold_fraction   : max 밝기의 몇 배 이상을 스팟으로 볼지 (기본 0.5)
    min_brightness       : 이 값 미만이면 스팟 없음으로 판정 (기본 20)
    min_area             : 검출할 최소 픽셀 면적 (기본 4)
    broad_area_threshold : 이 면적 이상이면 broad 패턴으로 분류 (기본 500)
    max_spots            : 최대 반환 스팟 수 (기본 10)
    blur_ksize           : Gaussian blur 커널 크기 (홀수, 기본 5)
    """

    def __init__(
        self,
        threshold_fraction: float = 0.5,
        min_brightness: float = 20.0,
        min_area: int = 4,
        max_area: int = 0,             # 0 = 프레임 크기의 60% 자동
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

    def detect(self, frame: np.ndarray) -> List[SpotResult]:
        """프레임에서 스팟 검출. 없으면 빈 리스트."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        max_area = self._max_area or int(gray.shape[0] * gray.shape[1] * 0.6)
        max_val = float(blurred.max())

        # 화면이 너무 어두우면 no spot
        if max_val < self.min_brightness:
            return []

        # max 기반 임계값: max * fraction
        thresh_val = max_val * self.threshold_fraction
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

        # Connected components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        results: List[SpotResult] = []
        for i in range(1, n_labels):  # 0 = background
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area or area > max_area:
                continue

            cx = float(centroids[i][0])
            cy = float(centroids[i][1])

            mask = (labels == i).astype(np.uint8)
            brightness = float(cv2.mean(blurred, mask=mask)[0])

            is_broad = area >= self.broad_area_threshold
            results.append(SpotResult(x=cx, y=cy, brightness=brightness,
                                      area=area, is_broad=is_broad))

        results.sort(key=lambda r: r.brightness, reverse=True)
        return results[: self.max_spots]

    def draw_spots(self, frame: np.ndarray, spots: List[SpotResult]) -> np.ndarray:
        """spot 오버레이 그리기. 원본의 복사본 반환."""
        out = frame.copy()
        for i, spot in enumerate(spots):
            cx, cy = int(round(spot.x)), int(round(spot.y))
            color = (0, 255, 0) if i == 0 else (0, 200, 200)
            marker_type = cv2.MARKER_SQUARE if spot.is_broad else cv2.MARKER_CROSS
            cv2.drawMarker(out, (cx, cy), color, marker_type, markerSize=24, thickness=2)
            cv2.circle(out, (cx, cy), 6, color, 1)
            tag = "B" if spot.is_broad else "D"
            label = f"#{i+1}[{tag}] ({cx},{cy}) Br={spot.brightness:.0f}"
            cv2.putText(out, label, (cx + 12, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        if not spots:
            h, w = out.shape[:2]
            cv2.putText(out, "NO SPOT", (w // 2 - 80, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        return out
