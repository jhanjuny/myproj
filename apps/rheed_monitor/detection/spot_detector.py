# apps/rheed_monitor/detection/spot_detector.py
"""
RHEED 스팟 검출기 + ROI 강도 추출기

■ 스팟 검출 전략 (SpotDetector.detect):
  - 녹색 채널(BGR[:,:,1]) 사용 — RHEED 인광 스크린이 녹색
  - max_val * threshold_fraction 임계값 (percentile은 대부분 배경 픽셀이
    0인 작은 dot에서 실패함)
  - dot / broad 모두 centroid로 처리

■ ROI 강도 추출 (SpotDetector.extract_roi_intensity):
  - RHEED 진동(oscillation) 측정을 위한 ROI 적분 강도
  - 논문 방식: specular spot 주변 고정 ROI의 평균 강도 vs. 시간
  - 최대값(포화)이 아닌 ROI 적분값 → 성장 중 진동 신호 획득 가능
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SpotResult:
    x: float           # centroid x (pixels, frame 좌표)
    y: float           # centroid y (pixels, frame 좌표)
    brightness: float  # 검출 영역 내 평균 밝기 (0–255)
    area: int          # 영역 넓이 (pixels)
    is_broad: bool     # True = 퍼진 패턴, False = dot


@dataclass
class RoiData:
    """ROI 강도 측정 결과."""
    cx: float          # ROI 중심 x (pixels)
    cy: float          # ROI 중심 y (pixels)
    roi_size: int      # ROI 반변 길이 (pixels), 실제 박스 = 2*roi_size x 2*roi_size
    mean_intensity: float   # ROI 내 녹색 채널 평균 강도
    sum_intensity: float    # ROI 내 녹색 채널 합산 강도
    pixel_count: int        # ROI 내 유효 픽셀 수
    rect: Tuple[int, int, int, int]  # (x1, y1, x2, y2) — 실제 ROI 영역


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

    # ── 스팟 검출 ──────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> List[SpotResult]:
        """
        프레임에서 RHEED 스팟 검출.
        녹색 채널 우선 사용 (인광 스크린이 녹색).
        """
        # 녹색 채널 추출
        if frame.ndim == 3:
            green = frame[:, :, 1].copy()  # BGR → G channel
        else:
            green = frame.copy()

        blurred = cv2.GaussianBlur(green, (self.blur_ksize, self.blur_ksize), 0)

        max_area = self._max_area or int(green.shape[0] * green.shape[1] * 0.6)
        max_val = float(blurred.max())

        if max_val < self.min_brightness:
            return []

        # max 기반 임계값 (percentile 대신)
        thresh_val = max_val * self.threshold_fraction
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

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

    # ── ROI 강도 추출 (RHEED 진동 측정 핵심) ──────────────────────────────
    def extract_roi_intensity(
        self,
        frame: np.ndarray,
        cx: float,
        cy: float,
        roi_size: int = 40,
    ) -> RoiData:
        """
        지정된 중심(cx, cy) 주변 ROI의 녹색 채널 적분 강도를 반환합니다.

        RHEED 진동 측정 원리:
          - 성장 중 결정 표면 → RHEED 스팟 강도가 주기적으로 변함 (1 monolayer = 1주기)
          - 최대 픽셀값은 포화(255)로 변화 없음 → ROI 적분값을 사용해야 진동 추출 가능
          - 논문(Fig 4.11~4.14 등)과 동일한 방식

        Parameters
        ----------
        frame    : BGR 또는 그레이스케일 프레임
        cx, cy   : ROI 중심 (pixel, float OK)
        roi_size : ROI 반변 길이 (실제 박스 = 2*roi_size × 2*roi_size px)

        Returns
        -------
        RoiData : mean_intensity, sum_intensity, rect, pixel_count
        """
        h, w = frame.shape[:2]

        # ROI 영역 계산 (경계 클리핑)
        x1 = max(0, int(round(cx - roi_size)))
        y1 = max(0, int(round(cy - roi_size)))
        x2 = min(w, int(round(cx + roi_size)))
        y2 = min(h, int(round(cy + roi_size)))

        if x2 <= x1 or y2 <= y1:
            return RoiData(cx=cx, cy=cy, roi_size=roi_size,
                           mean_intensity=0.0, sum_intensity=0.0,
                           pixel_count=0, rect=(x1, y1, x2, y2))

        # 녹색 채널 ROI 추출
        if frame.ndim == 3:
            roi_green = frame[y1:y2, x1:x2, 1].astype(np.float32)  # G channel
        else:
            roi_green = frame[y1:y2, x1:x2].astype(np.float32)

        pixel_count = roi_green.size
        sum_intensity = float(roi_green.sum())
        mean_intensity = float(roi_green.mean()) if pixel_count > 0 else 0.0

        return RoiData(
            cx=cx, cy=cy, roi_size=roi_size,
            mean_intensity=mean_intensity,
            sum_intensity=sum_intensity,
            pixel_count=pixel_count,
            rect=(x1, y1, x2, y2),
        )

    # ── 오버레이 그리기 ───────────────────────────────────────────────────
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

    def draw_roi(
        self,
        frame: np.ndarray,
        roi: RoiData,
        color: Tuple[int, int, int] = (0, 255, 255),
        label: Optional[str] = None,
    ) -> np.ndarray:
        """ROI 박스 오버레이 그리기."""
        out = frame.copy()
        x1, y1, x2, y2 = roi.rect
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.drawMarker(out, (int(round(roi.cx)), int(round(roi.cy))),
                       color, cv2.MARKER_CROSS, markerSize=16, thickness=1)
        if label is None:
            label = f"ROI  I={roi.mean_intensity:.1f}"
        cv2.putText(out, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return out
