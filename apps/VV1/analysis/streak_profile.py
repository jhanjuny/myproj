# apps/VV1/analysis/streak_profile.py
"""
스트립 프로파일 추출 → 배경 제거 → 3-피크(specular + 1차 회절 스트릭 2개)
검출/추적 → FWHM → 격자상수 d / apparent in-plane coherence length 변환.

arXiv:2609.15922 Methods "Strip-profile extraction..."과
"Streak width and apparent coherence length" 절의 방법(ALS 배경제거,
prominence 기반 피크검출 + 서브픽셀 파라볼릭 보정, half-max FWHM,
d = λL/Δx 소각근사)을 우리 코드베이스에 맞게 독립 구현한 것입니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import grey_opening, minimum_filter1d, uniform_filter1d
from scipy.signal import find_peaks

from .wavelength import lambda_cm

BG_METHODS = ("als", "rolling_ball", "polynomial", "moving_min", "none")


# ── 스트립 추출 ───────────────────────────────────────────────────────────

def strip_sum_profile(frame: np.ndarray, rect: tuple) -> np.ndarray:
    """ROI rect=(x1,y1,x2,y2) 내부를 세로로 합산해 1D 프로파일(가로 방향) 생성."""
    x1, y1, x2, y2 = rect
    if frame.ndim == 3:
        band = frame[y1:y2, x1:x2, 1]
    else:
        band = frame[y1:y2, x1:x2]
    return band.astype(np.float64).sum(axis=0)


# ── 배경 추정 ─────────────────────────────────────────────────────────────

def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Asymmetric Least Squares baseline."""
    L = len(y)
    if L < 5:
        return np.zeros_like(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    D = lam * D.dot(D.T)
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L).tocsc()
    z = y.copy()
    for _ in range(niter):
        W.setdiag(w)
        z = spsolve((W + D).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def baseline_rolling_ball(y: np.ndarray, radius: Optional[int] = None) -> np.ndarray:
    if radius is None:
        radius = max(5, len(y) // 12)
    return grey_opening(y, size=2 * radius + 1)


def baseline_polynomial(y: np.ndarray, degree: int = 4, iterations: int = 6) -> np.ndarray:
    x = np.arange(len(y), dtype=float)
    weights = np.ones(len(y))
    z = y.copy()
    for _ in range(iterations):
        coeffs = np.polyfit(x, y, degree, w=weights)
        z = np.polyval(coeffs, x)
        weights = np.where(y > z, 0.1, 1.0)
    return z


def baseline_moving_min(y: np.ndarray, window: Optional[int] = None) -> np.ndarray:
    if window is None:
        window = max(9, len(y) // 8)
    mn = minimum_filter1d(y, size=window, mode="nearest")
    return uniform_filter1d(mn, size=max(3, window // 3), mode="nearest")


_BASELINE_FUNCS = {
    "als": baseline_als,
    "rolling_ball": baseline_rolling_ball,
    "polynomial": baseline_polynomial,
    "moving_min": baseline_moving_min,
}


def subtract_background(profile: np.ndarray, method: str = "als", params: Optional[dict] = None):
    """Returns (baseline, subtracted). method='none' → baseline은 전부 0."""
    if method == "none":
        return np.zeros_like(profile), profile.copy()
    fn = _BASELINE_FUNCS.get(method)
    if fn is None:
        raise ValueError(f"Unknown background method: {method!r} (choose from {BG_METHODS})")
    baseline = fn(profile, **(params or {}))
    return baseline, profile - baseline


# ── 피크 검출 ──────────────────────────────────────────────────────────────

@dataclass
class StreakPeaks:
    specular_x: Optional[float]
    left_x: Optional[float]
    right_x: Optional[float]

    @property
    def avg_dx(self) -> Optional[float]:
        vals = []
        if self.specular_x is not None and self.left_x is not None:
            vals.append(abs(self.specular_x - self.left_x))
        if self.specular_x is not None and self.right_x is not None:
            vals.append(abs(self.right_x - self.specular_x))
        return sum(vals) / len(vals) if vals else None


def _subpixel_refine(y: np.ndarray, idx: int) -> float:
    """정수 피크 인덱스 주변 3점 파라볼릭 보간으로 서브픽셀 위치 산출."""
    L = len(y)
    if idx <= 0 or idx >= L - 1:
        return float(idx)
    y0, y1, y2 = y[idx - 1], y[idx], y[idx + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return float(idx)
    offset = 0.5 * (y0 - y2) / denom
    if abs(offset) > 1:
        return float(idx)
    return float(idx) + offset


def find_three_peaks(
    subtracted: np.ndarray,
    prev: Optional[StreakPeaks] = None,
    search_window: int = 20,
    min_prominence_frac: float = 0.05,
    min_signal_frac: float = 0.05,
) -> StreakPeaks:
    """
    Cold start(prev=None): scipy.find_peaks(prominence 기반)으로 처음부터 3피크 탐색
    — 가장 강한 피크를 specular로, 그 좌우에서 가장 가까운 피크를 1차 스트릭으로.

    Tracking(prev 있음): 이전 위치 주변 좁은 창(search_window)에서 argmax로
    프레임 간 지속 추적. 창 안에 신호가 없으면(min_signal_frac 미만) None 처리.
    """
    L = len(subtracted)
    mx = subtracted.max() if L else 0.0

    if prev is not None and prev.specular_x is not None and mx > 0:
        floor = max(mx * min_signal_frac, 1e-9)

        def pick(expected, lo_bound=0, hi_bound=L):
            if expected is None:
                return None
            lo = max(lo_bound, int(round(expected)) - search_window)
            hi = min(hi_bound, int(round(expected)) + search_window + 1)
            if hi <= lo:
                return None
            window = subtracted[lo:hi]
            if window.max() < floor:
                return None
            peak_idx = lo + int(np.argmax(window))
            return _subpixel_refine(subtracted, peak_idx)

        specular_x = pick(prev.specular_x)
        if specular_x is not None:
            gap = 4
            left_x = pick(prev.left_x, hi_bound=int(specular_x) - gap) if prev.left_x is not None else None
            right_x = pick(prev.right_x, lo_bound=int(specular_x) + gap + 1) if prev.right_x is not None else None
            return StreakPeaks(specular_x, left_x, right_x)
        # specular를 잃으면 콜드스타트로 폴백

    if mx <= 0:
        return StreakPeaks(None, None, None)
    prominence = max(mx * min_prominence_frac, 1e-9)
    idx, _ = find_peaks(subtracted, prominence=prominence)
    if len(idx) == 0:
        return StreakPeaks(None, None, None)

    specular_idx = idx[np.argmax(subtracted[idx])]
    left_c = idx[idx < specular_idx]
    right_c = idx[idx > specular_idx]
    left_idx = left_c[np.argmin(np.abs(left_c - specular_idx))] if len(left_c) else None
    right_idx = right_c[np.argmin(np.abs(right_c - specular_idx))] if len(right_c) else None

    return StreakPeaks(
        specular_x=_subpixel_refine(subtracted, int(specular_idx)),
        left_x=_subpixel_refine(subtracted, int(left_idx)) if left_idx is not None else None,
        right_x=_subpixel_refine(subtracted, int(right_idx)) if right_idx is not None else None,
    )


# ── FWHM → coherence length ─────────────────────────────────────────────

def fwhm_of_peak(subtracted: np.ndarray, peak_x: Optional[float]) -> Optional[float]:
    """배경제거된 프로파일에서 peak_x의 half-max FWHM(px), 선형보간으로 서브픽셀 정밀도."""
    if peak_x is None:
        return None
    L = len(subtracted)
    xp = int(round(peak_x))
    if xp <= 0 or xp >= L - 1:
        return None
    peak_val = float(subtracted[xp])
    if peak_val <= 0:
        return None
    half = peak_val / 2.0

    i = xp
    while i > 0 and subtracted[i] > half:
        i -= 1
    if subtracted[i] > half:
        return None
    yl0, yl1 = float(subtracted[i]), float(subtracted[i + 1])
    x_left = i + (half - yl0) / (yl1 - yl0) if yl1 != yl0 else float(i)

    j = xp
    while j < L - 1 and subtracted[j] > half:
        j += 1
    if subtracted[j] > half:
        return None
    yr0, yr1 = float(subtracted[j]), float(subtracted[j - 1])
    x_right = j - (half - yr0) / (yr1 - yr0) if yr1 != yr0 else float(j)

    width = x_right - x_left
    return width if width > 0 else None


# ── 픽셀 → 실공간 변환 ────────────────────────────────────────────────────

def d_from_dx(dx_px: Optional[float], scale_cm_px: float, l_cm: float, v_kev: float) -> Optional[float]:
    """
    d(Å) = λL / Δx  (소각 RHEED 관계식).
    격자상수(Δx=streak 간격)와 apparent coherence length(Δx=FWHM)에 동일하게 재사용.
    scale_cm_px가 0(미설정)이거나 dx_px가 없으면 None (픽셀 캘리브레이션 없이는
    Å 단위로 환산할 수 없음 — 이 경우 상위 호출부는 픽셀 값만 기록).
    """
    if dx_px is None or dx_px <= 0 or scale_cm_px <= 0:
        return None
    dx_cm = dx_px * scale_cm_px
    d_cm = (lambda_cm(v_kev) * l_cm) / dx_cm
    return d_cm * 1e8


def solve_scale_from_standard(dx_px: float, l_cm: float, v_kev: float, d_known_angstrom: float) -> Optional[float]:
    """
    기준 물질의 이미 알려진 d-spacing(d_known_angstrom)과, 그 물질이 찍힌 프레임에서
    측정한 스트릭 간격(dx_px)으로 픽셀 스케일(cm/px)을 역산:
        d_known = λL / (dx_px · scale)  ⇒  scale = λL / (d_known · dx_px)
    """
    if dx_px is None or dx_px <= 0 or d_known_angstrom is None or d_known_angstrom <= 0:
        return None
    dx_cm_total = (lambda_cm(v_kev) * l_cm) / (d_known_angstrom * 1e-8)
    return dx_cm_total / dx_px
