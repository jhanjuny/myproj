# apps/VV2/analysis/spectra.py
"""
ROI 강도 시계열 → (선택적 정규화) → 디트렌드 → Hann window → FFT → 성장률.

arXiv:2609.15922 Methods "Region-of-interest intensity and growth rate"
(Eq. 5-6)과 동일한 접근: 한 번의 강도 진동 = 1 monolayer 완성.
"""

from __future__ import annotations

import numpy as np


def normalize_series(y: np.ndarray, method: str = "max") -> np.ndarray:
    """
    강도 시계열 정규화. FFT의 피크 '주파수'는 스케일에 불변이지만, 여러 세션의
    진동 진폭을 비교하거나 결과를 사람이 읽기 좋게 만들려면 정규화가 유용합니다.

    method: "max"(0~1로 스케일) | "zscore"(평균0/표준편차1) | "none"
    """
    if method == "none" or len(y) == 0:
        return y
    if method == "zscore":
        std = y.std()
        return (y - y.mean()) / std if std > 1e-12 else y - y.mean()
    # "max" (default): 절대값 최대가 1이 되도록 스케일
    amp = np.max(np.abs(y))
    return y / amp if amp > 1e-12 else y


def _detrend(y: np.ndarray, t: np.ndarray, method: str, alpha: float) -> np.ndarray:
    if method == "mean":
        return y - y.mean()
    if method == "linear":
        coef = np.polyfit(t, y, 1)
        return y - np.polyval(coef, t)
    if method == "ema":
        s = np.empty_like(y)
        s[0] = y[0]
        a = float(alpha)
        for i in range(1, len(y)):
            s[i] = a * y[i] + (1 - a) * s[i - 1]
        return y - s
    return y - y.mean()


def intensity_fft(
    intensities,
    timestamps,
    detrend: str = "ema",
    alpha: float = 0.15,
    normalize: str = "max",
    layer_thickness_angstrom: float | None = None,
) -> dict:
    """
    Returns {freq, power, f0, period, growth_ml_per_s, growth_angstrom_s, dt}.
    growth_angstrom_s는 layer_thickness_angstrom이 주어졌을 때만 채워짐.
    """
    y = np.asarray(intensities, dtype=float)
    t = np.asarray(timestamps, dtype=float)
    n = len(y)
    if n < 8:
        raise ValueError("FFT에는 최소 8프레임 이상의 강도 시계열이 필요합니다")

    yd = _detrend(y, t, detrend, alpha)
    yd = normalize_series(yd, normalize)
    yw = yd * np.hanning(n)

    dt = float(np.mean(np.diff(t))) if n > 1 else 1.0
    if dt <= 0:
        dt = 1.0
    freqs = np.fft.rfftfreq(n, d=dt)
    power = np.abs(np.fft.rfft(yw))
    power[0] = 0.0  # DC(느린 드리프트) 제외

    peak_idx = int(np.argmax(power))
    f0 = float(freqs[peak_idx]) if power[peak_idx] > 0 else 0.0
    period = (1.0 / f0) if f0 > 0 else None
    growth_angstrom_s = (f0 * layer_thickness_angstrom) if (f0 > 0 and layer_thickness_angstrom) else None

    return {
        "freq": freqs.tolist(),
        "power": power.tolist(),
        "f0": f0,
        "period": period,
        "growth_ml_per_s": f0,
        "growth_angstrom_s": growth_angstrom_s,
        "dt": dt,
    }
