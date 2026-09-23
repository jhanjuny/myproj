# apps/VV2/analysis/report.py
"""
배치 분석 결과를 사람이 바로 볼 수 있는 PNG 한 장짜리 요약 리포트로 렌더링.

CSV/JSON만으로는 결과를 못 알아본다는 피드백에 따라 VV1에 추가된 기능.
arXiv:2609.15922 Fig 3/4 구성(대표 프레임+ROI, 강도 vs 시간, d(t)/coherence(t),
FFT 스펙트럼)을 참고해 우리 산출물 구조에 맞게 재구성했습니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 헤드리스 렌더링 — GUI 백엔드/이벤트루프 불필요
import matplotlib.pyplot as plt

# 기본 폰트(DejaVu Sans)에 한글 글리프가 없어 제목/라벨이 전부 깨져 나오는 문제
# (MBE 리포트 워크플로우에서도 겪은 것과 동일 클래스의 버그) — Windows 기본 탑재
# 폰트인 맑은고딕으로 강제 지정. 유니코드 마이너스 기호도 이 폰트에서 깨지므로 비활성화.
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def _draw_representative_overlay(
    frame: np.ndarray,
    rect: tuple,
    specular_x: Optional[float],
    left_x: Optional[float],
    right_x: Optional[float],
) -> np.ndarray:
    """ROI 박스 + 검출된 3피크 위치(로컬 오프셋 → 절대 프레임 좌표)를 오버레이."""
    out = frame.copy()
    x1, y1, x2, y2 = rect
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)

    def _vline(local_x: Optional[float], color, label):
        if local_x is None:
            return
        ax = x1 + int(round(local_x))
        cv2.line(out, (ax, y1), (ax, y2), color, 1)
        cv2.putText(out, label, (ax + 3, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    _vline(specular_x, (0, 255, 0), "0")
    _vline(left_x, (0, 200, 255), "-1")
    _vline(right_x, (0, 200, 255), "+1")
    return out


def render_summary_png(
    out_path: Path,
    records: list,
    fft_result: Optional[dict],
    summary: dict,
    representative_frame: Optional[np.ndarray] = None,
    representative_rect: Optional[tuple] = None,
    representative_peaks: Optional[dict] = None,
) -> None:
    """
    records   : worker._run()에서 쌓은 프레임별 dict 리스트
    fft_result: analysis.spectra.intensity_fft() 반환값 또는 None
    summary   : worker가 만든 summary dict (calibration, 평균값 등)
    representative_*: 대표 프레임 오버레이용(없으면 해당 패널은 안내 텍스트만 표시)
    """
    t = [r["t_sec"] for r in records]
    intensity = [r["roi_mean_intensity"] for r in records]
    dx = [r["dx_px"] for r in records]
    d_ang = [r["d_angstrom"] for r in records]
    coh_ang = [r["coherence_angstrom"] for r in records]

    has_calib = any(v is not None for v in d_ang)

    fig = plt.figure(figsize=(15, 9), dpi=130)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28, top=0.86, bottom=0.07)

    # ── (0,0) 대표 프레임 + ROI/피크 오버레이 ──
    ax_img = fig.add_subplot(gs[0, 0])
    if representative_frame is not None and representative_rect is not None:
        peaks = representative_peaks or {}
        overlay = _draw_representative_overlay(
            representative_frame, representative_rect,
            peaks.get("specular_x"), peaks.get("left_x"), peaks.get("right_x"),
        )
        ax_img.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax_img.set_title("대표 프레임 (ROI + 검출 피크)", fontsize=10)
    else:
        ax_img.text(0.5, 0.5, "대표 프레임 없음\n(ROI가 한 번도 검출되지 않음)",
                     ha="center", va="center", fontsize=10)
    ax_img.axis("off")

    # ── (0,1) ROI 강도 vs 시간 ──
    ax_i = fig.add_subplot(gs[0, 1])
    ax_i.plot(t, intensity, color="#00838f", lw=0.8)
    ax_i.set_title("ROI 강도 vs 시간 (RHEED 진동)", fontsize=10)
    ax_i.set_xlabel("시간 (s)")
    ax_i.set_ylabel("강도")

    # ── (0,2) FFT 파워 스펙트럼 ──
    ax_fft = fig.add_subplot(gs[0, 2])
    if fft_result is not None:
        ax_fft.plot(fft_result["freq"], fft_result["power"], color="#6a1b9a", lw=0.9)
        f0 = fft_result.get("f0") or 0.0
        if f0 > 0:
            ax_fft.axvline(f0, color="red", ls="--", lw=1)
            ax_fft.text(f0, max(fft_result["power"]) * 0.9, f" f0={f0:.4f} Hz",
                        color="red", fontsize=8)
        ax_fft.set_title("FFT 파워 스펙트럼", fontsize=10)
        ax_fft.set_xlabel("주파수 (Hz)")
    else:
        ax_fft.text(0.5, 0.5, "FFT 불가\n(유효 프레임 8개 미만)", ha="center", va="center")
        ax_fft.axis("off")

    # ── (1,0) 스트릭 간격 dx_px vs 시간 (항상 표시 가능 — 캘리브레이션 무관) ──
    ax_dx = fig.add_subplot(gs[1, 0])
    ax_dx.plot(t, dx, color="#2e7d32", lw=0.8)
    ax_dx.set_title("스트릭 간격 dx (px) vs 시간", fontsize=10)
    ax_dx.set_xlabel("시간 (s)")
    ax_dx.set_ylabel("dx (px)")

    # ── (1,1) 격자상수 d(t) ──
    ax_d = fig.add_subplot(gs[1, 1])
    if has_calib:
        ax_d.plot(t, d_ang, color="#c62828", lw=0.8)
        ax_d.set_title("격자상수 d(t) [Å]", fontsize=10)
        ax_d.set_xlabel("시간 (s)")
        ax_d.set_ylabel("d (Å)")
    else:
        ax_d.text(0.5, 0.5, "캘리브레이션 미설정\n(픽셀 스케일 0)\n→ dx_px 그래프 참고",
                    ha="center", va="center", fontsize=9)
        ax_d.axis("off")

    # ── (1,2) coherence length(t) ──
    ax_coh = fig.add_subplot(gs[1, 2])
    if has_calib:
        ax_coh.plot(t, coh_ang, color="#ef6c00", lw=0.8)
        ax_coh.set_title("apparent coherence length(t) [Å]", fontsize=10)
        ax_coh.set_xlabel("시간 (s)")
        ax_coh.set_ylabel("L (Å)")
    else:
        ax_coh.text(0.5, 0.5, "캘리브레이션 미설정", ha="center", va="center", fontsize=9)
        ax_coh.axis("off")

    # ── 상단 요약 텍스트 ──
    calib = summary.get("calibration", {})
    gr = summary.get("growth_rate") or {}
    d_mean = summary.get("d_angstrom_mean")
    coh_mean = summary.get("coherence_angstrom_mean")
    lines = [
        f"파일: {Path(summary.get('video', '')).name}   "
        f"분석 프레임: {summary.get('n_frames_with_roi')}/{summary.get('n_frames_total')}   "
        f"ROI 소스: {summary.get('roi_source')}",
        f"캘리브레이션: {calib.get('accel_voltage_kv')} kV, L={calib.get('camera_length_cm')} cm, "
        f"scale={calib.get('scale_cm_px')} cm/px, 층두께={calib.get('layer_thickness_angstrom')} Å"
        + ("" if has_calib else "  [미설정 → Å 값 없음]"),
        f"평균 d={d_mean:.4f} Å" if d_mean is not None else "평균 d = (캘리브레이션 없음)",
        f"  평균 coherence={coh_mean:.4f} Å" if coh_mean is not None else "",
        f"성장률: f0={gr.get('f0')} Hz (ML/s)"
        + (f", {gr.get('growth_angstrom_s'):.4f} Å/s" if gr.get("growth_angstrom_s") is not None else "")
        if gr else "성장률: 계산 불가",
    ]
    fig.suptitle("VV2 — RHEED 정량분석 요약", fontsize=13, fontweight="bold", y=0.985)
    # family="monospace"였다면 DejaVu Sans Mono로 되돌아가 한글이 다시 깨짐 — rcParams의
    # 맑은고딕을 그대로 쓰기 위해 family 지정을 생략한다(숫자 정렬은 약간 흐트러지지만
    # 한글이 읽히는 게 우선).
    fig.text(0.02, 0.955, "\n".join(l for l in lines if l), fontsize=8.5, va="top")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
