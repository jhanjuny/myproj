# apps/VV2/worker.py
"""
VV2 배치 분석 워커 (QThread).

실시간 미리보기 없이, 선택된 동영상 파일 전체를 한 번에 순회하며
  1) ROI 추적 (spot_data.csv 궤적 재사용 또는 SpotDetector+EMA 자동추적)
  2) 스트립 프로파일 → 배경제거 → 3피크 검출/추적 → d(t), FWHM→coherence(t)
  3) ROI 강도 시계열 누적
  4) (선택적) 정규화 → FFT → 성장률
을 계산하고 CSV/JSON으로 저장한 뒤 zip으로 압축합니다.
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt5.QtCore import QThread, pyqtSignal

from apps.VV2.capture.file_source import FileVideoSource
from apps.VV2.detection.spot_detector import (
    SpotDetector, RoiTracker, roi_rect, roi_mean_intensity,
)
from apps.VV2.analysis.streak_profile import (
    strip_sum_profile, subtract_background, find_three_peaks,
    fwhm_of_peak, d_from_dx, StreakPeaks,
)
from apps.VV2.analysis.spectra import intensity_fft
from apps.VV2.analysis.report import render_summary_png
from apps.VV2.storage.export import (
    write_lattice_csv, write_fft_csv, write_summary_json, zip_analysis_outputs,
)


def load_spot_csv_roi_trajectory(csv_path: Path) -> List[Optional[Tuple[float, float, Optional[float]]]]:
    """
    apps/rheed_monitor의 spot_data.csv를 읽어 프레임 순서대로
    (roi_cx, roi_cy, roi_mean_intensity) 목록을 만듭니다.

    CSV는 프레임마다 timestamp가 채워진 첫 행 하나 + 스팟 수만큼의 후속 행으로
    구성되며(session.py record_spots 참조), roi_* 값은 그 첫 행에만 있습니다.
    두 파일(video.mp4, spot_data.csv)은 같은 루프에서 한 프레임당 한 번씩
    같이 기록되므로, "새 프레임" 행의 등장 순서 = 비디오 프레임 순서로 간주합니다.
    """
    rows: List[Optional[Tuple[float, float, Optional[float]]]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            if not row.get("timestamp"):
                continue  # 같은 프레임의 후속 스팟 행 — 건너뜀
            cx = row.get("roi_cx") or ""
            cy = row.get("roi_cy") or ""
            mean_i = row.get("roi_mean_intensity") or ""
            if cx and cy:
                rows.append((float(cx), float(cy), float(mean_i) if mean_i else None))
            else:
                rows.append(None)
    return rows


class BatchAnalysisWorker(QThread):
    progress = pyqtSignal(int, int)      # (current_frame, total_frames)
    finished_ok = pyqtSignal(dict)       # summary dict
    failed = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        out_dir: str,
        roi_size_px: int,
        track_alpha: float,
        detector_cfg: dict,
        bg_method: str,
        bg_params: dict,
        v_kev: float,
        l_cm: float,
        scale_cm_px: float,
        layer_thickness_angstrom: float,
        fft_detrend: str,
        fft_alpha: float,
        fft_normalize: str,
        frame_normalize: bool,
        manual_roi_center: Optional[Tuple[float, float]] = None,
        spot_csv_path: Optional[str] = None,
        use_spot_csv: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = Path(video_path)
        self.out_dir = Path(out_dir)
        self.roi_size_px = roi_size_px
        self.track_alpha = track_alpha
        self.detector_cfg = detector_cfg
        self.bg_method = bg_method
        self.bg_params = bg_params
        self.v_kev = v_kev
        self.l_cm = l_cm
        self.scale_cm_px = scale_cm_px
        self.layer_thickness_angstrom = layer_thickness_angstrom
        self.fft_detrend = fft_detrend
        self.fft_alpha = fft_alpha
        self.fft_normalize = fft_normalize
        self.frame_normalize = frame_normalize
        self.manual_roi_center = manual_roi_center
        self.spot_csv_path = Path(spot_csv_path) if spot_csv_path else None
        self.use_spot_csv = use_spot_csv

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001 — 백그라운드 스레드 예외를 GUI로 전달
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")

    def _run(self) -> None:
        source = FileVideoSource(self.video_path)
        fps = source.fps
        total = source.frame_count

        csv_traj: Optional[list] = None
        if self.use_spot_csv and self.spot_csv_path and self.spot_csv_path.exists():
            csv_traj = load_spot_csv_roi_trajectory(self.spot_csv_path)

        detector = SpotDetector(**self.detector_cfg)
        tracker = RoiTracker(alpha=self.track_alpha, initial=self.manual_roi_center)

        records = []
        intensities: List[float] = []
        timestamps: List[float] = []
        prev_peaks: Optional[StreakPeaks] = None
        # 리포트용 대표 프레임: 영상 중간 지점(또는 그 직전 마지막 유효 프레임) 스냅샷
        rep_target_idx = (total // 2) if total > 0 else 0
        rep_snapshot: Optional[dict] = None

        idx = 0
        while True:
            frame = source.read()
            if frame is None:
                break
            h, w = frame.shape[:2]
            t_sec = idx / fps if fps > 0 else float(idx)

            # 1) ROI 위치: CSV 궤적 우선, 없으면 SpotDetector+EMA 자동추적.
            #    CSV가 비디오보다 짧아 도중에 소진되면 자동추적으로 넘어가되,
            #    그 사실이 로그(roi_source 컬럼)에 남도록 조용히 섞이지 않게 함.
            center = None
            csv_intensity = None
            frame_roi_source = None
            if csv_traj is not None and idx < len(csv_traj) and csv_traj[idx] is not None:
                cx, cy, csv_intensity = csv_traj[idx]
                center = (cx, cy)
                frame_roi_source = "csv"
            else:
                spots = detector.detect(frame, frame_normalize=self.frame_normalize)
                center = tracker.update(spots)
                if center is not None:
                    frame_roi_source = "csv_exhausted_auto" if csv_traj is not None else "auto"

            record = {
                "frame": idx, "t_sec": round(t_sec, 4),
                "roi_source": frame_roi_source,
                "roi_cx": None, "roi_cy": None, "dx_px": None,
                "specular_x": None, "left_x": None, "right_x": None,
                "d_angstrom": None, "fwhm_px": None, "coherence_angstrom": None,
                "roi_mean_intensity": None,
            }

            if center is not None:
                rect = roi_rect(center, self.roi_size_px, (h, w))
                record["roi_cx"], record["roi_cy"] = round(center[0], 2), round(center[1], 2)

                # 2) 스트립 프로파일 → 배경제거 → 3피크 추적
                profile = strip_sum_profile(frame, rect)
                _baseline, subtracted = subtract_background(profile, self.bg_method, self.bg_params)
                peaks = find_three_peaks(subtracted, prev=prev_peaks)
                prev_peaks = peaks if peaks.specular_x is not None else prev_peaks

                dx_px = peaks.avg_dx
                record["dx_px"] = round(dx_px, 3) if dx_px is not None else None
                record["specular_x"] = peaks.specular_x
                record["left_x"] = peaks.left_x
                record["right_x"] = peaks.right_x
                record["d_angstrom"] = d_from_dx(dx_px, self.scale_cm_px, self.l_cm, self.v_kev)

                fwhm = fwhm_of_peak(subtracted, peaks.specular_x)
                record["fwhm_px"] = round(fwhm, 3) if fwhm is not None else None
                record["coherence_angstrom"] = d_from_dx(fwhm, self.scale_cm_px, self.l_cm, self.v_kev)

                if rep_snapshot is None or idx <= rep_target_idx:
                    rep_snapshot = {
                        "frame": frame.copy(),
                        "rect": rect,
                        "peaks": {
                            "specular_x": peaks.specular_x,
                            "left_x": peaks.left_x,
                            "right_x": peaks.right_x,
                        },
                    }

                # 3) 강도 시계열 (CSV에 이미 있으면 재사용, 없으면 직접 계산)
                mean_i = csv_intensity if csv_intensity is not None else roi_mean_intensity(frame, rect)
                record["roi_mean_intensity"] = round(mean_i, 3)
                intensities.append(mean_i)
                timestamps.append(t_sec)

            records.append(record)
            idx += 1
            if idx % 10 == 0 or idx == total:
                self.progress.emit(idx, total or idx)

        source.release()

        # 4) FFT 성장률
        fft_result = None
        if len(intensities) >= 8:
            try:
                fft_result = intensity_fft(
                    intensities, timestamps,
                    detrend=self.fft_detrend, alpha=self.fft_alpha,
                    normalize=self.fft_normalize,
                    layer_thickness_angstrom=self.layer_thickness_angstrom,
                )
            except ValueError:
                fft_result = None

        # ── 결과 저장 ──
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stem = self.video_path.stem
        lattice_csv = self.out_dir / f"{stem}_lattice_data.csv"
        write_lattice_csv(lattice_csv, records)

        out_files = [lattice_csv]
        if fft_result is not None:
            fft_csv = self.out_dir / f"{stem}_fft_spectrum.csv"
            write_fft_csv(fft_csv, fft_result["freq"], fft_result["power"])
            out_files.append(fft_csv)

        d_vals = [r["d_angstrom"] for r in records if r["d_angstrom"] is not None]
        coh_vals = [r["coherence_angstrom"] for r in records if r["coherence_angstrom"] is not None]

        n_csv = sum(1 for r in records if r["roi_source"] == "csv")
        n_auto_fallback = sum(1 for r in records if r["roi_source"] == "csv_exhausted_auto")
        n_auto = sum(1 for r in records if r["roi_source"] == "auto")
        if csv_traj is None:
            roi_source_summary = "auto-track(SpotDetector+EMA)"
        elif n_auto_fallback == 0:
            roi_source_summary = "spot_data.csv"
        else:
            roi_source_summary = (
                f"spot_data.csv({n_csv} frames) + auto-track fallback({n_auto_fallback} frames) "
                f"— csv had only {len(csv_traj)} rows for {total} video frames"
            )

        summary = {
            "video": str(self.video_path),
            "n_frames_total": total,
            "n_frames_with_roi": len(intensities),
            "n_frames_csv_tracked": n_csv,
            "n_frames_auto_tracked": n_auto + n_auto_fallback,
            "roi_source": roi_source_summary,
            "calibration": {
                "accel_voltage_kv": self.v_kev,
                "camera_length_cm": self.l_cm,
                "scale_cm_px": self.scale_cm_px,
                "layer_thickness_angstrom": self.layer_thickness_angstrom,
            },
            "background_method": self.bg_method,
            "d_angstrom_mean": (sum(d_vals) / len(d_vals)) if d_vals else None,
            "d_angstrom_n": len(d_vals),
            "coherence_angstrom_mean": (sum(coh_vals) / len(coh_vals)) if coh_vals else None,
            "growth_rate": fft_result,
        }
        # 5) 사람이 바로 보는 PNG 요약 리포트 — 실패해도 CSV/JSON은 이미 확보했으니
        #    분석 전체를 실패시키지 않고 summary에 사유만 남긴다.
        try:
            report_png = self.out_dir / f"{stem}_summary.png"
            render_summary_png(
                report_png, records, fft_result, summary,
                representative_frame=rep_snapshot["frame"] if rep_snapshot else None,
                representative_rect=rep_snapshot["rect"] if rep_snapshot else None,
                representative_peaks=rep_snapshot["peaks"] if rep_snapshot else None,
            )
            out_files.append(report_png)
            summary["report_png"] = str(report_png)
        except Exception as exc:  # noqa: BLE001
            summary["report_png_error"] = str(exc)

        summary_json = self.out_dir / f"{stem}_summary.json"
        write_summary_json(summary_json, summary)
        out_files.append(summary_json)

        zip_path = zip_analysis_outputs(self.out_dir, stem, out_files)
        summary["zip_path"] = str(zip_path)
        self.finished_ok.emit(summary)
