# apps/VV1/storage/export.py
"""배치 분석 결과 → CSV/JSON 기록 + zip 압축."""

from __future__ import annotations

import csv
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict


def write_lattice_csv(path: Path, records: List[Dict]) -> None:
    """
    컬럼: frame, t_sec, roi_cx, roi_cy, dx_px, specular_x, left_x, right_x,
          d_angstrom, fwhm_px, coherence_angstrom, roi_mean_intensity
    """
    fieldnames = [
        "frame", "t_sec", "roi_source", "roi_cx", "roi_cy", "dx_px",
        "specular_x", "left_x", "right_x",
        "d_angstrom", "fwhm_px", "coherence_angstrom", "roi_mean_intensity",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})


def write_fft_csv(path: Path, freq: List[float], power: List[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["freq_hz", "power"])
        for fr, p in zip(freq, power):
            w.writerow([fr, p])


def write_summary_json(path: Path, summary: Dict) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def zip_analysis_outputs(out_dir: Path, video_stem: str, files: List[Path]) -> Path:
    """out_dir에 <video_stem>_VV1_analysis_<timestamp>.zip 생성 후 경로 반환."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = out_dir / f"{video_stem}_VV1_analysis_{ts}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in files:
            if fp.exists():
                zf.write(fp, arcname=fp.name)
    return zip_path
