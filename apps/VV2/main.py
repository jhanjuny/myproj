# apps/VV2/main.py
"""
VV2 실행 진입점.

    D:\\conda_envs\\torch\\python.exe -m apps.VV2.main

apps/rheed_monitor와 완전히 독립된 오프라인 정량분석 도구입니다 (실시간
카메라/미리보기 없음). 이미 녹화된 동영상 파일을 열어 정량분석을 실행합니다.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml


def _run_selftest(out_dir: str) -> None:
    """
    `VV2.exe --selftest <out_dir>` — GUI 없이 합성 영상으로 전체 배치
    파이프라인(ROI 추적 → 정량분석 → matplotlib PNG 리포트 → zip)이 이
    frozen exe 안에서 끝까지 정상 동작하는지 검증하는 내부 진단 모드.
    "창이 뜬다"만으로는 실제 크래시(예: PIL 누락)를 못 잡은 전례가 있어
    도입했다 — main_window/QApplication은 아예 만들지 않고 워커만 동기 실행.
    """
    import numpy as np
    import cv2
    from PyQt5.QtCore import QCoreApplication
    from apps.VV2.worker import BatchAnalysisWorker

    QCoreApplication(sys.argv)

    tmp_dir = Path(out_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    video_path = tmp_dir / "selftest.mp4"

    H, W, FPS, N = 300, 400, 20.0, 60
    yy, xx = np.mgrid[0:H, 0:W]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        osc = 0.5 + 0.5 * np.sin(2 * np.pi * i / 15.0)
        for cx, cy, sigma, amp in [(200, 150, 5.0, 80 + 120 * osc), (120, 150, 6.0, 40), (280, 150, 6.0, 40)]:
            g = amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
            frame[:, :, 1] = np.clip(frame[:, :, 1].astype(np.float32) + g, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()

    worker = BatchAnalysisWorker(
        video_path=str(video_path), out_dir=str(tmp_dir),
        roi_size_px=110, track_alpha=0.3,
        detector_cfg=dict(threshold_fraction=0.5, min_brightness=20.0, min_area=4,
                           max_area=0, broad_area_threshold=500, max_spots=10, blur_ksize=5),
        bg_method="als", bg_params={},
        v_kev=20.0, l_cm=10.5, scale_cm_px=0.007358, layer_thickness_angstrom=3.905,
        fft_detrend="ema", fft_alpha=0.15, fft_normalize="max", frame_normalize=False,
        manual_roi_center=(200.0, 150.0), spot_csv_path=None, use_spot_csv=False,
    )
    result = {}
    worker.finished_ok.connect(lambda s: result.setdefault("summary", s))
    worker.failed.connect(lambda m: result.setdefault("error", m))
    worker.run()

    if "error" in result:
        print(f"[SELFTEST FAILED]\n{result['error']}", file=sys.stderr)
        sys.exit(1)

    summary = result["summary"]
    print(f"[SELFTEST OK] report_png={summary.get('report_png')} "
          f"report_png_error={summary.get('report_png_error')} zip={summary.get('zip_path')}")
    if summary.get("report_png_error") or not summary.get("report_png"):
        print("[SELFTEST FAILED] report_png 생성 실패", file=sys.stderr)
        sys.exit(1)
    if not Path(summary["report_png"]).exists():
        print("[SELFTEST FAILED] report_png 파일이 실제로 없음", file=sys.stderr)
        sys.exit(1)
    print("[SELFTEST PASSED]")
    sys.exit(0)


if "--selftest" in sys.argv:
    idx = sys.argv.index("--selftest")
    _out = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else str(Path(sys.executable).parent / "_selftest_out")
    _run_selftest(_out)

from PyQt5.QtWidgets import QApplication
from apps.VV2.gui.main_window import MainWindow


def _resolve_config() -> Path:
    """
    EXE로 실행 시: PyInstaller onefile 엔트리스크립트는 패키지 경로가 보존되지
    않고 MEIPASS 루트에 풀리므로 __file__ 기준 상대경로가 깨진다 — apps/rheed_monitor
    /main.py의 _resolve_config()와 동일한 패턴으로 sys._MEIPASS를 직접 사용한다.
    EXE 옆에 VV2_config.yaml이 있으면 그걸 우선(재빌드 없이 설정 수정 가능).
    """
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        runtime = exe_dir / "VV2_config.yaml"
        if runtime.exists():
            return runtime
        return Path(sys._MEIPASS) / "apps/VV2/config.yaml"
    return REPO_ROOT / "apps/VV2/config.yaml"


def main() -> None:
    cfg_path = _resolve_config()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(cfg=cfg)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
