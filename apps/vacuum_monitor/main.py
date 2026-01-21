import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from apps.vacuum_monitor.readout.ocr_tesseract import read_value_tesseract



def clamp_rect(x, y, w, h, W, H):
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


def crop_frame(frame, rect):
    if rect is None:
        return frame
    x, y, w, h = rect
    H, W = frame.shape[:2]
    x, y, w, h = clamp_rect(x, y, w, h, W, H)
    return frame[y : y + h, x : x + w].copy()


def draw_rois(frame, rois, color=(0, 255, 0)):
    out = frame.copy()
    for r in rois or []:
        name = r.get("name", "roi")
        x, y, w, h = r.get("rect", [0, 0, 1, 1])
        H, W = out.shape[:2]
        x, y, w, h = clamp_rect(x, y, w, h, W, H)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, name, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def resolve_app_relative(app_dir: Path, p: str) -> Path:
    # config에 "data/xxx.mp4"처럼 들어오면 apps/vacuum_monitor 기준으로 해석
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (app_dir / pp).resolve()


class LoopingVideoCapture:
    def __init__(self, source):
        # source: 파일 경로(str) 또는 카메라 인덱스(int)
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            # 파일이면 처음으로 되감기
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Cannot read frame from source: {self.source}")
        return frame

    def release(self):
        self.cap.release()


def try_aruco_rectify(frame, ar_cfg):
    """
    ar_cfg 예:
      dict: DICT_4X4_50
      ids: [0,1,2,3]   (순서 가정: TL, TR, BR, BL)
      output_size: [1200, 400]  (W,H)
    실패하면 output_size로 resize만 수행(또는 그대로 반환).
    """
    if not ar_cfg:
        return frame

    out_size = ar_cfg.get("output_size", None)
    if out_size is None:
        return frame

    W_out, H_out = int(out_size[0]), int(out_size[1])

    aruco_mod = getattr(cv2, "aruco", None)
    if aruco_mod is None:
        return cv2.resize(frame, (W_out, H_out))

    dict_name = ar_cfg.get("dict", "DICT_4X4_50")
    ids_needed = ar_cfg.get("ids", [])
    if len(ids_needed) < 4:
        return cv2.resize(frame, (W_out, H_out))

    try:
        dictionary = aruco_mod.getPredefinedDictionary(getattr(aruco_mod, dict_name))
    except Exception:
        return cv2.resize(frame, (W_out, H_out))

    # OpenCV 버전 호환
    try:
        parameters = aruco_mod.DetectorParameters()
        detector = aruco_mod.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(frame)
    except Exception:
        parameters = aruco_mod.DetectorParameters_create()
        corners, ids, _ = aruco_mod.detectMarkers(frame, dictionary, parameters=parameters)

    if ids is None or len(ids) == 0:
        return cv2.resize(frame, (W_out, H_out))

    id_to_center = {}
    for c, i in zip(corners, ids.flatten()):
        pts = c[0]  # (4,2)
        center = pts.mean(axis=0)
        id_to_center[int(i)] = center

    if not all(int(i) in id_to_center for i in ids_needed[:4]):
        return cv2.resize(frame, (W_out, H_out))

    src = np.array([id_to_center[int(i)] for i in ids_needed[:4]], dtype=np.float32)
    dst = np.array([[0, 0], [W_out - 1, 0], [W_out - 1, H_out - 1], [0, H_out - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (W_out, H_out))
    return warped


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="apps/vacuum_monitor/config/config.yaml")
    ap.add_argument("--no_display", action="store_true")
    ap.add_argument("--dump_first_frames", action="store_true", help="Save first raw/cropped frames then exit")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]  # .../apps/vacuum_monitor/main.py -> repo root
    app_dir = Path(__file__).resolve().parent

    cfg_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    sampling = cfg.get("sampling", {})
    interval_sec = float(sampling.get("interval_sec", 1))
    excel_export_sec = float(sampling.get("excel_export_sec", 600))

    cameras = cfg.get("cameras", [])
    rois_cfg = cfg.get("rois", {})

    # outputs
    run_dir = repo_root / "outputs" / "vacuum_monitor" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(run_dir)
    ensure_dir(run_dir / "frames")
    ensure_dir(run_dir / "rois")

    # CSV log
    csv_path = run_dir / "records.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "camera_id", "roi_name", "value", "roi_path"])

    # 캡처를 소스별로 1번만 열기 (동일 source면 top/bottom이 같은 프레임을 공유)
    caps = {}  # key -> LoopingVideoCapture
    cam_specs = []
    for cam in cameras:
        cam_id = cam["id"]
        input_type = cam.get("input", "file")
        source = cam.get("source", "")

        if input_type == "file":
            src_path = resolve_app_relative(app_dir, source)
            cap_key = f"file::{str(src_path)}"
            if cap_key not in caps:
                caps[cap_key] = LoopingVideoCapture(str(src_path))
        else:
            # future: webcam/rtsp 등
            # 예: source가 숫자면 카메라 인덱스
            src = int(source) if str(source).isdigit() else source
            cap_key = f"cam::{str(src)}"
            if cap_key not in caps:
                caps[cap_key] = LoopingVideoCapture(src)

        crop_rect = cam.get("crop", None)  # [x,y,w,h] or None
        ar_cfg = cam.get("aruco", None)

        cam_specs.append(
            {
                "id": cam_id,
                "cap_key": cap_key,
                "crop": crop_rect,
                "aruco": ar_cfg,
                "rois": rois_cfg.get(cam_id, []),
            }
        )

    # 첫 프레임 덤프(좌표 튜닝용)
    if args.dump_first_frames:
        for key, cap in caps.items():
            frame = cap.read()
            out_path = run_dir / "frames" / f"first_frame_{key.replace(':', '_').replace('\\', '_').replace('/', '_')}.png"
            cv2.imwrite(str(out_path), frame)

        for spec in cam_specs:
            frame = caps[spec["cap_key"]].read()
            cropped = crop_frame(frame, spec["crop"])
            warped = try_aruco_rectify(cropped, spec["aruco"])
            out_path = run_dir / "frames" / f"first_{spec['id']}_processed.png"
            cv2.imwrite(str(out_path), warped)

        for cap in caps.values():
            cap.release()
        print(f"[dump] saved first frames under: {run_dir}")
        return

    last_sample_t = 0.0
    last_export_t = 0.0

    try:
        while True:
            # 소스별로 프레임 1회만 read
            frames_by_key = {k: cap.read() for k, cap in caps.items()}

            now = time.time()
            do_sample = (now - last_sample_t) >= interval_sec
            do_export = (now - last_export_t) >= excel_export_sec

            for spec in cam_specs:
                raw = frames_by_key[spec["cap_key"]]
                view = crop_frame(raw, spec["crop"])
                view = try_aruco_rectify(view, spec["aruco"])

                # ROI 샘플링(현재는 OCR 대신 ROI 이미지 저장 + value는 빈칸)
                if do_sample:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cam_roi_dir = run_dir / "rois" / spec["id"]
                    ensure_dir(cam_roi_dir)

                    with csv_path.open("a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        for r in spec["rois"] or []:
                            name = r.get("name", "roi")
                            rect = r.get("rect", [0, 0, 1, 1])
                            
                            
                            roi_img = crop_frame(view, rect)
                            roi_path = cam_roi_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.png"
                            cv2.imwrite(str(roi_path), roi_img)

                            res = read_value_tesseract(roi_img, cfg)  # cfg는 main()에서 읽은 전체 config dict
                            val_str = "" if (res.value is None) else f"{res.value:.6g}"

                            w.writerow([ts, spec["id"], name, val_str, str(roi_path)])


                # 디스플레이(ROI 박스 표시)
                if not args.no_display:
                    disp = draw_rois(view, spec["rois"])
                    cv2.imshow(f"vacuum_monitor::{spec['id']}", disp)

            if do_sample:
                last_sample_t = now

            # (선택) 엑셀 export는 일단 CSV 기반으로 운영하고, 필요하면 다음 단계에서 추가 구현 권장
            if do_export:
                last_export_t = now

            if not args.no_display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
        print(f"[done] logs/rois saved under: {run_dir}")


if __name__ == "__main__":
    main()
