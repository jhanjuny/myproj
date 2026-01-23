# --- MUST be at the very top (before any "from apps..." imports) ---
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../apps/vacuum_monitor/main.py -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# -------------------------------------------------------------------

import argparse
import csv
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import yaml

from apps.vacuum_monitor.readout.ocr_tesseract import read_value_tesseract
from apps.vacuum_monitor.alerts.email_smtp import send_email_smtp


# (선택) 이메일 모듈이 프로젝트에 이미 있다면 이 import 경로를 맞추세요.
# 없으면 아래 fallback이 동작하며 email_enabled=True여도 실제 발송은 안 됩니다.
try:
    from apps.vacuum_monitor.alerts.email_smtp import send_email_smtp  # type: ignore
except Exception:
    def send_email_smtp(cfg: dict, msg: str) -> None:
        raise RuntimeError(
            "send_email_smtp()를 import하지 못했습니다. "
            "apps/vacuum_monitor/alerts/email_smtp.py의 위치/함수명을 확인하세요."
        )


def clamp_rect(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


def crop_frame(frame: np.ndarray, rect) -> np.ndarray:
    if rect is None:
        return frame
    x, y, w, h = rect
    H, W = frame.shape[:2]
    x, y, w, h = clamp_rect(x, y, w, h, W, H)
    return frame[y : y + h, x : x + w].copy()


def draw_rois(frame: np.ndarray, rois, color=(0, 255, 0)) -> np.ndarray:
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
    """
    - 절대경로면 그대로
    - "data/xxx.mp4" 같이 오면 apps/vacuum_monitor 기준
    - "apps/vacuum_monitor/data/xxx.mp4" 같이 이미 앱 경로를 포함하면 repo 루트 기준
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp

    repo_root = app_dir.parent.parent
    parts = [x.lower() for x in pp.parts]
    if len(parts) >= 2 and parts[0] == "apps" and parts[1] == app_dir.name.lower():
        return (repo_root / pp).resolve()

    return (app_dir / pp).resolve()


class LoopingVideoCapture:
    def __init__(self, source):
        self.source = source

        # int(카메라 인덱스)인 경우 str로 바꾸면 깨질 수 있으므로 분기
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source)
        else:
            self.cap = cv2.VideoCapture(str(source))

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            # 파일이면 처음으로 되감기
            if isinstance(self.source, (str, Path)):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()

        if not ret:
            raise RuntimeError(f"Cannot read frame from source: {self.source}")

        return frame

    def release(self) -> None:
        self.cap.release()


def try_aruco_rectify(frame: np.ndarray, ar_cfg: Optional[dict]) -> np.ndarray:
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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class IssueInfo:
    kind: str  # "MISSING" | "ANOMALY"
    since_ts: float
    last_seen_ts: float
    last_value: Optional[float]
    baseline: Optional[float]
    ratio: Optional[float]
    text: str
    roi_path: str
    streak: int


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="apps/vacuum_monitor/config/config.yaml")
    ap.add_argument("--no_display", action="store_true")
    ap.add_argument("--dump_first_frames", action="store_true", help="Save first raw/cropped frames then exit")
    ap.add_argument("--max_samples", type=int, default=0, help="Stop after N sampling ticks (0 = no limit)")
    ap.add_argument("--duration_sec", type=float, default=0.0, help="Stop after N seconds (0 = no limit)")
    ap.add_argument("--save_roi_images", action="store_true", help="Save ROI PNGs under run_dir/rois (debug)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
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

    # alerts config
    alerts_cfg = cfg.get("alerts", {}) if isinstance(cfg, dict) else {}

    cooldown_sec = int(alerts_cfg.get("cooldown_sec", 3600))
    baseline_window_n = int(alerts_cfg.get("baseline_window_n", 60))
    warmup_n = int(alerts_cfg.get("warmup_n", 20))
    threshold_log10 = float(alerts_cfg.get("threshold_log10", 1.0))  # 1.0 => 10배
    skip_outliers = bool(alerts_cfg.get("skip_outliers", True))

    value_min = float(alerts_cfg.get("value_min", 1e-20))
    value_max = float(alerts_cfg.get("value_max", 1e3))

    # "원래 뜨다가 안 뜨면" 판단용 (연속 invalid)
    missing_k = int(alerts_cfg.get("missing_k", 3))

    email_cfg = (alerts_cfg.get("email", {}) or {}) if isinstance(alerts_cfg, dict) else {}
    email_enabled = bool(email_cfg.get("enabled", False))

    alert_path = run_dir / "alerts.log"
    alert_path.touch(exist_ok=True)

    # state per ROI
    baseline_logs: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=baseline_window_n))
    seen_valid = defaultdict(int)       # 유효값을 읽은 누적 횟수(= "원래 떴다" 판단)
    invalid_streak = defaultdict(int)   # 연속 invalid(=None/범위밖) 횟수
    active_issues: Dict[Tuple[str, str], IssueInfo] = {}

    last_summary_sent = 0.0  # 문제 상태 지속 시 1시간 간격 재발송 제어용

    def log_and_maybe_email(msg: str) -> None:
        with alert_path.open("a", encoding="utf-8") as af:
            af.write(msg + "\n")
        if email_enabled:
            send_email_smtp(cfg, msg)

    # 캡처를 소스별로 1번만 열기 (동일 source면 top/bottom이 같은 프레임을 공유)
    caps: Dict[str, LoopingVideoCapture] = {}
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
            src = int(source) if str(source).isdigit() else source
            cap_key = f"cam::{str(src)}"
            if cap_key not in caps:
                caps[cap_key] = LoopingVideoCapture(src)

        crop_rect = cam.get("crop", None)
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
        start_t = time.time()
        sample_count = 0

        while True:
            frames_by_key = {k: cap.read() for k, cap in caps.items()}

            now = time.time()
            do_sample = (now - last_sample_t) >= interval_sec
            do_export = (now - last_export_t) >= excel_export_sec

            if do_sample:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 이번 tick에서 “현재 문제”로 판정된 키 집합(정상화 판단용)
                current_bad = set()

                # CSV append는 tick마다 열어서 안정성 확보
                with csv_path.open("a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)

                    for spec in cam_specs:
                        raw = frames_by_key[spec["cap_key"]]
                        view = crop_frame(raw, spec["crop"])
                        view = try_aruco_rectify(view, spec["aruco"])

                        cam_roi_dir = run_dir / "rois" / spec["id"]
                        ensure_dir(cam_roi_dir)

                        for r in spec["rois"] or []:
                            name = r.get("name", "roi")
                            rect = r.get("rect", [0, 0, 1, 1])
                            key = (spec["id"], name)

                            roi_img = crop_frame(view, rect)

                            roi_path_str = ""
                            if args.save_roi_images:
                                roi_path = cam_roi_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.png"
                                cv2.imwrite(str(roi_path), roi_img)
                                roi_path_str = str(roi_path)

                            res = read_value_tesseract(roi_img, cfg)
                            val = res.value

                            # (항상) CSV 기록: None이면 value 칸은 빈칸
                            val_str = "" if (val is None) else f"{val:.6g}"
                            w.writerow([ts, spec["id"], name, val_str, roi_path_str])

                            # ---- 유효성 판정 ----
                            is_valid = (
                                (val is not None)
                                and (val > 0)
                                and (val >= value_min)
                                and (val <= value_max)
                            )

                            if not is_valid:
                                invalid_streak[key] += 1

                                # "아예 못읽는 값 무시" 정책:
                                #  - 이전에 정상 운용(=baseline warmup 이상)을 경험한 ROI만 missing 장애 대상으로 본다.
                                had_good = (len(baseline_logs[key]) >= warmup_n)

                                if had_good and invalid_streak[key] >= missing_k:
                                    current_bad.add(key)
                                    prev = active_issues.get(key)
                                    if prev is None or prev.kind != "MISSING":
                                        active_issues[key] = IssueInfo(
                                            kind="MISSING",
                                            since_ts=now,
                                            last_seen_ts=now,
                                            last_value=None,
                                            baseline=None,
                                            ratio=None,
                                            text=res.text or "",
                                            roi_path=roi_path_str,
                                            streak=invalid_streak[key],
                                        )
                                    else:
                                        prev.last_seen_ts = now
                                        prev.text = res.text or prev.text
                                        prev.roi_path = roi_path_str or prev.roi_path
                                        prev.streak = invalid_streak[key]

                                # invalid이면 baseline/이상탐지 없음
                                continue

                            # valid인 경우
                            invalid_streak[key] = 0
                            seen_valid[key] += 1

                            dq = baseline_logs[key]
                            baseline_log = None
                            if len(dq) >= warmup_n:
                                baseline_log = sum(dq) / len(dq)

                            # ---- 이상(10배) 판정: baseline이 준비된 뒤만 ----
                            is_anomaly = False
                            baseline_val = None
                            ratio = None
                            if baseline_log is not None:
                                logv = math.log10(val)
                                diff_log10 = abs(logv - baseline_log)
                                if diff_log10 >= threshold_log10:  # 1.0 => 10배
                                    is_anomaly = True
                                    baseline_val = 10 ** baseline_log
                                    ratio = (val / baseline_val) if baseline_val > 0 else float("inf")

                            if is_anomaly:
                                current_bad.add(key)
                                prev = active_issues.get(key)
                                if prev is None or prev.kind != "ANOMALY":
                                    active_issues[key] = IssueInfo(
                                        kind="ANOMALY",
                                        since_ts=now,
                                        last_seen_ts=now,
                                        last_value=val,
                                        baseline=baseline_val,
                                        ratio=ratio,
                                        text=res.text or "",
                                        roi_path=roi_path_str,
                                        streak=0,
                                    )
                                else:
                                    prev.last_seen_ts = now
                                    prev.last_value = val
                                    prev.baseline = baseline_val
                                    prev.ratio = ratio
                                    prev.text = res.text or prev.text
                                    prev.roi_path = roi_path_str or prev.roi_path

                                # outlier로 baseline 오염 방지
                                if skip_outliers:
                                    continue

                            # ---- baseline 업데이트(정상 값만, 딱 1회) ----
                            dq.append(math.log10(val))

                # ---- 정상화 처리: 이번 tick에 bad로 안 잡힌 active issue는 제거 ----
                for k in list(active_issues.keys()):
                    if k not in current_bad:
                        # 정상화되면 알람 중지
                        del active_issues[k]

                # ---- 요약 알림(메일/로그): 문제 지속 시 cooldown마다 반복 ----
                if active_issues:
                    # 첫 발생은 즉시, 이후는 cooldown_sec 간격
                    if (last_summary_sent == 0.0) or ((now - last_summary_sent) >= cooldown_sec):
                        lines = [f"{ts} VACUUM ALERT active={len(active_issues)} cooldown={cooldown_sec}s"]
                        for (cam_id, roi_name) in sorted(active_issues.keys()):
                            info = active_issues[(cam_id, roi_name)]
                            if info.kind == "MISSING":
                                lines.append(
                                    f"- MISSING camera={cam_id} roi={roi_name} "
                                    f"streak={info.streak} text={info.text!r} roi_path={info.roi_path}"
                                )
                            else:
                                v = info.last_value
                                b = info.baseline
                                r = info.ratio
                                lines.append(
                                    f"- ANOMALY camera={cam_id} roi={roi_name} "
                                    f"value={v:.6g} baseline={b:.6g} ratio={r:.3g} "
                                    f"text={info.text!r} roi_path={info.roi_path}"
                                )

                        msg = "\n".join(lines)
                        log_and_maybe_email(msg)
                        last_summary_sent = now
                else:
                    # 전부 정상화되면 타이머 리셋(다음 장애는 즉시 발송)
                    last_summary_sent = 0.0

                last_sample_t = now
                sample_count += 1

                # 종료 조건: 샘플 횟수
                if args.max_samples > 0 and sample_count >= args.max_samples:
                    break

            # 디스플레이
            if not args.no_display:
                for spec in cam_specs:
                    raw = frames_by_key[spec["cap_key"]]
                    view = crop_frame(raw, spec["crop"])
                    view = try_aruco_rectify(view, spec["aruco"])
                    disp = draw_rois(view, spec["rois"])
                    cv2.imshow(f"vacuum_monitor::{spec['id']}", disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            else:
                time.sleep(0.01)

            # 종료 조건: 전체 실행 시간
            if args.duration_sec > 0 and (time.time() - start_t) >= args.duration_sec:
                break

            # export tick
            if do_export:
                last_export_t = now

    finally:
        for cap in caps.values():
            cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print(f"[done] logs/rois saved under: {run_dir}")


if __name__ == "__main__":
    main()
