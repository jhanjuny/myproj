# apps/vacuum_monitor/main.py

# --- MUST be at the very top (before any "from apps..." imports) ---
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# -------------------------------------------------------------------

import argparse
import csv
import math
import shutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import yaml

from apps.vacuum_monitor.readout.ocr_tesseract import read_value_tesseract
# [수정됨] ScreenSource 추가
from apps.vacuum_monitor.capture.sources import FileSource, UvcSource, HikrobotMvsSource, ScreenSource


try:
    from apps.vacuum_monitor.alerts.email_smtp import send_email_smtp
except Exception:
    def send_email_smtp(cfg: dict, msg: str) -> None:
        pass


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


def resolve_app_relative(app_dir: Path, p: str):
    # [수정됨] 웹 주소(rtsp, http)인 경우 경로 변환 없이 그대로 문자열로 반환
    if "://" in str(p):
        return str(p)
        
    pp = Path(p)
    if pp.is_absolute():
        return pp
    repo_root = app_dir.parent.parent
    parts = [x.lower() for x in pp.parts]
    if len(parts) >= 2 and parts[0] == "apps" and parts[1] == app_dir.name.lower():
        return (repo_root / pp).resolve()
    return (app_dir / pp).resolve()


def try_aruco_rectify(frame: np.ndarray, ar_cfg: Optional[dict]) -> np.ndarray:
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
        pts = c[0]
        center = pts.mean(axis=0)
        id_to_center[int(i)] = center
    if not all(int(i) in id_to_center for i in ids_needed[:4]):
        return cv2.resize(frame, (W_out, H_out))
    src = np.array([id_to_center[int(i)] for i in ids_needed[:4]], dtype=np.float32)
    dst = np.array([[0, 0], [W_out - 1, 0], [W_out - 1, H_out - 1], [0, H_out - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (W_out, H_out))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _perform_cleanup(output_base: Path, current_run_dir: Path, retention_days: int) -> None:
    if retention_days <= 0:
        return
        
    cutoff_time = time.time() - (retention_days * 86400)
    print(f"[Cleanup] Running maintenance (Threshold: {retention_days} days)...")

    # 1. 과거 실행 폴더 청소
    if output_base.exists():
        for p in output_base.iterdir():
            if p.is_dir() and p.name.startswith("run_"):
                if p.resolve() == current_run_dir.resolve():
                    continue
                try:
                    if p.stat().st_mtime < cutoff_time:
                        shutil.rmtree(p)
                except Exception as e:
                    print(f"  [Error] Failed deleting folder {p.name}: {e}")

    # 2. 현재 실행 폴더 내부 파일 청소
    rois_base = current_run_dir / "rois"
    if rois_base.exists():
        for cam_dir in rois_base.iterdir():
            if cam_dir.is_dir():
                for img_file in cam_dir.glob("*.png"):
                    try:
                        if img_file.stat().st_mtime < cutoff_time:
                            img_file.unlink()
                    except Exception:
                        pass
    print("[Cleanup] Done.")


@dataclass
class IssueInfo:
    kind: str
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
    ap.add_argument("--dump_first_frames", action="store_true")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--duration_sec", type=float, default=0.0)
    ap.add_argument("--save_roi_images", action="store_true")
    ap.add_argument("--runtime_config", default="")
    ap.add_argument("--wizard", action="store_true")
    ap.add_argument("--retention_days", type=int, default=30, help="Log retention days")
    
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    app_dir = Path(__file__).resolve().parent

    def _default_runtime_config_path(repo_root: Path) -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent / "vacuum_monitor_config.yaml"
        return repo_root / "vacuum_monitor_config.yaml"

    runtime_path = Path(args.runtime_config).resolve() if args.runtime_config else _default_runtime_config_path(repo_root)
    
    if not runtime_path.exists():
        print("="*60)
        print(f"[Error] Configuration file not found!")
        print(f"Path: {runtime_path}")
        print("\nPlease run 'VacuumSetup.exe' first to create settings.")
        print("="*60)
        time.sleep(5)
        return

    if args.wizard:
        print("[Info] Use 'VacuumSetup.exe' to change settings.")
        time.sleep(3)
        return
    
    cfg = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))

    output_base = repo_root / "outputs" / "vacuum_monitor"
    ensure_dir(output_base)

    run_dir = output_base / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(run_dir)
    ensure_dir(run_dir / "frames")
    ensure_dir(run_dir / "rois")

    if args.retention_days > 0:
        _perform_cleanup(output_base, run_dir, args.retention_days)

    sampling = cfg.get("sampling", {})
    interval_sec = float(sampling.get("interval_sec", 1))
    excel_export_sec = float(sampling.get("excel_export_sec", 600))
    cameras = cfg.get("cameras", [])
    rois_cfg = cfg.get("rois", {})
    alerts_cfg = cfg.get("alerts", {}) if isinstance(cfg, dict) else {}

    csv_path = run_dir / "records.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "camera_id", "roi_name", "value", "roi_path"])

    cooldown_sec = int(alerts_cfg.get("cooldown_sec", 3600))
    baseline_window_n = int(alerts_cfg.get("baseline_window_n", 60))
    warmup_n = int(alerts_cfg.get("warmup_n", 20))
    threshold_log10 = float(alerts_cfg.get("threshold_log10", 1.0))
    skip_outliers = bool(alerts_cfg.get("skip_outliers", True))
    anomaly_k = int(alerts_cfg.get("anomaly_k", 2))
    value_min = float(alerts_cfg.get("value_min", 1e-20))
    value_max = float(alerts_cfg.get("value_max", 1e3))
    missing_k = int(alerts_cfg.get("missing_k", 3))
    
    email_cfg = (alerts_cfg.get("email", {}) or {}) if isinstance(alerts_cfg, dict) else {}
    email_enabled = bool(email_cfg.get("enabled", False))
    recovery_email = bool(alerts_cfg.get("recovery_email", True))

    alert_path = run_dir / "alerts.log"
    alert_path.touch(exist_ok=True)

    baseline_logs: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=baseline_window_n))
    seen_valid = defaultdict(int)
    invalid_streak = defaultdict(int)
    active_issues: Dict[Tuple[str, str], IssueInfo] = {}
    anomaly_streak = defaultdict(int)
    last_summary_sent = 0.0

    def log_and_maybe_email(msg: str):
        with alert_path.open("a", encoding="utf-8") as af:
            af.write(msg + "\n")
        if not email_enabled:
            return
        try:
            send_email_smtp(cfg, msg)
        except Exception as e:
            err = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EMAIL_ERROR {type(e).__name__}: {e}"
            with alert_path.open("a", encoding="utf-8") as af:
                af.write(err + "\n")

    caps = {}
    cam_specs = []

    for cam in cameras:
        cam_id = cam["id"]
        input_type = str(cam.get("input", "file")).lower()

        if input_type == "file":
            source = cam.get("source", "")
            src_path = resolve_app_relative(app_dir, source)
            cap_key = f"file::{str(src_path)}"
            if cap_key not in caps:
                caps[cap_key] = FileSource(src_path, loop=True)
        elif input_type in ("uvc", "webcam"):
            index = int(cam.get("index", 0))
            cap_key = f"uvc::{index}"
            if cap_key not in caps:
                caps[cap_key] = UvcSource(index=index)
        
        # [수정됨] 화면 캡처 모드 추가
        elif input_type == "screen":
            monitor_idx = int(cam.get("monitor", 1))
            cap_key = f"screen::{monitor_idx}"
            if cap_key not in caps:
                caps[cap_key] = ScreenSource(monitor_idx=monitor_idx)

        elif input_type in ("mvs", "hikrobot", "hik"):
            cap_key = f"mvs::{cam.get('device', 'default')}"
            if cap_key not in caps:
                caps[cap_key] = HikrobotMvsSource(cam)
        else:
            raise ValueError(f"Unknown camera input type: {input_type}")

        cam_specs.append({
            "id": cam_id,
            "cap_key": cap_key,
            "crop": cam.get("crop", None),
            "aruco": cam.get("aruco", None),
            "rois": rois_cfg.get(cam_id, []),
        })

    if args.dump_first_frames:
        for key, cap in caps.items():
            frame = cap.read()
            if frame is not None:
                out_path = run_dir / "frames" / f"first_frame_{key.replace(':', '_').replace(r'\\', '_').replace('/', '_')}.png"
                cv2.imwrite(str(out_path), frame)
        for cap in caps.values():
            cap.release()
        print(f"[dump] saved first frames under: {run_dir}")
        return

    last_sample_t = 0.0
    last_export_t = 0.0
    
    last_cleanup_t = time.time()
    cleanup_interval_sec = 3600

    print(f"[Start] Monitoring started. Output: {run_dir}")
    print(f"[Config] Retention: {args.retention_days} days (Check every 1h)")

    try:
        start_t = time.time()
        sample_count = 0

        while True:
            frames_by_key = {}
            for k, cap in caps.items():
                try:
                    frames_by_key[k] = cap.read()
                except Exception:
                    frames_by_key[k] = None

            now = time.time()
            do_sample = (now - last_sample_t) >= interval_sec
            do_export = (now - last_export_t) >= excel_export_sec
            
            if args.retention_days > 0 and (now - last_cleanup_t) >= cleanup_interval_sec:
                _perform_cleanup(output_base, run_dir, args.retention_days)
                last_cleanup_t = now

            if do_sample:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_bad = set()

                with csv_path.open("a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)

                    for spec in cam_specs:
                        raw = frames_by_key.get(spec["cap_key"])
                        if raw is None: continue

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
                                fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.png"
                                roi_path = cam_roi_dir / fname
                                cv2.imwrite(str(roi_path), roi_img)
                                roi_path_str = str(roi_path)

                            try:
                                res = read_value_tesseract(roi_img, cfg)
                                val = res.value
                            except Exception:
                                val = None
                                res = type('obj', (object,), {'text': ''})

                            val_str = "" if (val is None) else f"{val:.6g}"
                            w.writerow([ts, spec["id"], name, val_str, roi_path_str])

                            is_valid = (
                                (val is not None)
                                and (val > 0)
                                and (val >= value_min)
                                and (val <= value_max)
                            )

                            if not is_valid:
                                invalid_streak[key] += 1
                                anomaly_streak[key] = 0
                                had_good = (len(baseline_logs[key]) >= warmup_n)
                                if had_good and invalid_streak[key] >= missing_k:
                                    current_bad.add(key)
                                    prev = active_issues.get(key)
                                    if prev is None or prev.kind != "MISSING":
                                        active_issues[key] = IssueInfo(
                                            kind="MISSING", since_ts=now, last_seen_ts=now,
                                            last_value=None, baseline=None, ratio=None,
                                            text=getattr(res, 'text', ''), roi_path=roi_path_str,
                                            streak=invalid_streak[key]
                                        )
                                    else:
                                        prev.last_seen_ts = now
                                        prev.text = getattr(res, 'text', '') or prev.text
                                        prev.roi_path = roi_path_str or prev.roi_path
                                        prev.streak = invalid_streak[key]
                                continue

                            invalid_streak[key] = 0
                            seen_valid[key] += 1
                            dq = baseline_logs[key]
                            baseline_log = sum(dq)/len(dq) if len(dq) >= warmup_n else None

                            is_anomaly = False
                            baseline_val = None
                            ratio = None
                            if baseline_log is not None:
                                logv = math.log10(val)
                                if abs(logv - baseline_log) >= threshold_log10:
                                    is_anomaly = True
                                    baseline_val = 10 ** baseline_log
                                    ratio = (val / baseline_val) if baseline_val > 0 else float("inf")

                            if is_anomaly:
                                anomaly_streak[key] += 1
                            else:
                                anomaly_streak[key] = 0

                            if is_anomaly and anomaly_streak[key] >= anomaly_k:
                                current_bad.add(key)
                                prev = active_issues.get(key)
                                if prev is None or prev.kind != "ANOMALY":
                                    active_issues[key] = IssueInfo(
                                        kind="ANOMALY", since_ts=now, last_seen_ts=now,
                                        last_value=val, baseline=baseline_val, ratio=ratio,
                                        text=getattr(res, 'text', ''), roi_path=roi_path_str,
                                        streak=anomaly_streak[key]
                                    )
                                else:
                                    prev.last_seen_ts = now
                                    prev.last_value = val
                                    prev.baseline = baseline_val
                                    prev.ratio = ratio
                                    prev.text = getattr(res, 'text', '') or prev.text
                                    prev.roi_path = roi_path_str or prev.roi_path
                                    prev.streak = anomaly_streak[key]

                            if is_anomaly and skip_outliers:
                                continue

                            dq.append(math.log10(val))

                recovered = []
                for k in list(active_issues.keys()):
                    if k not in current_bad:
                        recovered.append((k, active_issues[k]))
                        del active_issues[k]

                if recovered and recovery_email:
                    lines = [f"{ts} VACUUM RECOVERED count={len(recovered)}"]
                    for (cam_id, roi_name), info in sorted(recovered, key=lambda x: x[0]):
                        dur_sec = max(0.0, now - float(info.since_ts))
                        if info.kind == "MISSING":
                            lines.append(f"- RECOVERED MISSING {cam_id}/{roi_name} duration={dur_sec:.0f}s")
                        else:
                            lines.append(f"- RECOVERED ANOMALY {cam_id}/{roi_name} duration={dur_sec:.0f}s value={info.last_value}")
                    log_and_maybe_email("\n".join(lines))

                if active_issues:
                    if (last_summary_sent == 0.0) or ((now - last_summary_sent) >= cooldown_sec):
                        lines = [f"{ts} VACUUM ALERT active={len(active_issues)}"]
                        for (cam_id, roi_name) in sorted(active_issues.keys()):
                            info = active_issues[(cam_id, roi_name)]
                            if info.kind == "MISSING":
                                lines.append(f"- MISSING {cam_id}/{roi_name} streak={info.streak} text={info.text!r}")
                            else:
                                lines.append(f"- ANOMALY {cam_id}/{roi_name} val={info.last_value} ratio={info.ratio:.1f}")
                        log_and_maybe_email("\n".join(lines))
                        last_summary_sent = now
                else:
                    last_summary_sent = 0.0

                last_sample_t = now
                sample_count += 1
                if args.max_samples > 0 and sample_count >= args.max_samples:
                    break

            if not args.no_display:
                for spec in cam_specs:
                    raw = frames_by_key.get(spec["cap_key"])
                    if raw is None: continue
                    view = crop_frame(raw, spec["crop"])
                    view = try_aruco_rectify(view, spec["aruco"])
                    disp = draw_rois(view, spec["rois"])
                    cv2.imshow(f"monitor::{spec['id']}", disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            else:
                time.sleep(0.01)

            if args.duration_sec > 0 and (time.time() - start_t) >= args.duration_sec:
                break
            
            if do_export:
                last_export_t = now

    finally:
        for cap in caps.values():
            try:
                cap.release()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print(f"[Done] Logs saved in: {run_dir}")

if __name__ == "__main__":
    main()