from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from apps.vacuum_monitor.wizard.config_io import load_yaml, save_yaml
from apps.vacuum_monitor.wizard.ui_cv2 import SetupWizardCV2


def _resolve_app_relative(app_dir: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp

    repo_root = app_dir.parent.parent
    parts = [x.lower() for x in pp.parts]
    if len(parts) >= 2 and parts[0] == "apps" and parts[1] == app_dir.name.lower():
        return (repo_root / pp).resolve()

    return (app_dir / pp).resolve()


def run_wizard(template_path: Path, out_path: Path) -> dict:
    """
    template_path를 기본값으로 읽고, UI 설정(크롭/ROI/메일)을 반영해 out_path에 저장.
    반환: 최종 cfg(dict)
    """
    template = load_yaml(template_path)

    # 최소 스키마 보장
    template.setdefault("sampling", {"interval_sec": 1, "excel_export_sec": 600})
    template.setdefault("cameras", [])
    template.setdefault("rois", {})
    template.setdefault("alerts", {})
    
    # 이메일 기본값 세팅 (UI에 전달하기 위해)
    alerts = template.get("alerts", {})
    email = alerts.setdefault("email", {})
    email.setdefault("to", [])
    email.setdefault("enabled", False)
    
    # 기본 SMTP 설정들 (자동 세팅)
    email.setdefault("from", "your_email@gmail.com")
    email.setdefault("subject_prefix", "[VACUUM ALERT]")
    email.setdefault("smtp_host", "smtp.gmail.com")
    email.setdefault("smtp_port", 587)
    email.setdefault("starttls", True)
    email.setdefault("username_env", "VACUUM_SMTP_USER")
    email.setdefault("password_env", "VACUUM_SMTP_PASS")
    alerts.setdefault("recovery_email", True)

    cameras = template.get("cameras", [])
    rois_cfg = template.get("rois", {})

    # 소스 열기
    from apps.vacuum_monitor.capture.sources import FileSource, UvcSource, HikrobotMvsSource

    app_dir = Path(__file__).resolve().parents[1]  # .../apps/vacuum_monitor
    caps: Dict[str, object] = {}
    cam_specs: List[dict] = []

    for cam in cameras:
        cam_id = cam["id"]
        input_type = str(cam.get("input", "file")).lower()

        if input_type == "file":
            source = cam.get("source", "")
            src_path = _resolve_app_relative(app_dir, source)
            cap_key = f"file::{str(src_path)}"
            if cap_key not in caps:
                caps[cap_key] = FileSource(src_path, loop=True)

        elif input_type in ("uvc", "webcam"):
            index = int(cam.get("index", 0))
            cap_key = f"uvc::{index}"
            if cap_key not in caps:
                caps[cap_key] = UvcSource(index=index)

        elif input_type in ("mvs", "hikrobot", "hik"):
            cap_key = f"mvs::{cam.get('device', 'default')}"
            if cap_key not in caps:
                caps[cap_key] = HikrobotMvsSource(cam)

        else:
            raise ValueError(f"Unknown camera input type: {input_type}")

        cam_specs.append(
            {
                "id": cam_id,
                "cap_key": cap_key,
                "crop": cam.get("crop", None),
                "aruco": cam.get("aruco", None),
                "rois": rois_cfg.get(cam_id, []),
            }
        )

    try:
        # 1) Wizard UI 실행 (이메일 설정도 여기서 함)
        ui = SetupWizardCV2(caps, cam_specs, email_cfg=email)
        updated_specs, ok = ui.run()
        
        if not ok:
            print("[Cancel] Wizard cancelled. No changes saved.")
            return template

        # 2) 반영: crop은 cameras에, rois는 rois 맵에 반영
        new_rois = {}
        for s in updated_specs:
            cam_id = s["id"]
            new_rois[cam_id] = s.get("rois", [])

            # cameras 리스트의 해당 cam crop 업데이트
            for cam in cameras:
                if cam.get("id") == cam_id:
                    cam["crop"] = s.get("crop", None)

        template["rois"] = new_rois
        
        # UI에서 수정된 이메일 설정을 템플릿에 반영
        template["alerts"]["email"] = ui.email_cfg

        # 3) 저장
        save_yaml(out_path, template)
        print(f"[Saved] Configuration saved to {out_path.resolve()}")
        
        # 안내 메시지
        print("\n[INFO] Email settings saved.")
        print(f"  Recipients: {ui.email_cfg.get('to')}")
        print("  Don't forget to set environment variables for SMTP auth:")
        print(f"  setx {ui.email_cfg['username_env']} \"your_email@gmail.com\"")
        print(f"  setx {ui.email_cfg['password_env']} \"your_app_password\"")
        
        return template

    finally:
        for cap in caps.values():
            try:
                cap.release()
            except Exception:
                pass


def run_wizard_cli() -> None:
    """
    __main__.py에서 호출하는 진입점 함수.
    이 함수가 반드시 파일 최상위 레벨(들여쓰기 없음)에 정의되어 있어야 합니다.
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="apps/vacuum_monitor/config/config.yaml")
    ap.add_argument("--out", default="vacuum_monitor_config.yaml")
    args = ap.parse_args()

    template_path = Path(args.template).resolve()
    out_path = Path(args.out).resolve()

    run_wizard(template_path, out_path)