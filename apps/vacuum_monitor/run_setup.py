# apps/vacuum_monitor/run_setup.py
import sys
import argparse
from pathlib import Path
import yaml

# --- 패키지 경로 인식용 ---
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# -------------------------

from apps.vacuum_monitor.wizard.run import run_wizard

def main():
    # exe 실행 시 경로 설정
    if getattr(sys, "frozen", False):
        base_path = Path(sys.executable).resolve().parent
    else:
        base_path = REPO_ROOT

    # 기본 설정 파일 경로 (exe 옆)
    config_path = base_path / "vacuum_monitor_config.yaml"

    print("="*50)
    print(" [ Vacuum Monitor Setup Wizard ]")
    print("="*50)

    # 1. 설정 파일이 아예 없으면 사용자가 요청한 커스텀 기본값으로 생성
    if not config_path.exists():
        print(f"[Info] Creating new configuration file: {config_path.name}")
        
        # === 사용자가 요청한 초기 세팅값 ===
        default_config = {
            "sampling": {
                "interval_sec": 1,
                "excel_export_sec": 600
            },
            "cameras": [
                {
                    "id": "top",
                    "input": "uvc",
                    "index": 0,
                    "crop": [87, 431, 724, 172]
                },
                {
                    "id": "bottom",
                    "input": "uvc",
                    "index": 0,
                    "crop": [73, 641, 889, 730]
                }
            ],
            "rois": {
                "top": [
                    {"name": "MG13_1_CH1", "rect": [283, 15, 171, 28]},
                    {"name": "MG13_1_CH2", "rect": [284, 35, 174, 32]},
                    {"name": "MG13_2_CH1", "rect": [39, 113, 168, 29]},
                    {"name": "MG13_2_CH2", "rect": [41, 138, 169, 29]},
                    {"name": "MG13_3_CH1", "rect": [553, 115, 167, 28]},
                    {"name": "MG13_3_CH2", "rect": [553, 142, 166, 26]}
                ],
                "bottom": [
                    {"name": "AN_HG", "rect": [242, 85, 124, 20]},
                    {"name": "AN_FG", "rect": [242, 128, 128, 23]},
                    {"name": "PP_HG", "rect": [683, 85, 123, 21]},
                    {"name": "PP_FG", "rect": [686, 124, 120, 26]},
                    {"name": "VV_FG1", "rect": [246, 359, 120, 25]},
                    {"name": "VV_FG2", "rect": [250, 401, 117, 24]},
                    {"name": "MN_HG", "rect": [685, 356, 119, 21]},
                    {"name": "MN_FG", "rect": [684, 395, 117, 23]},
                    {"name": "LL_HG", "rect": [250, 627, 122, 20]},
                    {"name": "LL_FG", "rect": [253, 668, 118, 23]},
                    {"name": "IS_HG", "rect": [669, 619, 109, 27]},
                    {"name": "IS_FG", "rect": [663, 656, 111, 26]}
                ]
            },
            "readout": {
                "engine": "tesseract",
                "tesseract_cmd": r"C:/Users/hanjunpy/AppData/Local/Programs/Tesseract-OCR/tesseract.exe",
                "psm": 7,
                "whitelist": "0123456789eE+-."
            },
            "alerts": {
                "threshold_log10": 1.0,
                "baseline_window_n": 60,
                "warmup_n": 20,
                "missing_k": 3,
                "skip_outliers": True,
                "cooldown_sec": 3600,
                "value_min": 1.0e-20,
                "value_max": 1000.0,
                "recovery_email": True,
                "email": {
                    "enabled": True,
                    "to": ["hanjun2217@skku.edu"],
                    "smtp_host": "smtp.gmail.com",
                    "smtp_port": 587,
                    "starttls": True,
                    "username_env": "VACUUM_SMTP_USER",
                    "password_env": "VACUUM_SMTP_PASS",
                    "subject_prefix": "[VACUUM ALERT]",
                    "subject_suffix": "vacuum monitor",
                    "debug": False
                }
            }
        }
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, sort_keys=False, allow_unicode=True)
    
    print(f"[Info] Loading config from: {config_path}")
    print("[Info] Opening Wizard UI... (Press '1','2','3' to switch modes)")

    try:
        # 템플릿과 저장 경로를 동일하게 주어 바로 수정되게 함
        run_wizard(template_path=config_path, out_path=config_path)
        print("\n[Success] Setup completed.")
        print("You can now run 'VacuumMonitor.exe' to start monitoring.")
    except Exception as e:
        print(f"\n[Error] Setup failed: {e}")
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()