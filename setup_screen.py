# setup_screen.py
import cv2
import numpy as np
import yaml
import mss
import sys
from pathlib import Path

# 설정 파일 경로
CONFIG_PATH = Path("vacuum_monitor_config.yaml")

def load_config():
    if not CONFIG_PATH.exists():
        print("설정 파일이 없습니다. vacuum_monitor_config.yaml을 확인하세요.")
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
    print("\n[저장 완료] 설정이 파일에 저장되었습니다!")

# 전역 변수
drawing = False
ix, iy = -1, -1
rects = []
mode = "crop" # 'crop' or 'roi'
current_crop = None
sct = mss.mss()
monitor = sct.monitors[1] # 기본 모니터 1번

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, rects, current_crop, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = param.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Screen Setup", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        w, h = abs(x - ix), abs(y - iy)
        rx, ry = min(ix, x), min(iy, y)
        
        if w < 5 or h < 5: return # 너무 작은 박스 무시

        if mode == "crop":
            current_crop = [rx, ry, w, h]
            print(f"[CROP 설정됨] {current_crop}")
        elif mode == "roi":
            # ROI 이름 자동 생성
            roi_name = f"roi_{len(rects)+1}"
            new_roi = {"name": roi_name, "rect": [rx, ry, w, h]}
            rects.append(new_roi)
            print(f"[ROI 추가됨] {roi_name}: {new_roi['rect']}")
        
        # 화면 갱신을 위해 메인 루프에서 다시 그림

def main():
    global rects, current_crop, mode
    
    cfg = load_config()
    
    # 기존 설정 불러오기
    if "cameras" in cfg and len(cfg["cameras"]) > 0:
        current_crop = cfg["cameras"][0].get("crop", None)
    
    # 기존 ROI 불러오기 (top 카메라 기준)
    if "rois" in cfg and "top" in cfg["rois"]:
        rects = cfg["rois"]["top"]
        if rects is None: rects = []

    cv2.namedWindow("Screen Setup")
    cv2.setMouseCallback("Screen Setup", mouse_callback)

    print("="*50)
    print(" [ 화면 캡처 설정 모드 ]")
    print(" 1: 전체 영역 자르기 (Crop) 모드")
    print(" 2: 숫자 영역 박스 (ROI) 추가 모드")
    print(" R: 마지막 ROI 지우기")
    print(" S: 저장 (Save)")
    print(" Q: 종료 (Quit)")
    print("="*50)

    while True:
        # 1. 화면 캡처
        img_grab = sct.grab(monitor)
        frame = np.array(img_grab)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # 시각화용 이미지
        disp = frame.copy()

        # 2. Crop 영역 그리기 (파란색)
        if current_crop:
            cx, cy, cw, ch = current_crop
            cv2.rectangle(disp, (cx, cy), (cx+cw, cy+ch), (255, 0, 0), 2)
            cv2.putText(disp, "CROP AREA", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 3. ROI 영역 그리기 (초록색)
        for r in rects:
            rx, ry, rw, rh = r["rect"]
            # 만약 Crop이 되어 있다면 좌표가 Crop 내부 기준일 수 있음 (복잡함 방지를 위해 절대 좌표로 표시)
            # 여기서는 편의상 화면 전체 기준 절대좌표로 저장/표시합니다.
            cv2.rectangle(disp, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
            cv2.putText(disp, r["name"], (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 상태 표시
        status = f"MODE: {mode.upper()} | ROIs: {len(rects)}"
        cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if not drawing: # 드래그 중이 아닐 때만 갱신 (깜빡임 방지)
            cv2.imshow("Screen Setup", disp)

        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            # 설정 저장
            if "cameras" in cfg and len(cfg["cameras"]) > 0:
                cfg["cameras"][0]["crop"] = current_crop
            
            # ROI 저장 (top 카메라에 몰아주기)
            if "rois" not in cfg: cfg["rois"] = {}
            cfg["rois"]["top"] = rects
            
            save_config(cfg)
            
        elif key == ord('1'):
            mode = "crop"
            print(">> 모드 변경: CROP (전체 영역 설정)")
        elif key == ord('2'):
            mode = "roi"
            print(">> 모드 변경: ROI (숫자 박스 추가)")
        elif key == ord('r'):
            if rects:
                removed = rects.pop()
                print(f"ROI 삭제됨: {removed['name']}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()