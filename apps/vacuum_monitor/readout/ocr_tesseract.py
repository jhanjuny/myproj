# apps/vacuum_monitor/readout/ocr_tesseract.py
import sys
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# pytesseract import 시도 (없으면 None 처리)
try:
    import pytesseract
except ImportError:
    pytesseract = None


# --- [핵심] Tesseract 경로 설정 로직 ---
def setup_tesseract_path():
    """
    1. exe 실행 시: exe 옆의 'tesseract_portable/tesseract.exe' 우선 확인
    2. 없으면: 시스템 기본 경로(C:/Program Files/...) 확인
    """
    if pytesseract is None:
        return

    # 1. 기준 경로 설정 (Frozen=exe실행, 아니면 현재파일 기준)
    if getattr(sys, 'frozen', False):
        # exe가 있는 폴더
        base_path = Path(sys.executable).parent
    else:
        # 개발 중 (repo root 추정)
        base_path = Path(__file__).resolve().parents[3]

    # 2. Portable 폴더 우선 탐색
    portable_path = base_path / "tesseract_portable" / "tesseract.exe"
    
    if portable_path.exists():
        pytesseract.pytesseract.tesseract_cmd = str(portable_path)
    else:
        # 3. 없으면 시스템 설치 경로 시도 (Fallback)
        default_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
        ]
        for dp in default_paths:
            if os.path.exists(dp):
                pytesseract.pytesseract.tesseract_cmd = dp
                break

# 파일 로드 시 경로 설정 즉시 실행
setup_tesseract_path()
# ------------------------------------


@dataclass
class OcrResult:
    text: str
    value: Optional[float]
    conf: float


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """OCR 인식률을 높이기 위한 이미지 전처리"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 노이즈 제거 및 대비 향상
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 이진화 (Adaptive가 보통 게이지/LCD에 유리)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return binary


def parse_float(text: str) -> Optional[float]:
    """텍스트에서 숫자(과학적 표기법 포함) 추출"""
    # 1. 공백 제거 및 소문자 변환
    text = text.strip().lower().replace(" ", "")
    
    # 2. 숫자, 점, E, -, + 만 남기기
    cleaned = re.sub(r"[^0-9e\+\-\.]", "", text)
    
    # 3. 예외 케이스 처리 (끝에 점이 붙는 경우 등)
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]

    try:
        return float(cleaned)
    except ValueError:
        return None


def read_value_tesseract(roi_img: np.ndarray, cfg: dict = None) -> OcrResult:
    """
    이미지에서 숫자를 읽어 반환.
    cfg: 호환성을 위해 남겨둠 (내부적으로 사용 안 함)
    """
    # pytesseract가 없거나 이미지가 비었으면 빈 결과 반환
    if (pytesseract is None) or (roi_img is None) or (roi_img.size == 0):
        return OcrResult("", None, 0.0)

    # 전처리
    proc_img = preprocess_for_ocr(roi_img)

    # Tesseract 설정 (숫자 위주 인식)
    # --psm 7: 한 줄의 텍스트로 취급
    # digits: 숫자만 허용 (whitelist)
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.eE+-'

    try:
        # 상세 데이터(신뢰도 포함) 가져오기
        data = pytesseract.image_to_data(proc_img, config=config, output_type=pytesseract.Output.DICT)
        
        # 신뢰도 높은 텍스트만 합치기
        text_parts = []
        conf_sum = 0.0
        conf_cnt = 0
        
        n_items = len(data['text'])
        for i in range(n_items):
            txt = data['text'][i].strip()
            # conf가 -1이거나 너무 낮은 값은 제외
            try:
                conf = float(data['conf'][i])
            except:
                conf = 0.0
                
            if txt and conf > 0:
                text_parts.append(txt)
                conf_sum += conf
                conf_cnt += 1
        
        full_text = "".join(text_parts)
        avg_conf = (conf_sum / conf_cnt) if conf_cnt > 0 else 0.0
        val = parse_float(full_text)

        # 구형 코드와의 호환성을 위해 OcrResult 객체가 text, value 속성을 가지도록 함
        # (위에 정의된 dataclass 사용)
        return OcrResult(full_text, val, avg_conf)

    except Exception:
        # Tesseract 에러 시
        return OcrResult("", None, 0.0)