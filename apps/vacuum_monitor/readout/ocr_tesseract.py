from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import pytesseract
except Exception:
    pytesseract = None


import os
import pytesseract

def _set_tesseract_cmd(cfg: dict):
     # 1) config.yaml에서 지정한 경로 우선
    cmd = None
    if isinstance(cfg, dict):
        cmd = (cfg.get("tesseract") or {}).get("cmd") or cfg.get("tesseract_cmd")
     # 2) 환경변수로도 가능
    cmd = cmd or os.environ.get("TESSERACT_CMD")
     # 3) 최종 적용
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd





_SCI_RE = re.compile(
    r"(?P<m>\d+(?:\.\d+)?)\s*[eE]\s*(?P<e>[+\-]?\s*\d+)"
)

_DEC_RE = re.compile(r"(?P<m>\d+(?:\.\d+)?)")


@dataclass
class ReadoutResult:
    text: str
    value: Optional[float]


def _preprocess(roi_bgr: np.ndarray) -> np.ndarray:
    # ROI를 OCR 친화적으로 전처리
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (3, 3), 0)

    # 조명/흔들림에도 견디도록 adaptive threshold
    th = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )

    # 글자가 흰색/검정색이 뒤집힌 경우를 대비해, 두 버전 중 “더 글자다운” 쪽 선택
    inv = 255 - th
    # 간단한 휴리스틱: 흰 픽셀 비율이 너무 크면 뒤집기
    if (th > 0).mean() > 0.80:
        th = inv

    return th


def _parse_value(text: str) -> Optional[float]:
    t = text.strip().replace(" ", "")
    if not t:
        return None

    # 장비 상태 텍스트 처리
    up = t.upper()
    if "NOSENSOR" in up or "NO" in up and "SENSOR" in up:
        return None

    # 과학적 표기 파싱 (예: 5.49E-10)
    m = _SCI_RE.search(t)
    if m:
        mant = float(m.group("m"))
        exp = int(m.group("e").replace(" ", ""))
        return mant * (10 ** exp)

    # 소수/정수만 있는 경우
    m2 = _DEC_RE.search(t)
    if m2:
        return float(m2.group("m"))

    return None


def read_value_tesseract(roi_bgr: np.ndarray, cfg: dict) -> ReadoutResult:
    _set_tesseract_cmd(cfg)

    if pytesseract is None:
        return ReadoutResult(text="", value=None)

    r_cfg = (cfg or {}).get("readout", {}) if isinstance(cfg, dict) else {}
    tcmd = r_cfg.get("tesseract_cmd")
    psm = int(r_cfg.get("psm", 7))
    whitelist = r_cfg.get("whitelist", "0123456789eE+-. ")

    if tcmd:
        pytesseract.pytesseract.tesseract_cmd = tcmd

    img = _preprocess(roi_bgr)

    # psm 7: single text line
    config = f'--psm {psm} -c tessedit_char_whitelist="{whitelist}"'
    text = pytesseract.image_to_string(img, config=config)
    text = text.strip().replace("\n", " ")

    val = _parse_value(text)
    return ReadoutResult(text=text, value=val)
