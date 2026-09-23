# apps/VV2/analysis/materials.py
"""
기준 물질 격자상수 데이터베이스 — 캘리브레이션(알려진 존축 d-spacing으로
픽셀 스케일 역산) 및 측정값 검증(기대값 대비 편차)에 사용.

필요한 물질이 없으면 이 표에 추가하거나, GUI에서 스케일(cm/px)을 직접
입력해 이 표를 거치지 않고도 캘리브레이션할 수 있습니다.
"""
import math

MATERIALS = {
    "SrTiO3":   {"a": 3.905, "system": "cubic"},
    "BaTiO3":   {"a": 3.994, "system": "cubic"},
    "LaAlO3":   {"a": 3.790, "system": "cubic"},
    "LSAT":     {"a": 3.868, "system": "cubic"},
    "DyScO3":   {"a": 3.944, "system": "cubic"},
    "GdScO3":   {"a": 3.967, "system": "cubic"},
    "KTaO3":    {"a": 3.989, "system": "cubic"},
    "MgO":      {"a": 4.211, "system": "cubic"},
    "NiO":      {"a": 4.177, "system": "cubic"},
    "Si":       {"a": 5.431, "system": "cubic"},
    "Ge":       {"a": 5.658, "system": "cubic"},
    "GaAs":     {"a": 5.653, "system": "cubic"},
    "Al2O3":    {"a": 4.759, "c": 12.991, "system": "hexagonal"},
    "GaN":      {"a": 3.189, "c": 5.185,  "system": "hexagonal"},
    "ZnO":      {"a": 3.250, "c": 5.207,  "system": "hexagonal"},
}


def find_material_key(raw: str):
    norm = (raw or "").strip().lower()
    for key in MATERIALS:
        if key.lower() == norm:
            return key
    return None


def expected_d(material: str, zone: str):
    """물질 + 존축(zone: '100'/'010'/'001'/'110'/'111', 괄호 없이)의 기대 면내 d-spacing(Å)."""
    m = MATERIALS.get(material)
    if m is None:
        return None
    a = m["a"]
    if m["system"] == "cubic":
        if zone in ("100", "001", "010"):
            return a
        if zone in ("110", "111"):
            return a / math.sqrt(2)
    elif m["system"] == "hexagonal":
        if zone in ("100", "010"):
            return a * math.sqrt(3) / 2
        if zone == "001":
            return a
    return None
