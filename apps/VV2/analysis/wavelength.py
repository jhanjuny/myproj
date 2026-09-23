# apps/VV2/analysis/wavelength.py
"""전자빔 상대론적 de Broglie 파장. (arXiv:2609.15922 Methods Eq. 2와 동일 관계식)"""


def lambda_angstrom(v_kev: float) -> float:
    """가속전압 v_kev(kV)에서 상대론적 전자 파장(Å)."""
    v_ev = v_kev * 1000.0
    return 12.2643 / (v_ev * (1 + 0.9788e-6 * v_ev)) ** 0.5


def lambda_cm(v_kev: float) -> float:
    """d = λL/Δx 계산에서 L(cm)과 단위를 맞추기 위한 cm 단위 파장."""
    return lambda_angstrom(v_kev) * 1e-8
