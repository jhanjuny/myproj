"""ARPES binding energy calculator (interactive, cmd).

Startup:  photon energy hv, work function phi, tolerance.
Loop:     enter electron kinetic energy E_kin ->
          E_B = hv - phi - E_kin, core-level candidates within tolerance,
          2nd-order light candidates (2hv), nearest level of pinned elements.
"""
import math
import sys

from binding_energies import BINDING_ENERGIES

PINNED = ["O", "C", "Sr", "Sn", "Ag", "Cu", "Mo"]
DEFAULT_TOL = 0.5

# flat list: (BE, symbol, Z, level)
LEVELS = sorted(
    (be, sym, z, lv)
    for sym, (z, levels) in BINDING_ENERGIES.items()
    for lv, be in levels.items()
)


def parse_float(s):
    """Finite float or None (rejects text, nan, inf)."""
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def ask_float(prompt, default=None, min_value=None):
    while True:
        s = input(prompt).strip().lstrip("﻿")
        if not s and default is not None:
            return default
        v = parse_float(s)
        if v is None:
            print("  숫자를 입력하세요.")
        elif min_value is not None and v < min_value:
            print(f"  {min_value} 이상의 값을 입력하세요.")
        else:
            return v


def candidates(eb, tol):
    hits = [(be - eb, sym, z, lv, be) for be, sym, z, lv in LEVELS if abs(be - eb) <= tol]
    return sorted(hits, key=lambda h: abs(h[0]))


def nearest(sym, eb):
    z, levels = BINDING_ENERGIES[sym]
    lv, be = min(levels.items(), key=lambda kv: abs(kv[1] - eb))
    return be - eb, sym, z, lv, be


def fmt(h, tol=None):
    d, sym, z, lv, be = h
    tag = "" if tol is None or abs(d) <= tol else "  (범위 밖)"
    return f"  {sym:<2} (Z={z:>2})  {lv:<6} {be:>9.1f} eV   dE={d:+8.2f}{tag}"


def report(ekin, hv, phi, tol):
    eb1 = hv - phi - ekin
    eb2 = 2 * hv - phi - ekin
    print(f"\n  E_kin = {ekin:.3f} eV  ->  E_B = {eb1:.3f} eV   (2nd order: E_B = {eb2:.3f} eV)")
    if eb1 < 0:
        print("  [주의] E_B < 0 : E_kin이 hv - phi 보다 큽니다 (Fermi level 위).")

    print(f"\n  [1차광 후보, ±{tol} eV]  dE = 표값 - 측정값")
    hits = candidates(eb1, tol)
    print("\n".join(fmt(h) for h in hits) if hits else "  (없음)")

    print(f"\n  [2차광(2hv = {2 * hv:.1f} eV) 후보, ±{tol} eV]")
    hits = candidates(eb2, tol)
    print("\n".join(fmt(h) for h in hits) if hits else "  (없음)")

    print("\n  [고정 원소 최근접 레벨 (1차광 기준)]")
    for sym in PINNED:
        print(fmt(nearest(sym, eb1), tol))
    print()


def main():
    print("=" * 64)
    print(" ARPES Binding Energy Calculator  (X-ray Data Booklet Table 1-1)")
    print("=" * 64)
    hv = ask_float(" Photon energy hv (eV): ")
    phi = ask_float(" Work function phi (eV): ")
    tol = ask_float(f" Tolerance ± (eV) [Enter = {DEFAULT_TOL}]: ", DEFAULT_TOL, min_value=0)
    print(f"\n hv = {hv} eV, phi = {phi} eV, tol = ±{tol} eV  (변경하려면 재시작)")
    print(" E_kin 입력 (종료: q)\n")

    while True:
        try:
            s = input("E_kin (eV) > ").strip().lstrip("﻿")
        except (EOFError, KeyboardInterrupt):
            break
        if s.lower() in ("q", "quit", "exit"):
            break
        if not s:
            continue
        ekin = parse_float(s)
        if ekin is None:
            print("  숫자를 입력하세요.")
            continue
        report(ekin, hv, phi, tol)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        pass
    sys.exit(0)
