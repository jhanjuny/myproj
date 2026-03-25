from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tb.model import TightBindingModel


def build_uniform_chain(t: float) -> TightBindingModel:
    onsite = np.array([[0.0]], dtype=np.complex128)
    hoppings = [
        (np.array([1.0]), np.array([[-t]], dtype=np.complex128)),
    ]
    return TightBindingModel("uniform 1D chain", 1, onsite, hoppings)


def build_dimerized_chain(t: float, delta: float) -> tuple[TightBindingModel, float, float]:
    t1 = t * (1.0 - delta)
    t2 = t * (1.0 + delta)
    onsite = np.zeros((2, 2), dtype=np.complex128)
    hoppings = [
        (
            np.array([0.0]),
            np.array(
                [
                    [0.0, t1],
                    [0.0, 0.0],
                ],
                dtype=np.complex128,
            ),
        ),
        (
            np.array([1.0]),
            np.array(
                [
                    [0.0, t2],
                    [0.0, 0.0],
                ],
                dtype=np.complex128,
            ),
        ),
    ]
    return TightBindingModel("dimerized 1D chain", 1, onsite, hoppings), t1, t2


def compute_dos(
    model: TightBindingModel,
    energy_axis: np.ndarray,
    k_grid: int,
    broadening: float,
) -> np.ndarray:
    k_points = np.linspace(-math.pi, math.pi, max(64, k_grid), endpoint=False)
    all_energies = np.concatenate([model.eigenvalues([k_value]) for k_value in k_points], axis=0)

    sigma = max(1e-4, float(broadening))
    diff = energy_axis[:, None] - all_energies[None, :]
    gaussian = np.exp(-0.5 * (diff / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    return gaussian.mean(axis=1)


def write_overlay_csv(path: Path, energies: np.ndarray, chain_dos: np.ndarray, dimer_dos: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["energy", "dos_uniform_chain", "dos_dimerized_chain"])
        for energy, uniform_value, dimer_value in zip(energies, chain_dos, dimer_dos):
            writer.writerow([f"{energy:.8f}", f"{uniform_value:.8f}", f"{dimer_value:.8f}"])


def _scale(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if domain_max == domain_min:
        return 0.5 * (range_min + range_max)
    alpha = (value - domain_min) / (domain_max - domain_min)
    return range_min + alpha * (range_max - range_min)


def write_dos_overlay_svg(path: Path, energies: np.ndarray, chain_dos: np.ndarray, dimer_dos: np.ndarray) -> None:
    width = 920
    height = 560
    left = 90
    right = 30
    top = 40
    bottom = 70

    x_min = float(energies.min())
    x_max = float(energies.max())
    y_min = 0.0
    y_max = float(max(chain_dos.max(), dimer_dos.max()))
    y_max = max(y_max, 1e-6)

    plot_left = left
    plot_right = width - right
    plot_top = top
    plot_bottom = height - bottom

    def make_points(values: np.ndarray) -> str:
        points = []
        for energy, density in zip(energies, values):
            svg_x = _scale(float(energy), x_min, x_max, plot_left, plot_right)
            svg_y = _scale(float(density), y_min, y_max, plot_bottom, plot_top)
            points.append(f"{svg_x:.2f},{svg_y:.2f}")
        return " ".join(points)

    y_ticks = []
    for fraction in np.linspace(0.0, 1.0, 5):
        value = y_min + fraction * (y_max - y_min)
        svg_y = _scale(value, y_min, y_max, plot_bottom, plot_top)
        y_ticks.append(
            f'<line x1="{plot_left}" y1="{svg_y:.2f}" x2="{plot_right}" y2="{svg_y:.2f}" '
            'stroke="#efefef" stroke-width="1" />'
        )
        y_ticks.append(
            f'<text x="{plot_left - 12}" y="{svg_y + 5:.2f}" text-anchor="end" '
            'font-family="Consolas, monospace" font-size="13">{value:.2f}</text>'
        )

    legend_x = plot_right - 230
    legend_y = plot_top + 12
    chain_points = make_points(chain_dos)
    dimer_points = make_points(dimer_dos)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="white" />
<text x="{width / 2:.2f}" y="24" text-anchor="middle" font-family="Consolas, monospace" font-size="20">1D chain DOS comparison</text>
<rect x="{plot_left}" y="{plot_top}" width="{plot_right - plot_left}" height="{plot_bottom - plot_top}" fill="none" stroke="black" stroke-width="1.5" />
{''.join(y_ticks)}
<polyline fill="none" stroke="#1f77b4" stroke-width="2.5" points="{chain_points}" />
<polyline fill="none" stroke="#d62728" stroke-width="2.5" points="{dimer_points}" />
<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="#1f77b4" stroke-width="3" />
<text x="{legend_x + 36}" y="{legend_y + 5}" font-family="Consolas, monospace" font-size="14">uniform chain</text>
<line x1="{legend_x}" y1="{legend_y + 24}" x2="{legend_x + 28}" y2="{legend_y + 24}" stroke="#d62728" stroke-width="3" />
<text x="{legend_x + 36}" y="{legend_y + 29}" font-family="Consolas, monospace" font-size="14">dimerized chain</text>
<text x="{width / 2:.2f}" y="{height - 8}" text-anchor="middle" font-family="Consolas, monospace" font-size="16">Energy</text>
<text x="22" y="{height / 2:.2f}" transform="rotate(-90 22,{height / 2:.2f})" text-anchor="middle" font-family="Consolas, monospace" font-size="16">DOS</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_real_space_svg(path: Path, cells: int, delta: float) -> None:
    width = 980
    height = 280
    left = 70
    top_uniform = 85
    top_dimer = 195
    atom_radius = 10

    uniform_spacing = 72
    short_bond = 52
    long_bond = 92
    if abs(delta) > 1.0:
        scale = min(1.6, 1.0 + 0.25 * abs(delta))
        short_bond /= scale
        long_bond *= scale

    uniform_positions = [left + index * uniform_spacing for index in range(cells)]
    dimer_positions = [left]
    for index in range(1, cells):
        bond = short_bond if index % 2 == 1 else long_bond
        dimer_positions.append(dimer_positions[-1] + bond)

    uniform_lines = []
    for x0, x1 in zip(uniform_positions[:-1], uniform_positions[1:]):
        uniform_lines.append(
            f'<line x1="{x0}" y1="{top_uniform}" x2="{x1}" y2="{top_uniform}" stroke="#444444" stroke-width="4" />'
        )

    dimer_lines = []
    for index, (x0, x1) in enumerate(zip(dimer_positions[:-1], dimer_positions[1:])):
        stroke = 5 if index % 2 == 0 else 3
        dimer_lines.append(
            f'<line x1="{x0}" y1="{top_dimer}" x2="{x1}" y2="{top_dimer}" stroke="#444444" stroke-width="{stroke}" />'
        )

    uniform_atoms = []
    for x in uniform_positions:
        uniform_atoms.append(
            f'<circle cx="{x}" cy="{top_uniform}" r="{atom_radius}" fill="#1f77b4" stroke="black" stroke-width="1.5" />'
        )

    dimer_atoms = []
    for x in dimer_positions:
        dimer_atoms.append(
            f'<circle cx="{x}" cy="{top_dimer}" r="{atom_radius}" fill="#d62728" stroke="black" stroke-width="1.5" />'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="white" />
<text x="{width / 2:.2f}" y="24" text-anchor="middle" font-family="Consolas, monospace" font-size="20">Real-space view: uniform vs dimerized 1D chain</text>
<text x="42" y="{top_uniform + 5}" text-anchor="end" font-family="Consolas, monospace" font-size="14">uniform</text>
<text x="42" y="{top_dimer + 5}" text-anchor="end" font-family="Consolas, monospace" font-size="14">dimerized</text>
{''.join(uniform_lines)}
{''.join(dimer_lines)}
{''.join(uniform_atoms)}
{''.join(dimer_atoms)}
<text x="{width - 160}" y="{top_dimer + 44}" font-family="Consolas, monospace" font-size="13">delta = {delta:.3f}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_model_summary(path: Path, t: float, delta: float, t1: float, t2: float) -> None:
    gap = 2.0 * abs(t1 - t2)
    text = f"""Uniform chain
-------------
H(k) = -2 t cos(k)
t = {t:.6f}

Dimerized chain
---------------
t1 = t (1 - delta) = {t1:.6f}
t2 = t (1 + delta) = {t2:.6f}
delta = {delta:.6f}

H(k) = [[0, t1 + t2 exp(-i k)],
        [t1 + t2 exp(+i k), 0]]

E_+(k), E_-(k) = +/- sqrt(t1^2 + t2^2 + 2 t1 t2 cos(k))
Band gap at k = pi: 2 |t1 - t2| = {gap:.6f}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--k-grid", type=int, default=240)
    parser.add_argument("--energy-points", type=int, default=480)
    parser.add_argument("--broadening", type=float, default=0.06)
    parser.add_argument("--cells", type=int, default=12)
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    structure_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve() if args.out_dir else structure_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    chain_model = build_uniform_chain(args.t)
    dimer_model, t1, t2 = build_dimerized_chain(args.t, args.delta)

    energy_limit = 2.6 * max(abs(args.t), abs(t1), abs(t2), 1.0)
    energy_axis = np.linspace(-energy_limit, energy_limit, max(120, args.energy_points))

    chain_dos = compute_dos(chain_model, energy_axis, args.k_grid, args.broadening)
    dimer_dos = compute_dos(dimer_model, energy_axis, args.k_grid, args.broadening)

    real_space_svg = out_dir / "real_space.svg"
    summary_txt = out_dir / "model_summary.txt"
    dos_csv = out_dir / "dos_overlay.csv"
    dos_svg = out_dir / "dos_overlay.svg"

    write_real_space_svg(real_space_svg, args.cells, args.delta)
    write_model_summary(summary_txt, args.t, args.delta, t1, t2)
    write_overlay_csv(dos_csv, energy_axis, chain_dos, dimer_dos)
    write_dos_overlay_svg(dos_svg, energy_axis, chain_dos, dimer_dos)

    print(f"Output directory: {out_dir}")
    print(f"[structure] {real_space_svg}")
    print(f"[summary]   {summary_txt}")
    print(f"[dos csv]   {dos_csv}")
    print(f"[dos svg]   {dos_svg}")
    print(f"Uniform DOS max:   {chain_dos.max():.4f}")
    print(f"Dimerized DOS max: {dimer_dos.max():.4f}")


if __name__ == "__main__":
    main()
