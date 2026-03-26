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
from tb.interactive_3d import export_atomic_scene_html, pyvista_is_available


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


def compute_bands(model: TightBindingModel, k_points: np.ndarray) -> np.ndarray:
    return np.vstack([model.eigenvalues([k_value]) for k_value in k_points])


def write_dos_overlay_csv(path: Path, energies: np.ndarray, chain_dos: np.ndarray, dimer_dos: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["energy", "dos_uniform_chain", "dos_dimerized_chain"])
        for energy, uniform_value, dimer_value in zip(energies, chain_dos, dimer_dos):
            writer.writerow([f"{energy:.8f}", f"{uniform_value:.8f}", f"{dimer_value:.8f}"])


def write_band_overlay_csv(
    path: Path,
    k_points: np.ndarray,
    chain_bands: np.ndarray,
    dimer_bands: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "k",
                "k_over_pi",
                "uniform_band_0",
                "dimerized_band_0",
                "dimerized_band_1",
            ]
        )
        for index, k_value in enumerate(k_points):
            writer.writerow(
                [
                    f"{k_value:.8f}",
                    f"{(k_value / math.pi):.8f}",
                    f"{chain_bands[index, 0]:.8f}",
                    f"{dimer_bands[index, 0]:.8f}",
                    f"{dimer_bands[index, 1]:.8f}",
                ]
            )


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


def write_reciprocal_space_svg(
    path: Path,
    k_points: np.ndarray,
    chain_bands: np.ndarray,
    dimer_bands: np.ndarray,
) -> None:
    width = 960
    height = 580
    left = 90
    right = 30
    top = 40
    bottom = 74

    plot_left = left
    plot_right = width - right
    plot_top = top
    plot_bottom = height - bottom

    x_min = float(k_points.min())
    x_max = float(k_points.max())
    y_min = float(min(chain_bands.min(), dimer_bands.min()))
    y_max = float(max(chain_bands.max(), dimer_bands.max()))
    padding = max(0.1, 0.08 * (y_max - y_min if y_max > y_min else 1.0))
    y_min -= padding
    y_max += padding

    def make_points(values: np.ndarray) -> str:
        points = []
        for k_value, energy in zip(k_points, values):
            svg_x = _scale(float(k_value), x_min, x_max, plot_left, plot_right)
            svg_y = _scale(float(energy), y_min, y_max, plot_bottom, plot_top)
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

    x_ticks = []
    tick_values = [-math.pi, -0.5 * math.pi, 0.0, 0.5 * math.pi, math.pi]
    tick_labels = ["-pi", "-pi/2", "0", "pi/2", "pi"]
    for tick_value, label in zip(tick_values, tick_labels):
        svg_x = _scale(float(tick_value), x_min, x_max, plot_left, plot_right)
        x_ticks.append(
            f'<line x1="{svg_x:.2f}" y1="{plot_top}" x2="{svg_x:.2f}" y2="{plot_bottom}" '
            'stroke="#dddddd" stroke-width="1" />'
        )
        x_ticks.append(
            f'<text x="{svg_x:.2f}" y="{height - 28}" text-anchor="middle" '
            'font-family="Consolas, monospace" font-size="14">{label}</text>'
        )

    legend_x = plot_right - 270
    legend_y = plot_top + 14

    uniform_points = make_points(chain_bands[:, 0])
    lower_points = make_points(dimer_bands[:, 0])
    upper_points = make_points(dimer_bands[:, 1])

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="white" />
<text x="{width / 2:.2f}" y="24" text-anchor="middle" font-family="Consolas, monospace" font-size="20">Reciprocal-space band comparison</text>
<rect x="{plot_left}" y="{plot_top}" width="{plot_right - plot_left}" height="{plot_bottom - plot_top}" fill="none" stroke="black" stroke-width="1.5" />
{''.join(y_ticks)}
{''.join(x_ticks)}
<polyline fill="none" stroke="#1f77b4" stroke-width="2.5" points="{uniform_points}" />
<polyline fill="none" stroke="#d62728" stroke-width="2.5" points="{lower_points}" />
<polyline fill="none" stroke="#ff7f0e" stroke-width="2.5" points="{upper_points}" />
<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="#1f77b4" stroke-width="3" />
<text x="{legend_x + 36}" y="{legend_y + 5}" font-family="Consolas, monospace" font-size="14">uniform chain</text>
<line x1="{legend_x}" y1="{legend_y + 24}" x2="{legend_x + 28}" y2="{legend_y + 24}" stroke="#d62728" stroke-width="3" />
<text x="{legend_x + 36}" y="{legend_y + 29}" font-family="Consolas, monospace" font-size="14">dimer lower band</text>
<line x1="{legend_x}" y1="{legend_y + 48}" x2="{legend_x + 28}" y2="{legend_y + 48}" stroke="#ff7f0e" stroke-width="3" />
<text x="{legend_x + 36}" y="{legend_y + 53}" font-family="Consolas, monospace" font-size="14">dimer upper band</text>
<text x="{width / 2:.2f}" y="{height - 8}" text-anchor="middle" font-family="Consolas, monospace" font-size="16">k</text>
<text x="22" y="{height / 2:.2f}" transform="rotate(-90 22,{height / 2:.2f})" text-anchor="middle" font-family="Consolas, monospace" font-size="16">Energy</text>
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


def build_real_space_scene(cells: int, delta: float) -> tuple[list[list[float]], list[str], list[float], list[tuple[int, int, float, str]], list[str], list[list[float]]]:
    left = -3.6
    uniform_y = 1.6
    dimer_y = -1.6
    uniform_spacing = 0.95
    short_bond = 0.72
    long_bond = 1.18

    if abs(delta) > 1.0:
        scale = min(1.6, 1.0 + 0.25 * abs(delta))
        short_bond /= scale
        long_bond *= scale

    positions: list[list[float]] = []
    colors: list[str] = []
    radii: list[float] = []
    bonds: list[tuple[int, int, float, str]] = []

    uniform_indices = []
    for index in range(cells):
        position = [left + index * uniform_spacing, uniform_y, 0.0]
        positions.append(position)
        colors.append("#1f77b4")
        radii.append(0.16)
        uniform_indices.append(len(positions) - 1)
    for start, end in zip(uniform_indices[:-1], uniform_indices[1:]):
        bonds.append((start, end, 0.06, "#406080"))

    dimer_indices = []
    current_x = left
    for index in range(cells):
        if index > 0:
            current_x += short_bond if index % 2 == 1 else long_bond
        positions.append([current_x, dimer_y, 0.0])
        colors.append("#d62728")
        radii.append(0.16)
        dimer_indices.append(len(positions) - 1)
    for index, (start, end) in enumerate(zip(dimer_indices[:-1], dimer_indices[1:])):
        bond_radius = 0.075 if index % 2 == 0 else 0.05
        bond_color = "#8f1d1d" if index % 2 == 0 else "#dd7777"
        bonds.append((start, end, bond_radius, bond_color))

    label_positions = [
        [left - 0.7, uniform_y, 0.0],
        [left - 0.7, dimer_y, 0.0],
    ]
    labels = [
        "uniform chain",
        f"dimerized chain (delta={delta:.2f})",
    ]
    return positions, colors, radii, bonds, labels, label_positions


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

Uniform dispersion: E(k) = -2 t cos(k)
E_+(k), E_-(k) = +/- sqrt(t1^2 + t2^2 + 2 t1 t2 cos(k))
Band gap at k = pi: 2 |t1 - t2| = {gap:.6f}
"""
    path.write_text(text, encoding="utf-8")


def write_formula_html(path: Path, t: float, delta: float, t1: float, t2: float) -> None:
    gap = 2.0 * abs(t1 - t2)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>1D Chain Calculation Formulas</title>
  <style>
    body {{
      margin: 24px;
      font-family: "Segoe UI", Arial, sans-serif;
      line-height: 1.65;
      color: #111;
      background: #fff;
    }}
    h1, h2 {{
      font-family: Consolas, monospace;
    }}
    .equation {{
      margin: 14px 0;
      padding: 14px 18px;
      border-left: 4px solid #1f6feb;
      background: #f6f8fa;
      font-family: "Times New Roman", serif;
      font-size: 1.12rem;
    }}
    .note {{
      padding: 14px 18px;
      border: 1px solid #d0d7de;
      border-radius: 10px;
      background: #fbfbfc;
    }}
  </style>
</head>
<body>
  <h1>1D Chain Calculation Formulas</h1>
  <p>This file writes out the actual mathematical pipeline behind the plots: Hamiltonian, eigenvalues, Brillouin-zone integrals, and the discrete approximations used in the code.</p>

  <h2>1. Uniform Chain</h2>
  <div class="equation">H(k) = -2 t cos(k)</div>
  <div class="equation">E(k) = -2 t cos(k)</div>
  <div class="equation">
    D(E) = (1 / 2&pi;) &int;<sub>-&pi;</sub><sup>&pi;</sup> &delta;(E + 2 t cos k) dk
  </div>
  <div class="equation">
    D(E) = 1 / [&pi; &radic;(4 t<sup>2</sup> - E<sup>2</sup>)] , &nbsp; |E| &lt; 2|t|
  </div>

  <h2>2. Dimerized Chain</h2>
  <p>The dimerized chain uses two alternating nearest-neighbor hoppings:</p>
  <div class="equation">t<sub>1</sub> = t (1 - &delta;) = {t1:.6f}</div>
  <div class="equation">t<sub>2</sub> = t (1 + &delta;) = {t2:.6f}</div>
  <div class="equation">
    H(k) =
    <span style="display: inline-block; vertical-align: middle;">
      [ [ 0,&nbsp; t<sub>1</sub> + t<sub>2</sub> e<sup>-ik</sup> ],
      <br/>
      &nbsp;&nbsp;[ t<sub>1</sub> + t<sub>2</sub> e<sup>+ik</sup>,&nbsp; 0 ] ]
    </span>
  </div>
  <div class="equation">
    E<sub>&plusmn;</sub>(k) = &plusmn; &radic;(t<sub>1</sub><sup>2</sup> + t<sub>2</sub><sup>2</sup> + 2 t<sub>1</sub> t<sub>2</sub> cos(k))
  </div>

  <h2>3. Band Gap Derivation</h2>
  <div class="equation">
    E<sub>g</sub> = min<sub>k</sub> [E<sub>+</sub>(k) - E<sub>-</sub>(k)]
    = 2 min<sub>k</sub> &radic;(t<sub>1</sub><sup>2</sup> + t<sub>2</sub><sup>2</sup> + 2 t<sub>1</sub> t<sub>2</sub> cos(k))
  </div>
  <div class="equation">
    cos(k) = -1 at k = &pi; &nbsp;&rArr;&nbsp;
    E<sub>g</sub> = 2 |t<sub>1</sub> - t<sub>2</sub>|
  </div>
  <div class="equation">E<sub>g</sub> = 2 |t<sub>1</sub> - t<sub>2</sub>| = 4 |t &delta;| = {gap:.6f}</div>
  <p>The gap appears at the Brillouin-zone boundary <strong>k = &pi;</strong>.</p>

  <h2>4. Brillouin-Zone Integrals and Numerical Approximation</h2>
  <div class="equation">
    D(E) = (1 / 2&pi;) &sum;<sub>n=&plusmn;</sub> &int;<sub>-&pi;</sub><sup>&pi;</sup> &delta;(E - E<sub>n</sub>(k)) dk
  </div>
  <div class="equation">
    k<sub>m</sub> = -&pi; + 2&pi; m / (N<sub>k</sub> - 1), &nbsp; m = 0, ..., N<sub>k</sub> - 1
  </div>
  <div class="equation">
    &delta;(x) &approx; L<sub>&eta;</sub>(x) = [&eta; / &pi;] / (x<sup>2</sup> + &eta;<sup>2</sup>)
  </div>
  <div class="equation">
    D(E) &approx; (1 / N<sub>k</sub>) &sum;<sub>m</sub> &sum;<sub>n</sub> L<sub>&eta;</sub>(E - E<sub>n</sub>(k<sub>m</sub>))
  </div>

  <h2>5. Band-Plot Sampling</h2>
  <div class="equation">
    k<sub>m</sub> = -&pi; + 2&pi; m / (N<sub>band</sub> - 1)
  </div>
  <div class="equation">
    The reciprocal-space band plot is obtained by evaluating E(k<sub>m</sub>) or E<sub>&plusmn;</sub>(k<sub>m</sub>) on this linearly spaced grid.
  </div>
  <div class="note">
    <strong>Parameters used in this run</strong><br/>
    &delta; = {delta:.6f}<br/>
    t = {t:.6f}<br/>
    t<sub>1</sub> = {t1:.6f}<br/>
    t<sub>2</sub> = {t2:.6f}
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_status_note(path: Path, message: str) -> None:
    path.write_text(message.rstrip() + "\n", encoding="utf-8")


def write_report_html(path: Path, interactive_available: bool) -> None:
    interactive_block = (
        '<iframe src="real_space_interactive.html" title="Interactive real-space view" '
        'style="width: 100%; height: 560px; border: 1px solid #d0d0d0; border-radius: 10px;"></iframe>'
        if interactive_available
        else '<div style="padding: 24px; border: 1px solid #d0d0d0; border-radius: 10px; background: #fafafa;">'
             '<strong>Interactive 3D view was not generated.</strong><br/>'
             'See <code>interactive_status.txt</code> for the dependency/runtime reason.'
             '</div>'
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>1D Chain Dimerization Report</title>
</head>
<body style="margin: 24px; font-family: Consolas, monospace; background: white; color: #111;">
  <h1 style="margin-bottom: 8px;">1D Chain Dimerization Report</h1>
  <p style="margin-top: 0;">Interactive real-space view plus reciprocal-space and DOS outputs.</p>
  <h2>Interactive Real Space</h2>
  {interactive_block}
  <h2 style="margin-top: 28px;">Static Real Space</h2>
  <img src="real_space.svg" alt="Real-space SVG" style="max-width: 100%; border: 1px solid #d0d0d0; border-radius: 10px;" />
  <h2 style="margin-top: 28px;">Reciprocal Space</h2>
  <img src="reciprocal_space.svg" alt="Reciprocal-space band plot" style="max-width: 100%; border: 1px solid #d0d0d0; border-radius: 10px;" />
  <h2 style="margin-top: 28px;">Calculation Formulas</h2>
  <p><a href="calculation_formulas.html">Open the formula file directly</a></p>
  <iframe src="calculation_formulas.html" title="Calculation formulas"
          style="width: 100%; height: 1200px; border: 1px solid #d0d0d0; border-radius: 10px;"></iframe>
  <h2 style="margin-top: 28px;">Density of States</h2>
  <img src="dos_overlay.svg" alt="DOS overlay" style="max-width: 100%; border: 1px solid #d0d0d0; border-radius: 10px;" />
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--k-grid", type=int, default=240)
    parser.add_argument("--energy-points", type=int, default=480)
    parser.add_argument("--broadening", type=float, default=0.06)
    parser.add_argument("--cells", type=int, default=12)
    parser.add_argument("--band-points", type=int, default=401)
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    structure_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve() if args.out_dir else structure_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    chain_model = build_uniform_chain(args.t)
    dimer_model, t1, t2 = build_dimerized_chain(args.t, args.delta)

    energy_limit = 2.6 * max(abs(args.t), abs(t1), abs(t2), 1.0)
    energy_axis = np.linspace(-energy_limit, energy_limit, max(120, args.energy_points))
    k_points = np.linspace(-math.pi, math.pi, max(101, args.band_points))

    chain_dos = compute_dos(chain_model, energy_axis, args.k_grid, args.broadening)
    dimer_dos = compute_dos(dimer_model, energy_axis, args.k_grid, args.broadening)
    chain_bands = compute_bands(chain_model, k_points)
    dimer_bands = compute_bands(dimer_model, k_points)

    real_space_svg = out_dir / "real_space.svg"
    real_space_html = out_dir / "real_space_interactive.html"
    summary_txt = out_dir / "model_summary.txt"
    formula_html = out_dir / "calculation_formulas.html"
    dos_csv = out_dir / "dos_overlay.csv"
    dos_svg = out_dir / "dos_overlay.svg"
    band_csv = out_dir / "band_overlay.csv"
    reciprocal_svg = out_dir / "reciprocal_space.svg"
    interactive_status = out_dir / "interactive_status.txt"
    report_html = out_dir / "report.html"

    write_real_space_svg(real_space_svg, args.cells, args.delta)
    write_model_summary(summary_txt, args.t, args.delta, t1, t2)
    write_formula_html(formula_html, args.t, args.delta, t1, t2)
    write_dos_overlay_csv(dos_csv, energy_axis, chain_dos, dimer_dos)
    write_dos_overlay_svg(dos_svg, energy_axis, chain_dos, dimer_dos)
    write_band_overlay_csv(band_csv, k_points, chain_bands, dimer_bands)
    write_reciprocal_space_svg(reciprocal_svg, k_points, chain_bands, dimer_bands)

    scene_positions, scene_colors, scene_radii, scene_bonds, scene_labels, scene_label_positions = build_real_space_scene(
        args.cells,
        args.delta,
    )
    interactive_ok, interactive_reason = pyvista_is_available(out_dir)
    if interactive_ok:
        try:
            export_atomic_scene_html(
                output_path=real_space_html,
                title="uniform vs dimerized 1D chain",
                atom_positions=scene_positions,
                atom_colors=scene_colors,
                atom_radii=scene_radii,
                bonds=scene_bonds,
                point_labels=scene_labels,
                label_positions=scene_label_positions,
            )
            write_status_note(interactive_status, "Interactive PyVista HTML generated successfully.")
        except Exception as exc:
            interactive_ok = False
            write_status_note(interactive_status, f"Interactive HTML export failed: {type(exc).__name__}: {exc}")
    else:
        write_status_note(interactive_status, f"PyVista unavailable: {interactive_reason}")

    write_report_html(report_html, interactive_available=interactive_ok)

    print(f"Output directory: {out_dir}")
    print(f"[structure] {real_space_svg}")
    print(f"[3d html]   {real_space_html}")
    print(f"[summary]   {summary_txt}")
    print(f"[formula]   {formula_html}")
    print(f"[band csv]  {band_csv}")
    print(f"[band svg]  {reciprocal_svg}")
    print(f"[dos csv]   {dos_csv}")
    print(f"[dos svg]   {dos_svg}")
    print(f"[report]    {report_html}")
    print(f"[3d status] {interactive_status}")
    print(f"Uniform band range: {chain_bands.min():.4f} .. {chain_bands.max():.4f}")
    print(f"Dimer band range:   {dimer_bands.min():.4f} .. {dimer_bands.max():.4f}")
    print(f"Uniform DOS max:   {chain_dos.max():.4f}")
    print(f"Dimerized DOS max: {dimer_dos.max():.4f}")


if __name__ == "__main__":
    main()
