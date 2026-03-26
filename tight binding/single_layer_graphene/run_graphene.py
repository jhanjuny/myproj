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

from tb.export import ensure_dir, write_band_csv, write_band_svg, write_dos_csv, write_dos_svg
from tb.interactive_3d import export_atomic_scene_html, export_reciprocal_surfaces_html, pyvista_is_available
from tb.kpath import sample_k_path


SQRT3 = math.sqrt(3.0)


def graphene_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a1 = np.array([SQRT3 / 2.0, 1.5], dtype=float)
    a2 = np.array([-SQRT3 / 2.0, 1.5], dtype=float)
    basis_a = np.array([0.0, 0.0], dtype=float)
    basis_b = np.array([0.0, 1.0], dtype=float)
    deltas = np.array(
        [
            [0.0, 1.0],
            [SQRT3 / 2.0, -0.5],
            [-SQRT3 / 2.0, -0.5],
        ],
        dtype=float,
    )
    return a1, a2, basis_a, basis_b, deltas


def reciprocal_vectors(a1: np.ndarray, a2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.column_stack([a1, a2])
    reciprocal = 2.0 * math.pi * np.linalg.inv(matrix).T
    return reciprocal[:, 0], reciprocal[:, 1]


def graphene_f(k_point: np.ndarray, deltas: np.ndarray) -> complex:
    phases = np.exp(1j * (deltas @ k_point))
    return phases.sum()


def graphene_bands(k_point: np.ndarray, t: float, deltas: np.ndarray) -> np.ndarray:
    amplitude = abs(graphene_f(k_point, deltas))
    energy = float(t) * amplitude
    return np.array([-energy, energy], dtype=float)


def compute_band_path(t: float, deltas: np.ndarray, k_path: list[tuple[str, np.ndarray]], samples_per_segment: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    k_points, distances, tick_positions, tick_labels = sample_k_path(k_path, samples_per_segment)
    bands = np.vstack([graphene_bands(k_point, t, deltas) for k_point in k_points])
    return distances, bands, tick_positions, tick_labels


def compute_dos(t: float, deltas: np.ndarray, b1: np.ndarray, b2: np.ndarray, grid_n: int, energy_axis: np.ndarray, broadening: float) -> np.ndarray:
    samples = max(40, int(grid_n))
    u_values = np.linspace(0.0, 1.0, samples, endpoint=False)
    v_values = np.linspace(0.0, 1.0, samples, endpoint=False)
    energies = []
    for u in u_values:
        for v in v_values:
            k_point = u * b1 + v * b2
            energies.append(graphene_bands(k_point, t, deltas))
    stacked = np.concatenate(energies, axis=0)

    sigma = max(1e-4, float(broadening))
    diff = energy_axis[:, None] - stacked[None, :]
    gaussian = np.exp(-0.5 * (diff / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    return gaussian.mean(axis=1)


def bz_hexagon_vertices(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    k1 = (b1 - b2) / 3.0
    k2 = (2.0 * b1 + b2) / 3.0
    k3 = (b1 + 2.0 * b2) / 3.0
    return np.array([k1, k2, k3, -k1, -k2, -k3], dtype=float)


def point_in_convex_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    signs = []
    for index in range(len(polygon)):
        current = polygon[index]
        nxt = polygon[(index + 1) % len(polygon)]
        edge = nxt - current
        rel = point - current
        cross = edge[0] * rel[1] - edge[1] * rel[0]
        signs.append(cross)
    return min(signs) >= -1e-9 or max(signs) <= 1e-9


def compute_reciprocal_map(t: float, deltas: np.ndarray, polygon: np.ndarray, grid_n: int) -> list[tuple[float, float, float, float]]:
    samples = max(35, int(grid_n))
    min_x, min_y = polygon.min(axis=0)
    max_x, max_y = polygon.max(axis=0)
    x_values = np.linspace(min_x, max_x, samples)
    y_values = np.linspace(min_y, max_y, samples)

    rows: list[tuple[float, float, float, float]] = []
    for y_value in y_values:
        for x_value in x_values:
            point = np.array([x_value, y_value], dtype=float)
            if not point_in_convex_polygon(point, polygon):
                continue
            lower, upper = graphene_bands(point, t, deltas)
            rows.append((x_value, y_value, lower, upper))
    return rows


def write_reciprocal_csv(path: Path, rows: list[tuple[float, float, float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kx", "ky", "valence_band", "conduction_band"])
        for kx, ky, lower, upper in rows:
            writer.writerow([f"{kx:.8f}", f"{ky:.8f}", f"{lower:.8f}", f"{upper:.8f}"])


def _scale(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if domain_max == domain_min:
        return 0.5 * (range_min + range_max)
    alpha = (value - domain_min) / (domain_max - domain_min)
    return range_min + alpha * (range_max - range_min)


def _lerp_color(stops: list[tuple[float, tuple[int, int, int]]], value: float) -> str:
    clipped = max(0.0, min(1.0, value))
    for index in range(len(stops) - 1):
        left_pos, left_color = stops[index]
        right_pos, right_color = stops[index + 1]
        if clipped <= right_pos:
            if right_pos == left_pos:
                alpha = 0.0
            else:
                alpha = (clipped - left_pos) / (right_pos - left_pos)
            rgb = [
                int(round((1.0 - alpha) * left_color[channel] + alpha * right_color[channel]))
                for channel in range(3)
            ]
            return "#{:02x}{:02x}{:02x}".format(*rgb)
    return "#{:02x}{:02x}{:02x}".format(*stops[-1][1])


def energy_color(value: float, energy_min: float, energy_max: float) -> str:
    normalized = 0.5 if energy_max == energy_min else (value - energy_min) / (energy_max - energy_min)
    stops = [
        (0.0, (40, 61, 137)),
        (0.5, (240, 248, 255)),
        (1.0, (197, 56, 43)),
    ]
    return _lerp_color(stops, normalized)


def write_reciprocal_svg(
    path: Path,
    polygon: np.ndarray,
    rows: list[tuple[float, float, float, float]],
    k_path: list[tuple[str, np.ndarray]],
    t: float,
) -> None:
    width = 1180
    height = 580
    panel_width = 430
    panel_height = 430
    left_margin = 80
    panel_gap = 120
    top_margin = 70

    min_x, min_y = polygon.min(axis=0)
    max_x, max_y = polygon.max(axis=0)
    span_x = max_x - min_x
    span_y = max_y - min_y
    pad_x = 0.08 * span_x
    pad_y = 0.08 * span_y
    x_min = min_x - pad_x
    x_max = max_x + pad_x
    y_min = min_y - pad_y
    y_max = max_y + pad_y

    left_panel_x = left_margin
    right_panel_x = left_margin + panel_width + panel_gap
    panel_y = top_margin

    lower_min = -3.0 * abs(t)
    lower_max = 0.0
    upper_min = 0.0
    upper_max = 3.0 * abs(t)

    def project(point: np.ndarray, panel_x: float) -> tuple[float, float]:
        return (
            _scale(float(point[0]), x_min, x_max, panel_x, panel_x + panel_width),
            _scale(float(point[1]), y_min, y_max, panel_y + panel_height, panel_y),
        )

    dx = span_x / max(12, math.sqrt(max(1, len(rows))))
    dy = span_y / max(12, math.sqrt(max(1, len(rows))))

    lower_tiles = []
    upper_tiles = []
    for kx, ky, lower, upper in rows:
        x0, y0 = project(np.array([kx - dx / 2.0, ky - dy / 2.0]), left_panel_x)
        x1, y1 = project(np.array([kx + dx / 2.0, ky + dy / 2.0]), left_panel_x)
        rect_x = min(x0, x1)
        rect_y = min(y0, y1)
        rect_w = abs(x1 - x0)
        rect_h = abs(y1 - y0)
        lower_tiles.append(
            f'<rect x="{rect_x:.2f}" y="{rect_y:.2f}" width="{rect_w:.2f}" height="{rect_h:.2f}" '
            f'fill="{energy_color(lower, lower_min, lower_max)}" stroke="none" />'
        )

        x0r, y0r = project(np.array([kx - dx / 2.0, ky - dy / 2.0]), right_panel_x)
        x1r, y1r = project(np.array([kx + dx / 2.0, ky + dy / 2.0]), right_panel_x)
        upper_tiles.append(
            f'<rect x="{min(x0r, x1r):.2f}" y="{min(y0r, y1r):.2f}" width="{abs(x1r - x0r):.2f}" height="{abs(y1r - y0r):.2f}" '
            f'fill="{energy_color(upper, upper_min, upper_max)}" stroke="none" />'
        )

    polygon_left = " ".join(f"{project(vertex, left_panel_x)[0]:.2f},{project(vertex, left_panel_x)[1]:.2f}" for vertex in polygon)
    polygon_right = " ".join(f"{project(vertex, right_panel_x)[0]:.2f},{project(vertex, right_panel_x)[1]:.2f}" for vertex in polygon)
    path_left = " ".join(f"{project(point, left_panel_x)[0]:.2f},{project(point, left_panel_x)[1]:.2f}" for _, point in k_path)
    path_right = " ".join(f"{project(point, right_panel_x)[0]:.2f},{project(point, right_panel_x)[1]:.2f}" for _, point in k_path)

    labels = []
    for label, point in k_path[:-1]:
        lx, ly = project(point, left_panel_x)
        rx, ry = project(point, right_panel_x)
        labels.append(
            f'<circle cx="{lx:.2f}" cy="{ly:.2f}" r="4.5" fill="black" />'
            f'<text x="{lx + 8:.2f}" y="{ly - 8:.2f}" font-family="Consolas, monospace" font-size="14">{label}</text>'
        )
        labels.append(
            f'<circle cx="{rx:.2f}" cy="{ry:.2f}" r="4.5" fill="black" />'
            f'<text x="{rx + 8:.2f}" y="{ry - 8:.2f}" font-family="Consolas, monospace" font-size="14">{label}</text>'
        )

    gamma_right_x, _ = project(k_path[0][1], right_panel_x)
    gamma_left_x, _ = project(k_path[0][1], left_panel_x)
    gamma_top = panel_y + panel_height + 42

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="white" />
<text x="{width / 2:.2f}" y="30" text-anchor="middle" font-family="Consolas, monospace" font-size="22">Graphene reciprocal-space band map</text>
<text x="{left_panel_x + panel_width / 2:.2f}" y="56" text-anchor="middle" font-family="Consolas, monospace" font-size="16">Valence band</text>
<text x="{right_panel_x + panel_width / 2:.2f}" y="56" text-anchor="middle" font-family="Consolas, monospace" font-size="16">Conduction band</text>
<rect x="{left_panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" fill="none" stroke="black" stroke-width="1.5" />
<rect x="{right_panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" fill="none" stroke="black" stroke-width="1.5" />
{''.join(lower_tiles)}
{''.join(upper_tiles)}
<polygon points="{polygon_left}" fill="none" stroke="#202020" stroke-width="2" />
<polygon points="{polygon_right}" fill="none" stroke="#202020" stroke-width="2" />
<polyline points="{path_left}" fill="none" stroke="#111111" stroke-width="2" stroke-dasharray="6,5" />
<polyline points="{path_right}" fill="none" stroke="#111111" stroke-width="2" stroke-dasharray="6,5" />
{''.join(labels)}
<text x="{gamma_left_x:.2f}" y="{gamma_top:.2f}" text-anchor="middle" font-family="Consolas, monospace" font-size="13">First Brillouin zone + G-K-M path</text>
<text x="{gamma_right_x:.2f}" y="{gamma_top:.2f}" text-anchor="middle" font-family="Consolas, monospace" font-size="13">Same path over conduction surface</text>
<text x="{left_panel_x + panel_width / 2:.2f}" y="{height - 12}" text-anchor="middle" font-family="Consolas, monospace" font-size="15">kx</text>
<text x="{right_panel_x + panel_width / 2:.2f}" y="{height - 12}" text-anchor="middle" font-family="Consolas, monospace" font-size="15">kx</text>
<text x="24" y="{panel_y + panel_height / 2:.2f}" transform="rotate(-90 24,{panel_y + panel_height / 2:.2f})" text-anchor="middle" font-family="Consolas, monospace" font-size="15">ky</text>
<text x="{left_panel_x + panel_width + 14:.2f}" y="{panel_y + 18:.2f}" font-family="Consolas, monospace" font-size="12">E in [-3t, 0]</text>
<text x="{right_panel_x + panel_width + 14:.2f}" y="{panel_y + 18:.2f}" font-family="Consolas, monospace" font-size="12">E in [0, 3t]</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def generate_real_space_patch(nx: int, ny: int) -> tuple[list[list[float]], list[str], list[float], list[tuple[int, int, float, str]], list[str], list[list[float]]]:
    a1, a2, basis_a, basis_b, _ = graphene_geometry()
    positions: list[list[float]] = []
    colors: list[str] = []
    radii: list[float] = []

    for iy in range(ny):
        for ix in range(nx):
            origin = ix * a1 + iy * a2
            a_pos = origin + basis_a
            b_pos = origin + basis_b
            positions.append([float(a_pos[0]), float(a_pos[1]), 0.0])
            colors.append("#1f77b4")
            radii.append(0.18)
            positions.append([float(b_pos[0]), float(b_pos[1]), 0.0])
            colors.append("#d62728")
            radii.append(0.18)

    bonds: list[tuple[int, int, float, str]] = []
    array_positions = np.asarray(positions, dtype=float)
    for left_index in range(len(array_positions)):
        for right_index in range(left_index + 1, len(array_positions)):
            distance = np.linalg.norm(array_positions[left_index] - array_positions[right_index])
            if 0.75 <= distance <= 1.05:
                bonds.append((left_index, right_index, 0.06, "#555555"))

    min_x = float(array_positions[:, 0].min())
    max_y = float(array_positions[:, 1].max())
    labels = ["A sublattice", "B sublattice"]
    label_positions = [
        [min_x - 1.2, max_y + 0.7, 0.0],
        [min_x - 1.2, max_y + 0.1, 0.0],
    ]
    return positions, colors, radii, bonds, labels, label_positions


def write_real_space_svg(path: Path, positions: list[list[float]], colors: list[str], bonds: list[tuple[int, int, float, str]]) -> None:
    width = 980
    height = 580
    left = 70
    right = 40
    top = 60
    bottom = 60

    array_positions = np.asarray(positions, dtype=float)
    min_x, min_y = array_positions[:, 0].min(), array_positions[:, 1].min()
    max_x, max_y = array_positions[:, 0].max(), array_positions[:, 1].max()
    span_x = max_x - min_x
    span_y = max_y - min_y
    x_min = min_x - 0.9
    x_max = max_x + 0.9
    y_min = min_y - 0.9
    y_max = max_y + 1.3

    def project(point: np.ndarray) -> tuple[float, float]:
        return (
            _scale(float(point[0]), x_min, x_max, left, width - right),
            _scale(float(point[1]), y_min, y_max, height - bottom, top),
        )

    bond_lines = []
    for start_index, end_index, _, _ in bonds:
        start = array_positions[start_index]
        end = array_positions[end_index]
        x0, y0 = project(start)
        x1, y1 = project(end)
        bond_lines.append(
            f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="#555555" stroke-width="4" />'
        )

    atom_circles = []
    for position, color in zip(array_positions, colors):
        x, y = project(position)
        atom_circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="11" fill="{color}" stroke="black" stroke-width="1.3" />'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="white" />
<text x="{width / 2:.2f}" y="30" text-anchor="middle" font-family="Consolas, monospace" font-size="22">Single-layer graphene real-space patch</text>
<text x="{left}" y="{height - 20}" font-family="Consolas, monospace" font-size="14">Blue: A sublattice   Red: B sublattice</text>
{''.join(bond_lines)}
{''.join(atom_circles)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_model_summary(path: Path, t: float, deltas: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> None:
    text = f"""Single-layer graphene nearest-neighbor model
--------------------------------------------
t = {t:.6f}

Nearest-neighbor bond vectors:
d1 = ({deltas[0, 0]:.6f}, {deltas[0, 1]:.6f})
d2 = ({deltas[1, 0]:.6f}, {deltas[1, 1]:.6f})
d3 = ({deltas[2, 0]:.6f}, {deltas[2, 1]:.6f})

Primitive reciprocal vectors:
b1 = ({b1[0]:.6f}, {b1[1]:.6f})
b2 = ({b2[0]:.6f}, {b2[1]:.6f})

Hamiltonian:
H(k) = [[0, -t f(k)],
        [-t f*(k), 0]]

f(k) = sum_j exp(i k·d_j)
Band energies: E_±(k) = ± t |f(k)|

This nearest-neighbor model produces Dirac cones at K/K' and zero DOS at E = 0.
"""
    path.write_text(text, encoding="utf-8")


def write_formula_html(path: Path, t: float, deltas: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> None:
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Single-Layer Graphene Calculation Formulas</title>
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
  <h1>Single-Layer Graphene Calculation Formulas</h1>
  <p>This file writes out the actual calculation flow used for the graphene plots: Bloch Hamiltonian, eigenvalues, k-path interpolation, Brillouin-zone integrals, and the discrete approximations used in the numerics.</p>

  <h2>1. Nearest-Neighbor Hamiltonian</h2>
  <div class="equation">
    H(k) =
    <span style="display: inline-block; vertical-align: middle;">
      [ [ 0,&nbsp; -t f(k) ],
      <br/>
      &nbsp;&nbsp;[ -t f*(k),&nbsp; 0 ] ]
    </span>
  </div>
  <div class="equation">f(k) = &sum;<sub>j=1</sub><sup>3</sup> exp[i k &middot; &delta;<sub>j</sub>]</div>
  <div class="equation">E<sub>&plusmn;</sub>(k) = &plusmn; t |f(k)|</div>

  <h2>2. Bond Vectors</h2>
  <div class="equation">&delta;<sub>1</sub> = ({deltas[0, 0]:.6f}, {deltas[0, 1]:.6f})</div>
  <div class="equation">&delta;<sub>2</sub> = ({deltas[1, 0]:.6f}, {deltas[1, 1]:.6f})</div>
  <div class="equation">&delta;<sub>3</sub> = ({deltas[2, 0]:.6f}, {deltas[2, 1]:.6f})</div>

  <h2>3. Reciprocal Lattice</h2>
  <div class="equation">b<sub>1</sub> = ({b1[0]:.6f}, {b1[1]:.6f})</div>
  <div class="equation">b<sub>2</sub> = ({b2[0]:.6f}, {b2[1]:.6f})</div>
  <div class="equation">K = (b<sub>1</sub> - b<sub>2</sub>) / 3,&nbsp;&nbsp; M = b<sub>1</sub> / 2</div>

  <h2>4. High-Symmetry Band Path</h2>
  <div class="equation">
    k(s) = (1 - s) k<sub>a</sub> + s k<sub>b</sub>, &nbsp; 0 &le; s &le; 1
  </div>
  <div class="equation">
    s<sub>m</sub> = m / (N<sub>seg</sub> - 1), &nbsp; m = 0, ..., N<sub>seg</sub> - 1
  </div>
  <div class="equation">
    The plotted path is G &rarr; K &rarr; M &rarr; G, with E<sub>&plusmn;</sub>(k(s<sub>m</sub>)) evaluated on each segment.
  </div>

  <h2>5. Brillouin-Zone Integral for DOS</h2>
  <div class="equation">
    D(E) = (1 / A<sub>BZ</sub>) &sum;<sub>n=&plusmn;</sub> &int;<sub>BZ</sub> &delta;(E - E<sub>n</sub>(k)) d<sup>2</sup>k
  </div>
  <div class="equation">
    &delta;(x) &approx; L<sub>&eta;</sub>(x) = [&eta; / &pi;] / (x<sup>2</sup> + &eta;<sup>2</sup>)
  </div>
  <div class="equation">
    D(E) &approx; (1 / N<sub>k</sub>) &sum;<sub>m</sub> &sum;<sub>n=&plusmn;</sub> L<sub>&eta;</sub>(E - E<sub>n</sub>(k<sub>m</sub>))
  </div>

  <h2>6. Reciprocal-Space Surface Sampling</h2>
  <div class="equation">
    k<sub>mn</sub> = u<sub>m</sub> b<sub>1</sub> + v<sub>n</sub> b<sub>2</sub>, &nbsp; (u<sub>m</sub>, v<sub>n</sub>) on a uniform grid clipped to the first Brillouin zone
  </div>
  <div class="equation">
    E<sub>lower</sub>(k<sub>mn</sub>) = E<sub>-</sub>(k<sub>mn</sub>), &nbsp; E<sub>upper</sub>(k<sub>mn</sub>) = E<sub>+</sub>(k<sub>mn</sub>)
  </div>

  <h2>7. Low-Energy Dirac Approximation</h2>
  <div class="equation">
    f(K + q) &approx; - (3 a / 2) (q<sub>x</sub> - i q<sub>y</sub>)
  </div>
  <div class="equation">
    E<sub>&plusmn;</sub>(K + q) &approx; &plusmn; v<sub>F</sub> |q|
  </div>

  <div class="note">
    <strong>Physical consequence</strong><br/>
    This model produces Dirac cones at K and K', so the Dirac-point gap is zero and the DOS vanishes linearly near E = 0.
  </div>
  <div class="note" style="margin-top: 14px;">
    <strong>Run parameter</strong><br/>
    t = {t:.6f}
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_status_note(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(line.rstrip() for line in lines).rstrip() + "\n", encoding="utf-8")


def write_report_html(path: Path, real_space_interactive: bool, reciprocal_interactive: bool) -> None:
    real_space_block = (
        '<iframe src="real_space_interactive.html" title="Interactive graphene view" '
        'style="width: 100%; height: 600px; border: 1px solid #d0d0d0; border-radius: 10px;"></iframe>'
        if real_space_interactive
        else '<div style="padding: 24px; border: 1px solid #d0d0d0; border-radius: 10px; background: #fafafa;">'
             '<strong>Interactive 3D view was not generated.</strong><br/>'
             'See <code>interactive_status.txt</code> for the dependency/runtime reason.'
             '</div>'
    )
    reciprocal_block = (
        '<iframe src="reciprocal_space_interactive.html" title="Interactive reciprocal-space graphene view" '
        'style="width: 100%; height: 760px; border: 1px solid #d0d0d0; border-radius: 10px;"></iframe>'
        if reciprocal_interactive
        else '<div style="padding: 24px; border: 1px solid #d0d0d0; border-radius: 10px; background: #fafafa;">'
             '<strong>Interactive reciprocal-space view was not generated.</strong><br/>'
             'See <code>interactive_status.txt</code> for the dependency/runtime reason.'
             '</div>'
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Single-Layer Graphene Report</title>
</head>
<body style="margin: 24px; font-family: Consolas, monospace; background: white; color: #111;">
  <h1 style="margin-bottom: 8px;">Single-Layer Graphene Report</h1>
  <p style="margin-top: 0;">Nearest-neighbor tight-binding outputs: real space, reciprocal-space map, band structure, and DOS.</p>
  <h2>Interactive Real Space</h2>
  {real_space_block}
  <h2 style="margin-top: 28px;">Static Real Space</h2>
  <img src="real_space.svg" alt="Graphene real-space patch" style="max-width: 100%; border: 1px solid #d0d0d0; border-radius: 10px;" />
  <h2 style="margin-top: 28px;">Band Structure (G-K-M-G)</h2>
  <img src="band_structure.svg" alt="Graphene band structure" style="max-width: 100%; border: 1px solid #d0d0d0; border-radius: 10px;" />
  <h2 style="margin-top: 28px;">2D Reciprocal Space</h2>
  <img src="reciprocal_space_map.svg" alt="Graphene reciprocal-space band map" style="max-width: 100%; border: 1px solid #d0d0d0; border-radius: 10px;" />
  <h2 style="margin-top: 28px;">Interactive 3D Reciprocal Space</h2>
  {reciprocal_block}
  <h2 style="margin-top: 28px;">Calculation Formulas</h2>
  <p><a href="calculation_formulas.html">Open the formula file directly</a></p>
  <iframe src="calculation_formulas.html" title="Graphene calculation formulas"
          style="width: 100%; height: 1350px; border: 1px solid #d0d0d0; border-radius: 10px;"></iframe>
  <h2 style="margin-top: 28px;">Density of States</h2>
  <img src="dos.svg" alt="Graphene DOS" style="max-width: 100%; border: 1px solid #d0d0d0; border-radius: 10px;" />
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--samples-per-segment", type=int, default=120)
    parser.add_argument("--dos-grid", type=int, default=150)
    parser.add_argument("--map-grid", type=int, default=95)
    parser.add_argument("--energy-points", type=int, default=420)
    parser.add_argument("--broadening", type=float, default=0.07)
    parser.add_argument("--nx", type=int, default=5)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    structure_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve() if args.out_dir else structure_dir / "outputs"
    ensure_dir(out_dir)

    a1, a2, _, _, deltas = graphene_geometry()
    b1, b2 = reciprocal_vectors(a1, a2)
    gamma = np.array([0.0, 0.0], dtype=float)
    k_point = (b1 - b2) / 3.0
    m_point = 0.5 * b1
    k_path = [
        ("G", gamma),
        ("K", k_point),
        ("M", m_point),
        ("G", gamma),
    ]

    distances, bands, tick_positions, tick_labels = compute_band_path(
        t=args.t,
        deltas=deltas,
        k_path=k_path,
        samples_per_segment=args.samples_per_segment,
    )

    energy_limit = 3.2 * max(abs(args.t), 1.0)
    energy_axis = np.linspace(-energy_limit, energy_limit, max(120, args.energy_points))
    dos = compute_dos(
        t=args.t,
        deltas=deltas,
        b1=b1,
        b2=b2,
        grid_n=args.dos_grid,
        energy_axis=energy_axis,
        broadening=args.broadening,
    )

    polygon = bz_hexagon_vertices(b1, b2)
    reciprocal_rows = compute_reciprocal_map(
        t=args.t,
        deltas=deltas,
        polygon=polygon,
        grid_n=args.map_grid,
    )

    positions, colors, radii, bonds, labels, label_positions = generate_real_space_patch(args.nx, args.ny)

    real_space_svg = out_dir / "real_space.svg"
    real_space_html = out_dir / "real_space_interactive.html"
    summary_txt = out_dir / "model_summary.txt"
    formula_html = out_dir / "calculation_formulas.html"
    band_csv = out_dir / "band_structure.csv"
    band_svg = out_dir / "band_structure.svg"
    dos_csv = out_dir / "dos.csv"
    dos_svg = out_dir / "dos.svg"
    reciprocal_csv = out_dir / "reciprocal_space_map.csv"
    reciprocal_svg = out_dir / "reciprocal_space_map.svg"
    reciprocal_html = out_dir / "reciprocal_space_interactive.html"
    report_html = out_dir / "report.html"
    interactive_status = out_dir / "interactive_status.txt"

    write_real_space_svg(real_space_svg, positions, colors, bonds)
    write_model_summary(summary_txt, args.t, deltas, b1, b2)
    write_formula_html(formula_html, args.t, deltas, b1, b2)
    write_band_csv(band_csv, distances, bands)
    write_band_svg(band_svg, "Single-layer graphene band structure", distances, bands, tick_positions, tick_labels)
    write_dos_csv(dos_csv, energy_axis, dos)
    write_dos_svg(dos_svg, "Single-layer graphene DOS", energy_axis, dos)
    write_reciprocal_csv(reciprocal_csv, reciprocal_rows)
    write_reciprocal_svg(reciprocal_svg, polygon, reciprocal_rows, k_path, args.t)

    interactive_ok, interactive_reason = pyvista_is_available(out_dir)
    real_space_interactive_ok = False
    reciprocal_interactive_ok = False
    status_lines: list[str] = []

    if interactive_ok:
        try:
            export_atomic_scene_html(
                output_path=real_space_html,
                title="single-layer graphene nearest-neighbor patch",
                atom_positions=positions,
                atom_colors=colors,
                atom_radii=radii,
                bonds=bonds,
                point_labels=labels,
                label_positions=label_positions,
            )
            real_space_interactive_ok = True
            status_lines.append("Real-space interactive HTML generated successfully.")
        except Exception as exc:
            status_lines.append(f"Real-space interactive HTML export failed: {type(exc).__name__}: {exc}")

        try:
            reciprocal_points = np.array([[row[0], row[1]] for row in reciprocal_rows], dtype=float)
            lower_energies = np.array([row[2] for row in reciprocal_rows], dtype=float)
            upper_energies = np.array([row[3] for row in reciprocal_rows], dtype=float)
            path_points = np.vstack([point for _, point in k_path])
            path_labels = [label for label, _ in k_path]
            export_reciprocal_surfaces_html(
                output_path=reciprocal_html,
                title="single-layer graphene reciprocal-space surfaces",
                xy_points=reciprocal_points,
                lower_energies=lower_energies,
                upper_energies=upper_energies,
                polygon_vertices=polygon,
                k_path=path_points,
                k_labels=path_labels,
                energy_limit=energy_limit,
            )
            reciprocal_interactive_ok = True
            status_lines.append("Reciprocal-space interactive HTML generated successfully.")
        except Exception as exc:
            status_lines.append(f"Reciprocal-space interactive HTML export failed: {type(exc).__name__}: {exc}")
    else:
        status_lines.append(f"PyVista unavailable: {interactive_reason}")

    write_status_note(interactive_status, status_lines)
    write_report_html(
        report_html,
        real_space_interactive=real_space_interactive_ok,
        reciprocal_interactive=reciprocal_interactive_ok,
    )

    print(f"Output directory: {out_dir}")
    print(f"[structure] {real_space_svg}")
    print(f"[3d html]   {real_space_html}")
    print(f"[summary]   {summary_txt}")
    print(f"[formula]   {formula_html}")
    print(f"[band csv]  {band_csv}")
    print(f"[band svg]  {band_svg}")
    print(f"[dos csv]   {dos_csv}")
    print(f"[dos svg]   {dos_svg}")
    print(f"[k-map csv] {reciprocal_csv}")
    print(f"[k-map svg] {reciprocal_svg}")
    print(f"[k-map 3d]  {reciprocal_html}")
    print(f"[report]    {report_html}")
    print(f"[3d status] {interactive_status}")
    print(f"Band range: {bands.min():.4f} .. {bands.max():.4f}")
    print(f"DOS max:    {dos.max():.4f}")
    print(f"K-point bands: {graphene_bands(k_point, args.t, deltas)[0]:.6f}, {graphene_bands(k_point, args.t, deltas)[1]:.6f}")


if __name__ == "__main__":
    main()
