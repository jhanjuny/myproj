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
A0 = 1.0


def base_graphene_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a1 = np.array([SQRT3 / 2.0, 1.5], dtype=float)
    a2 = np.array([-SQRT3 / 2.0, 1.5], dtype=float)
    bond = np.array([0.0, 1.0], dtype=float)
    return a1, a2, bond


def build_distorted_supercell(pair_shift: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[float]]:
    if not (0.0 <= pair_shift < 0.5):
        raise ValueError("pair_shift must satisfy 0 <= pair_shift < 0.5")

    a1, a2, bond = base_graphene_geometry()
    super_a1 = 2.0 * a1 - a2
    super_a2 = -a1 + 2.0 * a2

    cell_origins = [np.zeros(2, dtype=float), a1.copy(), a2.copy()]
    labels = ["A", "A'", "B", "B'", "C", "C'"]
    colors = ["#1f77b4", "#7fb7ff", "#d62728", "#ff9896", "#2ca02c", "#98df8a"]
    radii = [0.18] * 6

    positions: list[np.ndarray] = []
    shift_vec = pair_shift * bond
    for origin in cell_origins:
        positions.append(origin + shift_vec)
        positions.append(origin + bond - shift_vec)

    return super_a1, super_a2, np.vstack(positions), labels, colors, radii


def reciprocal_vectors(a1: np.ndarray, a2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.column_stack([a1, a2])
    reciprocal = 2.0 * math.pi * np.linalg.inv(matrix).T
    return reciprocal[:, 0], reciprocal[:, 1]


def distance_hopping(distance: float, t0: float, beta: float) -> float:
    return float(t0) * math.exp(-float(beta) * (distance / A0 - 1.0))


def build_neighbor_terms(
    positions: np.ndarray,
    super_a1: np.ndarray,
    super_a2: np.ndarray,
    t0: float,
    beta: float,
) -> list[tuple[int, int, int, int, float, float]]:
    terms: list[tuple[int, int, int, int, float, float]] = []

    for left_index in range(0, len(positions), 2):
        candidates: list[tuple[float, int, int, int]] = []
        for right_index in range(1, len(positions), 2):
            for shift_1 in (-1, 0, 1):
                for shift_2 in (-1, 0, 1):
                    shift = shift_1 * super_a1 + shift_2 * super_a2
                    distance = float(np.linalg.norm(positions[right_index] + shift - positions[left_index]))
                    candidates.append((distance, right_index, shift_1, shift_2))

        candidates.sort(key=lambda item: item[0])
        for distance, right_index, shift_1, shift_2 in candidates[:3]:
            hopping = distance_hopping(distance, t0, beta)
            terms.append((left_index, right_index, shift_1, shift_2, hopping, distance))

    return terms


def bloch_hamiltonian(
    k_point: np.ndarray,
    positions: np.ndarray,
    neighbor_terms: list[tuple[int, int, int, int, float, float]],
    super_a1: np.ndarray,
    super_a2: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((len(positions), len(positions)), dtype=complex)
    for left_index, right_index, shift_1, shift_2, hopping, _ in neighbor_terms:
        shift = shift_1 * super_a1 + shift_2 * super_a2
        phase_argument = positions[right_index] + shift - positions[left_index]
        phase = np.exp(1j * float(np.dot(k_point, phase_argument)))
        matrix[left_index, right_index] += -hopping * phase
        matrix[right_index, left_index] += -hopping * np.conjugate(phase)
    return matrix


def distorted_bands(
    k_point: np.ndarray,
    positions: np.ndarray,
    neighbor_terms: list[tuple[int, int, int, int, float, float]],
    super_a1: np.ndarray,
    super_a2: np.ndarray,
) -> np.ndarray:
    matrix = bloch_hamiltonian(k_point, positions, neighbor_terms, super_a1, super_a2)
    return np.linalg.eigvalsh(matrix)


def compute_band_path(
    band_fn,
    k_path: list[tuple[str, np.ndarray]],
    samples_per_segment: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    k_points, distances, tick_positions, tick_labels = sample_k_path(k_path, samples_per_segment)
    bands = np.vstack([band_fn(k_point) for k_point in k_points])
    return distances, bands, tick_positions, tick_labels


def compute_dos(
    band_fn,
    b1: np.ndarray,
    b2: np.ndarray,
    grid_n: int,
    energy_axis: np.ndarray,
    broadening: float,
) -> np.ndarray:
    samples = max(40, int(grid_n))
    u_values = np.linspace(0.0, 1.0, samples, endpoint=False)
    v_values = np.linspace(0.0, 1.0, samples, endpoint=False)
    energies = []
    for u_value in u_values:
        for v_value in v_values:
            k_point = u_value * b1 + v_value * b2
            energies.append(band_fn(k_point))
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


def valence_conduction_pair(bands: np.ndarray) -> tuple[float, float]:
    sorted_bands = np.sort(np.asarray(bands, dtype=float))
    half = len(sorted_bands) // 2
    return float(sorted_bands[half - 1]), float(sorted_bands[half])


def compute_reciprocal_map(
    band_fn,
    polygon: np.ndarray,
    grid_n: int,
) -> list[tuple[float, float, float, float]]:
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
            valence, conduction = valence_conduction_pair(band_fn(point))
            rows.append((x_value, y_value, valence, conduction))
    return rows


def write_reciprocal_csv(path: Path, rows: list[tuple[float, float, float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kx", "ky", "top_valence_band", "bottom_conduction_band"])
        for kx, ky, valence, conduction in rows:
            writer.writerow([f"{kx:.8f}", f"{ky:.8f}", f"{valence:.8f}", f"{conduction:.8f}"])


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
            alpha = 0.0 if right_pos == left_pos else (clipped - left_pos) / (right_pos - left_pos)
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
    title: str,
    polygon: np.ndarray,
    rows: list[tuple[float, float, float, float]],
    k_path: list[tuple[str, np.ndarray]],
    energy_limit: float,
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

    def project(point: np.ndarray, panel_x: float) -> tuple[float, float]:
        return (
            _scale(float(point[0]), x_min, x_max, panel_x, panel_x + panel_width),
            _scale(float(point[1]), y_min, y_max, panel_y + panel_height, panel_y),
        )

    dx = span_x / max(12, math.sqrt(max(1, len(rows))))
    dy = span_y / max(12, math.sqrt(max(1, len(rows))))
    lower_tiles = []
    upper_tiles = []
    for kx, ky, valence, conduction in rows:
        x0, y0 = project(np.array([kx - dx / 2.0, ky - dy / 2.0]), left_panel_x)
        x1, y1 = project(np.array([kx + dx / 2.0, ky + dy / 2.0]), left_panel_x)
        lower_tiles.append(
            f'<rect x="{min(x0, x1):.2f}" y="{min(y0, y1):.2f}" width="{abs(x1 - x0):.2f}" height="{abs(y1 - y0):.2f}" '
            f'fill="{energy_color(valence, -energy_limit, 0.0)}" stroke="none" />'
        )

        x0r, y0r = project(np.array([kx - dx / 2.0, ky - dy / 2.0]), right_panel_x)
        x1r, y1r = project(np.array([kx + dx / 2.0, ky + dy / 2.0]), right_panel_x)
        upper_tiles.append(
            f'<rect x="{min(x0r, x1r):.2f}" y="{min(y0r, y1r):.2f}" width="{abs(x1r - x0r):.2f}" height="{abs(y1r - y0r):.2f}" '
            f'fill="{energy_color(conduction, 0.0, energy_limit)}" stroke="none" />'
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

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="580" viewBox="0 0 1180 580">
<rect x="0" y="0" width="1180" height="580" fill="white" />
<text x="590" y="30" text-anchor="middle" font-family="Consolas, monospace" font-size="22">{title}</text>
<text x="{left_panel_x + panel_width / 2:.2f}" y="56" text-anchor="middle" font-family="Consolas, monospace" font-size="16">Top valence band</text>
<text x="{right_panel_x + panel_width / 2:.2f}" y="56" text-anchor="middle" font-family="Consolas, monospace" font-size="16">Bottom conduction band</text>
<rect x="{left_panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" fill="none" stroke="black" stroke-width="1.5" />
<rect x="{right_panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" fill="none" stroke="black" stroke-width="1.5" />
{''.join(lower_tiles)}
{''.join(upper_tiles)}
<polygon points="{polygon_left}" fill="none" stroke="#202020" stroke-width="2" />
<polygon points="{polygon_right}" fill="none" stroke="#202020" stroke-width="2" />
<polyline points="{path_left}" fill="none" stroke="#111111" stroke-width="2" stroke-dasharray="6,5" />
<polyline points="{path_right}" fill="none" stroke="#111111" stroke-width="2" stroke-dasharray="6,5" />
{''.join(labels)}
<text x="{left_panel_x + panel_width / 2:.2f}" y="565" text-anchor="middle" font-family="Consolas, monospace" font-size="15">kx</text>
<text x="{right_panel_x + panel_width / 2:.2f}" y="565" text-anchor="middle" font-family="Consolas, monospace" font-size="15">kx</text>
<text x="24" y="{panel_y + panel_height / 2:.2f}" transform="rotate(-90 24,{panel_y + panel_height / 2:.2f})" text-anchor="middle" font-family="Consolas, monospace" font-size="15">ky</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def build_real_space_patch(
    basis_positions: np.ndarray,
    labels: list[str],
    colors: list[str],
    radii: list[float],
    neighbor_terms: list[tuple[int, int, int, int, float, float]],
    super_a1: np.ndarray,
    super_a2: np.ndarray,
    nx: int,
    ny: int,
) -> tuple[list[list[float]], list[str], list[float], list[tuple[int, int, float, str]], list[str], list[list[float]]]:
    positions_3d: list[list[float]] = []
    patch_colors: list[str] = []
    patch_radii: list[float] = []
    index_map: dict[tuple[int, int, int], int] = {}

    for iy in range(max(1, ny)):
        for ix in range(max(1, nx)):
            cell_shift = ix * super_a1 + iy * super_a2
            for atom_index, point in enumerate(basis_positions):
                index_map[(ix, iy, atom_index)] = len(positions_3d)
                positions_3d.append([float(point[0] + cell_shift[0]), float(point[1] + cell_shift[1]), 0.0])
                patch_colors.append(colors[atom_index])
                patch_radii.append(radii[atom_index])

    max_hopping = max(term[4] for term in neighbor_terms)
    bonds: list[tuple[int, int, float, str]] = []
    for iy in range(max(1, ny)):
        for ix in range(max(1, nx)):
            for left_index, right_index, shift_1, shift_2, hopping, distance in neighbor_terms:
                target_x = ix + shift_1
                target_y = iy + shift_2
                if not (0 <= target_x < max(1, nx) and 0 <= target_y < max(1, ny)):
                    continue
                start = index_map[(ix, iy, left_index)]
                end = index_map[(target_x, target_y, right_index)]
                radius = 0.03 + 0.05 * (hopping / max_hopping)
                color = "#f28e2b" if distance < 0.95 else "#666666"
                bonds.append((start, end, radius, color))

    label_positions = [positions_3d[index_map[(0, 0, atom_index)]] for atom_index in range(len(labels))]
    return positions_3d, patch_colors, patch_radii, bonds, labels, label_positions


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
    for start_index, end_index, radius, color in bonds:
        start = array_positions[start_index]
        end = array_positions[end_index]
        x0, y0 = project(start)
        x1, y1 = project(end)
        stroke_width = 1.6 + 26.0 * float(radius)
        bond_lines.append(
            f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="{color}" stroke-width="{stroke_width:.2f}" />'
        )

    atom_circles = []
    for point, color in zip(array_positions, colors):
        x, y = project(point)
        atom_circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="10.5" fill="{color}" stroke="black" stroke-width="1.2" />'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="white" />
<text x="{width / 2:.2f}" y="30" text-anchor="middle" font-family="Consolas, monospace" font-size="22">Distorted AB-pair graphene real-space patch</text>
<text x="{left}" y="{height - 20}" font-family="Consolas, monospace" font-size="14">Orange: shortened AB pair bond   Gray: other nearest-neighbor bonds</text>
{''.join(bond_lines)}
{''.join(atom_circles)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def unique_bond_summary(neighbor_terms: list[tuple[int, int, int, int, float, float]]) -> list[tuple[float, float]]:
    rounded: dict[tuple[float, float], None] = {}
    for _, _, _, _, hopping, distance in neighbor_terms:
        key = (round(distance, 6), round(hopping, 6))
        rounded[key] = None
    return sorted(rounded.keys())


def direct_gap(reciprocal_rows: list[tuple[float, float, float, float]]) -> float:
    if not reciprocal_rows:
        return 0.0
    return min(conduction - valence for _, _, valence, conduction in reciprocal_rows)


def write_model_summary(
    path: Path,
    pair_shift: float,
    t0: float,
    beta: float,
    super_a1: np.ndarray,
    super_a2: np.ndarray,
    neighbor_terms: list[tuple[int, int, int, int, float, float]],
    gap_value: float,
) -> None:
    bond_lines = []
    for distance, hopping in unique_bond_summary(neighbor_terms):
        bond_lines.append(f"d = {distance:.6f}, t(d) = {hopping:.6f}")

    text = f"""Distorted AB-pair graphene cluster model
--------------------------------------
Interpretation used for this project:
- 6-site supercell with labels A, A', B, B', C, C'
- coordinates are explicitly distorted in real space
- each original A-B pair is pulled together inside the enlarged cell
- onsite energies are all equal to zero
- nearest-neighbor hopping only
- hopping is generated from the distorted bond length through
  t(d) = t0 * exp[-beta * (d/a0 - 1)]

pair_shift = {pair_shift:.6f}
t0 = {t0:.6f}
beta = {beta:.6f}

Supercell vectors:
T1 = ({super_a1[0]:.6f}, {super_a1[1]:.6f})
T2 = ({super_a2[0]:.6f}, {super_a2[1]:.6f})

6x6 Bloch Hamiltonian:
H_ij(k) = sum_R t_ij(R) exp[i k dot (R + r_j - r_i)]

Unique nearest-neighbor distances / hoppings:
{chr(10).join(bond_lines)}

Direct gap from reciprocal-space grid = {gap_value:.6f}
"""
    path.write_text(text, encoding="utf-8")


def write_status_note(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(line.rstrip() for line in lines).rstrip() + "\n", encoding="utf-8")


def write_report_html(
    path: Path,
    pair_shift: float,
    gap_value: float,
    real_space_interactive: bool,
    reciprocal_interactive: bool,
) -> None:
    real_space_block = (
        '<iframe src="real_space_interactive.html" title="Interactive distorted graphene view" '
        'style="width: 100%; height: 620px; border: 1px solid #d0d0d0; border-radius: 10px;"></iframe>'
        if real_space_interactive
        else '<div style="padding: 24px; border: 1px solid #d0d0d0; border-radius: 10px; background: #fafafa;">'
             '<strong>Interactive real-space view was not generated.</strong><br/>'
             'See <code>interactive_status.txt</code> for the dependency/runtime reason.'
             '</div>'
    )
    reciprocal_block = (
        '<iframe src="reciprocal_space_interactive.html" title="Interactive reciprocal-space view" '
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
  <title>Distorted AB-Pair Graphene Report</title>
</head>
<body style="margin: 24px; font-family: Consolas, monospace; background: white; color: #111;">
  <h1 style="margin-bottom: 8px;">Distorted AB-Pair Graphene Report</h1>
  <p style="margin-top: 0;">6-site unit cell with labels A, A', B, B', C, C'. Coordinates are explicitly distorted so that each AB pair is pulled together. pair_shift = {pair_shift:.6f}, direct gap = {gap_value:.6f}.</p>
  <h2>Interactive Real Space</h2>
  {real_space_block}
  <h2 style="margin-top: 28px;">Static Real Space</h2>
  <img src="real_space.svg" alt="Distorted AB-pair graphene real-space patch" style="max-width: 100%; border: 1px solid #d0d0d0; border-radius: 10px;" />
  <h2 style="margin-top: 28px;">Band Structure (G-K-M-G)</h2>
  <img src="band_structure.svg" alt="Distorted graphene band structure" style="max-width: 100%; border: 1px solid #d0d0d0; border-radius: 10px;" />
  <h2 style="margin-top: 28px;">2D Reciprocal Space</h2>
  <img src="reciprocal_space_map.svg" alt="Distorted graphene reciprocal-space map" style="max-width: 100%; border: 1px solid #d0d0d0; border-radius: 10px;" />
  <h2 style="margin-top: 28px;">Interactive 3D Reciprocal Space</h2>
  {reciprocal_block}
  <h2 style="margin-top: 28px;">Density of States</h2>
  <img src="dos.svg" alt="Distorted graphene DOS" style="max-width: 100%; border: 1px solid #d0d0d0; border-radius: 10px;" />
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=3.37)
    parser.add_argument("--pair-shift", type=float, default=0.12)
    parser.add_argument("--samples-per-segment", type=int, default=120)
    parser.add_argument("--dos-grid", type=int, default=140)
    parser.add_argument("--map-grid", type=int, default=85)
    parser.add_argument("--energy-points", type=int, default=420)
    parser.add_argument("--broadening", type=float, default=0.07)
    parser.add_argument("--nx", type=int, default=3)
    parser.add_argument("--ny", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    structure_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve() if args.out_dir else structure_dir / "outputs"
    ensure_dir(out_dir)

    super_a1, super_a2, basis_positions, labels, colors, radii = build_distorted_supercell(args.pair_shift)
    neighbor_terms = build_neighbor_terms(basis_positions, super_a1, super_a2, args.t0, args.beta)
    band_fn = lambda k_point: distorted_bands(k_point, basis_positions, neighbor_terms, super_a1, super_a2)

    b1, b2 = reciprocal_vectors(super_a1, super_a2)
    gamma = np.array([0.0, 0.0], dtype=float)
    k_point = (b1 - b2) / 3.0
    m_point = 0.5 * b1
    k_path = [("G", gamma), ("K", k_point), ("M", m_point), ("G", gamma)]

    distances, bands, tick_positions, tick_labels = compute_band_path(
        band_fn=band_fn,
        k_path=k_path,
        samples_per_segment=args.samples_per_segment,
    )

    max_hopping = max(term[4] for term in neighbor_terms)
    energy_limit = 3.8 * max(max_hopping, 1.0)
    energy_axis = np.linspace(-energy_limit, energy_limit, max(120, args.energy_points))
    dos = compute_dos(
        band_fn=band_fn,
        b1=b1,
        b2=b2,
        grid_n=args.dos_grid,
        energy_axis=energy_axis,
        broadening=args.broadening,
    )

    polygon = bz_hexagon_vertices(b1, b2)
    reciprocal_rows = compute_reciprocal_map(
        band_fn=band_fn,
        polygon=polygon,
        grid_n=args.map_grid,
    )
    gap_value = direct_gap(reciprocal_rows)

    patch_positions, patch_colors, patch_radii, patch_bonds, point_labels, label_positions = build_real_space_patch(
        basis_positions=basis_positions,
        labels=labels,
        colors=colors,
        radii=radii,
        neighbor_terms=neighbor_terms,
        super_a1=super_a1,
        super_a2=super_a2,
        nx=args.nx,
        ny=args.ny,
    )

    real_space_svg = out_dir / "real_space.svg"
    real_space_html = out_dir / "real_space_interactive.html"
    summary_txt = out_dir / "model_summary.txt"
    band_csv = out_dir / "band_structure.csv"
    band_svg = out_dir / "band_structure.svg"
    dos_csv = out_dir / "dos.csv"
    dos_svg = out_dir / "dos.svg"
    reciprocal_csv = out_dir / "reciprocal_space_map.csv"
    reciprocal_svg = out_dir / "reciprocal_space_map.svg"
    reciprocal_html = out_dir / "reciprocal_space_interactive.html"
    report_html = out_dir / "report.html"
    interactive_status = out_dir / "interactive_status.txt"

    write_real_space_svg(real_space_svg, patch_positions, patch_colors, patch_bonds)
    write_model_summary(summary_txt, args.pair_shift, args.t0, args.beta, super_a1, super_a2, neighbor_terms, gap_value)
    write_band_csv(band_csv, distances, bands)
    write_band_svg(band_svg, "Distorted AB-pair graphene band structure", distances, bands, tick_positions, tick_labels)
    write_dos_csv(dos_csv, energy_axis, dos)
    write_dos_svg(dos_svg, "Distorted AB-pair graphene DOS", energy_axis, dos)
    write_reciprocal_csv(reciprocal_csv, reciprocal_rows)
    write_reciprocal_svg(
        reciprocal_svg,
        "Distorted AB-pair graphene reciprocal-space map",
        polygon,
        reciprocal_rows,
        k_path,
        energy_limit,
    )

    interactive_ok, interactive_reason = pyvista_is_available(out_dir)
    real_space_interactive_ok = False
    reciprocal_interactive_ok = False
    status_lines: list[str] = []

    if interactive_ok:
        try:
            export_atomic_scene_html(
                output_path=real_space_html,
                title="distorted AB-pair graphene real-space patch",
                atom_positions=patch_positions,
                atom_colors=patch_colors,
                atom_radii=patch_radii,
                bonds=patch_bonds,
                point_labels=point_labels,
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
                title="distorted AB-pair graphene reciprocal-space surfaces",
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
        pair_shift=args.pair_shift,
        gap_value=gap_value,
        real_space_interactive=real_space_interactive_ok,
        reciprocal_interactive=reciprocal_interactive_ok,
    )

    k_bands = band_fn(k_point)
    print(f"Output directory: {out_dir}")
    print(f"[structure] {real_space_svg}")
    print(f"[3d html]   {real_space_html}")
    print(f"[summary]   {summary_txt}")
    print(f"[band csv]  {band_csv}")
    print(f"[band svg]  {band_svg}")
    print(f"[dos csv]   {dos_csv}")
    print(f"[dos svg]   {dos_svg}")
    print(f"[k-map csv] {reciprocal_csv}")
    print(f"[k-map svg] {reciprocal_svg}")
    print(f"[k-map 3d]  {reciprocal_html}")
    print(f"[report]    {report_html}")
    print(f"[3d status] {interactive_status}")
    print(f"Band range:  {bands.min():.4f} .. {bands.max():.4f}")
    print(f"DOS max:     {dos.max():.4f}")
    print(f"Direct gap:  {gap_value:.6f}")
    print(f"K valence/conduction: {valence_conduction_pair(k_bands)[0]:.6f}, {valence_conduction_pair(k_bands)[1]:.6f}")


if __name__ == "__main__":
    main()
