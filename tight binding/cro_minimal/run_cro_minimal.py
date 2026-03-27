from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tb.export import ensure_dir, write_band_csv, write_band_svg, write_dos_csv, write_dos_svg
from tb.interactive_3d import export_atomic_scene_html, export_reciprocal_surfaces_html, pyvista_is_available
from tb.kpath import sample_k_path


def lattice_matrix_from_lengths_angles(
    a: float,
    b: float,
    c: float,
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
) -> np.ndarray:
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)
    vector_a = np.array([a, 0.0, 0.0], dtype=float)
    vector_b = np.array([b * math.cos(gamma), b * math.sin(gamma), 0.0], dtype=float)
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / max(math.sin(gamma), 1e-12)
    cz_sq = max(c * c - cx * cx - cy * cy, 0.0)
    vector_c = np.array([cx, cy, math.sqrt(cz_sq)], dtype=float)
    return np.column_stack([vector_a, vector_b, vector_c])


def reciprocal_vectors(lattice_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reciprocal = 2.0 * math.pi * np.linalg.inv(lattice_matrix).T
    return reciprocal[:, 0], reciprocal[:, 1], reciprocal[:, 2]


def parse_float_tag(lines: list[str], tag: str) -> float:
    for line in lines:
        if line.startswith(tag):
            return float(line.split()[-1].strip("\"'"))
    raise KeyError(f"Missing CIF tag: {tag}")


def parse_string_tag(lines: list[str], tag: str, default: str = "") -> str:
    for line in lines:
        if line.startswith(tag):
            return line.split(maxsplit=1)[1].strip().strip("\"'")
    return default


def parse_fraction_or_float(token: str) -> float:
    token = token.strip()
    if "/" in token:
        left, right = token.split("/", maxsplit=1)
        return float(Fraction(int(left), int(right)))
    return float(token)


def eval_symmetry_coord(expr: str, x: float, y: float, z: float) -> float:
    cleaned = expr.replace(" ", "")
    total = 0.0
    for token in re.findall(r"[+-]?[^+-]+", cleaned):
        sign = 1.0
        if token.startswith("+"):
            token = token[1:]
        elif token.startswith("-"):
            sign = -1.0
            token = token[1:]
        if token == "x":
            value = x
        elif token == "y":
            value = y
        elif token == "z":
            value = z
        else:
            value = parse_fraction_or_float(token)
        total += sign * value
    total %= 1.0
    if abs(total) < 1e-10 or abs(total - 1.0) < 1e-10:
        return 0.0
    return total


def read_cif_structure(
    path: Path,
) -> tuple[np.ndarray, str, list[str], list[str], np.ndarray, np.ndarray]:
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]
    lattice = lattice_matrix_from_lengths_angles(
        a=parse_float_tag(lines, "_cell_length_a"),
        b=parse_float_tag(lines, "_cell_length_b"),
        c=parse_float_tag(lines, "_cell_length_c"),
        alpha_deg=parse_float_tag(lines, "_cell_angle_alpha"),
        beta_deg=parse_float_tag(lines, "_cell_angle_beta"),
        gamma_deg=parse_float_tag(lines, "_cell_angle_gamma"),
    )
    space_group = parse_string_tag(lines, "_symmetry_space_group_name_H-M", default="unknown")

    symmetry_ops: list[str] = []
    for index, line in enumerate(lines):
        if line.strip() == "_space_group_symop_operation_xyz":
            for row in lines[index + 1 :]:
                stripped = row.strip()
                if not stripped or stripped.startswith("loop_") or stripped.startswith("_"):
                    break
                parts = row.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    symmetry_ops.append(parts[1].strip())
            break

    atom_rows: list[str] = []
    for index, line in enumerate(lines):
        if line.strip() == "_atom_site_fract_symmform":
            for row in lines[index + 1 :]:
                stripped = row.strip()
                if not stripped or stripped.startswith("#"):
                    break
                atom_rows.append(stripped)
            break

    labels: list[str] = []
    species: list[str] = []
    frac_positions: list[np.ndarray] = []
    for row in atom_rows:
        parts = row.split()
        base_label = parts[0]
        atom_type = parts[1]
        x, y, z = map(float, parts[4:7])
        images_for_this_atom: list[np.ndarray] = []
        for op in symmetry_ops:
            expr_x, expr_y, expr_z = op.split(",")
            candidate = np.array(
                [
                    eval_symmetry_coord(expr_x, x, y, z),
                    eval_symmetry_coord(expr_y, x, y, z),
                    eval_symmetry_coord(expr_z, x, y, z),
                ],
                dtype=float,
            )
            if any(np.allclose(candidate, existing, atol=1e-8) for existing in images_for_this_atom):
                continue
            images_for_this_atom.append(candidate)
            labels.append(f"{base_label}_{len(images_for_this_atom)}")
            species.append(atom_type)
            frac_positions.append(candidate)

    frac_array = np.vstack(frac_positions)
    cart_array = (lattice @ frac_array.T).T
    return lattice, space_group, labels, species, frac_array, cart_array


def build_neighbor_terms(
    species: list[str],
    cart_positions: np.ndarray,
    lattice_matrix: np.ndarray,
    cutoff: float,
    t0: float,
    beta: float,
) -> tuple[list[tuple[int, int, tuple[int, int, int], float, float]], float]:
    lattice_vectors = [lattice_matrix[:, 0], lattice_matrix[:, 1], lattice_matrix[:, 2]]
    translations = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    shortest = float("inf")

    for left_index, left_species in enumerate(species):
        if left_species != "Cr":
            continue
        for right_index, right_species in enumerate(species):
            if right_species != "O":
                continue
            for n1, n2, n3 in translations:
                delta = (
                    cart_positions[right_index]
                    + n1 * lattice_vectors[0]
                    + n2 * lattice_vectors[1]
                    + n3 * lattice_vectors[2]
                    - cart_positions[left_index]
                )
                distance = float(np.linalg.norm(delta))
                if distance > 1e-8 and distance < shortest:
                    shortest = distance

    if not math.isfinite(shortest):
        raise RuntimeError("Could not determine the shortest Cr-O bond length from the CIF structure.")

    edges: list[tuple[int, int, tuple[int, int, int], float, float]] = []
    for left_index, left_species in enumerate(species):
        if left_species != "Cr":
            continue
        for right_index, right_species in enumerate(species):
            if right_species != "O":
                continue
            for n1, n2, n3 in translations:
                delta = (
                    cart_positions[right_index]
                    + n1 * lattice_vectors[0]
                    + n2 * lattice_vectors[1]
                    + n3 * lattice_vectors[2]
                    - cart_positions[left_index]
                )
                distance = float(np.linalg.norm(delta))
                if distance <= cutoff + 1e-9:
                    hopping = float(t0) * math.exp(-float(beta) * (distance / shortest - 1.0))
                    edges.append((left_index, right_index, (n1, n2, n3), hopping, distance))

    if not edges:
        raise RuntimeError("No Cr-O nearest-neighbor edges were found inside the chosen cutoff.")

    edges.sort(key=lambda item: (item[0], item[1], item[2]))
    return edges, shortest


def onsite_vector(species: list[str], onsite_cr: float, onsite_o: float) -> np.ndarray:
    return np.asarray([onsite_cr if atom == "Cr" else onsite_o for atom in species], dtype=float)


def bloch_hamiltonian(
    k_point: np.ndarray,
    onsite: np.ndarray,
    cart_positions: np.ndarray,
    lattice_matrix: np.ndarray,
    edges: list[tuple[int, int, tuple[int, int, int], float, float]],
) -> np.ndarray:
    matrix = np.diag(onsite.astype(complex))
    a1, a2, a3 = lattice_matrix[:, 0], lattice_matrix[:, 1], lattice_matrix[:, 2]
    for left_index, right_index, (n1, n2, n3), hopping, _distance in edges:
        shift = n1 * a1 + n2 * a2 + n3 * a3
        displacement = cart_positions[right_index] + shift - cart_positions[left_index]
        phase = np.exp(1j * float(np.dot(k_point, displacement)))
        matrix[left_index, right_index] += -hopping * phase
        matrix[right_index, left_index] += -hopping * np.conjugate(phase)
    return matrix


def band_values(
    k_point: np.ndarray,
    onsite: np.ndarray,
    cart_positions: np.ndarray,
    lattice_matrix: np.ndarray,
    edges: list[tuple[int, int, tuple[int, int, int], float, float]],
    energy_shift: float,
) -> np.ndarray:
    values = np.linalg.eigvalsh(bloch_hamiltonian(k_point, onsite, cart_positions, lattice_matrix, edges)).real
    return values - float(energy_shift)


def valence_conduction_pair(eigenvalues: np.ndarray) -> tuple[float, float]:
    ordered = np.sort(np.asarray(eigenvalues, dtype=float))
    half = len(ordered) // 2
    return float(ordered[half - 1]), float(ordered[half])


def compute_reference_energy_and_gap(
    onsite: np.ndarray,
    cart_positions: np.ndarray,
    lattice_matrix: np.ndarray,
    edges: list[tuple[int, int, tuple[int, int, int], float, float]],
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
    grid_n: int,
) -> tuple[float, float, np.ndarray]:
    samples = max(10, int(grid_n))
    u_values = np.linspace(0.0, 1.0, samples, endpoint=False)
    v_values = np.linspace(0.0, 1.0, samples, endpoint=False)
    w_values = np.linspace(0.0, 1.0, samples, endpoint=False)
    valence_max = -float("inf")
    conduction_min = float("inf")
    all_bands: list[np.ndarray] = []

    for u in u_values:
        for v in v_values:
            for w in w_values:
                k_point = u * b1 + v * b2 + w * b3
                eigenvalues = np.linalg.eigvalsh(
                    bloch_hamiltonian(k_point, onsite, cart_positions, lattice_matrix, edges)
                ).real
                all_bands.append(eigenvalues)
                valence, conduction = valence_conduction_pair(eigenvalues)
                valence_max = max(valence_max, valence)
                conduction_min = min(conduction_min, conduction)

    energy_shift = 0.5 * (valence_max + conduction_min)
    gap = conduction_min - valence_max
    return float(energy_shift), float(gap), np.vstack(all_bands)


def compute_dos(
    all_eigenvalues: np.ndarray,
    energy_axis: np.ndarray,
    broadening: float,
    energy_shift: float,
) -> np.ndarray:
    shifted = np.asarray(all_eigenvalues, dtype=float).ravel() - float(energy_shift)
    sigma = max(1e-4, float(broadening))
    diff = energy_axis[:, None] - shifted[None, :]
    gaussian = np.exp(-0.5 * (diff / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    return gaussian.mean(axis=1)


def compute_band_path(
    band_fn,
    k_path: list[tuple[str, np.ndarray]],
    samples_per_segment: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    k_points, distances, tick_positions, tick_labels = sample_k_path(k_path, samples_per_segment)
    bands = np.vstack([band_fn(k_point) for k_point in k_points])
    return distances, bands, tick_positions, tick_labels


def compute_slice_map(
    band_fn,
    bx: float,
    by: float,
    kz_value: float,
    grid_n: int,
) -> list[tuple[float, float, float, float]]:
    samples = max(35, int(grid_n))
    kx_values = np.linspace(-0.5 * bx, 0.5 * bx, samples)
    ky_values = np.linspace(-0.5 * by, 0.5 * by, samples)
    rows: list[tuple[float, float, float, float]] = []
    for ky in ky_values:
        for kx in kx_values:
            values = band_fn(np.array([kx, ky, kz_value], dtype=float))
            valence, conduction = valence_conduction_pair(values)
            rows.append((kx, ky, valence, conduction))
    return rows


def write_slice_csv(path: Path, rows: list[tuple[float, float, float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kx", "ky", "top_valence", "bottom_conduction"])
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


def write_slice_svg(
    path: Path,
    title: str,
    rows: list[tuple[float, float, float, float]],
    bx: float,
    by: float,
    path_points: list[tuple[str, np.ndarray]],
    energy_limit: float,
) -> None:
    width = 1180
    height = 580
    panel_width = 430
    panel_height = 430
    left_margin = 80
    panel_gap = 120
    top_margin = 70
    x_min = -0.5 * bx
    x_max = 0.5 * bx
    y_min = -0.5 * by
    y_max = 0.5 * by
    left_panel_x = left_margin
    right_panel_x = left_margin + panel_width + panel_gap
    panel_y = top_margin

    def project(point: np.ndarray, panel_x: float) -> tuple[float, float]:
        return (
            _scale(float(point[0]), x_min, x_max, panel_x, panel_x + panel_width),
            _scale(float(point[1]), y_min, y_max, panel_y + panel_height, panel_y),
        )

    sample_count = max(2, int(round(math.sqrt(len(rows)))))
    dx = (x_max - x_min) / sample_count
    dy = (y_max - y_min) / sample_count
    lower_tiles: list[str] = []
    upper_tiles: list[str] = []

    for kx, ky, valence, conduction in rows:
        x0, y0 = project(np.array([kx - 0.5 * dx, ky - 0.5 * dy]), left_panel_x)
        x1, y1 = project(np.array([kx + 0.5 * dx, ky + 0.5 * dy]), left_panel_x)
        lower_tiles.append(
            f'<rect x="{min(x0, x1):.2f}" y="{min(y0, y1):.2f}" width="{abs(x1 - x0):.2f}" height="{abs(y1 - y0):.2f}" '
            f'fill="{energy_color(valence, -energy_limit, 0.0)}" stroke="none" />'
        )
        x0r, y0r = project(np.array([kx - 0.5 * dx, ky - 0.5 * dy]), right_panel_x)
        x1r, y1r = project(np.array([kx + 0.5 * dx, ky + 0.5 * dy]), right_panel_x)
        upper_tiles.append(
            f'<rect x="{min(x0r, x1r):.2f}" y="{min(y0r, y1r):.2f}" width="{abs(x1r - x0r):.2f}" height="{abs(y1r - y0r):.2f}" '
            f'fill="{energy_color(conduction, 0.0, energy_limit)}" stroke="none" />'
        )

    square_vertices = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=float,
    )
    square_left = " ".join(f"{project(vertex, left_panel_x)[0]:.2f},{project(vertex, left_panel_x)[1]:.2f}" for vertex in square_vertices)
    square_right = " ".join(f"{project(vertex, right_panel_x)[0]:.2f},{project(vertex, right_panel_x)[1]:.2f}" for vertex in square_vertices)
    path_left = " ".join(f"{project(point[:2], left_panel_x)[0]:.2f},{project(point[:2], left_panel_x)[1]:.2f}" for _, point in path_points)
    path_right = " ".join(f"{project(point[:2], right_panel_x)[0]:.2f},{project(point[:2], right_panel_x)[1]:.2f}" for _, point in path_points)

    labels: list[str] = []
    for label, point in path_points[:-1]:
        lx, ly = project(point[:2], left_panel_x)
        rx, ry = project(point[:2], right_panel_x)
        labels.append(
            f'<circle cx="{lx:.2f}" cy="{ly:.2f}" r="4.5" fill="black" />'
            f'<text x="{lx + 8:.2f}" y="{ly - 8:.2f}" font-family="Consolas, monospace" font-size="14">{label}</text>'
        )
        labels.append(
            f'<circle cx="{rx:.2f}" cy="{ry:.2f}" r="4.5" fill="black" />'
            f'<text x="{rx + 8:.2f}" y="{ry - 8:.2f}" font-family="Consolas, monospace" font-size="14">{label}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="white" />
<text x="{width / 2:.2f}" y="30" text-anchor="middle" font-family="Consolas, monospace" font-size="22">{title}</text>
<text x="{left_panel_x + panel_width / 2:.2f}" y="56" text-anchor="middle" font-family="Consolas, monospace" font-size="16">Top valence band on k<sub>z</sub>=0 slice</text>
<text x="{right_panel_x + panel_width / 2:.2f}" y="56" text-anchor="middle" font-family="Consolas, monospace" font-size="16">Bottom conduction band on k<sub>z</sub>=0 slice</text>
<rect x="{left_panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" fill="none" stroke="black" stroke-width="1.5" />
<rect x="{right_panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" fill="none" stroke="black" stroke-width="1.5" />
{''.join(lower_tiles)}
{''.join(upper_tiles)}
<polygon points="{square_left}" fill="none" stroke="#202020" stroke-width="2" />
<polygon points="{square_right}" fill="none" stroke="#202020" stroke-width="2" />
<polyline points="{path_left}" fill="none" stroke="#111111" stroke-width="2" stroke-dasharray="6,5" />
<polyline points="{path_right}" fill="none" stroke="#111111" stroke-width="2" stroke-dasharray="6,5" />
{''.join(labels)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def build_real_space_scene(
    labels: list[str],
    species: list[str],
    cart_positions: np.ndarray,
    lattice_matrix: np.ndarray,
    edges: list[tuple[int, int, tuple[int, int, int], float, float]],
) -> tuple[np.ndarray, list[str], list[float], list[tuple[int, int, float, str]], list[str], np.ndarray]:
    display_keys: list[tuple[int, tuple[int, int, int]]] = [(index, (0, 0, 0)) for index in range(len(labels))]
    for _left, right, shift, _hopping, _distance in edges:
        key = (right, shift)
        if key not in display_keys:
            display_keys.append(key)

    key_to_index = {key: idx for idx, key in enumerate(display_keys)}
    a1, a2, a3 = lattice_matrix[:, 0], lattice_matrix[:, 1], lattice_matrix[:, 2]

    positions: list[np.ndarray] = []
    colors: list[str] = []
    radii: list[float] = []
    display_labels: list[str] = []
    for atom_index, shift in display_keys:
        n1, n2, n3 = shift
        position = cart_positions[atom_index] + n1 * a1 + n2 * a2 + n3 * a3
        positions.append(position)
        atom_species = species[atom_index]
        colors.append("#3a78d6" if atom_species == "Cr" else "#d64a3a")
        radii.append(0.26 if atom_species == "Cr" else 0.21)
        display_labels.append(
            labels[atom_index] if shift == (0, 0, 0) else f"{labels[atom_index]}[{n1},{n2},{n3}]"
        )

    bonds: list[tuple[int, int, float, str]] = []
    for left, right, shift, _hopping, distance in edges:
        left_draw = key_to_index[(left, (0, 0, 0))]
        right_draw = key_to_index[(right, shift)]
        bond_radius = 0.055 if distance <= 2.034 else 0.042
        bond_color = "#555555" if distance <= 2.034 else "#9a9a9a"
        bonds.append((left_draw, right_draw, bond_radius, bond_color))

    return np.vstack(positions), colors, radii, bonds, display_labels, np.vstack(positions)


def project_isometric(points: np.ndarray) -> np.ndarray:
    projection = np.array([[1.0, 0.45, 0.0], [-0.18, 0.22, -0.42]], dtype=float)
    return (projection @ points.T).T


def write_real_space_svg(
    path: Path,
    positions: np.ndarray,
    colors: list[str],
    bonds: list[tuple[int, int, float, str]],
    lattice_matrix: np.ndarray,
) -> None:
    projected = project_isometric(positions)
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            lattice_matrix[:, 0],
            lattice_matrix[:, 1],
            lattice_matrix[:, 2],
            lattice_matrix[:, 0] + lattice_matrix[:, 1],
            lattice_matrix[:, 0] + lattice_matrix[:, 2],
            lattice_matrix[:, 1] + lattice_matrix[:, 2],
            lattice_matrix[:, 0] + lattice_matrix[:, 1] + lattice_matrix[:, 2],
        ],
        dtype=float,
    )
    projected_corners = project_isometric(corners)
    all_points = np.vstack([projected, projected_corners])
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    width = 960
    height = 700
    margin = 90
    span = max(max_xy - min_xy)
    scale = (min(width, height) - 2 * margin) / max(span, 1e-6)

    def map_point(point: np.ndarray) -> tuple[float, float]:
        mapped = (point - min_xy) * scale
        return float(margin + mapped[0]), float(height - margin - mapped[1])

    cell_edges = [
        (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4),
        (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
    ]

    edge_lines: list[str] = []
    for start, end in cell_edges:
        x1, y1 = map_point(projected_corners[start])
        x2, y2 = map_point(projected_corners[end])
        edge_lines.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#b5b5b5" stroke-width="2" stroke-dasharray="6,5" />'
        )

    bond_lines: list[str] = []
    for start, end, radius, color in bonds:
        x1, y1 = map_point(projected[start])
        x2, y2 = map_point(projected[end])
        bond_lines.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{max(2.2, 60.0 * radius):.2f}" stroke-linecap="round" />'
        )

    atom_shapes: list[str] = []
    for atom_index in sorted(range(len(positions)), key=lambda idx: positions[idx, 2]):
        x, y = map_point(projected[atom_index])
        radius = 16 if colors[atom_index] == "#3a78d6" else 13
        atom_shapes.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{colors[atom_index]}" stroke="#1f1f1f" stroke-width="1.5" />'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="white" />
<text x="{width / 2:.2f}" y="32" text-anchor="middle" font-family="Consolas, monospace" font-size="22">CrO real-space unit cell and nearest Cr-O bonds</text>
{''.join(edge_lines)}
{''.join(bond_lines)}
{''.join(atom_shapes)}
<rect x="28" y="64" width="250" height="90" rx="12" fill="#fafafa" stroke="#d0d0d0" />
<circle cx="54" cy="96" r="10" fill="#3a78d6" stroke="#1f1f1f" stroke-width="1.2" />
<text x="72" y="101" font-family="Consolas, monospace" font-size="15">Cr site (effective d-like orbital)</text>
<circle cx="54" cy="128" r="8" fill="#d64a3a" stroke="#1f1f1f" stroke-width="1.2" />
<text x="72" y="133" font-family="Consolas, monospace" font-size="15">O site (effective p-like orbital)</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_model_summary(
    path: Path,
    cif_path: Path,
    space_group: str,
    lattice_matrix: np.ndarray,
    labels: list[str],
    species: list[str],
    edges: list[tuple[int, int, tuple[int, int, int], float, float]],
    reference_distance: float,
    onsite_cr: float,
    onsite_o: float,
    t0: float,
    beta: float,
    energy_shift: float,
    gap_value: float,
) -> None:
    unique_distances = sorted({round(edge[4], 6) for edge in edges})
    lines = [
        "CrO minimal structure-derived tight-binding model",
        f"Source CIF: {cif_path}",
        f"Space group: {space_group}",
        "",
        "Lattice matrix (Angstrom):",
    ]
    for row in lattice_matrix.T:
        lines.append("  " + "  ".join(f"{value:10.6f}" for value in row))
    lines.extend(
        [
            "",
            f"Orbitals in the basis: {len(labels)}",
            "One effective orbital is assigned to each crystallographic site.",
            f"Cr onsite energy: {onsite_cr:.6f}",
            f"O onsite energy:  {onsite_o:.6f}",
            f"Shortest Cr-O bond distance d0: {reference_distance:.6f} A",
            f"Cr-O edges included in the model: {len(edges)}",
            f"Distance-based hopping prefactor t0: {t0:.6f}",
            f"Distance decay beta: {beta:.6f}",
            "Unique Cr-O bond lengths in the model:",
        ]
    )
    for distance in unique_distances:
        lines.append(f"  {distance:.6f} A")
    lines.extend(
        [
            "",
            f"Presentation energy shift (mid-gap reference): {energy_shift:.6f}",
            f"Sampled direct gap: {gap_value:.6f}",
            "",
            "Site labels:",
        ]
    )
    for label, atom_species in zip(labels, species):
        lines.append(f"  {label}: {atom_species}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_formula_html(
    path: Path,
    labels: list[str],
    species: list[str],
    edges: list[tuple[int, int, tuple[int, int, int], float, float]],
    onsite_cr: float,
    onsite_o: float,
    reference_distance: float,
    beta: float,
    energy_shift: float,
    gap_value: float,
) -> None:
    edge_rows = []
    for left_index, right_index, (n1, n2, n3), hopping, distance in edges:
        edge_rows.append(
            "<tr>"
            f"<td>{labels[left_index]} ({species[left_index]})</td>"
            f"<td>{labels[right_index]} ({species[right_index]})</td>"
            f"<td>[{n1}, {n2}, {n3}]</td>"
            f"<td>{distance:.6f}</td>"
            f"<td>{hopping:.6f}</td>"
            "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CrO Minimal TB Formulas</title>
  <style>
    body {{ margin: 28px; font-family: Consolas, monospace; line-height: 1.55; color: #111; background: white; }}
    h1, h2 {{ color: #111; }}
    .equation {{ margin: 14px 0; padding: 14px 16px; background: #f6f8fb; border-left: 4px solid #4c78a8; }}
    .note {{ margin: 14px 0; padding: 14px 16px; background: #fafafa; border: 1px solid #dddddd; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #d8d8d8; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #f2f4f8; }}
  </style>
</head>
<body>
  <h1>CrO Minimal Structure-Derived Tight-Binding Formulas</h1>
  <p>This project is a <strong>minimal effective model</strong> extracted from the CIF geometry. It does <strong>not</strong> claim to be a fully fitted electronic-structure Hamiltonian. The rules used here are: one effective orbital per crystallographic site, Cr/O-dependent onsite energies, and nearest Cr-O hoppings that decay exponentially with bond length.</p>

  <h2>1. Basis and Bloch State</h2>
  <div class="equation">
    Basis size: N = {len(labels)}.<br/>
    |&psi;<sub>n</sub>(k)&rang; = &sum;<sub>&alpha;=1</sub><sup>{len(labels)}</sup> c<sub>&alpha;n</sub>(k) |&alpha;, k&rangle
  </div>
  <div class="note">
    Ordered basis:<br/>
    {", ".join(f"{label} ({atom})" for label, atom in zip(labels, species))}
  </div>

  <h2>2. Onsite Term</h2>
  <div class="equation">
    H<sub>onsite</sub> = diag(&epsilon;<sub>1</sub>, ..., &epsilon;<sub>{len(labels)}</sub>)<br/>
    &epsilon;<sub>Cr</sub> = {onsite_cr:.6f}, &nbsp; &epsilon;<sub>O</sub> = {onsite_o:.6f}
  </div>

  <h2>3. Distance-Dependent Cr-O Hopping</h2>
  <div class="equation">
    t<sub>ij</sub>(d<sub>ij</sub>) = t<sub>0</sub> exp[-&beta; (d<sub>ij</sub> / d<sub>0</sub> - 1)]
  </div>
  <div class="note">
    d<sub>0</sub> = {reference_distance:.6f} &Aring; (shortest Cr-O bond from the CIF)<br/>
    &beta; = {beta:.6f}
  </div>

  <h2>4. 8x8 Bloch Hamiltonian</h2>
  <div class="equation">
    H<sub>&alpha;&beta;</sub>(k) = &epsilon;<sub>&alpha;</sub> &delta;<sub>&alpha;&beta;</sub>
    + &sum;<sub>m</sub> t<sub>&alpha;&beta;</sub><sup>(m)</sup>
      exp[i k &middot; (&Delta;r<sub>&alpha;&beta;</sub><sup>(m)</sup>)] + h.c.
  </div>
  <div class="equation">
    &Delta;r<sub>&alpha;&beta;</sub><sup>(m)</sup> = r<sub>&beta;</sub> + R<sup>(m)</sup> - r<sub>&alpha;</sub>
  </div>

  <h2>5. Half-Filled Valence / Conduction Split</h2>
  <div class="equation">
    E<sub>1</sub>(k) &le; ... &le; E<sub>4</sub>(k) &le; E<sub>5</sub>(k) &le; ... &le; E<sub>8</sub>(k)<br/>
    E<sub>v</sub>(k) = E<sub>4</sub>(k), &nbsp; E<sub>c</sub>(k) = E<sub>5</sub>(k)
  </div>
  <div class="equation">
    E<sub>gap</sub><sup>dir</sup> = min<sub>k</sub> [E<sub>c</sub>(k)] - max<sub>k</sub> [E<sub>v</sub>(k)]
  </div>

  <h2>6. Mid-Gap Energy Reference Used in the Plots</h2>
  <div class="equation">
    E<sub>ref</sub> = 1/2 [max<sub>k</sub> E<sub>v</sub>(k) + min<sub>k</sub> E<sub>c</sub>(k)] = {energy_shift:.6f}
  </div>
  <div class="equation">
    &tilde;E<sub>n</sub>(k) = E<sub>n</sub>(k) - E<sub>ref</sub>
  </div>

  <h2>7. Brillouin-Zone Sampling and DOS</h2>
  <div class="equation">
    k(u,v,w) = u b<sub>1</sub> + v b<sub>2</sub> + w b<sub>3</sub>, &nbsp; u,v,w &in; [0,1)
  </div>
  <div class="equation">
    DOS(E) &approx; 1/N<sub>k</sub> &sum;<sub>k,n</sub>
    exp[-(E - &tilde;E<sub>n</sub>(k))<sup>2</sup> / 2&sigma;<sup>2</sup>] / (&sigma; &radic;(2&pi;))
  </div>

  <h2>8. k<sub>z</sub> = 0 Reciprocal-Space Slice</h2>
  <div class="equation">
    k = (k<sub>x</sub>, k<sub>y</sub>, 0), &nbsp;
    -|b<sub>1</sub>|/2 &le; k<sub>x</sub> &le; |b<sub>1</sub>|/2, &nbsp;
    -|b<sub>2</sub>|/2 &le; k<sub>y</sub> &le; |b<sub>2</sub>|/2
  </div>
  <div class="equation">
    The static and interactive reciprocal-space figures show E<sub>v</sub>(k<sub>x</sub>,k<sub>y</sub>,0) and E<sub>c</sub>(k<sub>x</sub>,k<sub>y</sub>,0).
  </div>

  <h2>9. Bond Table Used in This Run</h2>
  <table>
    <thead>
      <tr>
        <th>Left site</th>
        <th>Right site</th>
        <th>Lattice translation [n1,n2,n3]</th>
        <th>Distance (&Aring;)</th>
        <th>Hopping</th>
      </tr>
    </thead>
    <tbody>
      {''.join(edge_rows)}
    </tbody>
  </table>

  <div class="note">
    <strong>Current numerical result</strong><br/>
    Sampled direct gap = {gap_value:.6f}
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_status_note(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(line.rstrip() for line in lines).rstrip() + "\n", encoding="utf-8")


def write_report_html(
    path: Path,
    direct_gap: float,
    real_space_interactive: bool,
    reciprocal_interactive: bool,
) -> None:
    real_space_block = (
        '<iframe src="real_space_interactive.html" title="Interactive CrO real-space view" '
        'style="width: 100%; height: 660px; border: 1px solid #d0d0d0; border-radius: 10px;"></iframe>'
        if real_space_interactive
        else '<div style="padding: 24px; border: 1px solid #d0d0d0; border-radius: 10px; background: #fafafa;">'
             '<strong>Interactive real-space view was not generated.</strong><br/>'
             'See <code>interactive_status.txt</code> for the dependency/runtime reason.'
             '</div>'
    )
    reciprocal_block = (
        '<iframe src="reciprocal_space_interactive.html" title="Interactive CrO reciprocal-space view" '
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
  <title>CrO Minimal TB Report</title>
</head>
<body style="margin: 24px; font-family: Consolas, monospace; background: white; color: #111;">
  <h1 style="margin-bottom: 8px;">CrO Minimal Structure-Derived Tight-Binding Report</h1>
  <p style="margin-top: 0;">This report uses the supplied CIF file, expands the crystallographic symmetry to 4 Cr + 4 O sites, assigns one effective orbital to each site, and keeps only nearest Cr-O hoppings. The plotted direct gap from the sampled 3D Brillouin zone is {direct_gap:.6f}.</p>
  <h2>Interactive Real Space</h2>
  {real_space_block}
  <h2 style="margin-top: 28px;">Static Real Space</h2>
  <img src="real_space.svg" alt="CrO real-space structure" style="max-width: 100%; border: 1px solid #d0d0d0; border-radius: 10px;" />
  <h2 style="margin-top: 28px;">Band Structure (G-X-M-G-Z-R-A-Z)</h2>
  <img src="band_structure.svg" alt="CrO band structure" style="max-width: 100%; border: 1px solid #d0d0d0; border-radius: 10px;" />
  <h2 style="margin-top: 28px;">k<sub>z</sub> = 0 Reciprocal Slice</h2>
  <img src="reciprocal_space_map.svg" alt="CrO reciprocal-space slice" style="max-width: 100%; border: 1px solid #d0d0d0; border-radius: 10px;" />
  <h2 style="margin-top: 28px;">Interactive 3D Reciprocal Space</h2>
  {reciprocal_block}
  <h2 style="margin-top: 28px;">Calculation Formulas</h2>
  <p><a href="calculation_formulas.html">Open the formula file directly</a></p>
  <iframe src="calculation_formulas.html" title="CrO minimal TB formulas"
          style="width: 100%; height: 1650px; border: 1px solid #d0d0d0; border-radius: 10px;"></iframe>
  <h2 style="margin-top: 28px;">Density of States</h2>
  <img src="dos.svg" alt="CrO DOS" style="max-width: 100%; border: 1px solid #d0d0d0; border-radius: 10px;" />
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif", type=str, default="")
    parser.add_argument("--onsite-cr", type=float, default=1.2)
    parser.add_argument("--onsite-o", type=float, default=-1.2)
    parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--nn-cutoff", type=float, default=2.05)
    parser.add_argument("--samples-per-segment", type=int, default=90)
    parser.add_argument("--dos-grid", type=int, default=18)
    parser.add_argument("--slice-grid", type=int, default=55)
    parser.add_argument("--energy-points", type=int, default=360)
    parser.add_argument("--broadening", type=float, default=0.08)
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    structure_dir = Path(__file__).resolve().parent
    cif_path = Path(args.cif).resolve() if args.cif else structure_dir / "CrO_source.cif"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else structure_dir / "outputs"
    ensure_dir(out_dir)

    lattice_matrix, space_group, labels, species, frac_positions, cart_positions = read_cif_structure(cif_path)
    _ = frac_positions
    onsite = onsite_vector(species, args.onsite_cr, args.onsite_o)
    edges, reference_distance = build_neighbor_terms(
        species=species,
        cart_positions=cart_positions,
        lattice_matrix=lattice_matrix,
        cutoff=args.nn_cutoff,
        t0=args.t0,
        beta=args.beta,
    )

    b1, b2, b3 = reciprocal_vectors(lattice_matrix)
    energy_shift, gap_value, all_bands = compute_reference_energy_and_gap(
        onsite=onsite,
        cart_positions=cart_positions,
        lattice_matrix=lattice_matrix,
        edges=edges,
        b1=b1,
        b2=b2,
        b3=b3,
        grid_n=args.dos_grid,
    )

    band_fn = lambda k_point: band_values(
        k_point=k_point,
        onsite=onsite,
        cart_positions=cart_positions,
        lattice_matrix=lattice_matrix,
        edges=edges,
        energy_shift=energy_shift,
    )

    gamma = np.zeros(3, dtype=float)
    x_point = 0.5 * b1
    m_point = 0.5 * (b1 + b2)
    z_point = 0.5 * b3
    r_point = 0.5 * (b1 + b3)
    a_point = 0.5 * (b1 + b2 + b3)
    band_path = [("G", gamma), ("X", x_point), ("M", m_point), ("G", gamma), ("Z", z_point), ("R", r_point), ("A", a_point), ("Z", z_point)]
    distances, bands, tick_positions, tick_labels = compute_band_path(band_fn=band_fn, k_path=band_path, samples_per_segment=args.samples_per_segment)

    energy_limit = max(1.5, 1.15 * float(np.max(np.abs(bands))))
    energy_axis = np.linspace(-energy_limit, energy_limit, max(140, int(args.energy_points)))
    dos = compute_dos(all_eigenvalues=all_bands, energy_axis=energy_axis, broadening=args.broadening, energy_shift=energy_shift)

    bx = float(np.linalg.norm(b1))
    by = float(np.linalg.norm(b2))
    slice_rows = compute_slice_map(band_fn=band_fn, bx=bx, by=by, kz_value=0.0, grid_n=args.slice_grid)
    slice_path = [
        ("G", np.array([0.0, 0.0, 0.0], dtype=float)),
        ("X", np.array([0.5 * bx, 0.0, 0.0], dtype=float)),
        ("M", np.array([0.5 * bx, 0.5 * by, 0.0], dtype=float)),
        ("G", np.array([0.0, 0.0, 0.0], dtype=float)),
    ]

    scene_positions, scene_colors, scene_radii, scene_bonds, point_labels, label_positions = build_real_space_scene(
        labels=labels,
        species=species,
        cart_positions=cart_positions,
        lattice_matrix=lattice_matrix,
        edges=edges,
    )

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

    write_real_space_svg(real_space_svg, scene_positions, scene_colors, scene_bonds, lattice_matrix)
    write_model_summary(summary_txt, cif_path, space_group, lattice_matrix, labels, species, edges, reference_distance, args.onsite_cr, args.onsite_o, args.t0, args.beta, energy_shift, gap_value)
    write_formula_html(formula_html, labels, species, edges, args.onsite_cr, args.onsite_o, reference_distance, args.beta, energy_shift, gap_value)
    write_band_csv(band_csv, distances, bands)
    write_band_svg(band_svg, "CrO minimal structure-derived band structure", distances, bands, tick_positions, tick_labels)
    write_dos_csv(dos_csv, energy_axis, dos)
    write_dos_svg(dos_svg, "CrO minimal structure-derived DOS", energy_axis, dos)
    write_slice_csv(reciprocal_csv, slice_rows)
    write_slice_svg(reciprocal_svg, "CrO reciprocal-space slice (kz = 0)", slice_rows, bx, by, slice_path, energy_limit)

    interactive_ok, interactive_reason = pyvista_is_available(out_dir)
    real_space_interactive_ok = False
    reciprocal_interactive_ok = False
    status_lines: list[str] = []

    if interactive_ok:
        try:
            export_atomic_scene_html(
                output_path=real_space_html,
                title="CrO minimal real-space structure",
                atom_positions=scene_positions,
                atom_colors=scene_colors,
                atom_radii=scene_radii,
                bonds=scene_bonds,
                point_labels=point_labels,
                label_positions=label_positions.tolist(),
            )
            real_space_interactive_ok = True
            status_lines.append("Real-space interactive HTML generated successfully.")
        except Exception as exc:
            status_lines.append(f"Real-space interactive HTML export failed: {type(exc).__name__}: {exc}")

        try:
            reciprocal_points = np.array([[row[0], row[1]] for row in slice_rows], dtype=float)
            lower_energies = np.array([row[2] for row in slice_rows], dtype=float)
            upper_energies = np.array([row[3] for row in slice_rows], dtype=float)
            export_reciprocal_surfaces_html(
                output_path=reciprocal_html,
                title="CrO reciprocal-space surfaces on kz = 0",
                xy_points=reciprocal_points,
                lower_energies=lower_energies,
                upper_energies=upper_energies,
                polygon_vertices=np.array([[-0.5 * bx, -0.5 * by], [0.5 * bx, -0.5 * by], [0.5 * bx, 0.5 * by], [-0.5 * bx, 0.5 * by]], dtype=float),
                k_path=np.vstack([point[:2] for _, point in slice_path]),
                k_labels=[label for label, _ in slice_path],
                energy_limit=energy_limit,
            )
            reciprocal_interactive_ok = True
            status_lines.append("Reciprocal-space interactive HTML generated successfully.")
        except Exception as exc:
            status_lines.append(f"Reciprocal-space interactive HTML export failed: {type(exc).__name__}: {exc}")
    else:
        status_lines.append(f"PyVista unavailable: {interactive_reason}")

    write_status_note(interactive_status, status_lines)
    write_report_html(report_html, gap_value, real_space_interactive_ok, reciprocal_interactive_ok)

    gamma_bands = band_fn(gamma)
    gamma_valence, gamma_conduction = valence_conduction_pair(gamma_bands)
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
    print(f"Space group: {space_group}")
    print(f"Shortest Cr-O bond: {reference_distance:.6f} A")
    print(f"Included Cr-O edges: {len(edges)}")
    print(f"Band range (shifted): {bands.min():.6f} .. {bands.max():.6f}")
    print(f"Direct gap: {gap_value:.6f}")
    print(f"Gamma valence/conduction: {gamma_valence:.6f}, {gamma_conduction:.6f}")


if __name__ == "__main__":
    main()
