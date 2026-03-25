from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np


SVG_WIDTH = 900
SVG_HEIGHT = 560
LEFT_MARGIN = 80
RIGHT_MARGIN = 30
TOP_MARGIN = 40
BOTTOM_MARGIN = 70


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_band_csv(path: Path, distances: np.ndarray, bands: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["distance"] + [f"band_{index}" for index in range(bands.shape[1])]
        writer.writerow(header)
        for row_index in range(len(distances)):
            writer.writerow([f"{distances[row_index]:.8f}", *[f"{value:.8f}" for value in bands[row_index]]])


def write_dos_csv(path: Path, energies: np.ndarray, dos: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["energy", "dos"])
        for energy, density in zip(energies, dos):
            writer.writerow([f"{energy:.8f}", f"{density:.8f}"])


def _scale(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if domain_max == domain_min:
        return 0.5 * (range_min + range_max)
    alpha = (value - domain_min) / (domain_max - domain_min)
    return range_min + alpha * (range_max - range_min)


def write_band_svg(
    path: Path,
    title: str,
    distances: np.ndarray,
    bands: np.ndarray,
    tick_positions: np.ndarray,
    tick_labels: Sequence[str],
) -> None:
    x_min = float(distances.min())
    x_max = float(distances.max())
    y_min = float(bands.min())
    y_max = float(bands.max())
    padding = max(0.1, 0.08 * (y_max - y_min if y_max > y_min else 1.0))
    y_min -= padding
    y_max += padding

    plot_left = LEFT_MARGIN
    plot_right = SVG_WIDTH - RIGHT_MARGIN
    plot_top = TOP_MARGIN
    plot_bottom = SVG_HEIGHT - BOTTOM_MARGIN

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    lines: list[str] = []

    for band_index in range(bands.shape[1]):
        points = []
        for x_value, y_value in zip(distances, bands[:, band_index]):
            svg_x = _scale(float(x_value), x_min, x_max, plot_left, plot_right)
            svg_y = _scale(float(y_value), y_min, y_max, plot_bottom, plot_top)
            points.append(f"{svg_x:.2f},{svg_y:.2f}")
        color = colors[band_index % len(colors)]
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}" />'
        )

    ticks = []
    for position, label in zip(tick_positions, tick_labels):
        svg_x = _scale(float(position), x_min, x_max, plot_left, plot_right)
        ticks.append(
            f'<line x1="{svg_x:.2f}" y1="{plot_top}" x2="{svg_x:.2f}" y2="{plot_bottom}" '
            'stroke="#cccccc" stroke-width="1" />'
        )
        ticks.append(
            f'<text x="{svg_x:.2f}" y="{SVG_HEIGHT - 30}" text-anchor="middle" '
            'font-family="Consolas, monospace" font-size="16">{label}</text>'
        )

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

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">
<rect x="0" y="0" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="white" />
<text x="{SVG_WIDTH / 2:.2f}" y="24" text-anchor="middle" font-family="Consolas, monospace" font-size="20">{title}</text>
<rect x="{plot_left}" y="{plot_top}" width="{plot_right - plot_left}" height="{plot_bottom - plot_top}" fill="none" stroke="black" stroke-width="1.5" />
{''.join(y_ticks)}
{''.join(ticks)}
{''.join(lines)}
<text x="{SVG_WIDTH / 2:.2f}" y="{SVG_HEIGHT - 8}" text-anchor="middle" font-family="Consolas, monospace" font-size="16">k-path</text>
<text x="20" y="{SVG_HEIGHT / 2:.2f}" transform="rotate(-90 20,{SVG_HEIGHT / 2:.2f})" text-anchor="middle" font-family="Consolas, monospace" font-size="16">Energy</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_dos_svg(path: Path, title: str, energies: np.ndarray, dos: np.ndarray) -> None:
    x_min = float(energies.min())
    x_max = float(energies.max())
    y_min = 0.0
    y_max = float(dos.max())
    y_max = max(y_max, 1e-6)

    plot_left = LEFT_MARGIN
    plot_right = SVG_WIDTH - RIGHT_MARGIN
    plot_top = TOP_MARGIN
    plot_bottom = SVG_HEIGHT - BOTTOM_MARGIN

    points = []
    for energy, density in zip(energies, dos):
        svg_x = _scale(float(energy), x_min, x_max, plot_left, plot_right)
        svg_y = _scale(float(density), y_min, y_max, plot_bottom, plot_top)
        points.append(f"{svg_x:.2f},{svg_y:.2f}")

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

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">
<rect x="0" y="0" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="white" />
<text x="{SVG_WIDTH / 2:.2f}" y="24" text-anchor="middle" font-family="Consolas, monospace" font-size="20">{title}</text>
<rect x="{plot_left}" y="{plot_top}" width="{plot_right - plot_left}" height="{plot_bottom - plot_top}" fill="none" stroke="black" stroke-width="1.5" />
{''.join(y_ticks)}
<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{' '.join(points)}" />
<text x="{SVG_WIDTH / 2:.2f}" y="{SVG_HEIGHT - 8}" text-anchor="middle" font-family="Consolas, monospace" font-size="16">Energy</text>
<text x="20" y="{SVG_HEIGHT / 2:.2f}" transform="rotate(-90 20,{SVG_HEIGHT / 2:.2f})" text-anchor="middle" font-family="Consolas, monospace" font-size="16">DOS</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
