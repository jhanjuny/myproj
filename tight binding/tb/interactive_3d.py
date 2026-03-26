from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _ensure_runtime_cache(cache_root: Path) -> None:
    mpl_dir = cache_root / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def pyvista_is_available(cache_root: Path) -> tuple[bool, str]:
    _ensure_runtime_cache(cache_root)
    try:
        import pyvista  # noqa: F401
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def export_atomic_scene_html(
    output_path: Path,
    title: str,
    atom_positions: Sequence[Sequence[float]],
    atom_colors: Sequence[str],
    atom_radii: Sequence[float],
    bonds: Sequence[tuple[int, int, float, str]],
    point_labels: Sequence[str] | None = None,
    label_positions: Sequence[Sequence[float]] | None = None,
    window_size: tuple[int, int] = (1280, 860),
) -> None:
    cache_root = output_path.parent / ".runtime_cache"
    _ensure_runtime_cache(cache_root)

    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")

    positions = np.asarray(atom_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("atom_positions must be shaped (N, 3)")
    if not (len(atom_colors) == len(atom_radii) == len(positions)):
        raise ValueError("atom colors/radii must match atom count")

    for position, color, radius in zip(positions, atom_colors, atom_radii):
        sphere = pv.Sphere(radius=float(radius), center=tuple(position), theta_resolution=36, phi_resolution=36)
        plotter.add_mesh(
            sphere,
            color=color,
            smooth_shading=True,
            specular=0.15,
            name=f"atom-{position[0]:.3f}-{position[1]:.3f}-{position[2]:.3f}",
        )

    for start_index, end_index, radius, color in bonds:
        start = positions[int(start_index)]
        end = positions[int(end_index)]
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 1e-12:
            continue
        center = 0.5 * (start + end)
        cylinder = pv.Cylinder(
            center=tuple(center),
            direction=tuple(vector / length),
            radius=float(radius),
            height=length,
            resolution=36,
        )
        plotter.add_mesh(cylinder, color=color, smooth_shading=True, specular=0.1)

    if point_labels and label_positions:
        label_points = np.asarray(label_positions, dtype=float)
        plotter.add_point_labels(
            label_points,
            list(point_labels),
            point_size=0,
            font_size=18,
            always_visible=True,
            shape_opacity=0.15,
            text_color="black",
            fill_shape=True,
            shape_color="white",
        )

    plotter.add_axes(line_width=2, labels_off=False)
    plotter.add_text(title, position="upper_edge", font_size=14, color="black")
    plotter.view_isometric()
    plotter.camera.zoom(1.2)
    plotter.export_html(str(output_path))
    plotter.close()
