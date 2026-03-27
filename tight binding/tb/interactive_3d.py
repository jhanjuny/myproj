from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

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


def export_reciprocal_surfaces_html(
    output_path: Path,
    title: str,
    xy_points: Sequence[Sequence[float]],
    lower_energies: Sequence[float],
    upper_energies: Sequence[float],
    polygon_vertices: Sequence[Sequence[float]],
    k_path: Sequence[Sequence[float]],
    k_labels: Sequence[str],
    energy_limit: float,
    window_size: tuple[int, int] = (1380, 920),
) -> None:
    cache_root = output_path.parent / ".runtime_cache"
    _ensure_runtime_cache(cache_root)

    import pyvista as pv

    points_xy = np.asarray(xy_points, dtype=float)
    lower = np.asarray(lower_energies, dtype=float)
    upper = np.asarray(upper_energies, dtype=float)
    polygon = np.asarray(polygon_vertices, dtype=float)
    path_points = np.asarray(k_path, dtype=float)

    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("xy_points must be shaped (N, 2)")
    if not (len(points_xy) == len(lower) == len(upper)):
        raise ValueError("point and energy arrays must have the same length")

    base_points = np.column_stack([points_xy, np.zeros(len(points_xy), dtype=float)])
    base_cloud = pv.PolyData(base_points)
    triangulated = base_cloud.delaunay_2d(alpha=0.0, tol=1e-6, offset=1.0)
    triangulated["lower_energy"] = lower
    triangulated["upper_energy"] = upper

    lower_surface = triangulated.copy(deep=True)
    lower_surface.points = lower_surface.points.copy()
    lower_surface.points[:, 2] = lower_surface["lower_energy"]

    upper_surface = triangulated.copy(deep=True)
    upper_surface.points = upper_surface.points.copy()
    upper_surface.points[:, 2] = upper_surface["upper_energy"]

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")

    clim = (-float(energy_limit), float(energy_limit))
    plotter.add_mesh(
        lower_surface,
        scalars="lower_energy",
        cmap="coolwarm",
        clim=clim,
        smooth_shading=True,
        opacity=0.96,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Energy", "vertical": True},
    )
    plotter.add_mesh(
        upper_surface,
        scalars="upper_energy",
        cmap="coolwarm",
        clim=clim,
        smooth_shading=True,
        opacity=0.96,
        show_scalar_bar=False,
    )
    plotter.add_mesh(lower_surface.extract_feature_edges(), color="#1f1f1f", line_width=1.2, opacity=0.25)
    plotter.add_mesh(upper_surface.extract_feature_edges(), color="#1f1f1f", line_width=1.2, opacity=0.25)

    polygon_loop = np.vstack([polygon, polygon[0]])
    polygon_polyline = pv.Spline(np.column_stack([polygon_loop, np.zeros(len(polygon_loop))]), n_points=400)
    plotter.add_mesh(polygon_polyline, color="black", line_width=4)

    zero_plane_path = np.column_stack([path_points, np.zeros(len(path_points), dtype=float)])
    path_curve = pv.Spline(zero_plane_path, n_points=max(120, 80 * (len(path_points) - 1)))
    plotter.add_mesh(path_curve, color="#2d2d2d", line_width=5)

    path_labels_3d = np.column_stack([path_points, np.zeros(len(path_points), dtype=float)])
    plotter.add_point_labels(
        path_labels_3d,
        list(k_labels),
        point_size=12,
        font_size=18,
        always_visible=True,
        shape_opacity=0.12,
        fill_shape=True,
        shape_color="white",
        text_color="black",
    )

    plotter.add_axes(line_width=2, labels_off=False)
    plotter.add_text(title, position="upper_edge", font_size=14, color="black")
    plotter.view_isometric()
    plotter.camera.zoom(1.3)
    plotter.export_html(str(output_path))
    plotter.close()


def export_reciprocal_volume_html(
    output_path: Path,
    title: str,
    xyz_points: Sequence[Sequence[float]],
    valence_energies: Sequence[float],
    conduction_energies: Sequence[float],
    k_path: Sequence[Sequence[float]],
    k_labels: Sequence[str],
    energy_limit: float,
    window_size: tuple[int, int] = (1380, 920),
) -> None:
    cache_root = output_path.parent / ".runtime_cache"
    _ensure_runtime_cache(cache_root)

    import pyvista as pv

    points_xyz = np.asarray(xyz_points, dtype=float)
    valence = np.asarray(valence_energies, dtype=float)
    conduction = np.asarray(conduction_energies, dtype=float)
    path_points = np.asarray(k_path, dtype=float)

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("xyz_points must be shaped (N, 3)")
    if not (len(points_xyz) == len(valence) == len(conduction)):
        raise ValueError("point and energy arrays must have the same length")

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")

    clim = (-float(energy_limit), float(energy_limit))

    valence_cloud = pv.PolyData(points_xyz.copy())
    valence_cloud["energy"] = valence
    plotter.add_mesh(
        valence_cloud,
        scalars="energy",
        cmap="coolwarm",
        clim=clim,
        point_size=12,
        render_points_as_spheres=True,
        opacity=0.45,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Energy", "vertical": True},
    )

    conduction_cloud = pv.PolyData(points_xyz.copy())
    conduction_cloud["energy"] = conduction
    plotter.add_mesh(
        conduction_cloud,
        scalars="energy",
        cmap="coolwarm",
        clim=clim,
        point_size=6,
        render_points_as_spheres=True,
        opacity=0.9,
        show_scalar_bar=False,
    )

    bounds = (
        float(points_xyz[:, 0].min()),
        float(points_xyz[:, 0].max()),
        float(points_xyz[:, 1].min()),
        float(points_xyz[:, 1].max()),
        float(points_xyz[:, 2].min()),
        float(points_xyz[:, 2].max()),
    )
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box.extract_feature_edges(), color="black", line_width=2.5, opacity=0.55)

    path_curve = pv.Spline(path_points, n_points=max(180, 100 * (len(path_points) - 1)))
    plotter.add_mesh(path_curve, color="#1f1f1f", line_width=5)
    plotter.add_point_labels(
        path_points,
        list(k_labels),
        point_size=12,
        font_size=18,
        always_visible=True,
        shape_opacity=0.12,
        fill_shape=True,
        shape_color="white",
        text_color="black",
    )

    plotter.add_axes(line_width=2, labels_off=False)
    plotter.add_text(title, position="upper_edge", font_size=14, color="black")
    plotter.view_isometric()
    plotter.camera.zoom(1.15)
    plotter.export_html(str(output_path))
    plotter.close()
