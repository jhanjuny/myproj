from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tb.examples import build_model_spec, list_models
from tb.export import ensure_dir, write_band_csv, write_band_svg, write_dos_csv, write_dos_svg
from tb.kpath import sample_k_path


def compute_bands(spec, samples_per_segment: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    k_points, distances, tick_positions, tick_labels = sample_k_path(spec.k_path, samples_per_segment)
    bands = np.vstack([spec.model.eigenvalues(k_point) for k_point in k_points])
    return distances, bands, tick_positions, tick_labels


def compute_dos(spec, k_grid: int, broadening: float, energy_points: int) -> tuple[np.ndarray, np.ndarray]:
    axes = [
        np.linspace(bound_min, bound_max, max(8, k_grid), endpoint=False)
        for bound_min, bound_max in spec.bz_bounds
    ]

    all_energies = []
    for k_point in np.array(np.meshgrid(*axes, indexing="ij")).reshape(spec.model.dimension, -1).T:
        all_energies.append(spec.model.eigenvalues(k_point))
    energies = np.concatenate(all_energies, axis=0)

    window_min = min(spec.energy_window[0], float(energies.min()) - 0.5)
    window_max = max(spec.energy_window[1], float(energies.max()) + 0.5)
    energy_axis = np.linspace(window_min, window_max, max(80, energy_points))

    sigma = max(1e-4, float(broadening))
    diff = energy_axis[:, None] - energies[None, :]
    gaussian = np.exp(-0.5 * (diff / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    dos = gaussian.mean(axis=1)
    return energy_axis, dos


def default_output_dir(script_dir: Path, model_name: str) -> Path:
    return script_dir / "outputs" / model_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list_models(), default="square")
    parser.add_argument("--mode", choices=("band", "dos", "both"), default="both")
    parser.add_argument("--samples-per-segment", type=int, default=80)
    parser.add_argument("--k-grid", type=int, default=90)
    parser.add_argument("--energy-points", type=int, default=400)
    parser.add_argument("--broadening", type=float, default=0.08)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--t1", type=float, default=0.8)
    parser.add_argument("--t2", type=float, default=1.2)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    spec = build_model_spec(
        name=args.model,
        t=args.t,
        t1=args.t1,
        t2=args.t2,
        delta=args.delta,
    )

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_output_dir(script_dir, args.model)
    ensure_dir(out_dir)

    print(f"Model: {spec.model.name}")
    print(f"Orbitals: {spec.model.num_orbitals}")
    print(f"Output directory: {out_dir}")

    if args.mode in ("band", "both"):
        distances, bands, tick_positions, tick_labels = compute_bands(spec, args.samples_per_segment)
        band_csv = out_dir / "band_structure.csv"
        band_svg = out_dir / "band_structure.svg"
        write_band_csv(band_csv, distances, bands)
        write_band_svg(
            band_svg,
            title=f"{spec.model.name} band structure",
            distances=distances,
            bands=bands,
            tick_positions=tick_positions,
            tick_labels=tick_labels,
        )
        print(f"[band] energy range: {bands.min():.4f} .. {bands.max():.4f}")
        print(f"[band] csv: {band_csv}")
        print(f"[band] svg: {band_svg}")

    if args.mode in ("dos", "both"):
        energies, dos = compute_dos(spec, args.k_grid, args.broadening, args.energy_points)
        dos_csv = out_dir / "dos.csv"
        dos_svg = out_dir / "dos.svg"
        write_dos_csv(dos_csv, energies, dos)
        write_dos_svg(dos_svg, title=f"{spec.model.name} density of states", energies=energies, dos=dos)
        print(f"[dos] max density: {dos.max():.4f}")
        print(f"[dos] csv: {dos_csv}")
        print(f"[dos] svg: {dos_svg}")


if __name__ == "__main__":
    main()
