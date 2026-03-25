from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .model import TightBindingModel


@dataclass(frozen=True)
class ModelSpec:
    model: TightBindingModel
    k_path: list[tuple[str, np.ndarray]]
    bz_bounds: list[tuple[float, float]]
    energy_window: tuple[float, float]


def list_models() -> tuple[str, ...]:
    return ("chain", "square", "ssh")


def build_model_spec(
    name: str,
    t: float = 1.0,
    t1: float = 0.8,
    t2: float = 1.2,
    delta: float = 0.0,
) -> ModelSpec:
    key = name.strip().lower()
    if key == "chain":
        onsite = np.array([[0.0]])
        hoppings = [
            (np.array([1.0]), np.array([[-t]], dtype=np.complex128)),
        ]
        return ModelSpec(
            model=TightBindingModel("1D chain", 1, onsite, hoppings),
            k_path=[
                ("G", np.array([0.0])),
                ("X", np.array([np.pi])),
            ],
            bz_bounds=[(-np.pi, np.pi)],
            energy_window=(-2.5 * abs(t), 2.5 * abs(t)),
        )

    if key == "square":
        onsite = np.array([[0.0]])
        hoppings = [
            (np.array([1.0, 0.0]), np.array([[-t]], dtype=np.complex128)),
            (np.array([0.0, 1.0]), np.array([[-t]], dtype=np.complex128)),
        ]
        return ModelSpec(
            model=TightBindingModel("2D square lattice", 2, onsite, hoppings),
            k_path=[
                ("G", np.array([0.0, 0.0])),
                ("X", np.array([np.pi, 0.0])),
                ("M", np.array([np.pi, np.pi])),
                ("G", np.array([0.0, 0.0])),
            ],
            bz_bounds=[(-np.pi, np.pi), (-np.pi, np.pi)],
            energy_window=(-4.5 * abs(t), 4.5 * abs(t)),
        )

    if key == "ssh":
        onsite = np.array(
            [
                [delta, 0.0],
                [0.0, -delta],
            ],
            dtype=np.complex128,
        )
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
        energy_scale = max(abs(t1), abs(t2), abs(delta), 1.0)
        return ModelSpec(
            model=TightBindingModel("SSH chain", 1, onsite, hoppings),
            k_path=[
                ("G", np.array([0.0])),
                ("X", np.array([np.pi])),
            ],
            bz_bounds=[(-np.pi, np.pi)],
            energy_window=(-2.5 * energy_scale, 2.5 * energy_scale),
        )

    raise ValueError(f"Unknown model: {name}")
