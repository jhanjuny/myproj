from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class TightBindingModel:
    name: str
    dimension: int
    onsite: np.ndarray
    hoppings: list[tuple[np.ndarray, np.ndarray]]

    def __post_init__(self) -> None:
        self.onsite = np.asarray(self.onsite, dtype=np.complex128)
        normalized: list[tuple[np.ndarray, np.ndarray]] = []
        for displacement, hopping in self.hoppings:
            normalized.append(
                (
                    np.asarray(displacement, dtype=float),
                    np.asarray(hopping, dtype=np.complex128),
                )
            )
        self.hoppings = normalized

        if self.onsite.ndim != 2 or self.onsite.shape[0] != self.onsite.shape[1]:
            raise ValueError("onsite must be a square matrix")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")

        for displacement, hopping in self.hoppings:
            if displacement.shape != (self.dimension,):
                raise ValueError("displacement vector has the wrong dimension")
            if hopping.shape != self.onsite.shape:
                raise ValueError("hopping matrix shape must match onsite shape")

    @property
    def num_orbitals(self) -> int:
        return int(self.onsite.shape[0])

    def hamiltonian(self, k_point: Iterable[float]) -> np.ndarray:
        k_vector = np.asarray(tuple(k_point), dtype=float)
        if k_vector.shape != (self.dimension,):
            raise ValueError("k-point has the wrong dimension")

        hamiltonian = self.onsite.copy()
        for displacement, hopping in self.hoppings:
            phase = np.exp(1j * np.dot(k_vector, displacement))
            hamiltonian += hopping * phase
            hamiltonian += hopping.conj().T * np.conj(phase)
        return hamiltonian

    def eigenvalues(self, k_point: Iterable[float]) -> np.ndarray:
        return np.linalg.eigvalsh(self.hamiltonian(k_point)).real
