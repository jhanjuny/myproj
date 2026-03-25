from __future__ import annotations

from typing import Sequence

import numpy as np


def sample_k_path(
    labeled_points: Sequence[tuple[str, np.ndarray]],
    samples_per_segment: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if len(labeled_points) < 2:
        raise ValueError("At least two labeled points are required")

    samples = max(2, int(samples_per_segment))
    k_points: list[np.ndarray] = []
    distances: list[float] = []
    tick_positions: list[float] = [0.0]
    tick_labels: list[str] = [labeled_points[0][0]]

    cumulative = 0.0
    previous_k: np.ndarray | None = None

    for segment_index in range(len(labeled_points) - 1):
        _, start = labeled_points[segment_index]
        end_label, end = labeled_points[segment_index + 1]

        for step_index in range(samples):
            if segment_index > 0 and step_index == 0:
                continue

            alpha = step_index / (samples - 1)
            current_k = (1.0 - alpha) * start + alpha * end

            if previous_k is None:
                cumulative = 0.0
            else:
                cumulative += float(np.linalg.norm(current_k - previous_k))

            k_points.append(current_k)
            distances.append(cumulative)
            previous_k = current_k

        tick_positions.append(cumulative)
        tick_labels.append(end_label)

    return (
        np.asarray(k_points, dtype=float),
        np.asarray(distances, dtype=float),
        np.asarray(tick_positions, dtype=float),
        tick_labels,
    )
