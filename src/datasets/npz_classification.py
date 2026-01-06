from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class NpzClassificationDataset(Dataset):
    """
    Expect .npz with keys:
      - X: (N, D) float32/float64
      - y: (N,) int64/int32
    """
    def __init__(self, npz_path: str | Path):
        self.npz_path = Path(npz_path)
        data = np.load(self.npz_path)
        self.X = data["X"]
        self.y = data["y"]

        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D (N,D). got {self.X.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"y must be 1D (N,). got {self.y.shape}")
        if len(self.X) != len(self.y):
            raise ValueError(f"len(X) != len(y): {len(self.X)} vs {len(self.y)}")

    @property
    def input_dim(self) -> int:
        return int(self.X.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.y.max()) + 1

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
