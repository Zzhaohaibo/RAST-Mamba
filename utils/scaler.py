import numpy as np
import torch


class StandardScaler:
    """
    Per-channel Z-score scaler.

    For traffic data with shape [T, N], mean/std are shaped as [1, N],
    matching BasicTS norm_each_channel=True behavior.
    """

    def __init__(self, mean=None, std=None, eps=1e-5):
        self.mean = mean
        self.std = std
        self.eps = eps

    def fit(self, data: np.ndarray):
        if data.ndim != 2:
            raise ValueError(f"Expected data shape [T, N], got {data.shape}")

        self.mean = data.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = data.std(axis=0, keepdims=True).astype(np.float32)
        self.std = np.where(self.std < self.eps, 1.0, self.std).astype(np.float32)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return ((data - self.mean) / self.std).astype(np.float32)

    def inverse_transform_np(self, data: np.ndarray) -> np.ndarray:
        return (data * self.std + self.mean).astype(np.float32)

    def inverse_transform_tensor(self, data: torch.Tensor) -> torch.Tensor:
        """
        data: [B, H, N] or [B, T, N]
        """
        mean = torch.as_tensor(self.mean, dtype=data.dtype, device=data.device)
        std = torch.as_tensor(self.std, dtype=data.dtype, device=data.device)
        return data * std + mean
