import os
import numpy as np
import torch
from torch.utils.data import Dataset


def _derive_timestamps(length: int, offset: int = 0, points_per_day: int = 288):
    idx = np.arange(offset, offset + length)
    tid = idx % points_per_day
    diw = (idx // points_per_day) % 7
    return np.stack([tid, diw], axis=-1).astype(np.int64)


def _convert_timestamps(raw_ts, length: int, points_per_day: int = 288):
    """
    Convert BasicTS-style timestamps to integer indices:
    [:, 0] -> time-of-day index in [0, 287]
    [:, 1] -> day-of-week index in [0, 6]
    """
    if raw_ts is None:
        return _derive_timestamps(length, points_per_day=points_per_day)

    ts = np.asarray(raw_ts)

    if ts.ndim == 1:
        return _derive_timestamps(length, points_per_day=points_per_day)

    if ts.shape[0] != length:
        raise ValueError(f"Timestamp length {ts.shape[0]} != data length {length}")

    if ts.shape[-1] < 2:
        return _derive_timestamps(length, points_per_day=points_per_day)

    tid = ts[:, 0]
    diw = ts[:, 1]

    # time-of-day
    if np.nanmax(tid) <= 1.0:
        tid = np.rint(tid * points_per_day)
    tid = tid.astype(np.int64) % points_per_day

    # day-of-week
    if np.nanmax(diw) <= 1.0:
        diw = np.rint(diw * 7)
    diw = diw.astype(np.int64)

    # handle 1-7 style day index
    if diw.min() >= 1 and diw.max() <= 7:
        diw = diw - 1
    diw = diw % 7

    return np.stack([tid, diw], axis=-1).astype(np.int64)


class PEMSDataset(Dataset):
    """
    Dataset for BasicTS preprocessed files:
    train_data.npy / val_data.npy / test_data.npy
    train_timestamps.npy / val_timestamps.npy / test_timestamps.npy

    Returns:
        x_norm: [input_len, N]
        y_norm: [output_len, N]
        x_ts:   [input_len, 2]
        y_ts:   [output_len, 2]
        y_raw:  [output_len, N]
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        scaler,
        input_len: int = 12,
        output_len: int = 12,
        points_per_day: int = 288,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.split = split
        self.scaler = scaler
        self.input_len = input_len
        self.output_len = output_len
        self.points_per_day = points_per_day

        data_path = os.path.join(data_dir, f"{split}_data.npy")
        ts_path = os.path.join(data_dir, f"{split}_timestamps.npy")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Cannot find {data_path}")

        data = np.load(data_path).astype(np.float32)

        if data.ndim == 3 and data.shape[-1] == 1:
            data = data[..., 0]

        if data.ndim != 2:
            raise ValueError(f"Expected {split}_data shape [T, N], got {data.shape}")

        self.raw_data = data
        self.data = scaler.transform(data)

        if os.path.exists(ts_path):
            raw_ts = np.load(ts_path)
        else:
            raw_ts = None

        self.timestamps = _convert_timestamps(
            raw_ts,
            length=len(data),
            points_per_day=points_per_day,
        )

        self.num_samples = len(data) - input_len - output_len + 1
        if self.num_samples <= 0:
            raise ValueError(
                f"{split} data too short: len={len(data)}, "
                f"input_len={input_len}, output_len={output_len}"
            )

        self.num_nodes = data.shape[1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_begin = idx
        x_end = idx + self.input_len
        y_begin = x_end
        y_end = y_begin + self.output_len

        x_norm = self.data[x_begin:x_end]
        y_norm = self.data[y_begin:y_end]
        x_ts = self.timestamps[x_begin:x_end]
        y_ts = self.timestamps[y_begin:y_end]
        y_raw = self.raw_data[y_begin:y_end]

        return (
            torch.from_numpy(x_norm).float(),
            torch.from_numpy(y_norm).float(),
            torch.from_numpy(x_ts).long(),
            torch.from_numpy(y_ts).long(),
            torch.from_numpy(y_raw).float(),
        )


def build_datasets(data_dir, scaler, input_len=12, output_len=12, points_per_day=288):
    train_set = PEMSDataset(data_dir, "train", scaler, input_len, output_len, points_per_day)
    val_set = PEMSDataset(data_dir, "val", scaler, input_len, output_len, points_per_day)
    test_set = PEMSDataset(data_dir, "test", scaler, input_len, output_len, points_per_day)
    return train_set, val_set, test_set
