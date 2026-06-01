"""Time-series data utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticSeriesConfig:
    """Configuration for synthetic forecasting samples."""

    context_length: int = 48
    prediction_length: int = 12
    num_features: int = 3
    num_samples: int = 1024
    noise_std: float = 0.05
    seed: int = 42


class SyntheticSineDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Generate multivariate sine-wave sequences for forecasting smoke tests."""

    def __init__(self, config: SyntheticSeriesConfig) -> None:
        self.config = config
        if config.context_length <= 0 or config.prediction_length <= 0:
            raise ValueError("context_length and prediction_length must be positive.")
        if config.num_features <= 0 or config.num_samples <= 0:
            raise ValueError("num_features and num_samples must be positive.")

        rng = np.random.default_rng(config.seed)
        total_length = config.context_length + config.prediction_length
        time = np.arange(total_length, dtype=np.float32)

        contexts: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for _ in range(config.num_samples):
            features = []
            for feature_idx in range(config.num_features):
                period = rng.uniform(16.0, 64.0)
                phase = rng.uniform(0.0, 2.0 * np.pi)
                amplitude = rng.uniform(0.5, 1.5)
                trend = rng.uniform(-0.01, 0.01) * time
                seasonal = amplitude * np.sin((2.0 * np.pi * time / period) + phase)
                noise = rng.normal(0.0, config.noise_std, size=total_length)
                features.append(seasonal + trend + noise + 0.1 * feature_idx)

            series = np.stack(features, axis=-1).astype(np.float32)
            contexts.append(series[: config.context_length])
            targets.append(series[config.context_length :])

        self.contexts = torch.from_numpy(np.stack(contexts))
        self.targets = torch.from_numpy(np.stack(targets))

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.contexts[index], self.targets[index]


ETTH1_FEATURE_COLUMNS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
ETTH1_TARGET_COLUMN = "OT"


@dataclass(frozen=True)
class ETTh1Config:
    """Configuration for the ETTh1 forecasting benchmark."""

    data_path: Path
    context_length: int = 96
    prediction_length: int = 24
    feature_columns: tuple[str, ...] = ETTH1_FEATURE_COLUMNS
    target_column: str = ETTH1_TARGET_COLUMN
    train_hours: int = 12 * 30 * 24
    val_hours: int = 4 * 30 * 24
    test_hours: int = 8 * 30 * 24


@dataclass(frozen=True)
class StandardScaler:
    """Mean/std scaler fitted on the training portion only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler":
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)


class ETTh1Dataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """ETTh1 windows: 96 hours of 7 features to 24 hours of OT."""

    def __init__(
        self,
        config: ETTh1Config,
        split: str,
        scaler: StandardScaler | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test.")
        if config.context_length <= 0 or config.prediction_length <= 0:
            raise ValueError("context_length and prediction_length must be positive.")
        if config.target_column not in config.feature_columns:
            raise ValueError("target_column must be included in feature_columns.")

        raw_values = load_etth1_values(config.data_path, config.feature_columns)
        train_end = config.train_hours
        val_end = train_end + config.val_hours
        test_end = val_end + config.test_hours
        usable_end = min(test_end, len(raw_values))
        if usable_end < config.context_length + config.prediction_length:
            raise ValueError("ETTh1 data is too short for the requested window lengths.")

        self.context_length = config.context_length
        self.prediction_length = config.prediction_length
        self.target_index = config.feature_columns.index(config.target_column)
        self.scaler = scaler or StandardScaler.fit(raw_values[: min(train_end, len(raw_values))])
        self.values = torch.from_numpy(self.scaler.transform(raw_values[:usable_end]))
        self.start_indices = self._build_start_indices(config, split, len(self.values))
        if not self.start_indices:
            raise ValueError(f"No {split} windows can be built from the provided data.")

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.start_indices[index]
        context_end = start + self.context_length
        target_end = context_end + self.prediction_length
        context = self.values[start:context_end]
        target = self.values[context_end:target_end, self.target_index : self.target_index + 1]
        return context, target

    def _build_start_indices(self, config: ETTh1Config, split: str, length: int) -> list[int]:
        train_end = min(config.train_hours, length)
        val_end = min(config.train_hours + config.val_hours, length)
        test_end = min(config.train_hours + config.val_hours + config.test_hours, length)

        if split == "train":
            start = 0
            end = train_end
        elif split == "val":
            start = max(0, train_end - config.context_length)
            end = val_end
        else:
            start = max(0, val_end - config.context_length)
            end = test_end

        last_start = end - config.context_length - config.prediction_length
        return list(range(start, last_start + 1))


def load_etth1_values(
    data_path: Path,
    feature_columns: tuple[str, ...] = ETTH1_FEATURE_COLUMNS,
) -> np.ndarray:
    """Load ETTh1 numeric feature columns from a CSV file."""

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"ETTh1 CSV file was not found: {path}")

    rows: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV file does not contain a header row.")
        missing_columns = set(feature_columns) - set(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV file is missing required columns: {missing}")

        for row in reader:
            rows.append([float(row[column]) for column in feature_columns])

    if not rows:
        raise ValueError("CSV file does not contain any data rows.")
    return np.asarray(rows, dtype=np.float32)
