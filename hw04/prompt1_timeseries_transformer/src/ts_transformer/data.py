"""Synthetic time-series data utilities."""

from __future__ import annotations

from dataclasses import dataclass

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
    """Generate multivariate sine-wave sequences for forecasting experiments."""

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
