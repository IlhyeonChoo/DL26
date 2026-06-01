"""Transformer model for direct multi-step time-series forecasting."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal positional encodings to sequence embeddings."""

    def __init__(self, d_model: int, max_length: int = 4096) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_terms = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_terms)
        encoding[:, 1::2] = torch.cos(positions * div_terms[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected x to have shape (batch, sequence, features).")
        if x.size(1) > self.encoding.size(1):
            raise ValueError("Input sequence length exceeds max positional encoding length.")
        return x + self.encoding[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    """Forecast future time steps from a fixed context window."""

    def __init__(
        self,
        input_features: int,
        prediction_length: int,
        target_features: int | None = None,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_context_length: int = 4096,
    ) -> None:
        super().__init__()
        if input_features <= 0:
            raise ValueError("input_features must be positive.")
        if prediction_length <= 0:
            raise ValueError("prediction_length must be positive.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.prediction_length = prediction_length
        self.target_features = target_features or input_features

        self.input_projection = nn.Linear(input_features, d_model)
        self.position_encoding = SinusoidalPositionalEncoding(d_model, max_context_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, prediction_length * self.target_features),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Return direct multi-step forecasts.

        Args:
            context: Tensor with shape `(batch, context_length, input_features)`.

        Returns:
            Tensor with shape `(batch, prediction_length, target_features)`.
        """

        if context.ndim != 3:
            raise ValueError("Expected context shape (batch, context_length, input_features).")

        x = self.input_projection(context)
        x = self.position_encoding(x)
        x = self.encoder(x)
        pooled = self.norm(x[:, -1])
        forecast = self.forecast_head(pooled)
        return forecast.view(context.size(0), self.prediction_length, self.target_features)
