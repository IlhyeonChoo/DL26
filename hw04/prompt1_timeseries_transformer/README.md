# Time-Series Transformer Forecasting

This directory contains a small PyTorch implementation of a Transformer model for
multivariate time-series forecasting.

The model consumes a history window shaped as `(batch, context_length, features)`
and predicts the next `prediction_length` time steps shaped as
`(batch, prediction_length, target_features)`.

## Setup

```bash
uv venv --python 3.11
uv sync --extra dev
```

## Train on Synthetic Data

```bash
uv run train-ts-transformer --epochs 5 --context-length 48 --prediction-length 12
```

You can also run the module directly:

```bash
uv run python -m ts_transformer.train --epochs 5
```

## Smoke Test

```bash
uv run pytest
```

## Files

- `src/ts_transformer/model.py`: Transformer forecaster and positional encoding.
- `src/ts_transformer/data.py`: Synthetic sine-wave dataset with trend and noise.
- `src/ts_transformer/train.py`: Minimal training and validation loop.
- `tests/test_smoke.py`: Shape and one-step optimization checks.
