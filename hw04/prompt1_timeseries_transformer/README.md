# ETTh1 Transformer Forecasting

This directory contains a compact PyTorch Transformer for ETTh1 time-series
forecasting.

The ETTh1 setup is:

- Data frequency: hourly
- Variables: `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`
- Input: past 96 hours of all 7 variables
- Target: next 24 hours of `OT` only
- Split: train 12 months, validation 4 months, test 8 months

The model consumes `(batch, 96, 7)` and predicts `(batch, 24, 1)`.

## Setup

```bash
uv venv --python 3.11
uv sync --extra dev
```

## Train on ETTh1

Place or reference the ETTh1 CSV file, then run:

```bash
uv run train-ts-transformer --data-path /path/to/ETTh1.csv --epochs 5
```

Equivalent module command:

```bash
uv run python -m ts_transformer.train --data-path /path/to/ETTh1.csv --epochs 5
```

Useful options:

```bash
uv run train-ts-transformer   --data-path /path/to/ETTh1.csv   --context-length 96   --prediction-length 24   --batch-size 64   --epochs 10   --device cuda
```

## Tests

```bash
uv run pytest
```

The tests are smoke tests. They verify model tensor shapes, one optimization step,
and ETTh1-style CSV window construction. They are not a full benchmark.

## Files

- `src/ts_transformer/model.py`: Transformer forecaster and positional encoding.
- `src/ts_transformer/data.py`: ETTh1 CSV loader, scaler, split/window dataset, and synthetic smoke-test data.
- `src/ts_transformer/train.py`: ETTh1 training, validation, and final test evaluation loop.
- `tests/test_smoke.py`: Shape, optimization, and ETTh1 loader checks.
