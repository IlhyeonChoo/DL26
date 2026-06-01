from pathlib import Path

import torch
from torch import nn

from ts_transformer.data import (
    ETTh1Config,
    ETTh1Dataset,
    SyntheticSeriesConfig,
    SyntheticSineDataset,
)
from ts_transformer.model import TransformerForecaster


def test_model_output_shape() -> None:
    model = TransformerForecaster(
        input_features=7,
        prediction_length=24,
        target_features=1,
        d_model=32,
        num_heads=4,
        num_layers=1,
        dim_feedforward=64,
    )
    context = torch.randn(2, 96, 7)

    prediction = model(context)

    assert prediction.shape == (2, 24, 1)


def test_single_optimization_step() -> None:
    dataset = SyntheticSineDataset(
        SyntheticSeriesConfig(
            context_length=16,
            prediction_length=4,
            num_features=2,
            num_samples=8,
            seed=7,
        )
    )
    context, target = dataset[:]
    model = TransformerForecaster(
        input_features=2,
        prediction_length=4,
        d_model=32,
        num_heads=4,
        num_layers=1,
        dim_feedforward=64,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    prediction = model(context)
    loss = loss_fn(prediction, target)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_etth1_dataset_builds_ot_forecast_windows(tmp_path: Path) -> None:
    csv_path = tmp_path / "ETTh1.csv"
    rows = ["date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT"]
    for hour in range(80):
        rows.append(
            f"2016-07-01 {hour:02d}:00:00,"
            f"{hour + 0.1},{hour + 0.2},{hour + 0.3},{hour + 0.4},"
            f"{hour + 0.5},{hour + 0.6},{hour + 0.7}"
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    config = ETTh1Config(
        data_path=csv_path,
        context_length=8,
        prediction_length=4,
        train_hours=40,
        val_hours=20,
        test_hours=20,
    )

    train_dataset = ETTh1Dataset(config, split="train")
    val_dataset = ETTh1Dataset(config, split="val", scaler=train_dataset.scaler)
    test_dataset = ETTh1Dataset(config, split="test", scaler=train_dataset.scaler)
    context, target = train_dataset[0]

    assert context.shape == (8, 7)
    assert target.shape == (4, 1)
    assert len(train_dataset) == 29
    assert len(val_dataset) == 17
    assert len(test_dataset) == 17
