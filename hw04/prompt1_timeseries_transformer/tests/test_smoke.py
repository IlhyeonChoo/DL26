import torch
from torch import nn

from ts_transformer.data import SyntheticSeriesConfig, SyntheticSineDataset
from ts_transformer.model import TransformerForecaster


def test_model_output_shape() -> None:
    model = TransformerForecaster(
        input_features=3,
        prediction_length=5,
        d_model=32,
        num_heads=4,
        num_layers=1,
        dim_feedforward=64,
    )
    context = torch.randn(2, 16, 3)

    prediction = model(context)

    assert prediction.shape == (2, 5, 3)


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
