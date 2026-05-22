"""Train a Transformer forecaster on synthetic time-series data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from ts_transformer.data import SyntheticSeriesConfig, SyntheticSineDataset
from ts_transformer.model import TransformerForecaster


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration."""

    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 3e-4
    context_length: int = 48
    prediction_length: int = 12
    num_features: int = 3
    num_samples: int = 1024
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for context, target in loader:
        context = context.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(context)
        loss = loss_fn(prediction, target)
        loss.backward()
        optimizer.step()

        batch_size = context.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0

    for context, target in loader:
        context = context.to(device)
        target = target.to(device)
        prediction = model(context)
        loss = loss_fn(prediction, target)

        batch_size = context.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def run_training(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    dataset = SyntheticSineDataset(
        SyntheticSeriesConfig(
            context_length=config.context_length,
            prediction_length=config.prediction_length,
            num_features=config.num_features,
            num_samples=config.num_samples,
            seed=config.seed,
        )
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    model = TransformerForecaster(
        input_features=config.num_features,
        prediction_length=config.prediction_length,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        print(
            f"epoch={epoch:03d} train_mse={train_loss:.6f} "
            f"val_mse={val_loss:.6f} device={device}"
        )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--context-length", type=int, default=TrainConfig.context_length)
    parser.add_argument("--prediction-length", type=int, default=TrainConfig.prediction_length)
    parser.add_argument("--num-features", type=int, default=TrainConfig.num_features)
    parser.add_argument("--num-samples", type=int, default=TrainConfig.num_samples)
    parser.add_argument("--d-model", type=int, default=TrainConfig.d_model)
    parser.add_argument("--num-heads", type=int, default=TrainConfig.num_heads)
    parser.add_argument("--num-layers", type=int, default=TrainConfig.num_layers)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    return TrainConfig(**vars(parser.parse_args()))


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
