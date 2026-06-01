"""Train a Transformer forecaster on ETTh1."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ts_transformer.data import ETTH1_FEATURE_COLUMNS, ETTh1Config, ETTh1Dataset
from ts_transformer.model import TransformerForecaster


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration for ETTh1 forecasting."""

    data_path: Path
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 3e-4
    context_length: int = 96
    prediction_length: int = 24
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_workers: int = 0


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

    data_config = ETTh1Config(
        data_path=config.data_path,
        context_length=config.context_length,
        prediction_length=config.prediction_length,
    )
    train_dataset = ETTh1Dataset(data_config, split="train")
    val_dataset = ETTh1Dataset(data_config, split="val", scaler=train_dataset.scaler)
    test_dataset = ETTh1Dataset(data_config, split="test", scaler=train_dataset.scaler)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model = TransformerForecaster(
        input_features=len(ETTH1_FEATURE_COLUMNS),
        prediction_length=config.prediction_length,
        target_features=1,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    print(
        f"dataset_windows train={len(train_dataset)} val={len(val_dataset)} "
        f"test={len(test_dataset)} context={config.context_length} "
        f"prediction={config.prediction_length} target=OT device={device}"
    )
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        print(
            f"epoch={epoch:03d} train_mse={train_loss:.6f} "
            f"val_mse={val_loss:.6f} device={device}"
        )

    test_loss = evaluate(model, test_loader, loss_fn, device)
    print(f"test_mse={test_loss:.6f} device={device}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--context-length", type=int, default=TrainConfig.context_length)
    parser.add_argument("--prediction-length", type=int, default=TrainConfig.prediction_length)
    parser.add_argument("--d-model", type=int, default=TrainConfig.d_model)
    parser.add_argument("--num-heads", type=int, default=TrainConfig.num_heads)
    parser.add_argument("--num-layers", type=int, default=TrainConfig.num_layers)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    return TrainConfig(**vars(parser.parse_args()))


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
