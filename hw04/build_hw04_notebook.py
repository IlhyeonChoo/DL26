import json
from pathlib import Path


def lines(source: str) -> list[str]:
    return [line + "\n" for line in source.strip("\n").splitlines()]


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(source),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(source),
    }


cells = [
    markdown_cell(
        r"""
# DL Lab HW04 - ETTh1 Time Series Forecasting

이 노트북은 `Todo.md`의 문제 1~3을 수행하기 위한 실험 노트북이다.

ETTh1 데이터셋에서 과거 여러 시간의 7개 변수(`HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`)를 입력으로 사용하고, 미래 24시간의 `OT`를 예측한다.

Colab에서 실행할 때는 `런타임 > 런타임 유형 변경 > GPU`를 선택한 뒤 위에서부터 실행한다. Colab에서는 기본적으로 Google Drive를 마운트하고 결과를 `/content/drive/MyDrive/DL26/hw04/results_hw04`에 저장한다. CPU 런타임에서는 자동으로 fast debug mode가 켜져 epoch와 batch 수를 줄인 smoke 실행만 수행한다. 제출용 결과를 만들 때는 GPU 런타임에서 `COLAB_FAST_DEV_RUN = False`, `MAX_TRAIN_BATCHES = None`, `MAX_EVAL_BATCHES = None` 상태로 실행한다.

| 문제 | 비교 대상 | 고정 조건 |
| --- | --- | --- |
| 문제 1 | RNN vs LSTM vs GRU | input length 96, pred length 24, hidden dim 64, layer 1 |
| 문제 2 | 입력 길이 24/48/96/192, bidirectional False/True | 문제 1 best recurrent type 기준 |
| 문제 3 | TransformerEncoder | input length 96, pred length 24, d_model 64, nhead 4, layer 2 |
"""
    ),
    code_cell(
        r"""
import json
import math
import os
import random
import urllib.request
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


BASE_SEED = 42
FEATURE_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET_COLUMN = "OT"
PRED_LENGTH = 24
DEFAULT_INPUT_LENGTH = 96

try:
    import google.colab  # type: ignore[import-not-found]

    IN_COLAB = True
except ModuleNotFoundError:
    IN_COLAB = "COLAB_GPU" in os.environ


def set_global_seed(seed_value: int) -> None:
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_global_seed(BASE_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

NUM_WORKERS = 2 if device.type == "cuda" else 0
MOUNT_GOOGLE_DRIVE_IN_COLAB = IN_COLAB
GOOGLE_DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/DL26/hw04")

if IN_COLAB:
    data_dir = Path("/content/data")
    if MOUNT_GOOGLE_DRIVE_IN_COLAB:
        try:
            from google.colab import drive  # type: ignore[import-not-found]

            drive.mount("/content/drive")
            output_dir = GOOGLE_DRIVE_PROJECT_DIR / "results_hw04"
        except Exception as exc:
            print(f"Google Drive mount failed. Falling back to /content/results_hw04. Reason: {exc}")
            output_dir = Path("/content/results_hw04")
    else:
        output_dir = Path("/content/results_hw04")
else:
    project_dir = Path("hw04") if Path("hw04/Todo.md").exists() else Path(".")
    data_dir = project_dir / "data"
    output_dir = project_dir / "results_hw04"

data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

RUN_PROBLEM_1 = True
RUN_PROBLEM_2 = True
RUN_PROBLEM_3 = True

COLAB_FAST_DEV_RUN = IN_COLAB and device.type != "cuda"
MAX_TRAIN_BATCHES = 2 if COLAB_FAST_DEV_RUN else None
MAX_EVAL_BATCHES = 2 if COLAB_FAST_DEV_RUN else None
USE_AMP = torch.cuda.is_available()
AMP_INIT_SCALE = 1024.0


def experiment_epochs(full_epochs: int) -> int:
    return 1 if COLAB_FAST_DEV_RUN else full_epochs


def experiment_batch_size(full_batch_size: int, fast_batch_size: int = 32) -> int:
    return fast_batch_size if COLAB_FAST_DEV_RUN else full_batch_size


print("Using device:", device)
print("Running in Colab:", IN_COLAB)
print("Data directory:", data_dir.resolve())
print("Output directory:", output_dir.resolve())
print("DataLoader workers:", NUM_WORKERS)
print("MAX_TRAIN_BATCHES:", MAX_TRAIN_BATCHES)
print("MAX_EVAL_BATCHES:", MAX_EVAL_BATCHES)
print("Automatic mixed precision:", USE_AMP)
"""
    ),
    markdown_cell(
        r"""
## 데이터 로드 및 전처리

ETTh1 원본 CSV를 다운로드한 뒤, train 12개월 / validation 4개월 / test 8개월 기준으로 분할한다.

Scaling은 train 구간의 평균과 표준편차만 사용한다. 이는 validation/test 정보가 train preprocessing에 섞이는 data leakage를 막기 위함이다.
"""
    ),
    code_cell(
        r"""
ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
ETTH1_PATH = data_dir / "ETTh1.csv"


def download_etth1(path: Path = ETTH1_PATH) -> Path:
    if path.exists():
        print("ETTh1 already exists:", path)
        return path
    print("Downloading ETTh1 to:", path)
    urllib.request.urlretrieve(ETTH1_URL, path)
    return path


def load_etth1_dataframe(path: Path = ETTH1_PATH) -> pd.DataFrame:
    download_etth1(path)
    dataframe = pd.read_csv(path)
    missing_columns = sorted(set(FEATURE_COLUMNS + ["date"]) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"ETTh1 CSV is missing required columns: {missing_columns}")
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    return dataframe


raw_df = load_etth1_dataframe()
print(raw_df.head())
print("Rows:", len(raw_df))
print("Date range:", raw_df["date"].min(), "to", raw_df["date"].max())
"""
    ),
    code_cell(
        r"""
@dataclass(frozen=True)
class StandardScaler:
    mean: torch.Tensor
    std: torch.Tensor

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def inverse_target(self, values: torch.Tensor, target_index: int) -> torch.Tensor:
        return values * self.std[target_index] + self.mean[target_index]


def build_train_scaler(dataframe: pd.DataFrame) -> StandardScaler:
    train_end = 12 * 30 * 24
    train_values = torch.tensor(
        dataframe.loc[: train_end - 1, FEATURE_COLUMNS].to_numpy(dtype="float32"),
        dtype=torch.float32,
    )
    mean = train_values.mean(dim=0)
    std = train_values.std(dim=0).clamp_min(1e-6)
    return StandardScaler(mean=mean, std=std)


class ETTh1ForecastDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        split: str,
        input_length: int,
        pred_length: int,
        scaler: StandardScaler,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.input_length = input_length
        self.pred_length = pred_length
        self.target_index = FEATURE_COLUMNS.index(TARGET_COLUMN)

        split_boundaries = {
            "train": (0, 12 * 30 * 24),
            "val": (12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24),
            "test": (12 * 30 * 24 + 4 * 30 * 24, len(dataframe)),
        }
        self.start, self.end = split_boundaries[split]
        if self.end - self.start < input_length + pred_length:
            raise ValueError(
                f"Split {split} is too short for input_length={input_length}, "
                f"pred_length={pred_length}."
            )

        values = torch.tensor(dataframe[FEATURE_COLUMNS].to_numpy(dtype="float32"), dtype=torch.float32)
        self.values = scaler.transform(values)
        self.indices = list(range(self.start, self.end - input_length - pred_length + 1))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[index]
        input_end = start + self.input_length
        pred_end = input_end + self.pred_length
        x = self.values[start:input_end, :]
        y = self.values[input_end:pred_end, self.target_index]
        return x, y


scaler = build_train_scaler(raw_df)
target_index = FEATURE_COLUMNS.index(TARGET_COLUMN)


def make_dataset(split: str, input_length: int = DEFAULT_INPUT_LENGTH) -> ETTh1ForecastDataset:
    return ETTh1ForecastDataset(
        dataframe=raw_df,
        split=split,
        input_length=input_length,
        pred_length=PRED_LENGTH,
        scaler=scaler,
    )


def make_loader(
    split: str,
    input_length: int = DEFAULT_INPUT_LENGTH,
    batch_size: int = 128,
    shuffle: bool | None = None,
    seed_offset: int = 0,
) -> DataLoader:
    dataset = make_dataset(split=split, input_length=input_length)
    if shuffle is None:
        shuffle = split == "train"
    generator = torch.Generator().manual_seed(BASE_SEED + seed_offset) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False,
    )


for split_name in ["train", "val", "test"]:
    split_dataset = make_dataset(split_name, DEFAULT_INPUT_LENGTH)
    print(split_name, "samples:", len(split_dataset), "x/y shapes:", split_dataset[0][0].shape, split_dataset[0][1].shape)
"""
    ),
    markdown_cell(
        r"""
## 공통 모델 정의

RNN/LSTM/GRU 모델은 recurrent block의 종류만 바꾸고 `hidden_dim=64`, `num_layers=1` 조건을 동일하게 유지한다.

Transformer 모델은 `Input Projection + learnable positional embedding + TransformerEncoder + FC` 구조를 따른다.
"""
    ),
    code_cell(
        r"""
class RecurrentForecastModel(nn.Module):
    def __init__(
        self,
        rnn_type: str,
        input_dim: int = len(FEATURE_COLUMNS),
        hidden_dim: int = 64,
        num_layers: int = 1,
        pred_length: int = PRED_LENGTH,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        rnn_classes = {
            "RNN": nn.RNN,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
        }
        if rnn_type not in rnn_classes:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.recurrent = rnn_classes[rnn_type](
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=recurrent_dropout,
        )
        direction_multiplier = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * direction_multiplier),
            nn.Linear(hidden_dim * direction_multiplier, pred_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(x)
        last_state = output[:, -1, :]
        return self.head(last_state)


class TransformerForecastModel(nn.Module):
    def __init__(
        self,
        input_length: int = DEFAULT_INPUT_LENGTH,
        input_dim: int = len(FEATURE_COLUMNS),
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        pred_length: int = PRED_LENGTH,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, input_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, pred_length),
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.position_embedding.size(1):
            raise ValueError(
                f"Expected input length {self.position_embedding.size(1)}, got {x.size(1)}."
            )
        x = self.input_projection(x) + self.position_embedding
        encoded = self.encoder(x)
        return self.head(encoded[:, -1, :])


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


for name in ["RNN", "LSTM", "GRU"]:
    preview_model = RecurrentForecastModel(name)
    print(name, "parameters:", count_parameters(preview_model))
    del preview_model

preview_transformer = TransformerForecastModel()
print("Transformer parameters:", count_parameters(preview_transformer))
del preview_transformer
"""
    ),
    markdown_cell(
        r"""
## 공통 학습 및 평가 유틸리티

모든 실험은 같은 학습 루프와 같은 MSE/MAE 계산 함수를 사용한다. 그래프와 JSON 결과는 `results_hw04` 아래에 저장된다.
"""
    ),
    code_cell(
        r"""
criterion = nn.MSELoss()


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", path)


def autocast_context():
    if not (USE_AMP and device.type == "cuda"):
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler() -> object:
    enabled = USE_AMP and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device="cuda", enabled=enabled, init_scale=AMP_INIT_SCALE)
        except TypeError:
            try:
                return torch.amp.GradScaler("cuda", enabled=enabled, init_scale=AMP_INIT_SCALE)
            except TypeError:
                pass
    return torch.cuda.amp.GradScaler(enabled=enabled, init_scale=AMP_INIT_SCALE)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: object,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0

    for batch_index, (xb, yb) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            pred = model(xb)
            loss = criterion(pred, yb)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total += xb.size(0)

    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_elements = 0

    for batch_index, (xb, yb) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with autocast_context():
            pred = model(xb)
        pred = pred.float()
        yb = yb.float()
        total_squared_error += torch.sum((pred - yb) ** 2).item()
        total_absolute_error += torch.sum(torch.abs(pred - yb)).item()
        total_elements += yb.numel()

    return {
        "mse": total_squared_error / max(total_elements, 1),
        "mae": total_absolute_error / max(total_elements, 1),
    }


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    name: str,
) -> tuple[nn.Module, dict]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = make_grad_scaler()
    history = {
        "name": name,
        "parameters": count_parameters(model),
        "train_mse": [],
        "val_mse": [],
        "val_mae": [],
    }

    for epoch in range(1, epochs + 1):
        train_mse = train_one_epoch(model, train_loader, optimizer, scaler, MAX_TRAIN_BATCHES)
        val_metrics = evaluate(model, val_loader, MAX_EVAL_BATCHES)
        scheduler.step()
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_metrics["mse"])
        history["val_mae"].append(val_metrics["mae"])
        print(
            f"[{name}] epoch {epoch:02d}/{epochs} "
            f"train_mse={train_mse:.6f} val_mse={val_metrics['mse']:.6f} "
            f"val_mae={val_metrics['mae']:.6f}"
        )

    test_metrics = evaluate(model, test_loader, MAX_EVAL_BATCHES)
    history["test_mse"] = test_metrics["mse"]
    history["test_mae"] = test_metrics["mae"]
    return model, history


@torch.no_grad()
def collect_prediction_example(
    models: dict[str, nn.Module],
    loader: DataLoader,
    example_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    for model in models.values():
        model.eval()

    seen = 0
    for xb, yb in loader:
        batch_size = xb.size(0)
        if seen + batch_size <= example_index:
            seen += batch_size
            continue
        local_index = example_index - seen
        xb = xb.to(device)
        yb = yb.to(device)
        predictions = {
            name: model(xb[local_index : local_index + 1]).squeeze(0).detach().cpu()
            for name, model in models.items()
        }
        return (
            xb[local_index].detach().cpu(),
            yb[local_index].detach().cpu(),
            predictions,
        )
    raise IndexError(f"example_index {example_index} is out of range.")


def plot_loss_curves(histories: dict[str, dict], path: Path, metric_key: str = "val_mse") -> None:
    plt.figure(figsize=(8, 5))
    for name, history in histories.items():
        epochs = range(1, len(history[metric_key]) + 1)
        plt.plot(epochs, history[metric_key], marker="o", label=name)
    plt.xlabel("Epoch")
    plt.ylabel(metric_key)
    plt.title(metric_key.replace("_", " ").upper())
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.show()
    print("Saved:", path)


def plot_metric_bar(results: dict[str, dict], path: Path, metric: str = "test_mse") -> None:
    names = list(results.keys())
    values = [results[name][metric] for name in names]
    plt.figure(figsize=(8, 4))
    plt.bar(names, values)
    plt.ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.show()
    print("Saved:", path)


def plot_forecast_example(
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    path: Path,
    title: str,
) -> None:
    hours = list(range(1, len(target) + 1))
    plt.figure(figsize=(9, 5))
    plt.plot(hours, target.numpy(), marker="o", linewidth=2, label="Actual OT")
    for name, pred in predictions.items():
        plt.plot(hours, pred.numpy(), marker="o", alpha=0.8, label=name)
    plt.xlabel("Future hour")
    plt.ylabel("Scaled OT")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.show()
    print("Saved:", path)
"""
    ),
    markdown_cell(
        r"""
## 문제 1. RNN / LSTM / GRU 비교 실험

세 모델은 recurrent block만 다르게 두고 나머지 조건을 동일하게 둔다.
"""
    ),
    code_cell(
        r"""
problem1_models: dict[str, nn.Module] = {}
problem1_histories: dict[str, dict] = {}

if RUN_PROBLEM_1:
    batch_size = experiment_batch_size(128)
    epochs = experiment_epochs(20)
    train_loader_96 = make_loader("train", DEFAULT_INPUT_LENGTH, batch_size, shuffle=True, seed_offset=101)
    val_loader_96 = make_loader("val", DEFAULT_INPUT_LENGTH, batch_size, shuffle=False)
    test_loader_96 = make_loader("test", DEFAULT_INPUT_LENGTH, batch_size, shuffle=False)

    for rnn_type in ["RNN", "LSTM", "GRU"]:
        set_global_seed(BASE_SEED)
        model = RecurrentForecastModel(
            rnn_type=rnn_type,
            hidden_dim=64,
            num_layers=1,
            pred_length=PRED_LENGTH,
            bidirectional=False,
        )
        trained_model, history = fit_model(
            model=model,
            train_loader=train_loader_96,
            val_loader=val_loader_96,
            test_loader=test_loader_96,
            epochs=epochs,
            learning_rate=1e-3,
            weight_decay=1e-4,
            name=rnn_type,
        )
        problem1_models[rnn_type] = trained_model
        problem1_histories[rnn_type] = history

    save_json(problem1_histories, output_dir / "problem1_recurrent_results.json")
    plot_loss_curves(problem1_histories, output_dir / "problem1_recurrent_val_mse_curves.png", "val_mse")
    plot_metric_bar(problem1_histories, output_dir / "problem1_recurrent_test_mse.png", "test_mse")

    _, example_target, example_predictions = collect_prediction_example(problem1_models, test_loader_96, example_index=0)
    plot_forecast_example(
        target=example_target,
        predictions=example_predictions,
        path=output_dir / "problem1_recurrent_forecast_example.png",
        title="Problem 1 Forecast Example",
    )

problem1_summary = {
    name: {
        "parameters": history["parameters"],
        "test_mse": history["test_mse"],
        "test_mae": history["test_mae"],
    }
    for name, history in problem1_histories.items()
}
pd.DataFrame(problem1_summary).T.sort_values("test_mse")
"""
    ),
    markdown_cell(
        r"""
### 문제 1 분석 초안

- RNN은 hidden state가 단순 recurrent update로 누적되므로 긴 입력에서 gradient가 약해지거나 폭주하기 쉽다.
- LSTM은 input/forget/output gate와 cell state를 사용해 필요한 정보를 더 오래 유지하고 불필요한 정보를 지울 수 있다.
- GRU는 reset/update gate로 LSTM보다 단순한 구조를 유지하면서 장기 의존성을 RNN보다 잘 다룬다.
- 최종 서술에서는 위 표의 `test_mse`, `test_mae`, 파라미터 수와 예측 시각화를 근거로 가장 좋은 모델과 trade-off를 정리한다.
"""
    ),
    markdown_cell(
        r"""
## 문제 2. 입력 길이 비교 및 bidirectional 실험

문제 1에서 test MSE가 가장 낮은 recurrent model type을 기준으로 입력 길이 24/48/96/192를 비교한다. 이후 가장 좋은 입력 길이에서 bidirectional 적용 전/후를 비교한다.
"""
    ),
    code_cell(
        r"""
problem2_length_histories: dict[str, dict] = {}
problem2_bidirectional_histories: dict[str, dict] = {}

if RUN_PROBLEM_2:
    if problem1_histories:
        best_rnn_type = min(problem1_histories, key=lambda name: problem1_histories[name]["test_mse"])
    else:
        best_rnn_type = "GRU"
    print("Problem 2 base recurrent type:", best_rnn_type)

    batch_size = experiment_batch_size(128)
    epochs = experiment_epochs(15)
    input_lengths = [24, 48, 96, 192]

    for input_length in input_lengths:
        train_loader = make_loader("train", input_length, batch_size, shuffle=True, seed_offset=200 + input_length)
        val_loader = make_loader("val", input_length, batch_size, shuffle=False)
        test_loader = make_loader("test", input_length, batch_size, shuffle=False)
        set_global_seed(BASE_SEED + input_length)
        model = RecurrentForecastModel(
            rnn_type=best_rnn_type,
            hidden_dim=64,
            num_layers=1,
            pred_length=PRED_LENGTH,
            bidirectional=False,
        )
        _, history = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            learning_rate=1e-3,
            weight_decay=1e-4,
            name=f"{best_rnn_type}_len{input_length}",
        )
        problem2_length_histories[f"len_{input_length}"] = history

    best_length_key = min(
        problem2_length_histories,
        key=lambda name: problem2_length_histories[name]["test_mse"],
    )
    best_input_length = int(best_length_key.split("_")[1])
    print("Best input length for bidirectional comparison:", best_input_length)

    for bidirectional in [False, True]:
        train_loader = make_loader("train", best_input_length, batch_size, shuffle=True, seed_offset=300 + int(bidirectional))
        val_loader = make_loader("val", best_input_length, batch_size, shuffle=False)
        test_loader = make_loader("test", best_input_length, batch_size, shuffle=False)
        set_global_seed(BASE_SEED + int(bidirectional))
        name = f"{best_rnn_type}_len{best_input_length}_bidirectional_{bidirectional}"
        model = RecurrentForecastModel(
            rnn_type=best_rnn_type,
            hidden_dim=64,
            num_layers=1,
            pred_length=PRED_LENGTH,
            bidirectional=bidirectional,
        )
        _, history = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            learning_rate=1e-3,
            weight_decay=1e-4,
            name=name,
        )
        problem2_bidirectional_histories[f"bidirectional_{bidirectional}"] = history

    save_json(
        {
            "base_rnn_type": best_rnn_type,
            "length_results": problem2_length_histories,
            "bidirectional_results": problem2_bidirectional_histories,
        },
        output_dir / "problem2_length_bidirectional_results.json",
    )
    plot_metric_bar(problem2_length_histories, output_dir / "problem2_input_length_test_mse.png", "test_mse")
    plot_metric_bar(problem2_bidirectional_histories, output_dir / "problem2_bidirectional_test_mse.png", "test_mse")

pd.DataFrame({
    name: {
        "parameters": history["parameters"],
        "test_mse": history["test_mse"],
        "test_mae": history["test_mae"],
    }
    for name, history in problem2_length_histories.items()
}).T.sort_values("test_mse")
"""
    ),
    markdown_cell(
        r"""
### 문제 2 분석 초안

- 입력 길이가 너무 짧으면 하루 이상의 주기성이나 최근 추세를 충분히 볼 수 없다.
- 입력 길이가 길어지면 더 많은 과거 정보를 얻지만, recurrent model에서는 최적화가 어려워지고 오래된 정보가 noise처럼 작동할 수 있다.
- Bidirectional recurrent layer는 입력 구간 안에서는 양방향 정보를 모두 사용할 수 있으므로 representation은 강해질 수 있다.
- 다만 실제 forecasting에서는 미래 구간을 볼 수 없어야 하며, bidirectional은 관측된 과거 window 내부를 역방향으로 읽는 것만 허용된다. online/streaming 예측에서는 latency와 causality 관점에서 적절하지 않을 수 있다.
"""
    ),
    markdown_cell(
        r"""
## 문제 3. Transformer 기반 시계열 예측

`nn.TransformerEncoderLayer`와 `nn.TransformerEncoder`를 사용한다. Positional encoding은 학습 가능한 `nn.Parameter`로 구현한다.
"""
    ),
    code_cell(
        r"""
problem3_models: dict[str, nn.Module] = {}
problem3_histories: dict[str, dict] = {}

if RUN_PROBLEM_3:
    batch_size = experiment_batch_size(128)
    epochs = experiment_epochs(20)
    train_loader_96 = make_loader("train", DEFAULT_INPUT_LENGTH, batch_size, shuffle=True, seed_offset=401)
    val_loader_96 = make_loader("val", DEFAULT_INPUT_LENGTH, batch_size, shuffle=False)
    test_loader_96 = make_loader("test", DEFAULT_INPUT_LENGTH, batch_size, shuffle=False)

    set_global_seed(BASE_SEED)
    transformer = TransformerForecastModel(
        input_length=DEFAULT_INPUT_LENGTH,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        pred_length=PRED_LENGTH,
        dropout=0.1,
    )
    trained_transformer, transformer_history = fit_model(
        model=transformer,
        train_loader=train_loader_96,
        val_loader=val_loader_96,
        test_loader=test_loader_96,
        epochs=epochs,
        learning_rate=1e-3,
        weight_decay=1e-4,
        name="TransformerEncoder",
    )
    problem3_models["TransformerEncoder"] = trained_transformer
    problem3_histories["TransformerEncoder"] = transformer_history

    comparison_histories = dict(problem3_histories)
    comparison_models = dict(problem3_models)
    if problem1_histories:
        best_problem1_name = min(problem1_histories, key=lambda name: problem1_histories[name]["test_mse"])
        comparison_histories[best_problem1_name] = problem1_histories[best_problem1_name]
        comparison_models[best_problem1_name] = problem1_models[best_problem1_name]
        print("Comparing Transformer with best Problem 1 model:", best_problem1_name)

    save_json(
        {
            "transformer": problem3_histories,
            "comparison": comparison_histories,
        },
        output_dir / "problem3_transformer_results.json",
    )
    plot_loss_curves(comparison_histories, output_dir / "problem3_transformer_comparison_val_mse.png", "val_mse")
    plot_metric_bar(comparison_histories, output_dir / "problem3_transformer_comparison_test_mse.png", "test_mse")

    _, example_target, example_predictions = collect_prediction_example(comparison_models, test_loader_96, example_index=0)
    plot_forecast_example(
        target=example_target,
        predictions=example_predictions,
        path=output_dir / "problem3_transformer_forecast_example.png",
        title="Problem 3 Forecast Example",
    )

pd.DataFrame({
    name: {
        "parameters": history["parameters"],
        "test_mse": history["test_mse"],
        "test_mae": history["test_mae"],
    }
    for name, history in {**problem1_histories, **problem3_histories}.items()
}).T.sort_values("test_mse")
"""
    ),
    markdown_cell(
        r"""
### 문제 3 분석 초안

- Transformer는 self-attention으로 입력 window의 모든 시점 사이 관계를 직접 모델링한다.
- RNN 계열은 시간 순서대로 hidden state를 누적하므로 long-range dependency가 hidden state 압축에 의존한다.
- Transformer는 병렬성이 좋고 먼 시점 간 상호작용을 짧은 path로 연결하지만, 작은 데이터셋에서는 recurrent model보다 overfitting될 수 있다.
- 최종 서술에서는 `problem3_transformer_results.json`, test MSE/MAE 표, 예측 구간 그래프를 근거로 문제 1 best recurrent model과 비교한다.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


output_path = Path(__file__).with_name("DL_Lab0_HW04.ipynb")
output_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {output_path}")
