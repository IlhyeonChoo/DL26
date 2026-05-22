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
# DL Lab HW03 - CIFAR-10 CNN and Transfer Learning

이 노트북은 `Todo.md`의 문제 1~5를 수행하기 위한 실험 노트북이다.

HW02와 같은 원칙을 유지한다. 한 번에 비교하려는 요인만 바꾸고, 데이터 전처리, epoch 수, optimizer, scheduler, batch size는 문제별로 가능한 한 고정한다.

Colab에서 실행할 때는 `런타임 > 런타임 유형 변경 > GPU`를 선택한 뒤 위에서부터 실행한다. Colab에서는 기본적으로 Google Drive를 마운트하고 결과를 `/content/drive/MyDrive/DL26/hw03/results_hw03`에 저장한다. CPU 런타임에서는 자동으로 fast debug mode가 켜져 epoch, batch size, batch 수를 줄인 smoke 실행만 수행한다. 제출용 결과를 만들 때는 GPU 런타임에서 `COLAB_FAST_DEV_RUN = False`, `MAX_TRAIN_BATCHES = None`, `MAX_EVAL_BATCHES = None` 상태로 실행한다.

| 문제 | 비교 대상 | 고정 조건 |
| --- | --- | --- |
| 문제 1 | 직접 설계한 CNN | CIFAR-10, 32x32 입력, AdamW, cosine scheduler |
| 문제 2 | VGG-16 scratch vs ResNet-18 scratch | weights=None, 같은 optimizer/lr/scheduler/epoch |
| 문제 3 | ResNet-18 scratch vs feature extraction vs fine-tuning | 224x224 입력, 같은 epoch, transfer 조건만 변경 |
| 문제 4 | 문제 3 코드 검증 | shape, dummy forward, gradient/freeze, intentional error |
| 문제 5 | 실험 회고 | 실제 결과를 반영해 보완할 초안 |
"""
    ),
    code_cell(
        r"""
import json
import os
import random
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


BASE_SEED = 42
NUM_CLASSES = 10

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
GOOGLE_DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/DL26/hw03")

if IN_COLAB:
    data_dir = Path("/content/data")
    if MOUNT_GOOGLE_DRIVE_IN_COLAB:
        try:
            from google.colab import drive  # type: ignore[import-not-found]

            drive.mount("/content/drive")
            output_dir = GOOGLE_DRIVE_PROJECT_DIR / "results_hw03"
        except Exception as exc:
            print(f"Google Drive mount failed. Falling back to /content/results_hw03. Reason: {exc}")
            output_dir = Path("/content/results_hw03")
    else:
        output_dir = Path("/content/results_hw03")
else:
    project_dir = Path("hw03") if Path("hw03/Todo.md").exists() else Path(".")
    data_dir = project_dir / "data"
    output_dir = project_dir / "results_hw03"

data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Keep these True for the assignment run. For a quick syntax/debug pass, set
# MAX_TRAIN_BATCHES and MAX_EVAL_BATCHES to small integers instead of changing
# the experiment definitions.
RUN_PROBLEM_1 = True
RUN_PROBLEM_2 = True
RUN_PROBLEM_3 = True
RUN_PROBLEM_4 = True

COLAB_FAST_DEV_RUN = IN_COLAB and device.type != "cuda"
MAX_TRAIN_BATCHES = 2 if COLAB_FAST_DEV_RUN else None
MAX_EVAL_BATCHES = 2 if COLAB_FAST_DEV_RUN else None
USE_AMP = torch.cuda.is_available()
AMP_INIT_SCALE = 1024.0

if COLAB_FAST_DEV_RUN:
    print("Colab CPU runtime detected. Fast debug mode is enabled.")
    print("For assignment results, switch Colab runtime to GPU and set COLAB_FAST_DEV_RUN = False.")


def experiment_epochs(full_epochs: int) -> int:
    return 1 if COLAB_FAST_DEV_RUN else full_epochs


def experiment_batch_size(full_batch_size: int, fast_batch_size: int = 8) -> int:
    return fast_batch_size if COLAB_FAST_DEV_RUN else full_batch_size


print("Using device:", device)
print("Running in Colab:", IN_COLAB)
print("Mount Google Drive in Colab:", MOUNT_GOOGLE_DRIVE_IN_COLAB)
if IN_COLAB and MOUNT_GOOGLE_DRIVE_IN_COLAB:
    print("Google Drive result directory:", GOOGLE_DRIVE_PROJECT_DIR / "results_hw03")
print("DataLoader workers:", NUM_WORKERS)
print("Data directory:", data_dir.resolve())
print("Output directory:", output_dir.resolve())
print("MAX_TRAIN_BATCHES:", MAX_TRAIN_BATCHES)
print("MAX_EVAL_BATCHES:", MAX_EVAL_BATCHES)
print("Automatic mixed precision:", USE_AMP)
print("AMP initial scale:", AMP_INIT_SCALE)
"""
    ),
    markdown_cell(
        r"""
## 공통 데이터 구성

- CIFAR-10 train 50,000장과 test 10,000장을 사용한다.
- 문제 1과 문제 2는 CIFAR-10 원본 해상도인 32x32 입력을 사용한다.
- 문제 3은 ImageNet 사전학습 weight와 입력 조건을 맞추기 위해 224x224로 resize한다.
- train transform에는 horizontal flip 같은 약한 augmentation을 넣고, test transform에는 deterministic transform만 넣는다.
"""
    ),
    code_cell(
        r"""
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

train_transform_32 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

test_transform_32 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

train_transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

test_transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

train_dataset_32 = datasets.CIFAR10(
    root=str(data_dir),
    train=True,
    download=True,
    transform=train_transform_32,
)

test_dataset_32 = datasets.CIFAR10(
    root=str(data_dir),
    train=False,
    download=True,
    transform=test_transform_32,
)

train_dataset_224 = datasets.CIFAR10(
    root=str(data_dir),
    train=True,
    download=True,
    transform=train_transform_224,
)

test_dataset_224 = datasets.CIFAR10(
    root=str(data_dir),
    train=False,
    download=True,
    transform=test_transform_224,
)

train_indices = list(range(len(train_dataset_32)))


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    seed_offset: int,
    indices: list[int] | None = None,
) -> DataLoader:
    selected_dataset = Subset(dataset, indices) if indices is not None else dataset
    generator = torch.Generator().manual_seed(BASE_SEED + seed_offset) if shuffle else None
    return DataLoader(
        selected_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )


print(f"CIFAR-10 train samples: {len(train_dataset_32):,}")
print(f"CIFAR-10 test samples: {len(test_dataset_32):,}")
print("Classes:", class_names)
"""
    ),
    markdown_cell(
        r"""
## 공통 모델 정의

문제 1의 직접 설계 CNN은 다음 조건을 만족한다.

- Conv2d layer 3개
- MaxPool2d pooling
- BatchNorm2d를 각 convolution block 뒤에 적용
- Dropout2d와 Dropout을 feature extractor와 classifier에 적용
- ReLU activation

문제 2와 문제 3은 `torchvision.models`의 VGG-16, ResNet-18 구조를 사용하고, 마지막 classifier만 CIFAR-10의 10개 class에 맞게 수정한다.
"""
    ),
    code_cell(
        r"""
class CustomCIFARCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.35) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout * 0.5),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_custom_cnn() -> nn.Module:
    return CustomCIFARCNN(num_classes=NUM_CLASSES, dropout=0.35)


def build_vgg16_scratch() -> nn.Module:
    model = models.vgg16(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def build_resnet18_scratch(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet18_transfer(
    mode: str,
    num_classes: int = NUM_CLASSES,
    use_pretrained: bool = True,
) -> nn.Module:
    if mode not in {"scratch", "feature_extraction", "fine_tuning"}:
        raise ValueError(f"Unsupported transfer mode: {mode}")

    weights = None
    if use_pretrained and mode in {"feature_extraction", "fine_tuning"}:
        weights = ResNet18_Weights.IMAGENET1K_V1

    model = models.resnet18(weights=weights)

    if mode == "feature_extraction":
        for parameter in model.parameters():
            parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.transfer_mode = mode
    return model


def freeze_batchnorm_if_needed(model: nn.Module) -> None:
    if getattr(model, "transfer_mode", None) != "feature_extraction":
        return
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def model_parameter_report(model: nn.Module) -> dict[str, int]:
    return {
        "total_parameters": count_parameters(model, trainable_only=False),
        "trainable_parameters": count_parameters(model, trainable_only=True),
    }


preview_model_builders = {
    "custom_cnn": build_custom_cnn,
    "vgg16_scratch": build_vgg16_scratch,
    "resnet18_scratch": build_resnet18_scratch,
    "resnet18_feature_extraction": lambda: build_resnet18_transfer("feature_extraction", use_pretrained=False),
    "resnet18_fine_tuning": lambda: build_resnet18_transfer("fine_tuning", use_pretrained=False),
}

for preview_name, preview_builder in preview_model_builders.items():
    preview_model = preview_builder()
    print(preview_name, model_parameter_report(preview_model))
    del preview_model

del preview_model_builders
"""
    ),
    markdown_cell(
        r"""
## 공통 학습 및 시각화 유틸리티

모든 실험은 같은 학습 루프를 사용한다. 이렇게 하면 문제별 비교에서 training step, loss 계산, accuracy 계산 방식이 섞이지 않는다.

`MAX_TRAIN_BATCHES`와 `MAX_EVAL_BATCHES`는 빠른 디버깅용이다. 실제 제출 결과를 만들 때는 둘 다 `None`으로 둔다.
"""
    ),
    code_cell(
        r"""
criterion = nn.CrossEntropyLoss()


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
            return torch.amp.GradScaler(
                device="cuda",
                enabled=enabled,
                init_scale=AMP_INIT_SCALE,
            )
        except TypeError:
            try:
                return torch.amp.GradScaler(
                    "cuda",
                    enabled=enabled,
                    init_scale=AMP_INIT_SCALE,
                )
            except TypeError:
                pass
    return torch.cuda.amp.GradScaler(
        enabled=enabled,
        init_scale=AMP_INIT_SCALE,
    )


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.0,
) -> torch.optim.Optimizer:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("The model has no trainable parameters.")

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            trainable_parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> object | None:
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: object,
    max_batches: int | None = None,
) -> tuple[float, float, int]:
    model.train()
    freeze_batchnorm_if_needed(model)
    total_loss = 0.0
    correct = 0
    total = 0
    optimizer_steps = 0

    for batch_index, (xb, yb) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            logits = model(xb)
            loss = criterion(logits, yb)

        if scaler.is_enabled():
            scale_before_step = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before_step:
                optimizer_steps += 1
        else:
            loss.backward()
            optimizer.step()
            optimizer_steps += 1

        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += yb.size(0)

    return total_loss / total, correct / total, optimizer_steps


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_index, (xb, yb) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with autocast_context():
                logits = model(xb)
                loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)

    return total_loss / total, correct / total


def run_training_experiment(
    name: str,
    model_builder: Callable[[], nn.Module],
    config: dict,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> dict:
    set_global_seed(config.get("model_seed", BASE_SEED))
    model = model_builder().to(device)
    optimizer = build_optimizer(
        model=model,
        optimizer_name=config["optimizer"],
        learning_rate=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
        momentum=config.get("momentum", 0.0),
    )
    scheduler = build_scheduler(config.get("scheduler", "none"), optimizer, config["epochs"])
    scaler = make_grad_scaler()

    print(f"\n===== {name} =====")
    print("Config:", json.dumps(config, indent=2, ensure_ascii=False))
    print("Parameters:", model_parameter_report(model))

    history: list[dict] = []
    best_test_accuracy = -1.0
    best_epoch = 0

    for epoch in range(config["epochs"]):
        train_loss, train_accuracy, optimizer_steps = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            max_batches=MAX_TRAIN_BATCHES,
        )
        test_loss, test_accuracy = evaluate(
            model=model,
            loader=test_loader,
            max_batches=MAX_EVAL_BATCHES,
        )
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "optimizer_steps": optimizer_steps,
            }
        )

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_epoch = epoch + 1

        print(
            f"[{name}] Epoch {epoch + 1:02d}/{config['epochs']} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f} | "
            f"test_loss={test_loss:.4f}, test_acc={test_accuracy:.4f}"
        )

        if scheduler is not None and optimizer_steps > 0:
            scheduler.step()

    final_entry = history[-1]
    result = {
        "name": name,
        "config": deepcopy(config),
        "total_parameters": count_parameters(model, trainable_only=False),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "history": history,
        "best_epoch_by_test_accuracy": best_epoch,
        "best_test_accuracy": best_test_accuracy,
        "final_train_loss": final_entry["train_loss"],
        "final_train_accuracy": final_entry["train_accuracy"],
        "final_test_loss": final_entry["test_loss"],
        "final_test_accuracy": final_entry["test_accuracy"],
    }
    print(
        f"[{name}] best_epoch={best_epoch}, "
        f"best_test_acc={best_test_accuracy:.4f}, "
        f"final_test_acc={final_entry['test_accuracy']:.4f}"
    )
    return result


def plot_loss_accuracy(
    results: dict[str, dict],
    title: str,
    filename: str,
) -> None:
    if not results:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for run_name, result in results.items():
        history = result["history"]
        epochs = [entry["epoch"] for entry in history]

        axes[0].plot(
            epochs,
            [entry["train_loss"] for entry in history],
            linestyle="-",
            label=f"{run_name} train",
        )
        axes[0].plot(
            epochs,
            [entry["test_loss"] for entry in history],
            linestyle="--",
            label=f"{run_name} test",
        )
        axes[1].plot(
            epochs,
            [entry["train_accuracy"] for entry in history],
            linestyle="-",
            label=f"{run_name} train",
        )
        axes[1].plot(
            epochs,
            [entry["test_accuracy"] for entry in history],
            linestyle="--",
            label=f"{run_name} test",
        )

    axes[0].set_title(f"{title} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title(f"{title} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150)
    plt.show()


def summarize_results(results: dict[str, dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for run_name, result in results.items():
        trainable_millions = result["trainable_parameters"] / 1_000_000
        summary[run_name] = {
            "total_parameters": result["total_parameters"],
            "trainable_parameters": result["trainable_parameters"],
            "best_epoch_by_test_accuracy": result["best_epoch_by_test_accuracy"],
            "best_test_accuracy": result["best_test_accuracy"],
            "final_train_accuracy": result["final_train_accuracy"],
            "final_test_accuracy": result["final_test_accuracy"],
            "final_test_loss": result["final_test_loss"],
            "test_accuracy_per_million_trainable_parameters": (
                result["final_test_accuracy"] / trainable_millions
                if trainable_millions > 0
                else None
            ),
        }
    return summary
"""
    ),
    markdown_cell(
        r"""
## 문제 1. CNN 모델 직접 설계 및 학습

직접 설계한 CNN은 작은 CIFAR-10 이미지에 맞춰 VGG/ResNet보다 훨씬 작은 모델로 둔다.

- BatchNorm은 각 Conv2d 뒤에 배치한다. convolution output의 분포를 안정화해 learning rate와 초기화에 덜 민감하게 만들기 위해서다.
- Dropout2d는 pooling 이후 feature map에 적용한다. feature map 일부에 과도하게 의존하는 것을 줄이기 위해서다.
- classifier 앞의 Dropout은 최종 feature 조합의 overfitting을 줄이기 위해 적용한다.
"""
    ),
    code_cell(
        r"""
problem1_config = {
    "model": "CustomCIFARCNN",
    "optimizer": "adamw",
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "epochs": experiment_epochs(20),
    "batch_size": experiment_batch_size(128, fast_batch_size=16),
    "model_seed": BASE_SEED,
}

problem1_results: dict[str, dict] = {}

if RUN_PROBLEM_1:
    train_loader_32_p1 = make_loader(
        train_dataset_32,
        batch_size=problem1_config["batch_size"],
        shuffle=True,
        seed_offset=100,
    )
    test_loader_32_p1 = make_loader(
        test_dataset_32,
        batch_size=problem1_config["batch_size"],
        shuffle=False,
        seed_offset=101,
    )
    problem1_results["custom_cnn"] = run_training_experiment(
        name="custom_cnn",
        model_builder=build_custom_cnn,
        config=problem1_config,
        train_loader=train_loader_32_p1,
        test_loader=test_loader_32_p1,
    )

    problem1_notes = {
        "model_structure": [
            "Conv2d(3,64,3,padding=1) -> BatchNorm2d -> ReLU",
            "Conv2d(64,64,3,padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d -> Dropout2d",
            "Conv2d(64,128,3,padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d -> Dropout2d",
            "AdaptiveAvgPool2d -> Linear(128,128) -> ReLU -> Dropout -> Linear(128,10)",
        ],
        "batchnorm_position_reason": (
            "BatchNorm is placed after each convolution to stabilize feature distributions "
            "before ReLU activation."
        ),
        "dropout_position_reason": (
            "Dropout2d is placed after pooling blocks and Dropout is placed in the classifier "
            "to reduce co-adaptation and overfitting."
        ),
    }

    save_json(problem1_results, output_dir / "problem1_custom_cnn_results.json")
    save_json(problem1_notes, output_dir / "problem1_model_notes.json")
    plot_loss_accuracy(
        problem1_results,
        title="Problem 1 Custom CNN",
        filename="problem1_custom_cnn_curves.png",
    )
    problem1_summary = summarize_results(problem1_results)
    save_json(problem1_summary, output_dir / "problem1_summary.json")
    print(json.dumps(problem1_summary, indent=2, ensure_ascii=False))
else:
    print("Problem 1 run is disabled.")
"""
    ),
    markdown_cell(
        r"""
## 문제 2. Pre-defined 모델 구조 활용

`torchvision.models`의 VGG-16과 ResNet-18을 `weights=None`으로 불러와 처음부터 학습한다.

통제 조건:

- 두 모델 모두 CIFAR-10의 32x32 입력을 사용한다.
- pretrained weight는 사용하지 않는다.
- 마지막 classifier만 10-class 출력으로 바꾼다.
- optimizer, learning rate, weight decay, scheduler, epoch, batch size는 동일하게 둔다.

분석할 때는 단순 accuracy뿐 아니라 파라미터 수 대비 성능 효율도 함께 본다.
"""
    ),
    code_cell(
        r"""
problem2_shared_config = {
    "optimizer": "adamw",
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "epochs": experiment_epochs(15),
    "batch_size": experiment_batch_size(64, fast_batch_size=8),
    "model_seed": BASE_SEED,
    "weights": "None",
}

problem2_configs = {
    "vgg16_scratch": {
        **problem2_shared_config,
        "model": "torchvision.models.vgg16(weights=None)",
    },
    "resnet18_scratch": {
        **problem2_shared_config,
        "model": "torchvision.models.resnet18(weights=None)",
    },
}

problem2_builders = {
    "vgg16_scratch": build_vgg16_scratch,
    "resnet18_scratch": build_resnet18_scratch,
}

problem2_results: dict[str, dict] = {}

if RUN_PROBLEM_2:
    train_loader_32_p2 = make_loader(
        train_dataset_32,
        batch_size=problem2_shared_config["batch_size"],
        shuffle=True,
        seed_offset=200,
    )
    test_loader_32_p2 = make_loader(
        test_dataset_32,
        batch_size=problem2_shared_config["batch_size"],
        shuffle=False,
        seed_offset=201,
    )

    for run_name, run_config in problem2_configs.items():
        problem2_results[run_name] = run_training_experiment(
            name=run_name,
            model_builder=problem2_builders[run_name],
            config=run_config,
            train_loader=train_loader_32_p2,
            test_loader=test_loader_32_p2,
        )

    save_json(problem2_results, output_dir / "problem2_predefined_scratch_results.json")
    plot_loss_accuracy(
        problem2_results,
        title="Problem 2 VGG16 vs ResNet18 Scratch",
        filename="problem2_predefined_scratch_curves.png",
    )
    problem2_summary = summarize_results(problem2_results)
    save_json(problem2_summary, output_dir / "problem2_summary.json")
    print(json.dumps(problem2_summary, indent=2, ensure_ascii=False))

    if problem1_results:
        model_efficiency_comparison = summarize_results({
            **problem1_results,
            **problem2_results,
        })
        save_json(
            model_efficiency_comparison,
            output_dir / "problem2_custom_vgg_resnet_efficiency_summary.json",
        )
        print("Custom CNN / VGG-16 / ResNet-18 efficiency comparison")
        print(json.dumps(model_efficiency_comparison, indent=2, ensure_ascii=False))
else:
    print("Problem 2 run is disabled.")
"""
    ),
    markdown_cell(
        r"""
## 문제 3. Pre-trained Weight를 활용한 Transfer Learning

ResNet-18 기준으로 세 조건을 비교한다.

1. From Scratch: `weights=None`, 전체 모델 학습
2. Feature Extraction: `weights=IMAGENET1K_V1`, backbone freeze, 마지막 FC만 학습
3. Fine-tuning: `weights=IMAGENET1K_V1`, 전체 모델 학습

ImageNet weight는 224x224 입력을 기준으로 학습되었으므로, 이 문제에서는 CIFAR-10 이미지를 224x224로 resize하고 ImageNet mean/std로 normalize한다.

세 조건 모두 epoch 수는 동일하게 둔다. Fine-tuning은 이미 학습된 feature를 크게 망가뜨리지 않도록 scratch보다 낮은 learning rate를 사용한다.
"""
    ),
    code_cell(
        r"""
problem3_shared_config = {
    "optimizer": "adamw",
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "epochs": experiment_epochs(10),
    "batch_size": experiment_batch_size(64, fast_batch_size=8),
    "model_seed": BASE_SEED,
    "input_size": "224x224",
}

problem3_configs = {
    "resnet18_scratch_224": {
        **problem3_shared_config,
        "model": "resnet18",
        "transfer_mode": "scratch",
        "weights": "None",
        "learning_rate": 3e-4,
    },
    "resnet18_feature_extraction": {
        **problem3_shared_config,
        "model": "resnet18",
        "transfer_mode": "feature_extraction",
        "weights": "IMAGENET1K_V1",
        "learning_rate": 1e-3,
    },
    "resnet18_fine_tuning": {
        **problem3_shared_config,
        "model": "resnet18",
        "transfer_mode": "fine_tuning",
        "weights": "IMAGENET1K_V1",
        "learning_rate": 1e-4,
    },
}


def make_transfer_builder(mode: str) -> Callable[[], nn.Module]:
    def builder() -> nn.Module:
        return build_resnet18_transfer(mode=mode, use_pretrained=True)

    return builder


problem3_results: dict[str, dict] = {}

if RUN_PROBLEM_3:
    train_loader_224 = make_loader(
        train_dataset_224,
        batch_size=problem3_shared_config["batch_size"],
        shuffle=True,
        seed_offset=300,
    )
    test_loader_224 = make_loader(
        test_dataset_224,
        batch_size=problem3_shared_config["batch_size"],
        shuffle=False,
        seed_offset=301,
    )

    for run_name, run_config in problem3_configs.items():
        mode = run_config["transfer_mode"]
        problem3_results[run_name] = run_training_experiment(
            name=run_name,
            model_builder=make_transfer_builder(mode),
            config=run_config,
            train_loader=train_loader_224,
            test_loader=test_loader_224,
        )

    save_json(problem3_results, output_dir / "problem3_transfer_learning_results.json")
    plot_loss_accuracy(
        problem3_results,
        title="Problem 3 ResNet18 Transfer Learning",
        filename="problem3_transfer_learning_curves.png",
    )
    problem3_summary = summarize_results(problem3_results)
    save_json(problem3_summary, output_dir / "problem3_summary.json")
    print(json.dumps(problem3_summary, indent=2, ensure_ascii=False))

    early_convergence = {
        run_name: [
            {
                "epoch": entry["epoch"],
                "train_accuracy": entry["train_accuracy"],
                "test_accuracy": entry["test_accuracy"],
            }
            for entry in result["history"][:5]
        ]
        for run_name, result in problem3_results.items()
    }
    save_json(early_convergence, output_dir / "problem3_early_convergence_epochs_1_to_5.json")
    print("Early convergence, epochs 1 to 5")
    print(json.dumps(early_convergence, indent=2, ensure_ascii=False))
else:
    print("Problem 3 run is disabled.")
"""
    ),
    markdown_cell(
        r"""
## 문제 4. LLM 생성 코드의 검증

문제 3의 transfer learning 코드를 다음 방식으로 검증한다.

- 차원 검증: ResNet-18 주요 layer의 output shape를 hook으로 기록한다.
- 더미 테스트: 224x224 dummy input을 넣어 output shape가 `(batch_size, 10)`인지 확인한다.
- gradient/freeze 검증: feature extraction에서 backbone gradient가 생기지 않고 FC layer gradient만 생기는지 확인한다.
- 의도적 오류 삽입: 마지막 FC layer를 CIFAR-10에 맞게 바꾸지 않은 ResNet-18을 만들어 검증 함수가 오류를 잡는지 확인한다.

이 검증은 구조와 gradient 흐름을 확인한다. 실제 pretrained weight의 semantic quality나 최종 성능이 충분한지는 전체 학습 결과로 별도 확인해야 한다.
"""
    ),
    code_cell(
        r"""
def capture_resnet_shapes(model: nn.Module, input_shape: tuple[int, int, int, int]) -> dict[str, list[int]]:
    shape_report: dict[str, list[int]] = {}
    module_names = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
    hooks = []

    def make_hook(name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            shape_report[name] = list(output.shape)

        return hook

    for name, module in model.named_modules():
        if name in module_names:
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        dummy = torch.randn(*input_shape, device=device)
        model(dummy)

    for hook in hooks:
        hook.remove()

    return shape_report


def verify_transfer_model(mode: str) -> dict:
    model = build_resnet18_transfer(mode=mode, use_pretrained=False).to(device)
    shape_report = capture_resnet_shapes(model, input_shape=(2, 3, 224, 224))

    model.train()
    freeze_batchnorm_if_needed(model)
    model.zero_grad(set_to_none=True)

    dummy_x = torch.randn(2, 3, 224, 224, device=device)
    dummy_y = torch.tensor([0, 1], device=device)
    logits = model(dummy_x)
    loss = criterion(logits, dummy_y)
    loss.backward()

    fc_grad_exists = model.fc.weight.grad is not None and torch.isfinite(model.fc.weight.grad).all().item()
    frozen_backbone_grads = []
    trainable_backbone_grads = []
    for name, parameter in model.named_parameters():
        if name.startswith("fc."):
            continue
        if parameter.requires_grad and parameter.grad is not None:
            trainable_backbone_grads.append(name)
        if not parameter.requires_grad and parameter.grad is not None:
            frozen_backbone_grads.append(name)

    expected_trainable_backbone = mode in {"scratch", "fine_tuning"}
    return {
        "mode": mode,
        "weight_source_for_verification": "None; structure and freeze policy only",
        "output_shape": list(logits.shape),
        "output_shape_ok": list(logits.shape) == [2, NUM_CLASSES],
        "finite_output": torch.isfinite(logits).all().item(),
        "shape_report": shape_report,
        "total_parameters": count_parameters(model, trainable_only=False),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "fc_gradient_exists": bool(fc_grad_exists),
        "frozen_backbone_gradient_count": len(frozen_backbone_grads),
        "trainable_backbone_gradient_count": len(trainable_backbone_grads),
        "freeze_policy_ok": (
            len(frozen_backbone_grads) == 0
            and (len(trainable_backbone_grads) > 0) == expected_trainable_backbone
        ),
    }


def verify_intentional_error() -> dict:
    broken_model = models.resnet18(weights=None).to(device)
    broken_model.eval()
    with torch.no_grad():
        dummy_x = torch.randn(2, 3, 224, 224, device=device)
        logits = broken_model(dummy_x)

    return {
        "intentional_error": "ResNet-18 final fc layer was not changed from 1000 ImageNet classes to 10 CIFAR-10 classes.",
        "output_shape": list(logits.shape),
        "expected_shape": [2, NUM_CLASSES],
        "detected": list(logits.shape) != [2, NUM_CLASSES],
    }


verification_report: dict[str, dict] = {}

if RUN_PROBLEM_4:
    for mode_name in ["scratch", "feature_extraction", "fine_tuning"]:
        verification_report[mode_name] = verify_transfer_model(mode_name)

    verification_report["intentional_error_unmodified_fc"] = verify_intentional_error()
    save_json(verification_report, output_dir / "problem4_verification_report.json")
    print(json.dumps(verification_report, indent=2, ensure_ascii=False))

    print("\n검증 해석 가이드")
    print("- output_shape_ok가 True이면 CIFAR-10 class 수에 맞는 logits가 나온다.")
    print("- feature_extraction의 trainable_parameters가 전체보다 훨씬 작고 frozen_backbone_gradient_count가 0이면 freeze가 작동한 것이다.")
    print("- intentional_error_unmodified_fc.detected가 True이면 마지막 FC layer 미수정 오류를 검증 코드가 잡아낸 것이다.")
else:
    print("Problem 4 run is disabled.")
"""
    ),
    markdown_cell(
        r"""
## 문제 5. 실험 회고 및 고찰 초안

### 본인 코드에 대한 팀원의 검증 평가 또는 소감

팀원이 문제 3의 transfer learning 코드를 검증할 때 가장 중요한 지점은 코드가 실행되는지보다 “의도한 학습 조건이 실제로 적용되는지”다. 특히 feature extraction 조건에서는 backbone parameter가 freeze되어야 하고, 마지막 FC layer만 학습되어야 한다. 단순히 loss가 감소하는지만 확인하면 freeze 누락, 마지막 layer class 수 오류, 입력 resize 누락 같은 문제를 놓칠 수 있다.

### 팀원의 코드를 검증하는 과정에서의 한계점

shape와 gradient 검증은 구조적 오류를 잘 찾지만, pretrained feature가 실제로 CIFAR-10에 유효한지까지 보장하지는 않는다. 또한 짧은 dummy test는 data augmentation, scheduler, optimizer 설정이 장기 학습에서 적절한지 검증하지 못한다. 따라서 코드 검증은 shape/gradient 같은 정적 검증과 실제 train/test curve 분석을 함께 수행해야 한다.

### 실행은 되지만 올바르지 않은 코드를 검증하기 위해 필요한 지식

실행 가능한 코드가 항상 올바른 실험은 아니다. Transfer learning에서는 `requires_grad`, optimizer에 전달되는 parameter 목록, model train/eval mode, BatchNorm 동작, 입력 normalization, classifier 출력 차원을 이해해야 한다. 또한 학습 곡선을 해석하려면 overfitting, underfitting, 수렴 속도, 모델 capacity, 데이터셋 차이와 같은 딥러닝 실험 지식이 필요하다.

### Pre-trained weight의 의미와 CIFAR-10에서 유효한 이유

Pre-trained weight는 ImageNet 같은 큰 데이터셋에서 먼저 학습된 convolution filter와 feature hierarchy를 의미한다. 초기 layer는 edge, color contrast, texture 같은 일반적인 시각 feature를 학습하는 경향이 있고, 이런 feature는 CIFAR-10의 작은 자연 이미지에도 어느 정도 유효하다. 따라서 feature extraction은 적은 학습으로 빠르게 수렴할 수 있고, fine-tuning은 사전학습 feature를 CIFAR-10 분류 경계에 맞게 조정할 수 있다.
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
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


explanation = r"""
# DL_Lab0_HW03.ipynb 설계 설명서

이 문서는 `DL_Lab0_HW03.ipynb`의 구성과 `Todo.md` 요구사항의 대응 관계를 정리한다.

각 셀과 함수의 동작 방식, 모델 구조와 hyperparameter 선택 이유는 `DL_Lab0_HW03_detailed_explanation.md`에 더 자세히 정리했다.

## 설계 원칙

HW02의 핵심 원칙인 변인 통제를 유지했다. 문제 1은 직접 설계 CNN의 구조와 학습 결과를 기록하고, 문제 2는 pre-defined architecture를 같은 조건에서 scratch로 학습한다. 문제 3은 ResNet-18 transfer learning의 세 조건을 비교하며, 문제 4는 해당 코드가 의도대로 작동하는지 검증한다.

## 셀 구성

| 셀 | 종류 | 역할 | 관련 문제 |
| --- | --- | --- | --- |
| 0 | Markdown | 전체 실험 개요와 비교 조건 | 전체 |
| 1 | Code | import, seed, device, 실행 flag, 경로 설정 | 전체 |
| 2 | Markdown | CIFAR-10 데이터 구성 설명 | 전체 |
| 3 | Code | 32x32/224x224 transform, CIFAR-10 dataset, DataLoader 함수 | 문제 1~3 |
| 4 | Markdown | 공통 모델 정의 설명 | 문제 1~3 |
| 5 | Code | Custom CNN, VGG-16, ResNet-18, transfer 모델 builder | 문제 1~3 |
| 6 | Markdown | 학습/평가 유틸리티 설명 | 전체 |
| 7 | Code | optimizer, scheduler, train/eval loop, plot/summary 함수 | 전체 |
| 8 | Markdown | 직접 설계 CNN 설계 근거 | 문제 1 |
| 9 | Code | Custom CNN 학습, 그래프, summary 저장 | 문제 1 |
| 10 | Markdown | VGG-16/ResNet-18 scratch 비교 조건 | 문제 2 |
| 11 | Code | VGG-16과 ResNet-18 scratch 학습 및 효율 비교 | 문제 2 |
| 12 | Markdown | transfer learning 비교 조건 | 문제 3 |
| 13 | Code | ResNet-18 scratch/feature extraction/fine-tuning 학습 | 문제 3 |
| 14 | Markdown | LLM 생성 코드 검증 방법 | 문제 4 |
| 15 | Code | shape, dummy, gradient/freeze, intentional error 검증 | 문제 4 |
| 16 | Markdown | 회고 및 고찰 초안 | 문제 5 |

## 생성되는 결과 파일

노트북 실행 결과는 `results_hw03/` 아래에 저장된다.

| 파일 | 내용 |
| --- | --- |
| `problem1_custom_cnn_results.json` | 직접 설계 CNN의 epoch별 train/test 기록 |
| `problem1_custom_cnn_curves.png` | 직접 설계 CNN loss/accuracy 그래프 |
| `problem1_model_notes.json` | 모델 구조, BatchNorm/Dropout 적용 위치와 이유 |
| `problem1_summary.json` | 문제 1 성능 및 파라미터 요약 |
| `problem2_predefined_scratch_results.json` | VGG-16, ResNet-18 scratch 학습 기록 |
| `problem2_predefined_scratch_curves.png` | VGG-16, ResNet-18 scratch 그래프 |
| `problem2_summary.json` | 문제 2 성능 및 파라미터 요약 |
| `problem2_custom_vgg_resnet_efficiency_summary.json` | Custom CNN, VGG-16, ResNet-18의 파라미터 효율 비교 |
| `problem3_transfer_learning_results.json` | transfer learning 세 조건의 학습 기록 |
| `problem3_transfer_learning_curves.png` | transfer learning loss/accuracy 그래프 |
| `problem3_summary.json` | 문제 3 성능 및 학습 가능한 파라미터 요약 |
| `problem3_early_convergence_epochs_1_to_5.json` | 1~5 epoch 수렴 속도 비교용 기록 |
| `problem4_verification_report.json` | shape, dummy, gradient/freeze, intentional error 검증 결과 |

## 실행 방법

노트북을 재생성하려면 다음 명령을 사용한다.

```bash
cd hw03
python build_hw03_notebook.py
```

## Colab 실행 방법

1. `DL_Lab0_HW03.ipynb`를 Colab에 업로드한다.
2. `런타임 > 런타임 유형 변경 > GPU`를 선택한다.
3. 위에서부터 전체 실행한다.
4. 첫 설정 셀에서 Google Drive 권한 요청이 뜨면 승인한다.

노트북은 Colab을 감지하고, GPU가 없으면 `COLAB_FAST_DEV_RUN = True`로 전환해 epoch, batch size, batch 수를 줄인 smoke 실행만 수행한다. 이 모드는 코드 동작 확인용이다. 제출용 결과는 GPU 런타임에서 `COLAB_FAST_DEV_RUN = False`, `MAX_TRAIN_BATCHES = None`, `MAX_EVAL_BATCHES = None` 상태로 실행해야 한다.

로컬 repo root에서 실행하면 `hw03/data`, `hw03/results_hw03`를 사용한다. Colab에 노트북만 업로드해 실행하면 CIFAR-10 데이터는 `/content/data`에 저장하고, 결과 파일은 Google Drive의 `/content/drive/MyDrive/DL26/hw03/results_hw03`에 저장한다. Drive 마운트를 원하지 않으면 첫 설정 셀에서 `MOUNT_GOOGLE_DRIVE_IN_COLAB = False`로 바꾸면 결과가 `/content/results_hw03`에 저장된다.
"""


detailed_explanation = r"""
# DL_Lab0_HW03.ipynb 상세 코드 설명서

이 문서는 `DL_Lab0_HW03.ipynb`의 각 셀과 내부 함수가 어떻게 동작하는지, 왜 그런 구조와 옵션을 선택했는지 설명한다. 단순히 코드를 읽는 용도가 아니라, 과제 보고서에서 모델 구조, 실험 조건, 코드 검증 방법을 설명할 때 근거로 사용할 수 있도록 작성했다.

## 전체 설계 방향

HW03의 핵심은 CIFAR-10 이미지 분류를 CNN, pre-defined architecture, transfer learning 관점에서 비교하는 것이다. 그래서 노트북은 다음 기준으로 구성했다.

| 기준 | 적용 방식 | 이유 |
| --- | --- | --- |
| 데이터셋 | CIFAR-10 고정 | 모델 구조와 학습 방식 차이만 비교하기 위해서 |
| 문제 1 입력 | 32x32 | 직접 설계 CNN은 CIFAR-10 원본 해상도에 맞춘 작은 모델로 실험하기 위해서 |
| 문제 2 입력 | 32x32 | VGG-16, ResNet-18을 scratch로 동일 입력 조건에서 비교하기 위해서 |
| 문제 3 입력 | 224x224 | ImageNet pretrained weight의 원래 학습 조건과 맞추기 위해서 |
| 공통 출력 | 10 classes | CIFAR-10 class 수와 classifier 출력 차원을 맞추기 위해서 |
| 결과 저장 | JSON + PNG | 수치 분석과 그래프 분석을 모두 남기기 위해서 |
| Colab 대응 | Drive 결과 저장, CPU fast debug mode | 런타임 초기화로 결과가 사라지는 문제를 줄이고, CPU에서도 코드 동작 확인이 가능하게 하기 위해서 |

## Cell 0. 전체 개요 Markdown

이 셀은 노트북의 실험 목적과 문제별 비교 조건을 먼저 고정한다.

문제 1은 직접 설계 CNN 하나를 학습한다. 여기서는 모델 구조 자체가 핵심이므로 `Conv2d`, pooling, BatchNorm, Dropout, activation을 모두 포함한 작은 CNN을 사용한다.

문제 2는 `torchvision.models`에서 제공하는 VGG-16과 ResNet-18 구조를 `weights=None`으로 불러와 scratch 학습한다. 여기서는 pretrained weight를 쓰지 않으므로, 구조 차이와 파라미터 수 차이가 학습 결과에 어떤 영향을 주는지 보는 것이 목적이다.

문제 3은 ResNet-18으로 scratch, feature extraction, fine-tuning을 비교한다. 이 문제는 모델 구조를 ResNet-18로 고정하고, weight 초기화와 freeze 여부만 바꾸는 실험이다.

Colab 안내를 이 셀에 넣은 이유는 실행 전에 저장 위치와 GPU 설정을 먼저 확인해야 하기 때문이다. 특히 Colab의 `/content`는 런타임이 끊기면 사라질 수 있으므로, 결과 파일은 기본적으로 Google Drive에 저장하도록 했다.

## Cell 1. 환경 설정, 경로, 실행 옵션

### import 구성

`json`, `Path`, `deepcopy`는 결과 저장과 config 복사에 사용한다. 실험 결과를 JSON으로 저장하면 노트북 출력이 사라져도 수치가 남고, 그래프나 보고서 작성 시 재사용하기 쉽다.

`random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`은 재현성 확보를 위해 사용한다. 딥러닝 학습은 초기 weight, DataLoader shuffle, augmentation 순서에 따라 결과가 달라질 수 있으므로 seed를 한 곳에서 관리한다.

`Callable`은 모델 builder 함수의 타입을 명확히 하기 위해 사용한다. `run_training_experiment`는 모델 객체가 아니라 모델을 새로 만드는 함수를 받기 때문에 `Callable[[], nn.Module]` 형태가 적절하다.

`nullcontext`는 AMP를 사용하지 않는 CPU 환경에서도 같은 `with` 구조를 유지하기 위해 사용한다. CUDA에서는 autocast context를 열고, CPU에서는 아무 동작도 하지 않는 context를 반환한다.

`matplotlib.pyplot`은 loss/accuracy 그래프 저장에 사용한다. `torch`, `torch.nn`, `DataLoader`, `Subset`, `datasets`, `models`, `transforms`는 PyTorch 학습과 torchvision 데이터/모델 로딩의 핵심이다.

### `BASE_SEED`, `NUM_CLASSES`

`BASE_SEED = 42`는 모든 난수 생성의 기준값이다. 실험마다 seed를 다르게 흩어놓으면 결과 차이가 모델 때문인지 난수 때문인지 구분하기 어렵다.

`NUM_CLASSES = 10`은 CIFAR-10의 class 수다. 직접 설계 CNN의 마지막 Linear layer, VGG-16 classifier, ResNet-18 `fc` layer가 모두 이 값을 사용한다. 이렇게 상수로 두면 class 수 변경 시 한 곳만 수정하면 된다.

### Colab 감지 로직

```python
try:
    import google.colab
    IN_COLAB = True
except ModuleNotFoundError:
    IN_COLAB = "COLAB_GPU" in os.environ
```

Colab에서는 `google.colab` 모듈이 존재한다. 다만 환경에 따라 import 방식이 달라질 수 있어 `COLAB_GPU` 환경 변수도 fallback으로 확인한다. 이 값을 기준으로 Google Drive 마운트, `/content` 경로 사용, CPU fast debug mode를 결정한다.

### `set_global_seed(seed_value)`

이 함수는 Python random, PyTorch CPU seed, CUDA seed를 동시에 설정한다. 또한 cuDNN deterministic mode를 켜고 benchmark mode를 끈다.

`torch.backends.cudnn.deterministic = True`는 같은 입력에 대해 가능한 한 같은 연산 경로를 쓰게 한다. `benchmark = False`는 입력 크기에 따라 가장 빠른 커널을 자동 탐색하는 기능을 끄는 것이다. 속도는 약간 손해볼 수 있지만, 과제 실험에서는 재현성이 더 중요하다.

### `device`와 `torch.set_float32_matmul_precision("high")`

`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`로 GPU가 있으면 CUDA를 사용하고, 없으면 CPU를 사용한다. 특정 GPU 모델을 가정하지 않아 로컬, Colab T4/L4/A100 같은 환경에서 모두 실행 가능하다.

`torch.set_float32_matmul_precision("high")`는 CUDA 환경에서 matrix multiplication 성능을 개선할 수 있는 설정이다. 모델의 의미를 바꾸는 설정은 아니고, PyTorch가 허용하는 범위에서 연산 성능을 높이는 목적이다.

### DataLoader worker 선택

`NUM_WORKERS = 2 if device.type == "cuda" else 0`으로 설정했다.

GPU 환경에서는 학습 중 GPU가 쉬지 않도록 CPU worker가 데이터를 미리 준비하는 것이 유리하다. 반면 CPU 환경에서는 학습과 데이터 로딩이 같은 CPU 자원을 공유하므로 worker를 늘리는 이득이 작고, Colab CPU에서 multiprocessing 문제가 생길 수 있다. 그래서 CPU에서는 0으로 둔다.

### Google Drive 마운트와 경로

Colab에서는 다음 경로 정책을 사용한다.

| 항목 | 경로 | 이유 |
| --- | --- | --- |
| CIFAR-10 데이터 | `/content/data` | 학습 중 I/O가 잦으므로 Drive보다 로컬 디스크가 빠름 |
| 결과 파일 | `/content/drive/MyDrive/DL26/hw03/results_hw03` | 런타임 종료 후에도 결과를 보존하기 위해서 |
| Drive 마운트 실패 시 결과 | `/content/results_hw03` | 권한 문제나 사용자가 마운트를 거부해도 노트북이 중단되지 않게 하기 위해서 |

로컬에서는 repo root에서 실행할 때 `hw03/data`, `hw03/results_hw03`를 사용한다. `Path("hw03/Todo.md").exists()`로 repo root 실행인지 `hw03` 디렉터리 내부 실행인지 구분한다.

### 실행 flag

`RUN_PROBLEM_1`, `RUN_PROBLEM_2`, `RUN_PROBLEM_3`, `RUN_PROBLEM_4`는 문제별 실행 여부를 제어한다. 전체 실험은 시간이 오래 걸리므로 특정 문제만 다시 실행할 수 있게 분리했다.

### Colab fast debug mode

`COLAB_FAST_DEV_RUN = IN_COLAB and device.type != "cuda"`는 Colab CPU 런타임에서 자동으로 켜진다. CPU로 VGG-16이나 224x224 ResNet을 전체 학습하면 매우 오래 걸리므로, 이 모드에서는 epoch, batch size, train/eval batch 수를 줄여 코드가 실행되는지만 확인한다.

### `experiment_epochs(full_epochs)`

이 함수는 제출용 GPU 실행에서는 원래 epoch 수를 반환하고, fast debug mode에서는 1 epoch만 반환한다. 실험 config를 직접 고치지 않고도 환경에 따라 실행량을 줄이기 위한 함수다.

### `experiment_batch_size(full_batch_size, fast_batch_size)`

이 함수는 GPU 실행에서는 원래 batch size를 반환하고, fast debug mode에서는 작은 batch size를 반환한다. CPU나 메모리가 작은 환경에서 VGG/ResNet을 무리하게 실행하지 않기 위한 장치다.

## Cell 2. 공통 데이터 구성 Markdown

이 셀은 데이터 전처리의 큰 원칙을 설명한다.

문제 1과 문제 2는 32x32 입력을 사용한다. CIFAR-10의 원본 크기가 32x32이고, scratch 학습에서는 ImageNet 입력 크기를 맞출 필요가 없기 때문이다.

문제 3은 224x224 입력을 사용한다. ImageNet pretrained ResNet-18은 일반적으로 224x224 이미지와 ImageNet mean/std로 학습되었기 때문에, transfer learning 비교에서는 입력 조건을 맞추는 것이 더 타당하다.

## Cell 3. CIFAR-10 transform, dataset, DataLoader

### normalization 값

`cifar10_mean`, `cifar10_std`는 CIFAR-10 데이터셋에 맞춘 채널별 평균과 표준편차다. 문제 1, 2의 32x32 scratch 학습에 사용한다.

`imagenet_mean`, `imagenet_std`는 ImageNet pretrained model에 맞춘 표준 normalization 값이다. 문제 3에서 pretrained weight를 사용하는 조건과 scratch 224x224 조건 모두 같은 입력 스케일을 쓰게 하여 비교 조건을 맞춘다.

### `train_transform_32`

32x32 학습 transform은 다음 순서다.

1. `RandomCrop(32, padding=4)`
2. `RandomHorizontalFlip()`
3. `ToTensor()`
4. `Normalize(cifar10_mean, cifar10_std)`

`RandomCrop`은 작은 위치 변화에 대한 robustness를 높인다. CIFAR-10은 객체가 중앙 근처에 있는 경우가 많지만 위치가 완전히 고정되어 있지는 않기 때문에, padding 후 crop은 일반적인 augmentation이다.

`RandomHorizontalFlip`은 자동차, 동물, 배, 비행기 등 대부분 class에서 좌우 반전이 label을 바꾸지 않으므로 적절하다. 단, 숫자나 글자 분류처럼 좌우 반전이 의미를 바꾸는 데이터셋에서는 조심해야 한다.

### `test_transform_32`

test transform에는 random augmentation을 넣지 않는다. 평가가 실행할 때마다 바뀌면 성능 비교가 불안정해지기 때문이다. `ToTensor`와 normalization만 적용한다.

### `train_transform_224`, `test_transform_224`

문제 3용 transform은 `Resize((224, 224))`를 먼저 적용한다. CIFAR-10 원본은 32x32라 정보량이 적지만, pretrained ResNet-18의 구조와 ImageNet 학습 조건을 맞추기 위해 224x224로 키운다.

train에는 horizontal flip을 유지하고, test에는 deterministic resize만 사용한다. normalization은 ImageNet mean/std를 사용한다.

### dataset 4개를 따로 만드는 이유

`train_dataset_32`, `test_dataset_32`, `train_dataset_224`, `test_dataset_224`를 따로 만든 이유는 transform이 다르기 때문이다. 같은 CIFAR-10 원본 데이터를 쓰더라도 32x32 실험과 224x224 transfer 실험은 입력 전처리가 다르다.

### `make_loader(...)`

이 함수는 dataset을 DataLoader로 감싼다.

`indices`가 주어지면 `Subset`을 사용한다. 현재 노트북은 전체 CIFAR-10을 사용하지만, 나중에 일부 데이터만 쓰는 ablation 실험을 쉽게 추가할 수 있게 만든 구조다.

`generator = torch.Generator().manual_seed(BASE_SEED + seed_offset)`는 shuffle 순서를 재현 가능하게 한다. `seed_offset`을 문제별로 다르게 준 이유는 모든 실험의 shuffle 순서가 완전히 같을 필요는 없지만, 각 실험 내부에서는 재현 가능해야 하기 때문이다.

`pin_memory=torch.cuda.is_available()`는 GPU로 batch를 복사할 때 성능을 개선할 수 있다. CPU에서는 필요 없으므로 자동으로 꺼진다.

`persistent_workers=NUM_WORKERS > 0`는 worker를 epoch마다 다시 만들지 않고 유지한다. GPU 환경에서는 반복 학습 시 DataLoader overhead를 줄일 수 있다. CPU에서는 worker를 0으로 둬 Colab CPU 문제를 피한다.

## Cell 4. 공통 모델 정의 Markdown

이 셀은 문제 1의 custom CNN과 문제 2/3의 torchvision 모델 사용 방침을 설명한다.

과제 요구사항은 직접 설계 CNN에 Conv2d 2개 이상, pooling, BatchNorm, Dropout, activation이 모두 포함되어야 한다. 노트북의 custom CNN은 Conv2d 3개를 사용해 요구조건보다 약간 여유 있게 설계했다.

## Cell 5. 모델 정의와 파라미터 계산

### `CustomCIFARCNN`

직접 설계 CNN은 크게 `features`와 `classifier`로 나뉜다.

`features`는 convolution block으로 이미지에서 feature map을 뽑는다. `classifier`는 feature map을 class logits로 바꾼다.

#### Conv block 1

```python
nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
nn.BatchNorm2d(64)
nn.ReLU(inplace=True)
```

입력 채널 3은 RGB 이미지의 채널 수다. 출력 채널 64는 작은 CIFAR-10 모델에서 충분한 저수준 feature를 표현하면서도 파라미터 수를 과도하게 늘리지 않는 절충값이다.

`kernel_size=3`, `padding=1`을 사용하면 32x32 해상도를 유지한다. 3x3 convolution은 CNN에서 가장 일반적인 local feature extractor이고, 여러 개를 쌓으면 더 큰 receptive field를 비교적 적은 파라미터로 만들 수 있다.

`bias=False`인 이유는 바로 뒤에 BatchNorm이 있기 때문이다. BatchNorm에는 shift parameter가 있으므로 Conv bias가 중복 역할을 할 수 있다.

#### Conv block 2

두 번째 Conv2d도 64채널을 유지한다. 첫 block에서 얻은 edge, color contrast, simple texture를 조금 더 조합한 뒤 pooling으로 해상도를 줄인다.

```python
nn.MaxPool2d(kernel_size=2)
nn.Dropout2d(p=dropout * 0.5)
```

MaxPool2d는 32x32를 16x16으로 줄인다. pooling은 작은 위치 변화에 덜 민감한 feature를 만들고, 이후 layer의 계산량을 줄인다.

첫 Dropout2d 확률을 `dropout * 0.5`로 둔 이유는 초반 feature를 너무 강하게 끊으면 학습이 불안정해질 수 있기 때문이다. 초반에는 기본 edge/texture feature를 안정적으로 배우게 하고, 뒤쪽 block에서 더 강한 regularization을 적용한다.

#### Conv block 3

세 번째 Conv2d는 64채널에서 128채널로 늘린다. pooling으로 공간 해상도가 줄어든 뒤에는 더 많은 채널을 사용해 class 구분에 필요한 고수준 조합 feature를 표현한다.

Conv layer 수를 3개로 둔 이유는 다음과 같다.

- 과제 요구조건인 Conv2d 최소 2개 이상을 만족한다.
- CIFAR-10 32x32 이미지에는 너무 깊은 모델이 반드시 필요하지 않다.
- VGG-16, ResNet-18과 비교할 때 custom CNN을 작은 baseline으로 두어 파라미터 효율 분석이 가능하다.
- Colab에서도 비교적 빠르게 학습할 수 있다.

#### BatchNorm 위치

BatchNorm은 각 Conv2d 뒤, ReLU 앞에 둔다. Conv output의 scale과 distribution을 정규화한 뒤 activation을 통과시키면 학습이 안정된다. 특히 CNN에서는 layer가 깊어질수록 activation distribution이 변하는데, BatchNorm은 이 변화를 줄여 learning rate 선택에 덜 민감하게 만든다.

#### ReLU 선택 이유

ReLU는 양수 영역에서 gradient가 1에 가깝게 흐르므로 Sigmoid/Tanh보다 깊은 네트워크에서 gradient vanishing 문제가 덜하다. HW02에서 activation 비교를 했던 흐름과도 연결된다. CIFAR-10 CNN baseline에서는 특별한 이유가 없다면 ReLU가 단순하고 안정적인 선택이다.

#### Dropout2d와 Dropout 선택 이유

`Dropout2d`는 feature map 채널 단위로 일부 정보를 끊는다. CNN feature map에서는 인접 pixel이 강하게 correlated 되어 있으므로 일반 Dropout보다 channel 단위 Dropout이 더 의미 있는 regularization이 될 수 있다.

classifier의 `Dropout`은 최종 feature vector의 특정 조합에 과도하게 의존하는 것을 줄인다. CIFAR-10은 train set이 50,000장으로 충분하지만, 작은 custom CNN에서도 augmentation과 regularization을 함께 쓰면 test generalization을 안정화할 수 있다.

#### classifier

```python
nn.AdaptiveAvgPool2d((1, 1))
nn.Flatten()
nn.Linear(128, 128)
nn.ReLU(inplace=True)
nn.Dropout(p=dropout)
nn.Linear(128, num_classes)
```

`AdaptiveAvgPool2d((1, 1))`는 feature map 크기와 무관하게 채널별 평균 feature를 만든다. 이 구조는 flatten 후 큰 fully connected layer를 두는 것보다 파라미터가 훨씬 적다.

`Linear(128, 128)`은 convolution feature를 class 분류에 맞게 한 번 더 조합한다. 마지막 `Linear(128, 10)`은 CIFAR-10의 10개 class logits를 출력한다. 마지막 layer에 Softmax를 넣지 않는 이유는 `CrossEntropyLoss`가 내부적으로 log-softmax를 포함하기 때문이다.

### `forward`

`forward`는 입력 이미지를 `features`에 통과시키고, 그 결과를 `classifier`에 넣어 logits를 반환한다. 출력 shape는 `(batch_size, 10)`이다.

### `build_custom_cnn`

실험 함수는 모델 객체가 아니라 모델을 새로 만드는 builder를 받는다. `build_custom_cnn`은 custom CNN을 매번 같은 설정으로 새로 만들기 위한 함수다. 이렇게 하면 실험마다 이전 학습 weight가 섞이지 않는다.

### `build_vgg16_scratch`

`models.vgg16(weights=None)`으로 ImageNet pretrained weight 없이 구조만 가져온다. VGG-16의 원래 classifier 마지막 출력은 ImageNet 1000 class용이므로 `model.classifier[-1]`을 `Linear(in_features, 10)`으로 교체한다.

VGG-16은 파라미터 수가 매우 큰 모델이다. CIFAR-10 32x32에서는 과한 구조일 수 있지만, 과제에서 pre-defined architecture를 scratch로 학습하라고 했기 때문에 구조 비교 대상으로 사용한다.

### `build_resnet18_scratch`

`models.resnet18(weights=None)`으로 ResNet-18 구조를 가져오고, 마지막 `fc` layer를 10 class로 바꾼다. ResNet-18은 residual connection이 있어 깊은 모델에서도 gradient 흐름이 비교적 안정적이다. VGG-16보다 파라미터 수가 훨씬 적어 모델 크기 대비 성능 효율 비교에 적합하다.

### `build_resnet18_transfer`

이 함수는 문제 3의 세 조건을 모두 처리한다.

| mode | weights | freeze | 의미 |
| --- | --- | --- | --- |
| `scratch` | None | 없음 | ResNet-18 전체를 CIFAR-10으로 처음부터 학습 |
| `feature_extraction` | ImageNet | backbone freeze | pretrained feature는 고정하고 마지막 FC만 학습 |
| `fine_tuning` | ImageNet | 없음 | pretrained weight에서 시작해 전체 모델을 추가 학습 |

`feature_extraction`에서는 먼저 모든 parameter의 `requires_grad`를 False로 바꾼 뒤, 마지막 `fc`를 새 Linear layer로 교체한다. 새로 만든 `fc` layer는 기본적으로 `requires_grad=True`이므로 마지막 layer만 학습된다.

`model.transfer_mode = mode`를 저장하는 이유는 학습 루프에서 feature extraction일 때 BatchNorm을 eval mode로 유지하기 위해서다.

### `freeze_batchnorm_if_needed`

feature extraction에서는 backbone weight뿐 아니라 BatchNorm running mean/variance도 사실상 고정하는 것이 일반적이다. `model.train()`을 호출하면 BatchNorm layer가 train mode가 되어 running statistics가 업데이트될 수 있으므로, feature extraction 조건에서는 BatchNorm만 다시 `eval()`로 바꾼다.

이 함수가 없으면 parameter는 freeze되어도 BatchNorm 통계가 바뀌어 feature extractor가 완전히 고정되지 않는다.

### `count_parameters`

모델의 전체 파라미터 수 또는 학습 가능한 파라미터 수를 계산한다. 문제 1, 2, 3 모두 파라미터 수 기록이 요구되므로 같은 함수로 계산 방식을 통일했다.

`trainable_only=True`이면 `requires_grad=True`인 parameter만 센다. feature extraction 조건에서는 전체 파라미터 수와 학습 가능한 파라미터 수가 크게 다르므로 이 구분이 중요하다.

### `model_parameter_report`

전체 파라미터 수와 학습 가능한 파라미터 수를 dictionary로 묶어 출력한다. JSON 저장과 print에 같은 구조를 사용하기 위해 만든 helper다.

### preview model 출력

preview block은 모델을 실제 학습하기 전에 각 모델의 파라미터 수를 확인한다. 특히 VGG-16이 custom CNN보다 훨씬 큰 모델이라는 점, feature extraction의 trainable parameter가 마지막 FC layer 수준으로 작다는 점을 바로 확인할 수 있다.

## Cell 6. 학습/평가 유틸리티 Markdown

이 셀은 모든 문제에서 같은 학습 루프를 사용한다는 원칙을 설명한다. 실험마다 train loop가 다르면 accuracy 계산 방식, scheduler step 시점, AMP 사용 여부가 달라질 수 있어 비교가 불공정해진다.

## Cell 7. 공통 학습, 평가, 저장, 시각화 함수

### `criterion = nn.CrossEntropyLoss()`

CIFAR-10은 10-class single-label classification이므로 CrossEntropyLoss가 적절하다. 모델은 Softmax 확률이 아니라 raw logits를 출력하고, CrossEntropyLoss가 내부적으로 log-softmax와 negative log-likelihood를 계산한다.

### `save_json`

실험 결과 dictionary를 JSON 파일로 저장한다. `ensure_ascii=False`를 사용해 한글이 깨지지 않게 했다. 모든 결과 저장을 이 함수로 통일하면 파일 인코딩과 들여쓰기 형식이 일정하다.

### `autocast_context`

CUDA 환경에서는 AMP autocast를 사용해 일부 연산을 mixed precision으로 수행한다. 학습 속도와 GPU 메모리 사용량에 유리하다.

PyTorch 버전에 따라 `torch.amp.autocast(device_type="cuda")`와 `torch.cuda.amp.autocast()` API가 다를 수 있다. Colab의 PyTorch 버전이 로컬과 다를 수 있어 try/except fallback을 넣었다.

CPU 환경이나 AMP를 쓰지 않는 환경에서는 `nullcontext()`를 반환한다. 그러면 학습 코드는 항상 `with autocast_context():` 형태를 유지할 수 있다.

### `make_grad_scaler`

AMP 학습에서는 작은 gradient가 underflow되는 것을 막기 위해 GradScaler를 사용한다. 이 함수도 PyTorch 버전 차이를 흡수한다.

`AMP_INIT_SCALE = 1024.0`은 초기 loss scale이다. 너무 큰 scale은 첫 optimizer step이 skip될 수 있으므로 보수적인 값으로 둔다. smoke test에서도 optimizer step이 정상적으로 들어가도록 조정한 값이다.

### `build_optimizer`

이 함수는 모델의 `requires_grad=True` parameter만 optimizer에 넘긴다. feature extraction에서 frozen backbone parameter를 optimizer에 넘기지 않기 위해 중요하다.

지원 optimizer는 `sgd`와 `adamw`다. 현재 실험 config는 AdamW를 사용한다.

AdamW를 선택한 이유는 weight decay를 Adam의 adaptive update와 분리해 적용하기 때문이다. CNN/ResNet 학습에서 Adam보다 weight decay 해석이 명확하고, 과제 2의 optimizer/scheduler 경험을 반영한 안정적인 선택이다.

SGD 옵션을 남긴 이유는 비교 실험을 확장하거나 optimizer ablation을 추가할 때 학습 루프를 바꾸지 않기 위해서다.

### `build_scheduler`

`scheduler_name == "cosine"`이면 `CosineAnnealingLR`을 사용한다. Cosine scheduler는 학습 초반에는 비교적 큰 learning rate로 탐색하고, 후반에는 learning rate를 낮춰 안정적으로 수렴하게 한다.

`scheduler_name == "none"`도 지원한다. scheduler 효과를 끄고 싶을 때 config만 바꾸면 된다.

### `train_one_epoch`

이 함수는 한 epoch의 학습을 수행한다.

동작 순서는 다음과 같다.

1. `model.train()`으로 train mode 전환
2. feature extraction이면 BatchNorm만 eval mode로 되돌림
3. batch를 device로 이동
4. `optimizer.zero_grad(set_to_none=True)`로 gradient 초기화
5. autocast context 안에서 forward와 loss 계산
6. AMP 사용 시 scaler로 backward/step/update
7. AMP 미사용 시 일반 backward/optimizer step
8. loss와 accuracy 누적
9. 평균 train loss, train accuracy, optimizer step 수 반환

`optimizer_steps`를 반환하는 이유는 AMP에서 overflow가 감지되면 optimizer step이 skip될 수 있기 때문이다. optimizer step이 실제로 없었는데 scheduler만 진행하면 learning rate schedule이 한 단계 밀린다. 그래서 `run_training_experiment`에서 `optimizer_steps > 0`일 때만 scheduler를 step한다.

`max_batches`는 fast debug용이다. 전체 데이터셋을 다 돌지 않고 몇 batch만 실행해 코드 경로를 확인할 수 있다.

### `evaluate`

평가 함수는 `model.eval()`과 `torch.no_grad()`를 사용한다. Dropout은 꺼지고 BatchNorm은 running statistics를 사용한다. gradient를 계산하지 않아 메모리 사용량과 계산량도 줄어든다.

평가에는 random augmentation이 없는 test dataset loader를 사용한다. 따라서 test accuracy는 실행마다 동일한 test transform 기준으로 계산된다.

### `run_training_experiment`

이 함수는 하나의 실험 전체를 실행한다.

입력으로 모델 builder, config, train/test loader를 받는다. 함수 내부에서 모델을 새로 만들기 때문에 각 실험이 독립적이다.

동작 순서는 다음과 같다.

1. seed 설정
2. 모델 생성 후 device 이동
3. optimizer 생성
4. scheduler 생성
5. GradScaler 생성
6. epoch loop 실행
7. epoch별 train/test loss와 accuracy 저장
8. best test accuracy와 best epoch 기록
9. 최종 result dictionary 반환

`config`를 결과에 그대로 저장하는 이유는 나중에 그래프나 JSON만 보고도 어떤 hyperparameter로 학습했는지 알 수 있게 하기 위해서다.

### `plot_loss_accuracy`

여러 실험 결과를 한 그래프에 겹쳐 그린다. 왼쪽 subplot은 loss, 오른쪽 subplot은 accuracy다. train은 실선, test는 점선으로 표시한다.

문제 1은 custom CNN 하나만 그리지만, 문제 2와 문제 3은 여러 조건을 같은 그래프에서 비교해야 하므로 공통 plot 함수로 만들었다.

### `summarize_results`

각 실험의 핵심 지표만 추려 summary JSON을 만든다.

`test_accuracy_per_million_trainable_parameters`는 파라미터 수 대비 성능 효율을 보기 위한 값이다. 문제 2에서 custom CNN, VGG-16, ResNet-18을 비교할 때 단순 accuracy만 보면 큰 모델이 유리할 수 있다. 이 지표는 작은 모델이 얼마나 효율적으로 성능을 내는지 설명하는 데 사용한다.

## Cell 8. 문제 1 설계 설명 Markdown

이 셀은 custom CNN의 설계 이유를 보고서 문장 형태로 정리한다.

BatchNorm은 Conv2d 뒤에 둔다. Dropout2d는 pooling 이후 feature map에 적용한다. classifier 앞 Dropout은 최종 feature 조합의 overfitting을 줄이는 목적으로 적용한다.

## Cell 9. 문제 1 custom CNN 실험

### `problem1_config`

| 옵션 | 값 | 선택 이유 |
| --- | --- | --- |
| optimizer | AdamW | 안정적인 수렴과 weight decay 분리 적용 |
| learning_rate | 0.001 | 작은 CNN에서 AdamW 기본값으로 안정적인 시작점 |
| weight_decay | 1e-4 | 과도한 weight 성장을 줄이는 약한 regularization |
| scheduler | cosine | 후반 learning rate를 낮춰 수렴 안정화 |
| epochs | 20 | CIFAR-10 custom CNN 학습 곡선을 보기 위한 현실적인 길이 |
| batch_size | 128 | GPU 메모리와 학습 안정성의 절충 |

Colab CPU fast debug mode에서는 epoch와 batch size가 자동으로 줄어든다.

### DataLoader 구성

train loader는 shuffle을 켠다. 매 epoch 같은 class 순서로 학습하면 optimization에 불리할 수 있기 때문이다. test loader는 shuffle이 필요 없으므로 끈다.

### 결과 저장

`problem1_custom_cnn_results.json`에는 epoch별 history가 저장된다. `problem1_model_notes.json`에는 모델 구조와 BatchNorm/Dropout 적용 이유가 저장된다. `problem1_custom_cnn_curves.png`에는 train/test loss와 accuracy 그래프가 저장된다.

## Cell 10. 문제 2 설계 설명 Markdown

이 셀은 VGG-16과 ResNet-18 scratch 비교 조건을 명시한다.

두 모델 모두 `weights=None`으로 불러온다. 즉 ImageNet pretrained weight를 사용하지 않고 구조만 가져온다. 마지막 classifier는 CIFAR-10의 10 class에 맞게 바꾼다.

## Cell 11. 문제 2 VGG-16 / ResNet-18 scratch 실험

### `problem2_shared_config`

| 옵션 | 값 | 선택 이유 |
| --- | --- | --- |
| optimizer | AdamW | 두 모델에 동일하게 적용 가능한 안정적 optimizer |
| learning_rate | 3e-4 | VGG-16과 ResNet-18은 custom CNN보다 크므로 1e-3보다 보수적인 값 |
| weight_decay | 1e-4 | 큰 모델의 overfitting 완화 |
| scheduler | cosine | 동일한 learning rate decay 정책으로 비교 |
| epochs | 15 | scratch 대형 모델의 학습 비용을 고려한 타협 |
| batch_size | 64 | VGG-16 메모리 사용량을 고려한 값 |

VGG-16은 파라미터 수가 매우 많아 batch size 128이 부담될 수 있다. 그래서 문제 2는 batch size를 64로 둔다.

### config와 builder 분리

`problem2_configs`에는 실험 조건을 저장하고, `problem2_builders`에는 실제 모델 생성 함수를 저장한다. config와 builder를 분리하면 결과 JSON에는 사람이 읽을 수 있는 실험 조건이 남고, 실행 코드는 함수로 깔끔하게 연결된다.

### 효율 비교

문제 1 결과가 존재하면 custom CNN 결과와 문제 2 결과를 합쳐 `problem2_custom_vgg_resnet_efficiency_summary.json`을 저장한다. 이 파일은 모델 크기 대비 성능 효율을 분석할 때 사용한다.

VGG-16은 매우 큰 모델이라 최종 accuracy가 높더라도 파라미터 효율은 낮을 수 있다. ResNet-18은 residual connection 덕분에 VGG-16보다 적은 파라미터로 더 좋은 효율을 보일 가능성이 있다.

## Cell 12. 문제 3 transfer learning 설명 Markdown

이 셀은 transfer learning 세 조건을 설명한다.

Pretrained weight는 큰 데이터셋에서 미리 학습된 feature extractor를 의미한다. ImageNet에서 학습된 초기 convolution filter는 edge, color, texture 같은 일반 feature를 담고 있어 CIFAR-10에도 어느 정도 유효하다.

## Cell 13. 문제 3 ResNet-18 transfer learning 실험

### `problem3_shared_config`

| 옵션 | 값 | 선택 이유 |
| --- | --- | --- |
| model | ResNet-18 | 세 조건에서 구조를 고정하기 위해 |
| input_size | 224x224 | ImageNet pretrained weight와 입력 조건을 맞추기 위해 |
| optimizer | AdamW | 세 조건에 같은 optimizer 적용 |
| weight_decay | 1e-4 | overfitting 완화 |
| scheduler | cosine | 후반 learning rate 감소 |
| epochs | 10 | transfer learning 과제 권장 범위와 실행 비용 고려 |
| batch_size | 64 | 224x224 입력의 메모리 사용량 고려 |

### 조건별 learning rate

| 조건 | learning rate | 이유 |
| --- | --- | --- |
| scratch | 3e-4 | 전체 모델을 처음부터 학습하지만 224x224 입력과 ResNet 구조를 고려해 보수적으로 설정 |
| feature extraction | 1e-3 | 마지막 FC layer만 학습하므로 비교적 큰 learning rate를 써도 backbone이 망가지지 않음 |
| fine-tuning | 1e-4 | pretrained feature를 보존하면서 천천히 CIFAR-10에 맞게 조정하기 위해 낮게 설정 |

Fine-tuning에서 scratch와 같은 learning rate를 쓰면 pretrained feature가 빠르게 손상될 수 있다. 그래서 낮은 learning rate를 사용한다.

### `make_transfer_builder(mode)`

이 함수는 mode 값을 기억하는 builder 함수를 만든다. `run_training_experiment`는 인자가 없는 builder를 기대하므로, mode를 closure로 감싸는 구조가 필요하다.

### early convergence 저장

문제 3은 1~5 epoch의 수렴 속도 차이를 분석해야 한다. 그래서 각 조건의 첫 5 epoch train/test accuracy만 따로 추출해 `problem3_early_convergence_epochs_1_to_5.json`으로 저장한다.

Feature extraction은 초기 수렴이 빠를 수 있다. 이미 학습된 backbone feature를 사용하고 마지막 FC layer만 맞추면 되기 때문이다. Fine-tuning은 pretrained feature에서 시작하므로 scratch보다 초반 성능이 높을 가능성이 있다.

## Cell 14. 문제 4 검증 방법 Markdown

이 셀은 LLM이 생성한 문제 3 코드를 어떻게 검증할지 설명한다.

검증의 핵심은 실행 여부가 아니라 의도한 조건이 실제로 적용되는지다. 예를 들어 feature extraction에서 loss가 감소해도 backbone이 freeze되지 않았다면 실험 조건이 틀린 것이다.

## Cell 15. 문제 4 검증 코드

### `capture_resnet_shapes`

이 함수는 forward hook을 사용해 ResNet-18 주요 layer의 output shape를 기록한다.

hook을 거는 layer는 `conv1`, `bn1`, `relu`, `maxpool`, `layer1`, `layer2`, `layer3`, `layer4`, `avgpool`, `fc`다. 이 순서는 ResNet-18의 주요 forward 흐름과 같다.

더미 입력 shape는 `(2, 3, 224, 224)`다. batch size 2를 사용한 이유는 batch dimension이 유지되는지 확인할 수 있고, 너무 큰 메모리를 쓰지 않기 때문이다.

이 함수는 layer shape가 예상대로 이어지는지 확인한다. 특히 마지막 `fc` 출력이 `(2, 10)`인지 확인하는 근거가 된다.

### `verify_transfer_model(mode)`

이 함수는 transfer mode별로 다음을 검증한다.

| 검증 항목 | 의미 |
| --- | --- |
| `output_shape_ok` | 출력 logits shape가 `[2, 10]`인지 확인 |
| `finite_output` | NaN/Inf 없이 값이 정상인지 확인 |
| `shape_report` | 주요 layer의 중간 shape 기록 |
| `trainable_parameters` | mode별 학습 가능 파라미터 수 확인 |
| `fc_gradient_exists` | 마지막 FC layer에 gradient가 생기는지 확인 |
| `frozen_backbone_gradient_count` | freeze된 backbone에 gradient가 생기지 않는지 확인 |
| `trainable_backbone_gradient_count` | scratch/fine-tuning에서 backbone gradient가 생기는지 확인 |
| `freeze_policy_ok` | mode별 freeze 정책이 의도대로 적용됐는지 종합 판정 |

검증에서는 `use_pretrained=False`를 사용한다. 이유는 pretrained weight 다운로드 없이 구조와 freeze 정책만 빠르게 확인하기 위해서다. weight 값의 품질은 문제 3 실제 학습 결과로 확인하고, 문제 4에서는 코드 구조와 gradient 흐름을 검증한다.

### `verify_intentional_error`

이 함수는 일부러 마지막 FC layer를 10 class로 바꾸지 않은 ResNet-18을 만든다. 원래 ResNet-18은 ImageNet 1000 class 출력이므로 dummy input의 output shape는 `(2, 1000)`이 된다.

검증 함수가 이 오류를 `detected=True`로 잡아내면, 마지막 classifier 수정 누락 같은 흔한 LLM 생성 코드 오류를 발견할 수 있음을 보여준다.

### `verification_report`

검증 결과는 `problem4_verification_report.json`으로 저장한다. 보고서에서는 이 파일의 `output_shape_ok`, `freeze_policy_ok`, `intentional_error_unmodified_fc.detected` 값을 근거로 코드 검증 결과를 설명하면 된다.

## Cell 16. 문제 5 회고 Markdown

이 셀은 실험 회고 초안이다. 실제 실행 후에는 결과 JSON과 그래프를 보고 수치를 반영해 보완해야 한다.

핵심 메시지는 다음과 같다.

- 실행되는 코드와 올바른 실험 코드는 다르다.
- transfer learning에서는 freeze 여부, optimizer parameter 목록, BatchNorm mode, classifier 출력 차원, 입력 normalization을 확인해야 한다.
- shape/gradient 검증은 구조적 오류를 찾는 데 유용하지만, 최종 성능이나 pretrained feature의 유효성을 완전히 보장하지는 않는다.

## 주요 선택 옵션에 대한 종합 설명

### layer 수와 채널 수

Custom CNN은 Conv2d 3개를 사용한다. 최소 요구조건인 2개보다 하나 더 많아 feature hierarchy를 만들 수 있고, VGG/ResNet보다 훨씬 작아 모델 크기 대비 성능 효율 분석에 적합하다.

채널 수는 64, 64, 128로 늘린다. 초반에는 low-level feature를 충분히 잡기 위해 64채널을 사용하고, pooling 후에는 spatial resolution이 줄어든 만큼 channel capacity를 128로 늘려 class 구분 feature를 표현한다.

### activation function

ReLU를 사용한다. ReLU는 계산이 단순하고 양수 영역에서 gradient가 잘 흐른다. Sigmoid나 Tanh는 깊은 네트워크에서 gradient vanishing이 발생하기 쉬우므로 CNN baseline에서는 ReLU가 합리적이다.

### Batch Normalization

BatchNorm은 각 Conv2d 뒤에 둔다. activation 전에 feature distribution을 안정화해 학습을 빠르고 안정적으로 만든다. Conv2d의 `bias=False`도 BatchNorm이 shift 역할을 하기 때문에 선택한 것이다.

### Dropout

Dropout2d는 feature extractor의 pooling 뒤에 둔다. pooling 후 feature map은 더 abstract한 정보를 담으므로 이 시점의 channel dropout은 특정 feature map에 대한 과의존을 줄이는 데 유용하다.

classifier Dropout은 마지막 feature 조합의 overfitting을 줄인다. Dropout 비율 0.35는 너무 약하지도, 너무 강하지도 않은 중간값이다. 과도하게 큰 Dropout은 underfitting을 만들 수 있으므로 작은 CNN에서는 0.3~0.4 수준이 무난하다.

### optimizer와 scheduler

AdamW를 기본 optimizer로 사용한다. Adam의 adaptive learning rate 장점과 decoupled weight decay를 함께 사용할 수 있어 CNN/ResNet 학습에서 안정적이다.

CosineAnnealingLR은 후반 learning rate를 부드럽게 줄인다. 과제 2의 scheduler 실험 경험을 반영해, 초기에는 충분히 학습하고 후반에는 더 작은 step으로 수렴하도록 설정했다.

### learning rate

Custom CNN은 `1e-3`을 사용한다. 모델이 작고 AdamW를 쓰므로 일반적인 시작점이다.

VGG-16/ResNet-18 scratch는 `3e-4`를 사용한다. 모델이 크고 VGG-16은 특히 파라미터가 많아 너무 큰 learning rate가 불안정할 수 있다.

Fine-tuning은 `1e-4`를 사용한다. pretrained feature를 보존하면서 CIFAR-10에 맞게 조정하기 위해 낮은 값을 쓴다.

Feature extraction은 `1e-3`을 사용한다. backbone은 freeze되고 마지막 FC layer만 새로 학습되므로 비교적 큰 learning rate를 사용해 빠르게 classifier를 맞춘다.

### batch size

32x32 custom CNN은 batch size 128을 사용한다. 메모리 부담이 작고 gradient estimate도 안정적이다.

VGG-16/ResNet-18 scratch와 transfer learning은 batch size 64를 사용한다. 특히 VGG-16과 224x224 입력은 메모리를 많이 쓰므로 더 보수적인 batch size가 필요하다.

## 결과 파일을 보고서에 사용하는 방법

| 보고서 항목 | 참고 파일 |
| --- | --- |
| 문제 1 모델 구조와 파라미터 수 | `problem1_model_notes.json`, `problem1_summary.json` |
| 문제 1 train/test 그래프 | `problem1_custom_cnn_curves.png` |
| 문제 2 VGG/ResNet 파라미터 수와 성능 | `problem2_summary.json` |
| 문제 2 모델 크기 대비 효율 | `problem2_custom_vgg_resnet_efficiency_summary.json` |
| 문제 3 세 조건의 성능 | `problem3_summary.json` |
| 문제 3 초반 수렴 속도 | `problem3_early_convergence_epochs_1_to_5.json` |
| 문제 4 코드 검증 결과 | `problem4_verification_report.json` |

Colab에서 기본 설정으로 실행하면 이 파일들은 Google Drive의 `/content/drive/MyDrive/DL26/hw03/results_hw03` 아래에 저장된다.
"""


script_dir = Path(__file__).resolve().parent
(script_dir / "DL_Lab0_HW03.ipynb").write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False),
    encoding="utf-8",
)
(script_dir / "DL_Lab0_HW03_explanation.md").write_text(
    explanation.strip() + "\n",
    encoding="utf-8",
)
(script_dir / "DL_Lab0_HW03_detailed_explanation.md").write_text(
    detailed_explanation.strip() + "\n",
    encoding="utf-8",
)
print("Wrote DL_Lab0_HW03.ipynb")
print("Wrote DL_Lab0_HW03_explanation.md")
print("Wrote DL_Lab0_HW03_detailed_explanation.md")
