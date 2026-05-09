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
NUM_WORKERS = 2
NUM_CLASSES = 10


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

MAX_TRAIN_BATCHES = None
MAX_EVAL_BATCHES = None
USE_AMP = torch.cuda.is_available()
AMP_INIT_SCALE = 1024.0

print("Using device:", device)
print("Data directory:", data_dir.resolve())
print("Output directory:", output_dir.resolve())
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
    scaler: torch.amp.GradScaler,
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
        with torch.amp.autocast(
            device_type=device.type,
            enabled=USE_AMP and device.type == "cuda",
        ):
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
            with torch.amp.autocast(
                device_type=device.type,
                enabled=USE_AMP and device.type == "cuda",
            ):
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
    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=USE_AMP and device.type == "cuda",
        init_scale=AMP_INIT_SCALE,
    )

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
    "epochs": 20,
    "batch_size": 128,
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
    "epochs": 15,
    "batch_size": 64,
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
    "epochs": 10,
    "batch_size": 64,
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

실제 제출용 실험은 노트북의 `MAX_TRAIN_BATCHES = None`, `MAX_EVAL_BATCHES = None` 상태에서 실행한다. 빠른 디버깅이 필요하면 두 값을 작은 정수로 설정해 학습 루프만 확인할 수 있다.
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
print("Wrote DL_Lab0_HW03.ipynb")
print("Wrote DL_Lab0_HW03_explanation.md")
