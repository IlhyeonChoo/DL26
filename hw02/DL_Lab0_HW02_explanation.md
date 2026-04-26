# DL_Lab0_HW02.ipynb 설계 설명서

이 문서는 `DL_Lab0_HW02.ipynb`를 왜 현재 구조로 만들었는지, 각 셀이 어떤 역할을 하는지, `Todo.md`의 문제 1~5 중 무엇과 연결되는지 정리한 설명서이다.

핵심 설계 원칙은 **한 번에 하나의 독립변인만 바꾸는 것**이다. 지난 과제에서는 hidden layer, activation, learning rate, normalization 같은 요소가 동시에 바뀌어 성능 변화의 원인을 분리하기 어려웠다. HW02 노트북은 이 문제를 피하기 위해 공통 코드와 문제별 실험 코드를 분리하고, 각 실험의 고정 조건과 변경 조건을 코드상에서 명시했다.

## 1부. 전체 구조를 이렇게 나눈 이유

노트북은 총 20개 셀로 구성되어 있다.

| 셀 | 종류 | 제목 또는 시작 코드 | 관련 문제 | 역할 |
| --- | --- | --- | --- | --- |
| 0 | Markdown | `# DL Lab HW02 - FashionMNIST Controlled Experiments` | 전체 | 실험 원칙과 변인 통제 기준 설명 |
| 1 | Code | `import json` | 전체 | 라이브러리, seed, device, 경로, 실행 flag 설정 |
| 2 | Markdown | `## 공통 데이터 구성` | 전체, 문제 2 | 데이터 split 원칙 설명 |
| 3 | Code | `fashion_mnist_mean = ...` | 전체, 문제 2 | FashionMNIST 로드, 균등 샘플링 split 생성 |
| 4 | Markdown | `## 공통 모델과 학습 유틸리티` | 전체 | 공통 모델/학습 함수 사용 이유 설명 |
| 5 | Code | `def get_activation...` | 전체 | MLP 모델 정의, activation 선택, 파라미터 수 계산 |
| 6 | Code | `criterion = nn.CrossEntropyLoss()` | 전체 | optimizer, scheduler, 학습/평가 루프 정의 |
| 7 | Markdown | `## 문제 1. Deep Architecture 실험` | 문제 1 | deep MLP와 gradient norm 실험 설명 |
| 8 | Code | `problem1_shared_config = ...` | 문제 1 | Sigmoid/ReLU 실험 실행 |
| 9 | Code | `def plot_problem1_gradient_norms...` | 문제 1 | gradient norm 그래프와 요약 저장 |
| 10 | Markdown | `## 문제 2. Regularization 비교 실험` | 문제 2 | regularization 실험 조건 설명 |
| 11 | Code | `problem2_shared_config = ...` | 문제 2 | No reg, weight decay, dropout, both 실험 실행 |
| 12 | Code | `def plot_metric_comparison...` | 문제 2, 3 | loss/accuracy 그래프, overfitting 판정 함수 |
| 13 | Code | `data_size_results = ...` | 문제 2 | 학습 데이터 양 영향 비교 |
| 14 | Markdown | `## 문제 3. Optimizer 비교 실험` | 문제 3 | optimizer/scheduler 실험 조건 설명 |
| 15 | Code | `problem3_shared_config = ...` | 문제 3 | SGD, momentum, AdaGrad, Adam 실험 실행 |
| 16 | Code | `problem3_scheduler_results = ...` | 문제 3 | Adam vs Adam+cosine scheduler 비교 |
| 17 | Markdown | `## 문제 4. 최종 모델 구성 및 팀 내 비교` | 문제 4 | 최종 모델 선택 근거 설명 |
| 18 | Code | `final_model_config = ...` | 문제 4 | 최종 모델 학습, 팀 비교 template 저장 |
| 19 | Markdown | `## 문제 5. 실험 회고 및 고찰 초안` | 문제 5 | 실험 해석과 LLM 활용 회고 초안 |

### 셀을 나눈 이득

첫째, **실행 비용을 줄일 수 있다.** 데이터 로드나 공통 함수 정의는 한 번 실행하고, 문제 1, 문제 2, 문제 3 실험만 선택적으로 다시 실행할 수 있다.

둘째, **변인 통제가 명확해진다.** 예를 들어 문제 2에서는 모델 구조, activation, optimizer, learning rate를 공통 config에 묶고 regularization 값만 실험별로 바꾼다. 셀이 섞여 있으면 어떤 값이 어디서 바뀌었는지 추적하기 어렵다.

셋째, **그래프와 분석을 문제별로 독립적으로 만들 수 있다.** 문제 1의 gradient norm 그래프와 문제 2의 train/validation loss 그래프는 목적이 다르므로 별도 셀로 두는 것이 해석에 유리하다.

넷째, **보고서 작성이 쉬워진다.** 문제별 셀과 결과 JSON이 나뉘어 있어, 과제 답안에 필요한 그래프와 수치를 바로 찾을 수 있다.

다섯째, **디버깅이 쉽다.** 문제가 발생했을 때 전체 노트북을 처음부터 다시 실행하지 않고, 관련 셀만 고쳐 재실행할 수 있다.

## 2부. 공통 준비 셀 설명

### Cell 0. 전체 실험 원칙 설명

관련 문제: 전체, 특히 문제 1~3

이 셀은 코드 셀이 아니라 Markdown 셀이다. 노트북 전체의 실험 철학을 먼저 설명한다.

핵심 내용은 다음과 같다.

- 문제 1에서는 activation만 바꾼다.
- 문제 2에서는 regularization만 바꾼다.
- 문제 3에서는 optimizer만 바꾼다.
- scheduler 비교는 optimizer 비교와 섞지 않고 별도로 둔다.
- 데이터 수 비교는 regularization 비교와 섞지 않고 별도로 둔다.

이 셀이 필요한 이유는 실험을 실행하기 전에 “무엇을 고정하고 무엇을 바꾸는지”를 명시하기 위해서다. 딥러닝 실험에서는 성능 차이보다 성능 차이의 원인을 설명하는 것이 중요하다. 이 셀은 그 기준을 노트북 맨 앞에서 고정한다.

### Cell 1. 라이브러리, seed, device, 경로 설정

관련 문제: 전체

이 셀은 모든 문제에서 공통으로 사용하는 실행 환경을 준비한다.

주요 코드 의미는 다음과 같다.

```python
import json
import random
from copy import deepcopy
from pathlib import Path
```

`json`은 실험 결과를 파일로 저장하기 위해 사용한다. `random`과 `torch.manual_seed`는 재현 가능한 실험을 위해 사용한다. `deepcopy`는 validation 성능이 가장 좋았던 모델 상태를 안전하게 저장하기 위해 필요하다. `Path`는 `data`, `results_hw02` 같은 경로를 OS에 덜 의존적으로 다루기 위해 사용한다.

```python
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
```

`matplotlib`은 그래프 저장과 시각화에 사용한다. `torch`와 `torch.nn`은 모델, loss, optimizer, tensor 연산의 핵심 라이브러리다.

```python
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
```

`CosineAnnealingLR`은 문제 3의 scheduler 비교와 문제 4 최종 모델에 사용한다. `DataLoader`는 batch 단위 학습을 위해 필요하고, `Subset`은 5,000장 균등 샘플링 데이터셋을 만들기 위해 필요하다. `datasets`, `transforms`는 FashionMNIST 다운로드와 전처리에 사용한다.

```python
BASE_SEED = 42
NUM_WORKERS = 2
```

`BASE_SEED`는 실험 재현성을 위한 기준 seed다. `NUM_WORKERS`는 DataLoader가 데이터를 읽는 worker 수다. 너무 크게 잡으면 환경에 따라 문제가 생길 수 있어 보수적인 값인 2를 사용했다.

```python
def set_global_seed(seed_value: int) -> None:
    ...
```

이 함수는 Python random, PyTorch CPU seed, CUDA seed, cuDNN 설정을 한 번에 맞춘다.

함수로 만든 이유는 다음과 같다.

- 여러 실험에서 같은 초기화 조건을 재사용할 수 있다.
- 실험별 model seed를 명시적으로 적용할 수 있다.
- seed 설정 코드가 흩어지는 것을 막아 재현성 관리가 쉬워진다.
- 나중에 seed 정책을 바꾸려면 함수 하나만 수정하면 된다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

GPU가 있으면 CUDA를 사용하고, 없으면 CPU를 사용한다. 특정 GPU 모델을 하드코딩하지 않아 다른 환경에서도 실행 가능하다.

```python
RUN_PROBLEM_1 = True
RUN_PROBLEM_2 = True
RUN_PROBLEM_3 = True
RUN_FINAL_MODEL = True
```

각 문제별 실행 flag다. 전체 실험은 시간이 오래 걸릴 수 있으므로, 특정 문제만 다시 실행하고 싶을 때 해당 flag만 조정하면 된다.

이 셀을 따로 둔 이득은 모든 실험의 환경 조건이 한 곳에서 관리된다는 점이다. 경로, seed, device, 실행 여부가 문제별 코드 안에 섞여 있으면 실험 조건을 추적하기 어렵다.

### Cell 2. 공통 데이터 구성 설명

관련 문제: 전체, 특히 문제 2

이 Markdown 셀은 데이터 split 원칙을 설명한다.

문제 2에서는 FashionMNIST train 60,000장 중 5,000장만 사용해야 하고, 클래스별 비율이 균등해야 한다. 따라서 클래스당 500장씩 뽑는 규칙을 명시했다.

또한 validation split은 학습 데이터와 분리한다. test 데이터는 원본 test set을 그대로 유지한다. 이렇게 해야 validation은 모델 선택과 실험 비교에 사용하고, test는 최종 일반화 성능 확인에 사용할 수 있다.

이 셀을 따로 둔 이유는 데이터 split이 모든 실험 해석의 기준이기 때문이다. 데이터 구성이 모호하면 문제 2의 overfitting 분석과 문제 3의 optimizer 비교 모두 신뢰하기 어렵다.

### Cell 3. FashionMNIST 로드와 stratified split 생성

관련 문제: 전체, 특히 문제 2

이 셀은 실제 데이터셋을 만들고, 실험에 사용할 index split을 생성한다.

```python
fashion_mnist_mean = (0.2860,)
fashion_mnist_std = (0.3530,)
```

FashionMNIST의 평균과 표준편차다. 입력 정규화를 통해 학습 안정성을 높인다. 모든 실험에 동일하게 적용되므로 normalization은 독립변인이 아니다.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(fashion_mnist_mean, fashion_mnist_std),
])
```

이미지를 PyTorch tensor로 바꾸고 정규화한다. 문제 1~4 전체에서 같은 transform을 사용하므로, 실험 결과 차이가 normalization 때문에 발생하지 않는다.

```python
train_dataset_full = datasets.FashionMNIST(...)
test_dataset = datasets.FashionMNIST(...)
```

FashionMNIST train/test 데이터를 다운로드하거나 로드한다. train set은 split을 나누는 데 사용하고, test set은 원본 그대로 최종 평가에 사용한다.

```python
targets = torch.as_tensor(train_dataset_full.targets)
```

각 이미지의 class label을 tensor로 저장한다. 클래스별 균등 샘플링을 하려면 label 정보가 필요하다.

#### `stratified_sample_indices`

```python
def stratified_sample_indices(...):
    ...
```

이 함수는 각 class에서 같은 개수만큼 sample index를 뽑는다.

함수로 만든 이유는 다음과 같다.

- 문제 2 요구사항인 “클래스당 500장”을 코드로 보장한다.
- validation split도 클래스별 균등하게 만들 수 있다.
- `excluded_indices`를 통해 train과 validation이 겹치지 않도록 만들 수 있다.
- 데이터 수 실험에서 추가 학습 데이터를 뽑을 때도 같은 규칙을 재사용할 수 있다.

그냥 한 번만 코드로 작성하지 않고 함수화한 이유는 split 생성이 여러 번 반복되기 때문이다. 반복 코드를 직접 복사하면 class별 샘플 수나 seed를 실수로 다르게 적용할 위험이 커진다.

#### `shuffled_indices`

```python
def shuffled_indices(indices: list[int], seed_value: int) -> list[int]:
    ...
```

선택된 index 목록을 seed 기반으로 섞는다. class별로 뽑은 index가 class 순서대로 정렬되어 있으면 DataLoader shuffle을 사용하더라도 split 자체가 보기 좋지 않고 디버깅도 어렵다.

함수화 이득은 seed 기반 섞기 로직을 재사용할 수 있다는 점이다.

#### split 변수

```python
small_train_indices = ...
validation_indices = ...
additional_train_indices = ...
large_train_indices = ...
full_train_indices = ...
```

각 split의 의미는 다음과 같다.

- `small_train_indices`: 문제 2 regularization 실험용 5,000장 학습 데이터
- `validation_indices`: 모든 실험에서 공통으로 사용하는 validation 데이터
- `additional_train_indices`: 데이터 수 비교를 위해 추가로 뽑은 데이터
- `large_train_indices`: 20,000장 데이터 수 비교 실험용 학습 데이터
- `full_train_indices`: validation을 제외한 나머지 train 데이터, 문제 1과 최종 모델에 사용

데이터 수 비교를 문제 2 regularization 실험과 분리한 이유는, regularization과 train data size가 동시에 바뀌면 어떤 요인이 overfitting에 영향을 줬는지 알 수 없기 때문이다.

#### `class_count`, `print_split_summary`

```python
def class_count(indices: list[int]) -> dict[int, int]:
    ...

def print_split_summary(name: str, indices: list[int]) -> None:
    ...
```

이 함수들은 split이 정말 균등한지 출력한다.

함수로 만든 이유는 검증 절차를 반복 가능하게 하기 위해서다. 문제 2의 핵심 조건이 class-balanced sampling이므로, split을 만들고 바로 class count를 확인하는 구조가 안전하다.

#### `make_train_loader`, `make_test_loader`

```python
def make_train_loader(...):
    ...

def make_test_loader(...):
    ...
```

이 함수들은 index split을 실제 DataLoader로 바꾼다.

함수로 만든 이유는 다음과 같다.

- batch size와 shuffle 여부를 일관되게 적용할 수 있다.
- GPU가 있으면 `pin_memory`를 자동으로 켤 수 있다.
- train/validation/test loader 생성 방식이 실험마다 달라지는 것을 막는다.
- shuffle seed를 관리해 실험 간 차이를 줄인다.

이 셀을 따로 둔 이득은 데이터 생성과 모델 학습 코드가 분리된다는 점이다. 데이터 split이 바뀌지 않는다면 이후 문제별 실험 셀만 반복 실행하면 된다.

## 3부. 공통 모델과 학습 함수 셀 설명

### Cell 4. 공통 모델과 학습 유틸리티 설명

관련 문제: 전체

이 Markdown 셀은 같은 모델 클래스와 같은 학습 함수를 모든 문제에서 사용한다고 설명한다.

이 방식의 장점은 실험 조건 통제다. 문제마다 모델 클래스를 따로 만들면 구조가 조금씩 달라질 가능성이 있다. 하나의 `ControlledMLP`를 사용하면 config 값만 바꿔 실험할 수 있다.

### Cell 5. ControlledMLP 모델 정의

관련 문제: 전체, 특히 문제 1, 2, 3, 4

이 셀은 공통 MLP 모델과 모델 관련 유틸리티를 정의한다.

#### `get_activation`

```python
def get_activation(name: str) -> type[nn.Module]:
    ...
```

문자열 이름을 PyTorch activation class로 바꾼다.

예를 들어 `"relu"`는 `nn.ReLU`, `"sigmoid"`는 `nn.Sigmoid`로 변환된다.

함수로 만든 이유는 다음과 같다.

- 실험 config에서 activation을 문자열로 명확하게 기록할 수 있다.
- 문제 1에서 Sigmoid와 ReLU만 바꾸는 구조가 쉬워진다.
- 지원하지 않는 activation을 넣으면 바로 error가 발생해 실수를 빨리 찾을 수 있다.

#### `ControlledMLP`

```python
class ControlledMLP(nn.Module):
    ...
```

FashionMNIST 이미지를 784차원 vector로 펼친 뒤, hidden layer를 통과시키고 10개 class logits을 출력하는 MLP다.

주요 인자는 다음과 같다.

- `hidden_dims`: hidden layer 크기 목록
- `activation_name`: activation 종류
- `dropout`: dropout 비율

이 클래스를 만든 이유는 하나의 모델 정의로 문제 1~4 전체를 처리하기 위해서다.

문제별 사용 방식은 다음과 같다.

- 문제 1: `hidden_dims=[256]*8`, activation만 Sigmoid/ReLU로 변경
- 문제 2: `hidden_dims=[512,256,128]`, ReLU 고정, dropout/weight decay만 변경
- 문제 3: 같은 모델 구조와 ReLU 고정, optimizer만 변경
- 문제 4: 실험 결과를 반영한 최종 모델 구성

`forward` 함수는 입력 이미지를 펼치고 모델에 넣는다.

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.view(x.size(0), -1)
    return self.model(x)
```

FashionMNIST는 `(batch, 1, 28, 28)` 형태이므로 MLP에 넣기 위해 `(batch, 784)`로 바꾼다.

`linear_layers` 함수는 모델 안의 `nn.Linear` layer만 골라낸다.

```python
def linear_layers(self) -> list[nn.Linear]:
    return [module for module in self.model if isinstance(module, nn.Linear)]
```

이 함수는 문제 1의 gradient norm 측정에 필요하다. gradient norm은 각 linear layer의 weight 기준으로 측정해야 하기 때문이다.

#### `count_trainable_parameters`

```python
def count_trainable_parameters(model: nn.Module) -> int:
    ...
```

학습 가능한 파라미터 수를 계산한다.

문제 1은 모델 구조 설명과 파라미터 수 기록을 요구한다. 이 함수를 만들면 모든 실험에서 파라미터 수를 같은 방식으로 계산할 수 있다.

#### `model_structure_string`

```python
def model_structure_string(hidden_dims: list[int]) -> str:
    ...
```

`[512, 256, 128]` 같은 hidden dimension 목록을 `784 -> 512 -> 256 -> 128 -> 10` 형태의 문자열로 바꾼다.

함수로 만든 이유는 결과 출력과 JSON 저장에서 모델 구조 표현을 통일하기 위해서다.

### Cell 6. 학습, 평가, 결과 저장 함수 정의

관련 문제: 전체

이 셀은 모든 실험에서 사용하는 학습 루프와 평가 루프를 정의한다. 가장 중요한 공통 셀이다.

```python
criterion = nn.CrossEntropyLoss()
```

FashionMNIST는 10-class classification 문제이므로 cross entropy loss를 사용한다. 마지막 layer에 softmax를 넣지 않는 이유는 `CrossEntropyLoss`가 내부적으로 log-softmax를 포함하기 때문이다.

#### `build_optimizer`

```python
def build_optimizer(...):
    ...
```

문자열 config에 따라 optimizer를 생성한다.

지원 optimizer는 다음과 같다.

- SGD
- SGD with momentum
- AdaGrad
- Adam

함수로 만든 이유는 문제 3에서 optimizer만 바꿔야 하기 때문이다. optimizer 생성 코드가 실험마다 따로 있으면 learning rate나 weight decay가 실수로 달라질 수 있다. 함수로 통합하면 optimizer 이름만 바꾸고 나머지 조건은 config에서 동일하게 유지할 수 있다.

#### `build_scheduler`

```python
def build_scheduler(...):
    ...
```

scheduler 이름에 따라 scheduler를 생성한다. 현재는 `none`과 `cosine`을 지원한다.

문제 3에서는 learning rate scheduling 적용 전/후 비교가 필요하다. scheduler를 optimizer 비교와 섞지 않기 위해, `scheduler` 값을 config로 분리했다.

#### `weight_gradient_norms`

```python
def weight_gradient_norms(model: ControlledMLP) -> list[float]:
    ...
```

각 linear layer의 weight gradient L2 norm을 계산한다.

문제 1에서 요구한 `param.grad.norm()`을 구현한 함수다. layer별 gradient 흐름을 보기 위해 list 형태로 저장한다.

함수화 이득은 다음과 같다.

- gradient 측정 방식이 Sigmoid와 ReLU 실험에서 완전히 동일해진다.
- 나중에 norm 계산 방식을 바꾸려면 함수 하나만 수정하면 된다.
- 학습 루프 내부가 지나치게 길어지는 것을 막는다.

#### `train_one_epoch`

```python
def train_one_epoch(...):
    ...
```

한 epoch 동안 학습을 수행한다.

작업 순서는 다음과 같다.

1. 모델을 train mode로 전환한다.
2. batch를 device로 보낸다.
3. optimizer gradient를 초기화한다.
4. forward pass로 logits를 계산한다.
5. loss를 계산한다.
6. backward pass로 gradient를 계산한다.
7. 문제 1에서 필요하면 gradient norm을 기록한다.
8. optimizer step으로 parameter를 업데이트한다.
9. epoch 평균 loss와 accuracy를 계산한다.

함수로 만든 이유는 모든 실험에서 동일한 학습 절차를 사용하기 위해서다. 문제별로 학습 코드를 따로 쓰면 accuracy 계산 방식이나 gradient 초기화 방식이 달라질 수 있다.

#### `evaluate`

```python
def evaluate(model: ControlledMLP, loader: DataLoader) -> tuple[float, float]:
    ...
```

validation 또는 test 데이터를 평가한다.

`model.eval()`과 `torch.no_grad()`를 사용한다. 이것은 dropout을 평가 모드로 바꾸고, gradient 계산을 끄기 위해 필요하다. 특히 문제 2에서 dropout을 비교하므로 train mode와 eval mode를 정확히 구분해야 한다.

#### `run_controlled_training`

```python
def run_controlled_training(...):
    ...
```

이 노트북의 핵심 실행 함수다. 하나의 실험 config를 받아서 DataLoader 생성, 모델 생성, optimizer 생성, scheduler 생성, 학습, validation, best model 저장, test 평가, 결과 반환까지 수행한다.

함수로 만든 이유는 다음과 같다.

- 문제 1~4의 모든 실험을 같은 방식으로 실행할 수 있다.
- config만 바꿔 독립변인 변경을 명확히 할 수 있다.
- best validation 성능 기준으로 모델을 저장하고 test 평가하는 절차가 일관된다.
- 출력과 JSON 저장에 필요한 결과 형식이 통일된다.
- 실험 재현성과 비교 가능성이 높아진다.

이 함수를 쓰지 않고 각 실험을 직접 작성하면 코드가 반복되고, 실험별로 작은 차이가 생길 가능성이 커진다. 이번 과제의 핵심이 변인 통제이므로 학습 실행 로직은 반드시 공통화하는 것이 좋다.

#### `save_json`

```python
def save_json(data: dict, path: Path) -> None:
    ...
```

실험 결과를 JSON 파일로 저장한다.

함수로 만든 이유는 결과 저장 방식을 통일하고, 인코딩 문제를 피하기 위해서다. `ensure_ascii=False`를 사용해 한글이 깨지지 않도록 했다.

## 4부. 문제별 실험 셀 설명

### Cell 7. 문제 1 설명

관련 문제: 문제 1

이 Markdown 셀은 deep architecture 실험의 목적과 통제 조건을 적는다.

문제 1의 요구사항은 다음과 같다.

- 8층 이상 MLP
- Sigmoid activation
- 동일 구조의 ReLU activation
- 각 layer weight gradient norm을 epoch별로 기록
- activation 외 조건은 동일하게 유지

이 셀은 위 요구사항을 코드 실행 전에 문장으로 고정한다. 특히 `hidden_dims`, optimizer, learning rate, batch size, data split이 같고 activation만 바뀐다는 점을 명시한다.

### Cell 8. 문제 1 실험 실행

관련 문제: 문제 1

이 셀은 Sigmoid deep MLP와 ReLU deep MLP를 실행한다.

```python
problem1_shared_config = {
    "hidden_dims": DEEP_HIDDEN_DIMS,
    "optimizer": "adam",
    "learning_rate": 0.001,
    ...
}
```

공통 조건을 먼저 dictionary로 만든다. 여기에 들어간 값은 Sigmoid와 ReLU 실험에서 동일하게 유지된다.

```python
problem1_configs = {
    "sigmoid_deep": {
        **problem1_shared_config,
        "activation": "sigmoid",
    },
    "relu_deep": {
        **problem1_shared_config,
        "activation": "relu",
    },
}
```

두 실험의 차이는 activation뿐이다. `**problem1_shared_config`를 사용하면 공통 조건이 그대로 복사되고, activation 값만 덮어쓴다. 이것이 변인 통제 측면에서 중요하다.

```python
record_gradients=True
```

문제 1은 gradient norm 기록이 필요하므로 이 값을 켠다. 문제 2, 3에서는 필요 없으므로 끈다. 이렇게 하면 문제별로 필요한 추가 계산만 수행할 수 있다.

### Cell 9. 문제 1 gradient norm 그래프와 요약

관련 문제: 문제 1

이 셀은 문제 1 결과를 시각화하고 요약한다.

#### `plot_problem1_gradient_norms`

```python
def plot_problem1_gradient_norms(results: dict[str, dict], selected_epochs: list[int]) -> None:
    ...
```

선택한 epoch에서 layer 위치별 gradient norm을 그린다. Sigmoid와 ReLU를 같은 그래프에 겹쳐 표시한다.

`axis.set_yscale("log")`를 사용하는 이유는 gradient norm 차이가 매우 클 수 있기 때문이다. Sigmoid에서 gradient vanishing이 발생하면 앞쪽 layer의 norm이 매우 작아져 linear scale에서는 보이지 않을 수 있다.

함수로 만든 이유는 epoch 1, 6, 12처럼 여러 시점의 그래프를 같은 방식으로 그리기 위해서다.

#### `summarize_problem1`

```python
def summarize_problem1(results: dict[str, dict]) -> dict:
    ...
```

최종 epoch의 앞쪽 hidden layer, 뒤쪽 hidden layer, output layer gradient norm을 요약한다. 또한 파라미터 수와 validation/test accuracy도 저장한다.

문제 1에서 요구하는 “모델 구조 설명 및 파라미터 수”, “gradient vanishing 여부 분석”에 사용할 수 있는 수치를 만든다.

### Cell 10. 문제 2 설명

관련 문제: 문제 2

이 Markdown 셀은 regularization 비교 실험의 목적과 조건을 적는다.

문제 2 요구사항은 다음과 같다.

- 60,000장 중 5,000장만 학습에 사용
- 클래스당 500장 균등 샘플링
- test 데이터는 원본 그대로 사용
- No regularization, Weight decay, Dropout 비교
- Dropout과 Weight decay 동시 적용 결과 비교
- 학습 데이터 양이 overfitting에 미치는 영향 분석

이 셀은 모델 구조, activation, optimizer, learning rate를 모두 고정하고 regularization만 바꾼다고 명시한다.

### Cell 11. 문제 2 regularization 실험 실행

관련 문제: 문제 2

이 셀은 regularization 조건 4개를 실행한다.

```python
problem2_shared_config = {
    "hidden_dims": BASE_HIDDEN_DIMS,
    "activation": "relu",
    "optimizer": "adam",
    "learning_rate": 0.001,
    ...
}
```

공통 모델과 학습 조건이다. 이 값들은 모든 regularization 실험에서 동일하다.

```python
"no_regularization": {
    "dropout": 0.0,
    "weight_decay": 0.0,
}
```

regularization을 적용하지 않는 기준선이다. overfitting 발생 여부와 발생 시점을 판단하는 기준이 된다.

```python
"weight_decay": {
    "dropout": 0.0,
    "weight_decay": 1e-4,
}
```

L2 regularization만 적용한다. Dropout은 0으로 유지하므로 weight decay 효과만 볼 수 있다.

```python
"dropout": {
    "dropout": 0.3,
    "weight_decay": 0.0,
}
```

Dropout만 적용한다. Weight decay는 0으로 유지하므로 dropout 효과만 볼 수 있다.

```python
"dropout_weight_decay": {
    "dropout": 0.3,
    "weight_decay": 1e-4,
}
```

두 regularization을 동시에 적용한다. 과제에서 요구한 “동시 적용 결과 비교”를 위한 조건이다. 이 조건은 단일 변인 실험은 아니지만, 앞의 세 조건을 먼저 수행한 뒤 조합 효과를 확인하기 위한 추가 실험으로 분리했다.

이 셀은 `small_train_indices`를 사용한다. 즉 모든 regularization 실험은 정확히 같은 5,000장 학습 데이터로 실행된다.

### Cell 12. 문제 2/3 공통 그래프와 overfitting 분석 함수

관련 문제: 문제 2, 문제 3

이 셀은 train/validation loss와 accuracy를 그리는 함수, overfitting 시작 시점을 추정하는 함수, 최종 metric 요약 함수를 정의한다.

#### `plot_metric_comparison`

```python
def plot_metric_comparison(results: dict[str, dict], title: str, filename: str) -> None:
    ...
```

여러 실험의 train loss, validation loss, train accuracy, validation accuracy를 한 번에 그린다.

문제 2와 문제 3 모두 “각 조건별 Train/Validation Loss 및 Accuracy 그래프”가 필요하므로 공통 함수로 만들었다.

함수로 만든 이득은 다음과 같다.

- 그래프 스타일과 축 이름이 통일된다.
- 문제 2와 문제 3에서 같은 형식의 비교 그래프를 만들 수 있다.
- 그래프 저장 위치와 파일명을 명확히 관리할 수 있다.

#### `detect_overfitting_onset`

```python
def detect_overfitting_onset(...):
    ...
```

No regularization 조건에서 overfitting이 시작된 epoch를 추정한다.

판정 기준은 다음 두 가지다.

- train accuracy와 validation accuracy의 gap이 일정 기준 이상 커진다.
- best validation loss 이후 validation loss가 다시 증가한다.

이 함수는 완벽한 수학적 판정이 아니라 보고서 작성을 위한 일관된 기준이다. 사람이 그래프를 보고 판단하는 것과 함께 사용해야 한다.

함수로 만든 이유는 overfitting 판단 기준을 코드에 명시하기 위해서다. 기준이 없으면 실험자가 임의로 epoch를 고르는 문제가 생길 수 있다.

#### `summarize_final_metrics`

```python
def summarize_final_metrics(results: dict[str, dict]) -> dict[str, dict]:
    ...
```

각 실험의 best epoch, best validation accuracy, test accuracy, final train/validation loss와 accuracy, final gap을 정리한다.

문제 2와 문제 3의 결과 비교표에 바로 사용할 수 있는 형태다.

### Cell 13. 문제 2 학습 데이터 양 비교

관련 문제: 문제 2

이 셀은 학습 데이터 양이 overfitting에 미치는 영향을 분석한다.

```python
data_size_results["no_regularization_5000"] = problem2_results["no_regularization"]
```

이미 실행한 5,000장 No regularization 결과를 재사용한다. 같은 결과를 다시 학습하지 않아 실행 시간을 줄인다.

```python
data_size_results["no_regularization_20000"] = run_controlled_training(...)
```

20,000장 학습 데이터 조건을 새로 실행한다.

중요한 점은 regularization을 적용하지 않는다는 것이다. 데이터 양만 바꿔야 “학습 데이터 양이 overfitting에 미치는 영향”을 볼 수 있다. 만약 데이터 양과 dropout을 동시에 바꾸면 어떤 변화가 gap을 줄였는지 알 수 없다.

### Cell 14. 문제 3 설명

관련 문제: 문제 3

이 Markdown 셀은 optimizer 비교 실험의 조건을 설명한다.

문제 3 요구사항은 다음과 같다.

- SGD
- SGD with momentum
- AdaGrad
- Adam
- optimizer 중 하나에 learning rate scheduling 적용
- train/validation loss 및 accuracy 그래프
- 수렴 속도와 최종 성능 비교
- scheduler 적용 전/후 비교

이 셀에서는 optimizer 비교와 scheduler 비교를 분리한다고 명시한다. scheduler까지 동시에 바꾸면 optimizer 효과와 scheduler 효과가 섞이기 때문이다.

### Cell 15. 문제 3 optimizer 비교 실행

관련 문제: 문제 3

이 셀은 optimizer 4개를 실행한다.

```python
problem3_shared_config = {
    "hidden_dims": BASE_HIDDEN_DIMS,
    "activation": "relu",
    "dropout": 0.0,
    "weight_decay": 0.0,
    "learning_rate": 0.005,
    ...
}
```

모델 구조, activation, regularization, learning rate, epoch, batch size를 모두 고정한다.

```python
"sgd": {"optimizer": "sgd", "momentum": 0.0}
"sgd_momentum": {"optimizer": "sgd", "momentum": 0.9}
"adagrad": {"optimizer": "adagrad"}
"adam": {"optimizer": "adam"}
```

차이는 optimizer 종류와 momentum 여부뿐이다.

주의할 점은 SGD with momentum은 optimizer 자체는 SGD이지만 momentum이 0.9로 바뀐다. 과제에서 별도 조건으로 요구했기 때문에 이것은 “optimizer 조건”의 일부로 취급했다.

### Cell 16. 문제 3 scheduler 비교

관련 문제: 문제 3

이 셀은 Adam과 Adam + cosine scheduler를 비교한다.

```python
problem3_scheduler_results["adam_no_scheduler"] = problem3_optimizer_results["adam"]
```

이미 실행한 Adam 결과를 scheduler 없는 기준선으로 재사용한다.

```python
adam_cosine_config = {
    **problem3_optimizer_configs["adam"],
    "scheduler": "cosine",
}
```

Adam 조건에서 scheduler만 `cosine`으로 바꾼다. optimizer와 initial learning rate는 동일하게 유지한다.

이 분리가 중요한 이유는 문제 3에서 optimizer 비교와 scheduler 비교를 동시에 하면 해석이 어려워지기 때문이다. 이 셀은 scheduler의 효과만 보기 위한 구조다.

## 5부. 최종 모델과 회고 셀 설명

### Cell 17. 문제 4 설명

관련 문제: 문제 4

이 Markdown 셀은 문제 1~3 결과를 종합해 최종 모델을 선택하는 논리를 설명한다.

최종 모델 후보는 다음 가정을 반영한다.

- Sigmoid보다 ReLU가 깊은 모델에서 gradient flow에 유리하다.
- 작은 데이터셋에서는 regularization이 overfitting 완화에 필요하다.
- Adam은 일반적으로 MLP에서 안정적인 수렴을 보인다.
- Cosine scheduler는 후반부 learning rate를 낮춰 안정적인 수렴을 기대할 수 있다.

팀원 비교는 현재 노트북에 팀원 결과가 없으므로 template 형태로 둔다.

### Cell 18. 최종 모델 실행과 팀 비교 template

관련 문제: 문제 4

이 셀은 최종 모델을 학습하고, 팀원 비교용 template JSON을 저장한다.

```python
final_model_config = {
    "hidden_dims": BASE_HIDDEN_DIMS,
    "activation": "relu",
    "dropout": 0.3,
    "weight_decay": 1e-4,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "scheduler": "cosine",
    ...
}
```

최종 모델은 문제 1~3에서 얻을 수 있는 일반적인 결론을 조합한다. 단, 이 조합은 여러 요소를 동시에 적용하는 것이므로 문제 1~3의 단일 변인 실험과 성격이 다르다. 문제 4는 최종 모델 구성 단계이므로 조합이 허용된다.

```python
train_indices=full_train_indices
```

최종 모델은 validation split을 제외한 train data 전체를 사용한다. 문제 2의 regularization 비교는 5,000장 조건이지만, 최종 모델은 성능을 높이는 목적이므로 사용 가능한 학습 데이터를 더 많이 사용한다.

```python
team_comparison_template = [...]
```

팀원별 모델 구조, activation, regularization, optimizer, scheduler, validation/test accuracy를 채울 수 있는 template이다.

이 template을 만든 이유는 문제 4에서 팀원 간 최종 모델 선택 차이와 성능 차이를 분석해야 하기 때문이다. 형식을 먼저 만들어두면 나중에 팀원 결과를 같은 기준으로 비교할 수 있다.

### Cell 19. 문제 5 회고 초안

관련 문제: 문제 5

이 Markdown 셀은 실험 회고와 고찰의 초안이다.

다루는 내용은 다음과 같다.

- 모델 성능에 큰 영향을 미친 변화
- 상대적으로 영향이 작거나 조건부인 변화
- 모델을 개선한다는 것의 의미
- 과제 1 대비 LLM 활용 방식 변화

이 셀을 마지막에 둔 이유는 문제 1~4의 실험 결과를 보고 수정해야 하기 때문이다. 현재 내용은 초안이며, 실제 그래프와 summary JSON 결과를 확인한 뒤 수치를 반영해 보완하는 것이 좋다.

## 6부. 함수화한 이유 정리

이 노트북은 함수가 많다. 이유는 단순히 코드를 짧게 만들기 위해서가 아니라, 실험 통제를 위해서다.

| 함수 | 사용 위치 | 함수화 이유 |
| --- | --- | --- |
| `set_global_seed` | 전체 | 재현성 설정을 한 곳에서 관리 |
| `stratified_sample_indices` | 데이터 split | 클래스 균등 샘플링 보장 |
| `shuffled_indices` | 데이터 split | seed 기반 shuffle 재사용 |
| `class_count` | 데이터 검증 | split 균등성 확인 |
| `print_split_summary` | 데이터 검증 | split 정보를 같은 형식으로 출력 |
| `make_train_loader` | 전체 실험 | DataLoader 설정 통일 |
| `make_test_loader` | 전체 실험 | test loader 설정 통일 |
| `get_activation` | 모델 생성 | activation을 config 문자열로 관리 |
| `ControlledMLP` | 전체 실험 | 같은 모델 class에서 config만 바꿔 실험 |
| `count_trainable_parameters` | 문제 1, 결과 요약 | 파라미터 수 계산 방식 통일 |
| `model_structure_string` | 결과 출력 | 모델 구조 표현 통일 |
| `build_optimizer` | 문제 3 중심 | optimizer 생성 방식 통일 |
| `build_scheduler` | 문제 3, 4 | scheduler 적용 여부를 config로 통제 |
| `weight_gradient_norms` | 문제 1 | gradient norm 측정 방식 통일 |
| `train_one_epoch` | 전체 실험 | 학습 절차 통일 |
| `evaluate` | 전체 실험 | validation/test 평가 절차 통일 |
| `run_controlled_training` | 문제 1~4 | 실험 실행 절차 전체 통일 |
| `save_json` | 전체 결과 저장 | 결과 저장 방식 통일 |
| `plot_problem1_gradient_norms` | 문제 1 | gradient norm 그래프 형식 통일 |
| `summarize_problem1` | 문제 1 | gradient/파라미터 요약 자동화 |
| `plot_metric_comparison` | 문제 2, 3 | loss/accuracy 그래프 형식 통일 |
| `detect_overfitting_onset` | 문제 2 | overfitting 판정 기준 명시 |
| `summarize_final_metrics` | 문제 2, 3 | 비교표용 metric 요약 자동화 |

함수화의 가장 큰 장점은 **실험 간 차이를 config 차이로 제한할 수 있다는 것**이다. 이번 과제에서 가장 중요한 요구사항은 “다른 조건은 동일하게 유지”하는 것이므로, 반복되는 학습 코드를 함수로 묶는 것이 실험 설계상 더 안전하다.

## 7부. 실행과 결과 파일 구조

노트북 실행 결과는 `results_hw02` 폴더에 저장된다.

주요 결과 파일은 다음과 같다.

| 파일 | 관련 문제 | 내용 |
| --- | --- | --- |
| `problem1_deep_activation_results.json` | 문제 1 | Sigmoid/ReLU deep MLP 전체 history |
| `problem1_gradient_norms.png` | 문제 1 | layer별 gradient norm 그래프 |
| `problem1_summary.json` | 문제 1 | gradient norm, 파라미터 수, 성능 요약 |
| `problem2_regularization_results.json` | 문제 2 | regularization 조건별 전체 history |
| `problem2_regularization_curves.png` | 문제 2 | regularization train/validation 그래프 |
| `problem2_summary.json` | 문제 2 | overfitting onset 및 성능 요약 |
| `problem2_data_size_curves.png` | 문제 2 | 5,000장 vs 20,000장 비교 그래프 |
| `problem2_data_size_summary.json` | 문제 2 | 데이터 수 비교 요약 |
| `problem3_optimizer_results.json` | 문제 3 | optimizer별 전체 history |
| `problem3_optimizer_curves.png` | 문제 3 | optimizer별 train/validation 그래프 |
| `problem3_scheduler_curves.png` | 문제 3 | Adam vs Adam+cosine 비교 그래프 |
| `problem3_optimizer_summary.json` | 문제 3 | optimizer 성능 요약 |
| `problem3_scheduler_summary.json` | 문제 3 | scheduler 성능 요약 |
| `problem4_final_model_result.json` | 문제 4 | 최종 모델 결과 |
| `problem4_team_comparison_template.json` | 문제 4 | 팀원 비교 template |

결과를 JSON과 PNG로 저장하는 이유는 노트북 출력이 사라져도 보고서 작성에 필요한 수치와 그래프가 남도록 하기 위해서다.

## 8부. 보고서 작성 시 활용 방법

문제 1 보고서에는 다음을 사용하면 된다.

- `problem1_summary.json`의 파라미터 수와 최종 gradient norm
- `problem1_gradient_norms.png`
- Sigmoid의 앞쪽 layer gradient norm이 ReLU보다 작아지는지 여부

문제 2 보고서에는 다음을 사용하면 된다.

- `problem2_regularization_curves.png`
- `problem2_summary.json`의 overfitting onset epoch
- No regularization, Weight decay, Dropout, Both의 final gap 비교
- `problem2_data_size_curves.png`와 `problem2_data_size_summary.json`

문제 3 보고서에는 다음을 사용하면 된다.

- `problem3_optimizer_curves.png`
- `problem3_optimizer_summary.json`
- `problem3_scheduler_curves.png`
- Adam과 Adam+cosine scheduler의 수렴 속도와 최종 성능 비교

문제 4 보고서에는 다음을 사용하면 된다.

- `problem4_final_model_result.json`
- `problem4_team_comparison_template.json`
- 팀원 결과를 추가한 뒤 모델 선택 차이 분석

문제 5 보고서에는 Cell 19의 초안을 기반으로 실제 실험 결과 수치를 반영하면 된다.

## 9부. 설계상 주의점과 한계

첫째, 모든 실험은 seed를 고정했지만 GPU 연산은 환경에 따라 완전히 동일하지 않을 수 있다. 따라서 작은 성능 차이는 반복 실행으로 확인하는 것이 더 안전하다.

둘째, overfitting onset 판정은 코드 기준과 그래프 해석을 함께 사용해야 한다. 함수가 반환한 epoch는 보조 지표이며, 최종 보고서에서는 그래프상 validation loss 증가와 train-validation gap을 함께 설명하는 것이 좋다.

셋째, 문제 4의 최종 모델은 여러 개선 요소를 조합한다. 이것은 문제 1~3의 통제 실험과 다르게 “최종 성능을 위한 구성”이므로, 어떤 단일 요소가 성능을 올렸다고 주장하면 안 된다. 단일 요소의 효과는 문제 1~3 결과로 설명하고, 최종 모델은 그 결과를 종합한 선택으로 설명해야 한다.

넷째, Dropout 비율과 weight decay 값은 하나의 합리적 선택일 뿐이다. 더 정확한 최종 모델을 만들려면 dropout rate나 weight decay strength를 별도의 단일 변인 실험으로 추가 비교할 수 있다.

## 10부. 요약

`DL_Lab0_HW02.ipynb`는 단순히 과제 코드를 실행하는 노트북이 아니라, 실험 설계를 검증할 수 있게 만든 노트북이다.

가장 중요한 설계 의도는 다음과 같다.

- 공통 데이터, 공통 모델, 공통 학습 루프를 먼저 만들었다.
- 문제별로 바뀌는 변인을 config에만 드러나게 했다.
- 결과를 JSON과 PNG로 저장해 보고서 작성에 바로 사용할 수 있게 했다.
- 문제 1~3은 단일 변인 비교로 설계했고, 문제 4에서만 여러 요소를 종합했다.
- 문제 5는 실험 결과를 반영해 수정할 수 있는 회고 초안으로 남겼다.

따라서 이 노트북 구조의 핵심 이득은 **성능 변화의 원인을 설명할 수 있다는 것**이다. 이것이 지난 과제에서 발생했던 “무엇 때문에 성능이 바뀌었는지 알 수 없는 문제”를 해결하기 위한 가장 중요한 개선점이다.
