# DL Lab HW05 Notebook Documentation

이 문서는 `DL_Lab0_HW05.ipynb`의 각 셀이 어떤 역할을 하는지, 그리고 셀 내부의 주요 함수와 클래스가 어떻게 동작하는지를 설명한다. 노트북은 `hw05/Todo.md`의 문제 1, 문제 2, 문제 3을 수행하도록 작성되어 있다.

## 전체 흐름

노트북의 실험 흐름은 다음과 같다.

1. 실행 환경, 경로, seed, GPU/AMP 설정을 초기화한다.
2. FashionMNIST train/test dataset과 DataLoader를 만든다.
3. Autoencoder와 Variational Autoencoder 모델을 정의한다.
4. 학습, 평가, 저장, 시각화 함수를 정의한다.
5. 문제 1의 AE 실험을 latent dimension 2, 8, 32에 대해 실행한다.
6. 문제 2의 VAE 실험을 latent dimension 2, 8, 32에 대해 실행한다.
7. AE와 VAE의 최종 reconstruction loss를 비교할 수 있는 JSON 결과를 저장한다.
8. 문제 3의 classifier 기반 생성 이미지 평가를 실행한다.

## Cell 1. 노트북 제목 및 실험 범위 안내

**Cell type:** Markdown

이 셀은 노트북의 목적과 실행 방법을 설명한다.

- FashionMNIST 데이터셋을 사용한다.
- 문제 1은 Autoencoder(AE)를 다룬다.
- 문제 2는 Variational Autoencoder(VAE)를 다룬다.
- 문제 3은 팀 논의 이미지 `q3.png`의 품질, 다양성, 커버리지 기준을 반영해 classifier 기반 평가로 수행한다.
- Colab GPU 실행을 권장한다.
- CPU Colab 환경에서는 fast debug mode로 실행 batch와 epoch를 줄인다.
- 실험 결과는 로컬에서는 `hw05/results_hw05`, Colab에서는 `/content/drive/MyDrive/DL26/hw05/results_hw05`에 저장된다.

핵심 통제 조건은 AE와 VAE 모두 hidden 구조를 `784 -> 256 -> 128 -> latent -> 128 -> 256 -> 784`로 맞추고, latent dimension만 2, 8, 32로 바꾸는 것이다.

## Cell 2. import, seed, 실행 환경, 경로 설정

**Cell type:** Code

이 셀은 노트북 전체에서 사용할 library, 상수, seed, device, data/output directory, 실행 flag를 설정한다.

### 주요 상수

- `BASE_SEED = 42`: 실험 재현성을 위한 기본 seed이다.
- `LATENT_DIMS = [2, 8, 32]`: AE/VAE 실험에서 비교할 latent dimension 목록이다.
- `IMAGE_SHAPE = (1, 28, 28)`: FashionMNIST 이미지 shape이다.
- `FLAT_DIM = 28 * 28`: MLP 입력을 위해 이미지를 flatten했을 때의 차원이다.
- `NUM_CLASSES = 10`: FashionMNIST class 개수이다.
- `RUN_PROBLEM_1 = True`: 문제 1 AE 실험 실행 여부이다.
- `RUN_PROBLEM_2 = True`: 문제 2 VAE 실험 실행 여부이다.
- `RUN_PROBLEM_3 = True`: 문제 3 classifier 기반 생성 이미지 검증 실행 여부이다.
- `USE_AMP`: CUDA 사용 가능 시 automatic mixed precision을 켠다.
- `MAX_TRAIN_BATCHES`, `MAX_EVAL_BATCHES`: fast debug mode에서 학습/평가 batch 수를 제한한다.

### Colab 감지 로직

`google.colab` import 가능 여부 또는 `COLAB_GPU` 환경 변수를 통해 Colab 실행 여부를 판단한다.

- Colab이면 `data_dir = /content/data`를 사용한다.
- Google Drive mount가 성공하면 결과를 Drive 아래에 저장한다.
- Drive mount가 실패하면 `/content/results_hw05`를 fallback output directory로 사용한다.
- 로컬 실행이면 repo 안의 `hw05/data`, `hw05/results_hw05`를 사용한다.

### `set_global_seed(seed_value: int) -> None`

실험 재현성을 높이기 위한 함수이다.

동작:

- Python `random` module의 seed를 고정한다.
- PyTorch CPU random seed를 고정한다.
- CUDA가 가능하면 모든 CUDA device의 random seed를 고정한다.
- `torch.backends.cudnn.deterministic = True`로 설정해 deterministic kernel 사용을 유도한다.
- `torch.backends.cudnn.benchmark = False`로 설정해 입력 shape에 따라 cudnn이 다른 알고리즘을 선택하는 변동성을 줄인다.

주의:

- 이 설정은 재현성을 높이지만, 모든 GPU 연산이 완전히 bit-level deterministic임을 보장하지는 않는다.
- deterministic 설정은 일부 환경에서 학습 속도를 낮출 수 있다.

### `experiment_epochs(full_epochs: int) -> int`

실험 epoch 수를 환경에 따라 조절한다.

- 일반 실행에서는 `full_epochs`를 그대로 반환한다.
- Colab CPU fast debug mode에서는 `1`을 반환한다.

목적:

- CPU 환경에서 전체 학습을 오래 기다리지 않고 코드가 실행되는지만 빠르게 확인할 수 있게 한다.

### `experiment_batch_size(full_batch_size: int, fast_batch_size: int = 64) -> int`

batch size를 환경에 따라 조절한다.

- 일반 실행에서는 `full_batch_size`를 반환한다.
- fast debug mode에서는 `fast_batch_size`를 반환한다.

목적:

- GPU에서는 큰 batch size로 효율적으로 학습한다.
- CPU debug에서는 작은 batch size로 빠르게 smoke test를 수행한다.

## Cell 3. 공통 데이터 구성 설명

**Cell type:** Markdown

이 셀은 FashionMNIST 데이터 전처리와 실험 통제 조건을 설명한다.

주요 내용:

- train 60,000장, test 10,000장을 사용한다.
- AE/VAE decoder 마지막에 `Sigmoid`를 쓰므로 입력도 `[0, 1]` 범위로 둔다.
- 따라서 `Normalize`를 적용하지 않고 `transforms.ToTensor()`만 사용한다.
- AE와 VAE의 비교에서 DataLoader, optimizer 계열, learning rate, scheduler, epoch 수를 가능한 한 동일하게 유지한다.

## Cell 4. FashionMNIST dataset과 DataLoader 생성

**Cell type:** Code

이 셀은 FashionMNIST dataset을 다운로드하고, train/test DataLoader를 만든다.

### `class_names`

FashionMNIST class id를 사람이 읽을 수 있는 label로 변환하기 위한 목록이다.

순서:

0. `T-shirt/top`
1. `Trouser`
2. `Pullover`
3. `Dress`
4. `Coat`
5. `Sandal`
6. `Shirt`
7. `Sneaker`
8. `Bag`
9. `Ankle boot`

### `transform = transforms.ToTensor()`

PIL image를 PyTorch tensor로 변환한다.

결과:

- shape: `(1, 28, 28)`
- value range: `[0.0, 1.0]`

이 range는 decoder의 `Sigmoid` 출력 range와 맞기 때문에 reconstruction loss 계산이 자연스럽다.

### `train_dataset`, `test_dataset`

`torchvision.datasets.FashionMNIST`로 생성한다.

- `train_dataset`: FashionMNIST train split
- `test_dataset`: FashionMNIST test split
- `download=True`이므로 데이터가 없으면 자동 다운로드한다.
- `root=str(data_dir)`에 저장한다.

### `make_loader(dataset, batch_size, shuffle, seed_offset) -> DataLoader`

DataLoader를 만드는 helper 함수이다.

입력:

- `dataset`: PyTorch dataset 객체
- `batch_size`: 한 batch에 포함할 sample 수
- `shuffle`: sample 순서를 섞을지 여부
- `seed_offset`: shuffle generator에 사용할 seed offset

동작:

- `shuffle=True`이면 `BASE_SEED + seed_offset`으로 `torch.Generator`를 만든다.
- 이 generator를 DataLoader에 넘겨 shuffle 순서를 재현 가능하게 만든다.
- `num_workers`는 GPU 환경에서 2, CPU 환경에서 0으로 설정된다.
- CUDA가 가능하면 `pin_memory=True`를 사용해 host-to-device transfer 효율을 높인다.
- worker가 있을 때만 `persistent_workers=True`로 설정한다.

결과:

- `train_loader`: train dataset용 DataLoader, shuffle enabled
- `test_loader`: test dataset용 DataLoader, shuffle disabled

## Cell 5. 모델 구조 설명

**Cell type:** Markdown

이 셀은 AE와 VAE의 모델 구조를 설명한다.

공통 구조:

- Encoder: `784 -> 256 -> 128 -> latent_dim`
- Decoder: `latent_dim -> 128 -> 256 -> 784`

VAE 차이점:

- encoder가 latent vector를 직접 하나만 출력하지 않는다.
- `mu`와 `log_var`를 각각 출력한다.
- `z = mu + sigma * eps` 형태의 reparameterization trick으로 latent sample을 만든다.

## Cell 6. AE, VAE 모델 클래스 정의

**Cell type:** Code

이 셀은 실험에 사용할 model class와 parameter counting 함수를 정의한다.

### `class Autoencoder(nn.Module)`

FashionMNIST image를 latent vector로 압축한 뒤 다시 복원하는 deterministic autoencoder이다.

#### `Autoencoder.__init__(latent_dim: int) -> None`

모델 layer를 초기화한다.

Encoder:

- `nn.Linear(FLAT_DIM, 256)`: 784차원 입력을 256차원 hidden representation으로 변환한다.
- `nn.ReLU()`: 비선형성을 추가한다.
- `nn.Linear(256, 128)`: hidden dimension을 128로 줄인다.
- `nn.ReLU()`
- `nn.Linear(128, latent_dim)`: 최종 latent vector를 만든다.

Decoder:

- `nn.Linear(latent_dim, 128)`: latent vector를 hidden representation으로 확장한다.
- `nn.ReLU()`
- `nn.Linear(128, 256)`
- `nn.ReLU()`
- `nn.Linear(256, FLAT_DIM)`: 784차원 이미지 vector로 복원한다.
- `nn.Sigmoid()`: output pixel 값을 `[0, 1]` 범위로 제한한다.

#### `Autoencoder.forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`

입력 image를 복원하고 latent vector를 함께 반환한다.

입력:

- `x`: shape `(batch_size, 1, 28, 28)`

동작:

1. `x.view(x.size(0), -1)`로 image를 `(batch_size, 784)`로 flatten한다.
2. encoder를 통과시켜 `z`를 만든다.
3. decoder를 통과시켜 reconstruction vector를 만든다.
4. `.view(x.size(0), *IMAGE_SHAPE)`로 `(batch_size, 1, 28, 28)` shape로 되돌린다.

출력:

- `reconstruction`: 복원 이미지
- `z`: latent representation

### `class VariationalAutoencoder(nn.Module)`

입력을 확률적 latent distribution으로 encoding하고, sampling된 latent vector로 이미지를 복원하는 VAE이다.

#### `VariationalAutoencoder.__init__(latent_dim: int) -> None`

VAE layer를 초기화한다.

Encoder backbone:

- `nn.Linear(FLAT_DIM, 256)`
- `nn.ReLU()`
- `nn.Linear(256, 128)`
- `nn.ReLU()`

Latent distribution heads:

- `self.mu_layer = nn.Linear(128, latent_dim)`: latent Gaussian의 평균 `mu`를 출력한다.
- `self.log_var_layer = nn.Linear(128, latent_dim)`: latent Gaussian의 log variance `log_var`를 출력한다.

Decoder:

- AE decoder와 같은 구조이다.
- `latent_dim -> 128 -> 256 -> 784 -> Sigmoid`

#### `VariationalAutoencoder.encode(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`

입력을 Gaussian latent distribution parameter로 변환한다.

입력:

- `x`: shape `(batch_size, 1, 28, 28)`

동작:

1. 이미지를 flatten한다.
2. `encoder_backbone`을 통과시켜 128차원 hidden vector를 만든다.
3. `mu_layer`로 평균 `mu`를 계산한다.
4. `log_var_layer`로 log variance `log_var`를 계산한다.

출력:

- `mu`: shape `(batch_size, latent_dim)`
- `log_var`: shape `(batch_size, latent_dim)`

#### `VariationalAutoencoder.reparameterize(mu, log_var) -> torch.Tensor`

VAE의 핵심인 reparameterization trick을 수행한다.

수식:

```text
std = exp(0.5 * log_var)
eps ~ N(0, I)
z = mu + std * eps
```

왜 필요한가:

- sampling 자체는 gradient가 직접 지나가기 어렵다.
- `eps`를 외부 random noise로 분리하고, `mu`와 `std`에 대한 differentiable 연산으로 `z`를 만들면 backpropagation이 가능해진다.

출력:

- `z`: sampled latent vector, shape `(batch_size, latent_dim)`

#### `VariationalAutoencoder.decode(z: torch.Tensor) -> torch.Tensor`

latent vector를 이미지로 복원한다.

입력:

- `z`: shape `(batch_size, latent_dim)`

출력:

- reconstruction image, shape `(batch_size, 1, 28, 28)`

#### `VariationalAutoencoder.forward(x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`

VAE 전체 forward pass를 수행한다.

동작:

1. `encode(x)`로 `mu`, `log_var`를 얻는다.
2. `reparameterize(mu, log_var)`로 `z`를 sampling한다.
3. `decode(z)`로 이미지를 복원한다.

출력:

- `reconstruction`: 복원 이미지
- `mu`: latent Gaussian 평균
- `log_var`: latent Gaussian log variance
- `z`: sampled latent vector

### `count_parameters(model: nn.Module, trainable_only: bool = False) -> int`

모델 parameter 수를 계산한다.

입력:

- `model`: PyTorch model
- `trainable_only`: `True`이면 `requires_grad=True`인 parameter만 계산한다.

동작:

- `model.parameters()`를 순회하며 `param.numel()`을 합산한다.

출력:

- parameter 개수 integer

이 함수는 latent dimension별 AE/VAE 모델 크기를 비교하는 데 사용된다.

## Cell 7. 학습/평가/시각화 함수 섹션 설명

**Cell type:** Markdown

이 셀은 이후 code cell들이 공통 학습, 평가, 결과 저장, 시각화 함수를 정의한다는 것을 안내한다.

## Cell 8. Loss, 학습, 평가, 저장 함수 정의

**Cell type:** Code

이 셀은 AE/VAE 학습과 평가에 필요한 핵심 함수들을 정의한다.

### `reconstruction_loss(reconstruction, x) -> torch.Tensor`

복원 이미지와 원본 이미지 사이의 MSE loss를 계산한다.

입력:

- `reconstruction`: model output image
- `x`: original image

동작:

- `F.mse_loss(reconstruction, x, reduction="mean")`을 반환한다.

의미:

- pixel 단위 평균 제곱 오차이다.
- 값이 낮을수록 원본에 가까운 복원이다.

### `vae_loss(reconstruction, x, mu, log_var, beta) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

VAE total loss를 계산하고, reconstruction loss와 KL divergence를 분리해 반환한다.

수식:

```text
loss = reconstruction_loss + beta * KL_divergence
KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
```

입력:

- `reconstruction`: VAE decoder output
- `x`: original image
- `mu`: latent Gaussian mean
- `log_var`: latent Gaussian log variance
- `beta`: KL divergence 가중치

출력:

- `loss`: 학습에 사용하는 total loss
- `recon`: reconstruction MSE
- `kl`: batch 평균 KL divergence

해석:

- reconstruction loss는 이미지 복원 품질을 높인다.
- KL divergence는 latent distribution이 표준정규분포에 가까워지도록 regularization한다.
- `beta`가 커질수록 latent space 정규화가 강해지지만, reconstruction 품질이 흐려질 수 있다.

### `amp_context()`

automatic mixed precision context manager를 반환한다.

동작:

- CUDA 사용 시 `torch.autocast(device_type=device.type, dtype=torch.float16)`을 반환한다.
- CUDA가 아니면 `nullcontext()`를 반환한다.

목적:

- GPU에서는 float16 mixed precision으로 학습 속도와 memory 효율을 개선한다.
- CPU에서는 동일한 코드 흐름을 유지하되 mixed precision을 사용하지 않는다.

### `train_ae(latent_dim, epochs, learning_rate, weight_decay) -> tuple[Autoencoder, dict]`

특정 latent dimension의 AE를 학습한다.

입력:

- `latent_dim`: latent vector 차원
- `epochs`: 학습 epoch 수
- `learning_rate`: AdamW learning rate
- `weight_decay`: AdamW weight decay

동작:

1. seed를 `BASE_SEED + latent_dim`으로 다시 고정한다.
2. `Autoencoder(latent_dim)` 모델을 생성하고 device로 이동한다.
3. `AdamW` optimizer를 만든다.
4. `CosineAnnealingLR` scheduler를 사용한다.
5. CUDA 사용 시 `GradScaler`로 AMP gradient scaling을 적용한다.
6. epoch마다 train loader를 순회한다.
7. reconstruction loss를 계산하고 backpropagation한다.
8. epoch 종료 후 `evaluate_ae`로 test reconstruction loss를 계산한다.
9. train/test reconstruction loss를 `history`에 저장한다.

출력:

- 학습 완료된 AE model
- 학습 기록 dictionary

`history` 주요 key:

- `latent_dim`
- `parameters`
- `train_recon_loss`
- `test_recon_loss`

### `evaluate_ae(model: Autoencoder) -> float`

AE의 test reconstruction loss를 계산한다.

동작:

- `model.eval()`과 `torch.no_grad()`로 evaluation mode를 사용한다.
- test loader를 순회하며 reconstruction MSE를 sample 수로 weighted average한다.
- fast debug mode에서는 `MAX_EVAL_BATCHES`만큼만 평가한다.

출력:

- test set 평균 reconstruction MSE

### `train_vae(latent_dim, epochs, learning_rate, weight_decay, beta) -> tuple[VariationalAutoencoder, dict]`

특정 latent dimension의 VAE를 학습한다.

입력:

- `latent_dim`: latent vector 차원
- `epochs`: 학습 epoch 수
- `learning_rate`: AdamW learning rate
- `weight_decay`: AdamW weight decay
- `beta`: KL divergence 가중치

동작:

1. seed를 `BASE_SEED + 100 + latent_dim`으로 다시 고정한다.
2. `VariationalAutoencoder(latent_dim)` 모델을 생성하고 device로 이동한다.
3. optimizer, scheduler, AMP scaler를 설정한다.
4. train loader를 순회한다.
5. model forward로 `reconstruction`, `mu`, `log_var`, `z`를 얻는다.
6. `vae_loss`로 total/reconstruction/KL loss를 계산한다.
7. total loss 기준으로 backpropagation한다.
8. epoch 종료 후 `evaluate_vae`로 test total/reconstruction/KL loss를 계산한다.
9. 각 loss component를 history에 저장한다.

출력:

- 학습 완료된 VAE model
- 학습 기록 dictionary

`history` 주요 key:

- `latent_dim`
- `parameters`
- `beta`
- `train_total_loss`
- `train_recon_loss`
- `train_kl_loss`
- `test_total_loss`
- `test_recon_loss`
- `test_kl_loss`

### `evaluate_vae(model: VariationalAutoencoder, beta: float) -> dict[str, float]`

VAE의 test loss를 계산한다.

동작:

- evaluation mode와 no-grad context를 사용한다.
- test loader를 순회한다.
- `vae_loss`를 사용해 total, reconstruction, KL loss를 계산한다.
- sample 수 기준 weighted average를 구한다.

출력 dictionary:

- `"total"`: 평균 total loss
- `"recon"`: 평균 reconstruction loss
- `"kl"`: 평균 KL divergence

### `save_json(payload: dict, path: Path) -> None`

실험 결과 dictionary를 JSON 파일로 저장한다.

동작:

- `json.dumps(..., indent=2, ensure_ascii=False)`로 사람이 읽기 쉬운 JSON 문자열을 만든다.
- UTF-8 encoding으로 파일을 저장한다.
- 저장 경로를 출력한다.

사용 목적:

- 학습 결과를 노트북 output에만 의존하지 않고 파일로 남긴다.
- 최종 보고서 작성 시 수치 결과를 다시 참조할 수 있게 한다.

## Cell 9. 시각화 및 latent 추출 함수 정의

**Cell type:** Code

이 셀은 loss curve, reconstruction image grid, 2D latent space plot을 그리는 함수들을 정의한다.

### `plot_ae_loss_curves(histories, path) -> None`

AE의 train/test reconstruction loss curve를 latent dimension별로 한 그래프에 그린다.

입력:

- `histories`: latent dimension을 key로 갖는 AE history dictionary
- `path`: 저장할 image path

동작:

- 각 latent dimension에 대해 train loss는 dashed line으로 그린다.
- test loss는 solid line으로 그린다.
- x축은 epoch, y축은 MSE reconstruction loss이다.
- figure를 `path`에 PNG로 저장하고 화면에도 표시한다.

해석:

- latent dimension이 커질수록 test reconstruction loss가 낮아지는지 확인할 수 있다.
- train/test loss 간 차이를 통해 overfitting 여부도 대략 확인할 수 있다.

### `plot_vae_loss_curves(histories, path) -> None`

VAE의 total loss, reconstruction loss, KL divergence를 나란히 시각화한다.

입력:

- `histories`: latent dimension을 key로 갖는 VAE history dictionary
- `path`: 저장할 image path

동작:

- 1행 3열 subplot을 만든다.
- 첫 번째 subplot: total loss
- 두 번째 subplot: reconstruction loss
- 세 번째 subplot: KL divergence
- 각 subplot에서 train/test curve를 latent dimension별로 함께 그린다.

해석:

- reconstruction 품질과 latent regularization의 trade-off를 분리해서 볼 수 있다.
- KL divergence가 너무 작으면 posterior collapse 가능성을 의심할 수 있고, 너무 크면 reconstruction 품질 저하를 확인해야 한다.

### `collect_reconstructions(model, model_type, sample_count=8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

test batch에서 일부 이미지를 가져와 원본과 복원 결과를 수집한다.

입력:

- `model`: AE 또는 VAE model
- `model_type`: `"ae"` 또는 `"vae"`
- `sample_count`: 시각화할 sample 수

동작:

- test loader의 첫 batch를 가져온다.
- 앞에서 `sample_count`개 이미지만 선택한다.
- `model_type == "ae"`이면 `model(x)`에서 reconstruction을 얻는다.
- `model_type == "vae"`이면 VAE forward output에서 reconstruction을 얻는다.
- CPU tensor로 변환해 반환한다.

출력:

- `originals`: 원본 이미지 tensor
- `reconstructions`: 복원 이미지 tensor
- `labels`: class label tensor

### `plot_reconstruction_grid(originals, reconstructions, labels, path, title) -> None`

원본 이미지와 복원 이미지를 2행 grid로 시각화한다.

입력:

- `originals`: 원본 이미지 tensor
- `reconstructions`: 복원 이미지 tensor
- `labels`: label tensor
- `path`: 저장할 image path
- `title`: plot title

동작:

- 첫 번째 행에는 원본 이미지를 표시한다.
- 두 번째 행에는 복원 이미지를 표시한다.
- 각 열은 같은 sample의 원본/복원 pair이다.
- 원본 이미지 위에는 class name을 표시한다.

해석:

- latent dimension별 복원 품질을 눈으로 비교할 수 있다.
- AE와 VAE가 같은 latent dimension에서 얼마나 선명하게 복원하는지 확인할 수 있다.

### `collect_latent_2d(model, model_type, max_batches=None) -> tuple[torch.Tensor, torch.Tensor]`

test dataset을 encoder에 통과시켜 2D latent vector와 label을 수집한다.

입력:

- `model`: AE 또는 VAE model
- `model_type`: `"ae"` 또는 `"vae"`
- `max_batches`: 처리할 최대 batch 수. `None`이면 전체 test loader를 사용한다.

동작:

- AE에서는 `model(x)`의 `z`를 사용한다.
- VAE에서는 random sample `z` 대신 `model.encode(x)`의 `mu`를 사용한다.

VAE에서 `mu`를 쓰는 이유:

- sampling된 `z`는 random noise에 따라 매번 조금 달라진다.
- latent space 구조를 안정적으로 시각화하려면 distribution 중심인 `mu`가 더 적합하다.

출력:

- `latents`: shape `(num_samples, 2)`
- `labels`: class labels

전제:

- 이 함수는 latent dimension이 2인 모델에 사용하도록 작성되어 있다.

### `plot_latent_space(latents, labels, path, title) -> None`

2D latent vector를 scatter plot으로 시각화한다.

입력:

- `latents`: 2D latent tensor
- `labels`: class label tensor
- `path`: 저장할 image path
- `title`: plot title

동작:

- `latents[:, 0]`을 x축, `latents[:, 1]`을 y축으로 사용한다.
- class label을 color로 구분한다.
- `tab10` colormap과 colorbar를 사용한다.

해석:

- class별 cluster가 분리되는지 확인한다.
- AE와 VAE latent space의 분포 차이를 비교한다.

## Cell 10. 문제 1 섹션 제목

**Cell type:** Markdown

문제 1 AE 실험이 시작됨을 알리는 셀이다.

내용:

- latent dimension 2, 8, 32에 대해 같은 AE 구조를 학습한다.
- 비교의 핵심은 latent dimension 증가가 reconstruction loss와 복원 이미지 품질에 미치는 영향이다.

## Cell 11. 문제 1 AE 실험 실행

**Cell type:** Code

이 셀은 AE를 latent dimension별로 학습하고 결과를 저장/시각화한다.

### 주요 설정

- `AE_EPOCHS = experiment_epochs(15)`: 일반 실행에서는 15 epoch, fast debug에서는 1 epoch이다.
- `AE_LR = 1e-3`: AdamW learning rate이다.
- `AE_WEIGHT_DECAY = 1e-5`: AdamW weight decay이다.
- `ae_models`: 학습된 AE model을 latent dimension별로 저장한다.
- `ae_histories`: AE 학습 기록을 latent dimension별로 저장한다.

### 실행 흐름

`RUN_PROBLEM_1`이 true일 때 다음을 수행한다.

1. `LATENT_DIMS = [2, 8, 32]`를 순회한다.
2. 각 `latent_dim`에 대해 `train_ae`를 호출한다.
3. 학습된 model과 history를 각각 `ae_models`, `ae_histories`에 저장한다.
4. `problem1_ae_results.json`으로 history를 저장한다.
5. `problem1_ae_reconstruction_loss.png`로 train/test reconstruction loss curve를 저장한다.
6. latent dimension별로 원본 vs 복원 이미지 grid를 저장한다.
7. `latent_dim=2` AE 모델의 test latent space를 scatter plot으로 저장한다.

### 생성 산출물

- `problem1_ae_results.json`
- `problem1_ae_reconstruction_loss.png`
- `problem1_ae_latent_2_reconstructions.png`
- `problem1_ae_latent_8_reconstructions.png`
- `problem1_ae_latent_32_reconstructions.png`
- `problem1_ae_latent_2_space.png`

### 결과 해석 포인트

- test reconstruction loss가 낮을수록 복원 품질이 좋다.
- `latent_dim=2`는 정보 병목이 강해 복원이 흐릴 가능성이 크다.
- `latent_dim=32`는 더 많은 정보를 보존할 수 있어 일반적으로 가장 선명한 복원을 기대할 수 있다.
- 2D latent plot에서는 시각적으로 유사한 class들이 겹치는지 확인한다.

## Cell 12. 문제 1 분석 초안

**Cell type:** Markdown

이 셀은 AE 실험 결과를 보고서에 서술할 때 사용할 분석 방향을 제시한다.

핵심 주장:

- latent dimension이 커질수록 표현 용량이 증가한다.
- 표현 용량 증가는 reconstruction loss 감소와 이미지 선명도 개선으로 이어질 수 있다.
- `latent_dim=2`는 시각화에는 좋지만 정보 손실이 크다.
- FashionMNIST에서 상의류처럼 형태가 유사한 class는 latent space에서 겹치기 쉽다.

최종 보고서에서는 이 초안을 그대로 쓰기보다 실제 저장된 loss graph, reconstruction image, latent space plot을 근거로 수치를 보완해야 한다.

## Cell 13. 문제 2 섹션 제목

**Cell type:** Markdown

문제 2 VAE 실험이 시작됨을 알리는 셀이다.

내용:

- AE와 같은 hidden 구조를 사용한다.
- encoder output을 `mu`, `log_var`로 분리한다.
- VAE loss를 reconstruction loss와 KL divergence로 나누어 기록한다.

## Cell 14. 문제 2 VAE 실험 실행

**Cell type:** Code

이 셀은 VAE를 latent dimension별로 학습하고 결과를 저장/시각화한다.

### 주요 설정

- `VAE_EPOCHS = experiment_epochs(15)`: 일반 실행에서는 15 epoch, fast debug에서는 1 epoch이다.
- `VAE_LR = 1e-3`: AdamW learning rate이다.
- `VAE_WEIGHT_DECAY = 1e-5`: AdamW weight decay이다.
- `VAE_BETA = 1e-4`: KL divergence 가중치이다.
- `vae_models`: 학습된 VAE model을 latent dimension별로 저장한다.
- `vae_histories`: VAE 학습 기록을 latent dimension별로 저장한다.

### `VAE_BETA = 1e-4`의 의미

VAE의 KL divergence는 reconstruction MSE보다 scale이 클 수 있다. `beta`를 작게 두면 학습 초기에 reconstruction 품질을 지나치게 희생하지 않으면서 latent regularization을 적용할 수 있다.

trade-off:

- `beta`가 너무 작으면 latent space가 표준정규분포에 잘 맞지 않을 수 있다.
- `beta`가 너무 크면 복원 이미지가 흐려질 수 있다.

### 실행 흐름

`RUN_PROBLEM_2`가 true일 때 다음을 수행한다.

1. `LATENT_DIMS = [2, 8, 32]`를 순회한다.
2. 각 `latent_dim`에 대해 `train_vae`를 호출한다.
3. 학습된 model과 history를 각각 `vae_models`, `vae_histories`에 저장한다.
4. `problem2_vae_results.json`으로 history를 저장한다.
5. `problem2_vae_loss_components.png`로 total/reconstruction/KL loss curve를 저장한다.
6. latent dimension별 원본 vs 복원 이미지 grid를 저장한다.
7. `latent_dim=2` VAE 모델의 `mu` latent space를 scatter plot으로 저장한다.

### 생성 산출물

- `problem2_vae_results.json`
- `problem2_vae_loss_components.png`
- `problem2_vae_latent_2_reconstructions.png`
- `problem2_vae_latent_8_reconstructions.png`
- `problem2_vae_latent_32_reconstructions.png`
- `problem2_vae_latent_2_space.png`

### 결과 해석 포인트

- VAE의 reconstruction loss는 AE보다 높을 수 있다.
- 이는 VAE가 reconstruction뿐 아니라 latent distribution regularization도 동시에 만족해야 하기 때문이다.
- VAE latent space는 AE보다 연속적이고 원점 주변에 모이는 경향을 보일 수 있다.
- KL loss curve를 함께 봐야 VAE가 단순 AE처럼 동작하는지, 또는 posterior collapse가 의심되는지 판단할 수 있다.

## Cell 15. AE/VAE 비교 table 생성 및 저장

**Cell type:** Code

이 셀은 문제 1과 문제 2의 최종 수치를 latent dimension별로 비교 가능한 형태로 정리한다.

### `build_ae_vae_comparison_table(ae_histories, vae_histories) -> list[dict]`

AE와 VAE의 최종 test metric을 table 형태의 list로 만든다.

입력:

- `ae_histories`: 문제 1에서 생성한 AE history dictionary
- `vae_histories`: 문제 2에서 생성한 VAE history dictionary

동작:

1. `LATENT_DIMS`를 순회한다.
2. 각 latent dimension에 대해 row dictionary를 만든다.
3. AE 결과가 있으면 다음 값을 추가한다.
   - `ae_test_recon_loss`
   - `ae_parameters`
4. VAE 결과가 있으면 다음 값을 추가한다.
   - `vae_test_total_loss`
   - `vae_test_recon_loss`
   - `vae_test_kl_loss`
   - `vae_parameters`
5. row들을 list로 반환한다.

출력 예시 구조:

```python
[
    {
        "latent_dim": 2,
        "ae_test_recon_loss": ...,
        "ae_parameters": ...,
        "vae_test_total_loss": ...,
        "vae_test_recon_loss": ...,
        "vae_test_kl_loss": ...,
        "vae_parameters": ...,
    },
    ...
]
```

### 실행 흐름

1. `comparison_rows = build_ae_vae_comparison_table(ae_histories, vae_histories)`를 실행한다.
2. `problem2_ae_vae_comparison.json`으로 저장한다.
3. 마지막 줄에 `comparison_rows`를 두어 Jupyter output으로 table-like list가 표시되게 한다.

### 생성 산출물

- `problem2_ae_vae_comparison.json`

### 결과 해석 포인트

- 같은 latent dimension에서 AE와 VAE의 reconstruction loss를 비교한다.
- AE/VAE parameter 수 차이를 함께 확인한다.
- VAE는 `mu_layer`와 `log_var_layer`를 따로 가지므로 AE보다 parameter 수가 약간 많다.
- reconstruction loss만 보면 AE가 유리할 수 있지만, VAE는 sampling 가능한 latent distribution을 얻는 것이 추가 목적이다.

## Cell 16. 문제 2 분석 초안

**Cell type:** Markdown

이 셀은 VAE와 AE 비교 결과를 보고서에 서술할 때 사용할 분석 방향을 제시한다.

핵심 주장:

- AE는 deterministic latent code를 사용한다.
- VAE는 `mu`, `log_var`가 정의하는 확률분포에서 latent vector를 sampling한다.
- KL divergence 때문에 VAE latent space는 표준정규분포에 가까운 연속적 구조를 갖도록 유도된다.
- 같은 latent dimension에서 VAE reconstruction은 AE보다 흐릴 수 있다.
- `latent_dim=2` 시각화에서 VAE는 AE보다 더 regularized된 분포를 보일 수 있다.

최종 보고서에서는 다음 파일을 근거로 분석을 보완해야 한다.

- `problem1_ae_results.json`
- `problem2_vae_results.json`
- `problem2_ae_vae_comparison.json`
- AE/VAE reconstruction image
- AE/VAE 2D latent space plot

## Cell 17. 문제 3 생성 모델 검증 기준 및 평가 방법 설명

**Cell type:** Markdown

이 셀은 `q3.png`에 담긴 팀 논의 결과를 노트북에 연결한다.

팀 논의 기준:

- 품질(Quality): 생성 이미지가 실제 FashionMNIST 이미지처럼 보이고, 사람이 보거나 classifier가 판단했을 때 클래스 특징이 분명해야 한다.
- 다양성(Diversity): 특정 패턴만 반복하지 않고 상의, 바지, 신발, 가방 등 다양한 class와 형태를 생성해야 한다.
- 커버리지(Coverage): 실제 데이터에 존재하는 class 또는 mode를 고르게 생성해야 한다.

노트북에서 선택한 구현 방법은 classifier 기반 평가이다.

평가 방식:

1. FashionMNIST train split으로 CNN classifier를 학습한다.
2. VAE decoder에 `z ~ N(0, I)`를 입력해 이미지를 생성한다.
3. 생성 이미지를 classifier에 넣어 예측 확률을 얻는다.
4. 평균 confidence와 high-confidence 비율로 품질을 평가한다.
5. 예측 class distribution과 entropy로 다양성을 평가한다.
6. 생성 결과에서 관찰된 class 개수로 coverage를 평가한다.

## Cell 18. FashionMNIST classifier 정의 및 학습/평가 함수

**Cell type:** Code

이 셀은 문제 3에서 생성 이미지를 평가할 classifier를 정의하고 학습하는 함수를 포함한다.

### `class FashionMNISTClassifier(nn.Module)`

FashionMNIST 이미지를 10개 class로 분류하는 CNN이다.

구조:

- `Conv2d(1, 32, kernel_size=3, padding=1)`
- `BatchNorm2d(32)`
- `ReLU`
- `MaxPool2d(2)`
- `Conv2d(32, 64, kernel_size=3, padding=1)`
- `BatchNorm2d(64)`
- `ReLU`
- `MaxPool2d(2)`
- `Flatten`
- `Linear(64 * 7 * 7, 128)`
- `ReLU`
- `Dropout(0.25)`
- `Linear(128, NUM_CLASSES)`

입력 shape:

- `(batch_size, 1, 28, 28)`

출력 shape:

- `(batch_size, 10)` logits

이 classifier는 생성 모델 자체가 아니라 평가 도구이다. 따라서 test accuracy를 함께 기록해 평가 도구가 실제 FashionMNIST를 충분히 잘 분류하는지 확인한다.

### `FashionMNISTClassifier.forward(x: torch.Tensor) -> torch.Tensor`

classifier의 forward pass이다.

동작:

1. `self.features(x)`로 convolution feature map을 만든다.
2. `self.classifier(...)`로 logits를 계산한다.

출력:

- softmax 이전 logits

주의:

- loss 계산에는 `F.cross_entropy`를 사용하므로 forward 함수 내부에서 softmax를 적용하지 않는다.

### `train_classifier(epochs, learning_rate, weight_decay) -> tuple[FashionMNISTClassifier, dict]`

FashionMNIST classifier를 학습한다.

입력:

- `epochs`: classifier 학습 epoch 수
- `learning_rate`: AdamW learning rate
- `weight_decay`: AdamW weight decay

동작:

1. seed를 `BASE_SEED + 300`으로 고정한다.
2. `FashionMNISTClassifier`를 생성하고 device로 이동한다.
3. `AdamW` optimizer와 `CosineAnnealingLR` scheduler를 설정한다.
4. train loader를 순회하며 cross entropy loss로 학습한다.
5. epoch마다 `evaluate_classifier`로 test loss와 test accuracy를 계산한다.
6. train/test loss와 accuracy를 `history`에 저장한다.

출력:

- 학습된 classifier
- classifier 학습 기록 dictionary

`history` 주요 key:

- `parameters`
- `train_loss`
- `train_accuracy`
- `test_loss`
- `test_accuracy`

### `evaluate_classifier(model: FashionMNISTClassifier) -> dict[str, float]`

classifier의 test 성능을 평가한다.

동작:

- `model.eval()`과 `torch.no_grad()`로 evaluation mode를 사용한다.
- test loader를 순회하며 cross entropy loss와 accuracy를 계산한다.
- fast debug mode에서는 `MAX_EVAL_BATCHES`만큼만 평가한다.

출력:

- `"loss"`: test 평균 cross entropy loss
- `"accuracy"`: test accuracy

문제 3에서 classifier test accuracy는 평가 방법의 신뢰도를 확인하는 보조 지표이다.

## Cell 19. VAE 생성 이미지 평가 및 시각화 함수

**Cell type:** Code

이 셀은 VAE로 이미지를 생성하고, classifier 예측을 바탕으로 품질/다양성/커버리지 지표를 계산하고 시각화하는 함수들을 정의한다.

### `generate_vae_images(model, latent_dim, sample_count, batch_size) -> torch.Tensor`

VAE decoder로 생성 이미지를 만든다.

입력:

- `model`: 학습된 `VariationalAutoencoder`
- `latent_dim`: 해당 VAE의 latent dimension
- `sample_count`: 생성할 이미지 수
- `batch_size`: generation batch size

동작:

1. `z = torch.randn(current_batch_size, latent_dim)`으로 표준정규분포 latent sample을 만든다.
2. `model.decode(z)`로 이미지를 생성한다.
3. 생성 이미지를 `[0, 1]` 범위로 clamp한다.
4. 모든 batch를 concat해 CPU tensor로 반환한다.

출력:

- generated images, shape `(sample_count, 1, 28, 28)`

### `evaluate_generated_images_with_classifier(classifier, images, batch_size, confidence_threshold=0.8) -> dict`

생성 이미지를 classifier로 평가한다.

입력:

- `classifier`: 학습된 FashionMNIST classifier
- `images`: VAE 생성 이미지 tensor
- `batch_size`: 평가 batch size
- `confidence_threshold`: high-confidence image로 볼 기준 확률

동작:

1. 생성 이미지를 classifier에 넣어 logits를 얻는다.
2. softmax로 class probability를 계산한다.
3. 각 이미지의 최대 class probability를 confidence로 사용한다.
4. argmax class를 predicted class로 사용한다.
5. predicted class count와 distribution을 계산한다.
6. class distribution의 entropy와 normalized entropy를 계산한다.
7. confidence, class distribution, coverage metric을 dictionary로 반환한다.

출력 metric:

- `sample_count`: 평가한 생성 이미지 수
- `mean_confidence`: 평균 최대 softmax 확률
- `median_confidence`: 중앙값 confidence
- `high_confidence_threshold`: high-confidence 기준값
- `high_confidence_rate`: confidence가 threshold 이상인 비율
- `coverage_class_count`: 생성 결과에서 관찰된 predicted class 개수
- `class_entropy`: predicted class distribution entropy
- `normalized_class_entropy`: entropy를 `log(10)`으로 나눈 값
- `class_counts`: class별 예측 개수
- `class_distribution`: class별 예측 비율

해석:

- `mean_confidence`와 `high_confidence_rate`가 높으면 classifier가 생성 이미지를 더 명확한 class로 인식했다는 뜻이다.
- `normalized_class_entropy`가 높으면 class 분포가 더 고르다는 뜻이다.
- `coverage_class_count`가 10에 가까우면 더 많은 class가 생성 결과에 나타났다는 뜻이다.

### `plot_generated_samples(images, path, title, sample_count=40) -> None`

생성 이미지 sample을 grid로 저장한다.

동작:

- 생성 이미지 중 앞 `sample_count`개를 선택한다.
- 10열 grid로 이미지를 표시한다.
- PNG 파일로 저장한다.

사용 목적:

- classifier 수치만으로 알 수 없는 시각적 품질을 사람이 직접 확인한다.

### `plot_class_distribution(metrics_by_name, path) -> None`

VAE latent dimension별 predicted class distribution을 bar plot으로 그린다.

입력:

- `metrics_by_name`: VAE 이름을 key로 하고 평가 metric dictionary를 value로 갖는 dictionary
- `path`: 저장할 PNG 경로

해석:

- 특정 class 막대만 크면 mode collapse 또는 coverage 부족을 의심할 수 있다.
- 여러 class에 고르게 분포하면 다양성과 coverage가 상대적으로 좋다고 해석할 수 있다.

### `plot_problem3_metric_bars(metrics_by_name, path) -> None`

문제 3의 핵심 metric을 bar chart로 비교한다.

시각화 metric:

- `mean_confidence`
- `high_confidence_rate`
- `normalized_class_entropy`

해석:

- confidence metric은 품질 기준과 연결된다.
- normalized entropy는 다양성/커버리지 기준과 연결된다.
- 같은 그래프에서 latent dimension별 VAE 생성 품질 차이를 비교할 수 있다.

## Cell 20. 문제 3 classifier 기반 평가 실행

**Cell type:** Code

이 셀은 문제 3 전체 실험을 실행한다.

### 주요 설정

- `CLASSIFIER_EPOCHS = experiment_epochs(5)`: classifier 학습 epoch 수이다.
- `CLASSIFIER_LR = 1e-3`: classifier AdamW learning rate이다.
- `CLASSIFIER_WEIGHT_DECAY = 1e-4`: classifier weight decay이다.
- `GENERATED_SAMPLE_COUNT = 1000`: 일반 실행에서 VAE별 생성 이미지 수이다.
- `GENERATION_BATCH_SIZE = experiment_batch_size(256)`: generation/evaluation batch size이다.

### 실행 흐름

`RUN_PROBLEM_3`이 true일 때 다음을 수행한다.

1. `vae_models`가 비어 있으면 문제 2를 먼저 실행하라는 error를 발생시킨다.
2. `train_classifier`로 FashionMNIST classifier를 학습한다.
3. 각 VAE latent dimension에 대해 `generate_vae_images`로 이미지를 생성한다.
4. `evaluate_generated_images_with_classifier`로 생성 이미지를 평가한다.
5. `plot_generated_samples`로 생성 이미지 grid를 저장한다.
6. 논의 요약, classifier history, VAE별 생성 평가 metric을 JSON으로 저장한다.
7. class distribution plot을 저장한다.
8. confidence/entropy metric bar plot을 저장한다.

### 생성 산출물

- `problem3_classifier_based_evaluation.json`
- `problem3_generated_samples_latent_2.png`
- `problem3_generated_samples_latent_8.png`
- `problem3_generated_samples_latent_32.png`
- `problem3_generated_class_distribution.png`
- `problem3_generation_metric_bars.png`

### `problem3_classifier_based_evaluation.json`에 저장되는 내용

- `discussion_summary`: `q3.png`의 품질, 다양성, 커버리지 논의 요약
- `evaluation_method`: 사용한 평가 방식
- `classifier_history`: classifier train/test loss와 accuracy 기록
- `generated_metrics`: VAE latent dimension별 생성 이미지 평가 metric

## Cell 21. 문제 3 분석 초안

**Cell type:** Markdown

이 셀은 문제 3 결과를 보고서에 서술할 때 사용할 분석 방향을 제시한다.

핵심 연결:

- 품질은 `mean_confidence`, `high_confidence_rate`로 분석한다.
- 다양성은 class distribution과 `normalized_class_entropy`로 분석한다.
- 커버리지는 `coverage_class_count`와 class distribution으로 분석한다.

방법의 장점:

- 사람이 모든 이미지를 직접 평가하지 않아도 자동으로 수치화할 수 있다.
- classifier test accuracy를 함께 제시하면 평가 도구 자체의 신뢰도를 어느 정도 설명할 수 있다.
- class별 분포를 통해 mode collapse나 class 편향을 확인할 수 있다.

방법의 한계:

- classifier가 틀리거나 생성 이미지에 대해 과신할 수 있다.
- 높은 confidence가 항상 사람이 보기에 좋은 이미지를 의미하지는 않는다.
- class 내부의 세부 mode 다양성까지 충분히 측정하지 못한다.
- FashionMNIST classifier가 학습한 기준에 맞는 이미지만 높게 평가될 수 있다.

## 함수와 클래스 의존 관계 요약

주요 실행 의존 관계는 다음과 같다.

```text
set_global_seed
  -> train_ae
  -> train_vae
  -> train_classifier

experiment_epochs / experiment_batch_size
  -> AE_EPOCHS, VAE_EPOCHS, CLASSIFIER_EPOCHS, BATCH_SIZE, GENERATION_BATCH_SIZE

make_loader
  -> train_loader
  -> test_loader

Autoencoder
  -> train_ae
  -> evaluate_ae
  -> collect_reconstructions
  -> collect_latent_2d

VariationalAutoencoder
  -> train_vae
  -> evaluate_vae
  -> collect_reconstructions
  -> collect_latent_2d
  -> generate_vae_images

FashionMNISTClassifier
  -> train_classifier
  -> evaluate_classifier
  -> evaluate_generated_images_with_classifier

reconstruction_loss
  -> train_ae
  -> evaluate_ae
  -> vae_loss

vae_loss
  -> train_vae
  -> evaluate_vae

generate_vae_images
  -> VAE generated sample tensors

evaluate_generated_images_with_classifier
  -> problem3 quality/diversity/coverage metrics

plot_* functions
  -> PNG result files

save_json
  -> JSON result files
```

## 제출 전 확인 사항

제출용 결과를 만들 때는 다음을 확인한다.

1. GPU 런타임에서 실행했는지 확인한다.
2. `COLAB_FAST_DEV_RUN = False`인지 확인한다.
3. `MAX_TRAIN_BATCHES = None`, `MAX_EVAL_BATCHES = None`인지 확인한다.
4. 문제 1, 문제 2, 문제 3의 JSON, PNG 결과가 모두 생성되었는지 확인한다.
5. `problem3_classifier_based_evaluation.json`에 classifier test accuracy와 generated metric이 들어 있는지 확인한다.
6. `q3.png`의 논의 기준인 품질, 다양성, 커버리지가 문제 3 분석에 각각 연결되어 있는지 확인한다.
