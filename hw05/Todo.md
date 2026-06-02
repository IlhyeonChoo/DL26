## 문제 1. Autoencoder 구현

FashionMNIST 데이터에 대해 Autoencoder(AE)를 구현하고, 잠재 공간(latent space)의 특성을 분석하시오.

아래 3가지 latent dimension에 대해 각각 AE를 구현하고 학습하시오.

 (a)2 (b)8 (c)32

※ Encoder/Decoder 구조: 784 → 256 → 128 → latent_dim → 128 → 256 → 784 (권장)

※ 모든 조건에서 latent_dim 외의 설정은 동일하게 유지할 것

각 팀원은 자신의 모델에 대해 다음을 기록하시오.

- 각 조건별 Train/Test Reconstruction Loss 그래프

- 각 조건별 원본 이미지 vs 복원 이미지 비교 (5장 이상)

- Latent dimension이 복원 품질에 미치는 영향 분석

- (latent_dim=2인 경우) Test 데이터를 Encoder에 통과시킨 후, 2D latent space에 점으로 표시하고 클래스별로 색을 구분하여 시각화하고, 어떤 패턴이 관찰되는지 분석

## 문제 2. Variational Autoencoder 구현

VAE를 구현하고, AE와의 차이를 비교하시오. 다음 사항을 포함하여 VAE를 구현하시오.

- Encoder: 입력 → hidden layers → μ (mean) 와 log σ² (log variance) 를 각각 출력

- Reparameterization trick: z = μ + σ · ε (ε ~ N(0, 1))

- Decoder: z → hidden layers → 복원 이미지

- Loss = Reconstruction Loss + β · KL Divergence

※ Encoder/Decoder 구조는 문제 1과 동일하게 유지할 것을 권장

문제 1과 동일한 3가지 latent dimension (2, 8, 32)에 대해 각각 학습하시오.

각 팀원은 자신의 모델에 대해 다음을 기록하시오.

- 각 조건별 Train/Test Loss 그래프 (Reconstruction Loss와 KL Divergence를 분리하여 표시)

- 각 조건별 원본 이미지 vs 복원 이미지 비교

- 문제 1의 AE 결과와 동일 latent dimension에서의 복원 품질 비교

- (latent_dim=2인 경우) 2D latent space 시각화 및 문제 1의 AE latent space와 비교: 분포의 형태가 어떻게 다른지 분석

## 문제 3. 생성 모델의 검증 (팀별)

VAE가 생성한 이미지를 어떻게 평가하고 검증할 수 있는지를 팀 내에서 논의하고, 최소 1가지 방법을 직접 구현하여 적용하시오.

분류 모델(classifier)은 Test Accuracy로 성능을 측정할 수 있다. 그렇다면 생성 모델은 무엇을 기준으로 좋다고 판단할 수 있는지 팀 내에서 논의하고, 아래 관점을 포함하여 서술하시오.

 (a) 품질(Quality): 생성된 이미지가 실제 이미지와 얼마나 유사한가?

 (b) 다양성(Diversity): 생성된 이미지가 다양한가, 아니면 특정 패턴만 반복하는가?

 (c) 커버리지(Coverage): 실제 데이터의 모든 클래스/모드를 고르게 생성하는가?

아래 방법 중 최소 1가지를 선택하여 직접 구현하고, VAE의 생성 결과에 적용하시오. (다른 방법을 제안하고 구현해도 됨)

 (a) Classifier 기반 평가: 사전 학습된 MNIST 분류기로 생성 이미지를 분류하여, 분류 정확도와 클래스별 분포를 측정

 (b) Reconstruction 기반 평가: 생성 이미지를 다시 Encoder에 통과시킨 후 Decoder로 복원했을 때, 원본 생성 이미지와의 차이를 측정

 (c) 실제 데이터와의 거리 기반 평가: 생성 이미지와 실제 이미지를 latent space에서 비교하여 거리 분포를 측정

 (d) 사람 평가: 팀원들이 직접 생성 이미지를 보고 숫자를 맞추어, 인식률을 측정

각 팀원은 다음을 기록하시오.

- 팀에서 논의한 "좋은 생성 모델의 기준"에 대한 정리

- 선택한 검증 방법의 구체적인 구현 과정 및 코드

- 검증 결과 (수치 및 시각화)

- 해당 검증 방법의 한계: 이 방법으로 측정할 수 없는 측면은 무엇인지
