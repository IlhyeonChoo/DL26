## 문제 1. Deep architecture 실험

각 팀원은 동일한 baseline 모델(Fashion MNIST 데이터셋에 대한 classification 모델)을 출발점으로 하여 더 깊은 구조로 확장하고, 깊은 모델에서 발생하는 문제를 직접 확인하시오.

Gradient vanishing 관찰을 위해 아래 두 모델을 각각 학습하고 각 layer의 gradient norm을 epoch별로 기록하시오.

 (a) 8층 이상 MLP 모델, sigmoid activation function

 (b) 동일 구조, ReLU activation function

※ hidden dimension, learning rate, optimizer 등 activation 외의 조건은 동일하게 유지할 것

※ Gradient norm 기록 방법: 각 layer의 weight에 대한 gradient의 L2 norm을 epoch마다 계산하여 저장하시오. (param.grad.norm())

각 팀원은 자신의 모델에 대해 다음을 기록하시오.

 - 모델 (a), (b)의 구조 설명 및 파라미터 수

 - 각 layer별 gradient norm 변화 그래프 (x축: layer 위치, y축: gradient norm, 모델 (a)와 (b)를 겹쳐서 표현)

 - Sigmoid 모델에서 gradient vanishing이 발생하는지 여부와, 그래프상 어떻게 확인되는지 분석

 - ReLU로 변경했을 때 gradient 흐름이 어떻게 달라지는지 분석



## 문제 2. Regularization 비교 실험

Overfitting을 명확하게 관찰하기 위해, 학습 데이터를 축소한 환경에서 regularization 효과를 비교하시오.

Fashion MNIST 학습 데이터 60,000장 중 5,000장만 무작위로 샘플링하여 학습 데이터로 사용하시오. (Test 데이터는 원본 그대로 사용할 것)

※ 샘플링 시 클래스별 비율이 균등하도록 할 것 (클래스당 500장)

다음 3가지 조건에 대해 각각 학습하시오. (optimizer, learning rate 등 나머지 조건은 동일하게 고정)

 (a) No regularization

 (b) Weight decay (L2 regularization) 적용

 (c) Dropout 적용 (비율은 본인이 선택)

각 팀원은 자신의 모델에 대해 다음을 기록하시오.

 - 각 조건별 Train/Validation Loss 및 Accuracy 그래프 (하나의 그래프에 3개 조건을 겹쳐서 표현)

 - (a)에서 overfitting이 발생하는지 여부 및 발생 시점 (epoch 기준)

 - 각 regularization 기법이 overfitting 양상에 미친 영향 분석

 - Dropout과 Weight decay를 동시에 적용했을 때의 결과 비교

 - 훈련 데이터 양이 overfitting에 미치는 영향 분석



## 문제 3. Optimizer 비교 실험

다음 4가지 조건에 대해 각각 학습하시오. (optimizer, learning rate 등 나머지 조건은 동일하게 고정)

 (a) SGD

 (b) SGD with momentum

 (c) AdaGrad

 (d) Adam

위 optimizer 중 하나를 선택하여 임의의 learning rate scheduling 전략을 적용하시오.

각 팀원은 자신의 모델에 대해 다음을 기록하시오.

 - 각 optimizer별 Train/Validation Loss 및 Accuracy 그래프

 - optimizer에 따른 수렴 속도와 최종 성능 비교

 - Learning rate scheduling 적용 전/후 비교 분석



## 문제 4. 최종 모델 구성 및 팀 내 비교

문제 1~3의 실험 결과를 종합하여, 본인의 최종 모델을 구성하시오.

 - 모델 구조, activation, regularization, optimizer, scheduler 등 결정

팀원의 최종 모델을 비교하고, 다음을 분석하시오.

 - 최종 모델 선택의 차이가 발생한 원인 분석

 - 모델 선택에 따른 성능 차이가 발생한 주요 원인 분석



## 문제 5. 실험 회고 및 고찰

다음 질문에 대해 서술하시오.

 - 모델의 성능에 큰 영향을 미치는 변화와 큰 영향을 미치지 않는 변화는 무엇인가?

 - 모델을 개선한다는 것은 무엇을 의미하는지에 대한 본인의 생각은 무엇인가?

 - 과제 1 대비 LLM 활용 방식이 어떻게 달라졌는가?

