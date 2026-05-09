## 문제 1. CNN 모델 직접 설계 및 학습

CIFAR-10 데이터셋에 대한 분류 task를 수행하는 CNN 모델을 직접 설계하고 학습하시오.
아래 요소를 모두 포함하여 CNN 모델을 설계하시오.

 (a) Conv2d layer (최소 2개 이상)

 (b) Pooling layer

 (c) Batch Normalization

 (d) Dropout

 (e) 적절한 activation function

※ 모델 구조는 본인이 자유롭게 설계할 것.

※ Optimizer, learning rate, scheduler 등은 과제 2의 경험을 바탕으로 선택할 것

각 팀원은 자신의 모델에 대해 다음을 기록하시오.

 - 모델 구조 및 파라미터 수

 - Batch Normalization과 Dropout의 적용 위치 및 그 이유

 - Train/Test Loss 및 Accuracy 그래프



## 문제 2. Pre-defined 모델 구조 활용

PyTorch torchvision.models에서 제공하는 모델 구조를 불러와 CIFAR-10에 대해 처음부터(from scratch) 학습하시오.

이 문제에서는 사전 학습된 weight를 사용하지 않고, 모델 구조만 가져와서 학습하시오.

아래 2가지 모델을 각각 학습하시오.

 (a) VGG-16 (torchvision.models.vgg16(weights=None))

 (b) ResNet-18 (torchvision.models.resnet18(weights=None))

※ 두 모델 모두 ImageNet용으로 설계되어 있으므로, 마지막 FC layer를 CIFAR-10에 맞게 수정해야 함

※ Optimizer, learning rate 등은 본인이 선택 (두 모델에 동일하게 적용할 것)

각 팀원은 자신의 모델에 대해 다음을 기록하시오.

 - 각 모델의 파라미터 수

 - 각 모델의 Train/Test Accuracy 그래프

 - VGG-16과 ResNet-18의 성능 비교 및 차이 원인 분석

 - 문제 1의 직접 설계 CNN과 VGG-16, ResNet-18의 성능 비교 및 차이 분석(모델 크기 대비 성능 효율 관점 분석 포함)



## 문제 3. Pre-trained Weight를 활용한 Transfer Learning

ImageNet으로 사전 학습된 weight를 불러와 CIFAR-10에 대해 transfer learning을 수행하시오.

※ Pre-trained 모델은 ImageNet의 224×224 크기 이미지로 학습되었으나, CIFAR-10의 원본 이미지는 32×32이므로 입력을 224×224로 리사이징한 후 모델에 입력해야 함.

※ 224×224 입력은 메모리를 많이 사용하므로 batch size를 줄여야 할 수 있음 (예: 64)

※ 학습에 많은 시간이 소요되므로 10~20 epoch만 수행하는 것을 권장

ResNet-18을 기준으로 아래 3가지 방식을 비교하시오.

 (a) From Scratch (weights=None): 전체 모델을 CIFAR-10 데이터를 통해 처음부터 학습

 (b) Feature Extraction (weights="IMAGENET1K_V1"): 마지막 FC layer만 CIFAR-10 데이터를 통해 추가 학습하고 나머지 layer는 freeze

 (c) Fine-tuning (weights="IMAGENET1K_V1"): 전체 모델 파라미터를 CIFAR-10 데이터를 통해 추가 학습

※ 조건 (b)에서 layer freeze 방법: param.requires_grad = False

※ (c)에서는 (a)보다 낮은 learning rate를 사용할 것을 권장

※ 3가지 조건 모두 동일한 epoch 수로 학습할 것

각 팀원은 자신의 모델에 대해 다음을 기록하시오.

 - 3가지 조건의 Train/Test Accuracy 그래프 (하나의 그래프에 겹쳐서 표현)

 - 각 조건별 학습 가능한 파라미터 수

 - 학습 초반(1~5 epoch)의 수렴 속도 차이

 - 각 방식의 최종 성능, 수렴 속도의 차이가 발생하는 이유 분석

 - Pre-trained weight가 의미하는 것은 무엇인지, 그리고 ImageNet에서 학습한 feature가 CIFAR-10에도 유효한 이유 분석



## 문제 4. LLM 생성 코드의 검증 (팀별)

문제 3에서 세 가지 방식을 비교하기 위해 팀원이 작성한 코드를 어떻게 검증할 수 있는지를 직접 고찰하고 실험을 통해 검증을 수행하시오.

검증을 위해 아래 방법을 참고하시오. (다른 방법도 가능)

 (a) 차원 검증: 각 layer의 입출력 shape를 수동으로 추적하여 올바른지 확인

 (b) 더미 테스트: 더미(dummy) 입력을 넣어 출력 shape와 값의 범위 확인

 (c) 중간값 출력: forward 과정의 중간 feature map shape 또는 gradient 유무 등을 출력하여 확인

 (d) 의도적 오류 삽입: 코드의 일부를 의도적으로 잘못 변경한 후 이에 대한 디버깅을 요청하여 오류를 찾아내는지 확인

팀원의 코드에 대한 검증 결과에 대해 다음을 기록하시오.

 - 검증 대상으로 선택한 코드에 대한 검증을 LLM에게 요청한 과정

 - 적용한 검증 방법과 구체적인 검증 과정

 - 검증 결과 발견된 문제점 (있는 경우)

 - 문제점이 발견되지 않은 경우, 해당 검증 방법의 한계는 무엇인지



## 문제 5. 실험 회고 및 고찰

다음 질문에 대해 서술하시오.

 - 본인 코드에 대한 팀원의 검증에 대한 평가 또는 소감

 - 팀원의 코드를 검증하는 과정에서의 한계점

 - 실행은 되지만 올바르지 않은 코드를 검증하기 위해 어떤 능력 또는 지식이 필요한 지에 대한 생각