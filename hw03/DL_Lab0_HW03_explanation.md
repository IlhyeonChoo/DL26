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
