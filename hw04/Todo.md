데이터셋 소개

ETTh1은 전력 변압기의 온도 및 관련 변수를 1시간 단위로 기록한 시계열 데이터셋입니다.

기간: 약 2년 (2016.7 ~ 2018.6, 총 17,420개 시점)

변수: 7개 (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)

예측 대상: OT (Oil Temperature)

입력: 과거 96시간의 7개 변수 → 출력: 미래 24시간의 OT 값

※ 데이터 분할: Train (12개월) / Validation (4개월) / Test (8개월)

※ 데이터 로드 코드는 아래 제공

문제 1. RNN / LSTM / GRU 비교 실험

ETTh1 데이터셋에 대해 RNN, LSTM, GRU 기반 시계열 예측 모델을 각각 구현하고 비교하시오.

아래 3가지 모델을 구현하고 각각 학습하시오.

 (a) RNN: 7개 변수 입력 → RNN → FC → OT 예측

 (b) LSTM: 7개 변수 입력 → LSTM → FC → OT 예측

 (c) GRU: 7개 변수 입력 → GRU → FC → OT 예측

※ hidden_dim, num_layers 등 RNN/LSTM/GRU 외의 조건은 동일하게 유지할 것

※ 권장 설정: hidden_dim = 64, num_layers = 1

※ 입력: (batch, 96, 7), 출력: (batch, 24) — 과거 96시간 → 미래 24시간 OT

※ 평가 지표: MSE (Mean Squared Error) 및 MAE (Mean Absolute Error)

각 팀원은 자신의 모델에 대해 다음을 기록하시오.

- 각 모델의 파라미터 수

- 각 모델의 Train/Test Loss(MSE) 그래프 (하나의 그래프에 3개 모델을 겹쳐서 표현)

- 세 모델의 최종 MSE, MAE 비교

- 특정 구간에 대한 예측값 vs 실제값 시각화 (3개 모델을 겹쳐서 표현)

- LSTM/GRU가 RNN 대비 어떤 이점이 있는지 구조적 차이를 근거로 서술

문제 2. 입력 길이 비교 및 bidirectional 실험

문제 1에서 가장 성능이 좋았던 모델을 기준으로, 입력 시퀀스 길이와 bidirectional 설정에 따른 성능 변화를 분석하시오.

아래의 input_length에 대해 각각 학습하시오. (출력 길이 24시간은 고정)

 (a) 24

 (b) 48

 (c) 96

 (d) 192

input_length를 하나 선택하여 다음 2가지 조건을 비교하시오.

 (a) bidirectional = False

 (b) bidirectional = True

각 팀원은 자신의 모델에 대해 다음을 기록하시오.

- 입력 길이별 Test MSE, MAE 비교 그래프 또는 표

- 입력 길이가 길어질수록 성능이 어떻게 변화하는지, 그리고 그 이유 분석

- Bidirectional 적용 전/후 성능 비교 및 차이 원인 분석

- 시계열 예측 task에서 bidirectional이 적절한지에 대한 본인의 생각

문제 3. Transformer 기반 시계열 예측

PyTorch의 nn.TransformerEncoder를 활용하여 시계열 예측 모델을 구현하고 학습하시오.

아래 구조의 예측 모델을 구현하시오.

Input Projection (Linear) + Positional Encoding → TransformerEncoder → FC → Output

※ nn.TransformerEncoderLayer와 nn.TransformerEncoder를 사용할 것

※ 권장 설정: d_model=64, nhead=4, num_layers=2, dim_feedforward=128

※ Positional Encoding은 학습 가능한 방식(nn.Embedding 또는 nn.Parameter)으로 구현하시오

※ 입력/출력 형태는 문제 1과 동일 (96시간 → 24시간)

각 팀원은 자신의 모델에 대해 다음을 기록하시오.

- 모델 파라미터 수

- Train/Test Loss(MSE) 그래프

- 문제 1 모델과의 성능 비교 (MSE, MAE)

- 특정 구간에 대한 예측값 vs 실제값 시각화 (문제 1 모델과 Transformer를 겹쳐서 표현)
