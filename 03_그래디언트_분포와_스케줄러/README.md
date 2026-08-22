# 그래디언트 분포와 학습률 스케줄러 실습

`grad_distribution_and_lr_scheduler.ipynb` 파일에 대한 소개입니다.

## 실습 목적

MNIST 손글씨 분류를 위한 MLP를 학습시키면서, **He 초기화 + OneCycleLR 스케줄러**를 함께 적용했을 때 학습이 어떻게 진행되는지 확인합니다. 특히 **역전파 훅(hook)**을 이용해 학습 중 각 레이어의 그래디언트 크기 변화를 직접 관찰하는 것이 핵심입니다.

## 진행 흐름

1. **데이터 준비**
   MNIST 데이터셋을 `transforms.ToTensor()`로 전처리(0~1 스케일링)하고, `DataLoader`로 배치(train: 256, test: 512) 단위 로딩 구성

2. **모델 정의 (MLP)**
   `28*28 → 512 → 256 → 10` 구조의 완전연결 신경망(ReLU 활성화)
   - 가중치(weight)는 **He 초기화**(`kaiming_normal_`) 적용 → ReLU에 적합
   - 편향(bias)은 0으로 초기화

3. **Optimizer & 스케줄러**
   - `AdamW` optimizer 사용
   - `OneCycleLR` 스케줄러로 학습률을 낮게 시작 → `max_lr`까지 상승 → 다시 매우 낮게 하강시키는 패턴 적용 (배치마다 `schd.step()` 호출)

4. **그래디언트 모니터링 (hook)**
   각 `nn.Linear` 레이어에 `register_full_backward_hook`을 등록해, 역전파가 일어날 때마다 가중치 그래디언트의 평균 절댓값을 `grads` 리스트에 자동 기록

5. **학습/평가 루프**
   - `train_epoch()`: 배치별로 forward → loss 계산 → backward → optimizer/scheduler 업데이트, 훈련 정확도 계산
   - `eval_epoch()`: 테스트 데이터로 정확도 평가 (`torch.no_grad()`로 그래디언트 계산 생략)
   - 5 에폭 동안 반복하며 에폭별 평균 그래디언트 크기(`hist_grad`)와 테스트 정확도(`hist_acc`)를 기록

6. **결과 시각화**
   - 에폭별 평균 그래디언트 크기 추이 그래프
   - 에폭별 테스트 정확도 추이 그래프
   - 관찰이 끝난 후 등록해둔 훅 제거(`h.remove()`)

## 핵심 포인트

- **He 초기화**: ReLU 계열 네트워크에서 층을 통과해도 신호 분산이 유지되도록 가중치를 초기화하는 방법
- **OneCycleLR**: 학습률을 한 사이클(상승 → 하강) 패턴으로 자동 조절해 빠르고 안정적인 수렴을 유도
- **Backward hook**: 역전파 시점에 개입해 그래디언트 크기를 실시간으로 기록 → vanishing/exploding gradient 여부를 눈으로 확인 가능
- 그래디언트 크기 추이와 정확도 추이를 함께 비교하면, 학습이 안정적으로 진행되는지(그래디언트가 너무 작아지거나 커지지 않는지) 진단할 수 있음