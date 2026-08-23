# 다중 분류(Multi-class Classification) 실습

`multiclass_classification_pytorch.ipynb` 파일에 대한 소개입니다.

## 실습 목적

**Iris(붓꽃) 데이터셋**을 이용해 PyTorch로 **다중 분류 모델**을 처음부터 직접 구현하고, `CrossEntropyLoss`의 내부 동작(`NLLLoss`, `LogSoftmax`, `Softmax`)까지 단계별로 뜯어보며 이해하는 실습입니다.

## 진행 흐름

1. **데이터 준비**
   `sklearn.datasets.load_iris()`로 붓꽃 데이터(150개 샘플, 4개 특성, 3개 클래스) 로드 → 이해를 돕기 위해 초반엔 특성 2개(sepal length, petal length)만 추출해 사용
   `train_test_split`으로 훈련/검증 데이터 분할(75/75), 클래스별 산포도로 데이터 분포 시각화

2. **모델 정의**
   `nn.Linear(n_input, n_output)` 하나로 구성된 **2입력 3출력 로지스틱 회귀 모델** (다중 분류의 가장 단순한 형태)
   `torchinfo.summary`, `torchviz.make_dot`으로 모델 구조와 계산 그래프 확인

3. **손실 함수 & 옵티마이저**
   `nn.CrossEntropyLoss()` + `optim.SGD`로 학습 준비

4. **경사하강법 학습 루프**
   - 훈련 페이즈: `optimizer.zero_grad()` → forward → loss 계산 → `backward()` → `optimizer.step()`
   - 검증 페이즈: `net.eval()` + `torch.no_grad()`로 감싸서 불필요한 그래디언트 계산 방지
   - `torch.max(outputs, 1)[1]`로 예측 클래스(인덱스) 추출, 정확도 계산

5. **결과 확인**
   에폭별 loss/accuracy 학습 곡선 시각화, 모델이 실제로 출력한 값(softmax 확률)과 가중치 행렬/바이어스 값 직접 확인

6. **입력 변수 4차원화**
   특성을 4개(전체 Iris 특성) 모두 사용한 모델로 확장해 재학습

7. **칼럼: NLLLoss의 동작 원리**
   NLLLoss(Negative Log Likelihood)가 "정답 클래스의 로그 확률만 뽑아서 부호를 뒤집는" 방식으로 손실을 계산하는 과정을 간단한 예시로 직접 확인

8. **칼럼: 다중 분류 모델의 세 가지 구현 패턴 비교**
   - 패턴1: 모델은 로짓(logit)만 출력, `CrossEntropyLoss`가 Softmax+NLLLoss를 내부에서 처리
   - 패턴2: 모델 안에 `LogSoftmax`를 포함하고 `NLLLoss` 사용
   - 패턴3: 모델 안에 `Softmax`를 포함 (일반적으로 권장되지 않는 방식, 비교용)
   - 마지막으로 `Softmax(dim=0)`과 `Softmax(dim=1)`의 계산 축 차이도 실습으로 확인

## 핵심 포인트

- 다중 분류에서 **정답 label은 클래스 인덱스이므로 반드시 정수(`long`) 타입**이어야 함
- `CrossEntropyLoss` = `LogSoftmax` + `NLLLoss`가 결합된 형태라는 것을 세 가지 구현 패턴 비교를 통해 확인
- 검증(평가) 단계는 `net.eval()` + `torch.no_grad()`로 감싸는 것이 정석
- 모델 구조, 계산 그래프, 학습 곡선, 가중치까지 다각도로 들여다보며 다중 분류의 내부 동작 원리를 이해하는 데 초점