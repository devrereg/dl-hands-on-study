# dl-hands-on-study

딥러닝 개론 강의를 따라가며 PyTorch로 직접 구현한 실습 기록입니다.
이론으로만 넘기지 않고, 주요 개념들을 직접 손코딩하면서 이해한 과정을 정리합니다.

## 다루는 내용

- 이미지 전처리 (`torchvision.transforms`)
- 데이터셋 / 데이터로더 구성
- CNN 모델 설계 및 학습 루프 구현
- 학습 결과 시각화 (loss, accuracy)

## 폴더 구조

```
dl-hands-on-study/
├── 01_딥러닝_미리보기&이미지_인식/
└── README.md
```

> 폴더 구성은 진행 상황에 따라 계속 업데이트됩니다.

## 환경

- Python 3.x
- PyTorch
- torchvision
- matplotlib, numpy

## 참고

이 저장소는 학습 목적으로 작성되었으며, 강의 자료를 기반으로 직접 코드를 작성하고
필요한 부분에 주석과 설명을 덧붙여 정리한 것입니다.

colab 환경에서 실행되는 코드입니다.
VSCODE 사용시 확장프로그램 설치 후 실행환경 colab으로 변경 후 사용가능 (계정 연결 필요)