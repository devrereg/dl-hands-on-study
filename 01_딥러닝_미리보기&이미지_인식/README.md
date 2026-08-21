# 01. 딥러닝 미리보기 & 이미지 인식

`강의_10기_AI개론_1차시_01_intro.ipynb`는 앞으로 진행할 딥러닝 학습(개/늑대 이미지 분류)의 전체 흐름을 미리 훑어보는 예고편 성격의 노트북입니다. 세부 이론보다는 "데이터 → 전처리 → 데이터셋/로더 → 학습 → 시각화"로 이어지는 파이토치 실습의 큰 그림을 잡는 데 목적이 있습니다.

## 실행 환경

- Google Colab 기준으로 작성됨 (나눔 폰트 설치, `apt`/`pip` 명령 등 Colab 환경 의존)
- VSCode에서 실행하려면 Colab 확장 설치 후 실행 환경을 Colab으로 변경 (계정 연결 필요)

## 노트북 흐름

1. **환경 준비**
   - 나눔고딕 폰트 설치 및 matplotlib 캐시 초기화 (그래프 한글 깨짐 방지)
   - `torchviz`, `tree` 등 실습에 필요한 패키지/커맨드 설치
2. **라이브러리 임포트 & 공통 설정**
   - `torch`, `torchvision`, `matplotlib`, `numpy` 등 임포트
   - 그래프 폰트/사이즈/그리드 등 matplotlib 기본값 설정, GPU/CPU 디바이스 확인
   - [wikibook/pythonlibs](https://github.com/wikibook/pythonlibs) 저장소를 클론하여 `torch_lib1`의 공통 함수(`fit`, `show_images_labels`, `torch_seed` 등) 로드
3. **학습 데이터 준비**
   - 개(dog) / 늑대(wolf) 이미지 압축 파일을 다운로드하고 `train`/`test` 폴더로 압축 해제
   - `tree` 명령으로 데이터 폴더 구조 확인
4. **이미지 전처리 (`torchvision.transforms`)**
   - 검증/테스트용: `Resize → CenterCrop → ToTensor → Normalize`로 기본 정규화만 적용
   - 학습용: 위 과정에 `RandomHorizontalFlip`, `RandomErasing`을 추가해 데이터 증강(augmentation) 적용
5. **Dataset / DataLoader 구성**
   - `ImageFolder`로 폴더명을 라벨로 자동 매핑
   - 증강 적용 학습용 로더, 증강 없는 평가용 로더, 테스트용 로더 등 목적별로 여러 `DataLoader` 정의
6. **데이터 확인**
   - `show_images_labels` 함수로 로딩된 이미지와 라벨을 그리드로 시각화
7. **공통 함수 살펴보기 (참고)**
   - `fit`: 학습/검증 루프를 한 번에 수행하는 학습 함수
   - `show_images_labels`: 이미지 + 정답/예측 라벨을 함께 시각화하는 함수
   - `torch_seed`: 재현성을 위한 난수 고정 함수

## 배울 점

- 딥러닝 실습(특히 이미지 분류)의 전체 파이프라인이 어떤 단계로 구성되는지 감을 잡을 수 있음
- `torchvision.transforms`를 이용한 이미지 전처리 및 데이터 증강 기법의 역할과 사용법
- 앞으로 반복해서 사용할 공통 함수(`fit`, `show_images_labels`, `torch_seed`)의 구조 미리보기
