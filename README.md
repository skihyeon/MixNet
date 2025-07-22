MixNet
---
This is my code for MixNet: Toward Accurate Detection of Challenging Scene Text in the Wild

ref
---
[paper](https://arxiv.org/abs/2308.12817)
<br>
[github](https://github.com/D641593/MixNet)
# MixNet - 텍스트 검출 모델 인수인계 문서

## 프로젝트 개요

MixNet은 딥러닝 기반의 텍스트 검출(Text Detection) 모델입니다. 이미지에서 텍스트 영역을 정확하게 검출하고 다각형(polygon) 형태로 경계를 예측합니다.

### 주요 특징
- **FSNet 기반 백본**: 효율적인 특징 추출을 위한 FSNet 백본 아키텍처 사용
- **Evolution 모듈**: Boundary Progressive Network(BPN)을 통한 텍스트 경계 점진적 개선
- **Transformer 기반 GCN**: 그래프 합성곱 네트워크로 텍스트 경계 정제
- **다중 데이터셋 지원**: 6개 주요 공개 데이터셋 동시 학습 가능
- **분산 학습**: Accelerate를 이용한 다중 GPU 학습 지원

## 모델 구조

### 1. 전체 아키텍처 (TextNet)
```
입력 이미지 → FSNet 백본 → FPN → 세그멘테이션 헤드 → Evolution 모듈 → 최종 다각형 예측
```

### 2. 핵심 컴포넌트

#### FSNet 백본
- `network/layers/FSNet.py`: 특징 추출 네트워크
- FPN과 연결되어 다중 스케일 특징맵 생성

#### Evolution 모듈 (BPN)
- `network/evolution.py`: 텍스트 경계 점진적 개선
- Transformer 기반 GCN으로 100개 제어점 진화
- 학습 시 3회, 추론 시 1회 반복

#### 손실 함수 (TextLoss)
- `network/loss.py`: 7가지 손실 함수 조합
  - Classification Loss: 텍스트/배경 분류
  - Distance Loss: 텍스트 중심선까지의 거리
  - Direction Loss: 텍스트 방향 예측
  - Norm Loss: 법선 벡터 예측
  - Angle Loss: 각도 예측
  - Point Loss: 제어점 위치 손실
  - Energy Loss: 경계 에너지 손실

## 지원 데이터셋

### 공개 데이터셋 (Open Datasets)
1. **TotalText**: 곡선 텍스트 검출 데이터셋
2. **CTW-1500**: 중국어 곡선 텍스트
3. **MSRA-TD500**: 다국어 텍스트 검출
4. **FUNSD**: 양식 이해 데이터셋
5. **XFUND**: 다국어 양식 이해 (7개 언어)
6. **SROIE2019**: 영수증 텍스트 검출

### 커스텀 데이터셋
- JSON 형식의 annotation 지원
- `data/custom_data_root/` 경로에 배치
- Train/Test 폴더 구조: `images/`, `gt/`

### 데이터 형식
```json
{
  "fields": [
    {
      "boundingPoly": [
        {"x": 100, "y": 50},
        {"x": 200, "y": 50},
        {"x": 200, "y": 80},
        {"x": 100, "y": 80}
      ],
      "text": "텍스트 내용"
    }
  ]
}
```

## 설정 (Configuration)

### 주요 설정 파일
- `cfglib/config.py`: 기본 설정
- `cfglib/option.py`: 명령행 인자 처리

### 핵심 설정 옵션

#### 학습 설정
```python
config.batch_size = 2          # 배치 크기
config.max_epoch = 1          # 최대 에포크
config.lr = 1e-4              # 학습률
config.num_points = 100       # 제어점 개수
config.input_size = 640       # 입력 이미지 크기
```

#### 데이터셋 설정
```python
config.open_data_root = "data/open_datas"     # 공개 데이터셋 경로
config.custom_data_root = "data/custom_data"  # 커스텀 데이터셋 경로
config.select_open_data = "totaltext,ctw1500" # 선택할 공개 데이터셋
config.select_custom_data = "/"               # 선택할 커스텀 데이터셋
```

#### 모델 설정
```python
config.net = "FSNet_H_M"      # 백본 네트워크
config.mid = False            # 중간선 모드
config.embed = False          # 임베딩 모드
```

## 학습 방법

### 1. 환경 설정
```bash
# 필요한 패키지 설치
pip install torch torchvision
pip install accelerate wandb
pip install opencv-python pillow
pip install easydict scipy
```

### 2. 데이터셋 준비
```bash
# 데이터 폴더 구조
MixNet/
├── data/
│   ├── open_datas/          # 공개 데이터셋
│   │   ├── totaltext/
│   │   ├── ctw1500/
│   │   └── ...
│   └── custom_data/         # 커스텀 데이터셋
│       ├── dataset1/
│       │   ├── Train/
│       │   │   ├── images/
│       │   │   └── gt/
│       │   └── Test/
│       └── ...
```

### 3. 학습 실행
```bash
cd MixNet
python train.py \
    --exp_name "experiment_name" \
    --batch_size 4 \
    --max_epoch 20 \
    --lr 1e-4 \
    --net "FSNet_H_M" \
    --select_open_data "totaltext,ctw1500" \
    --wandb True
```

### 4. 주요 학습 특징

#### 배치 샘플링
- `BalancedBatchSampler`: annotation 수 기반 균형 잡힌 배치 생성
- 많은 텍스트가 있는 이미지를 우선적으로 배치

#### 옵티마이저 & 스케줄러
- **옵티마이저**: AdamW
- **스케줄러**: CosineAnnealingWarmRestarts
- **그래디언트 클리핑**: 25.0

#### 분산 학습
- Accelerate 라이브러리 사용
- 다중 GPU 자동 지원
- Mixed precision 학습 지원

## 평가 방법

### 1. 평가 스크립트
```bash
python eval.py \
    --exp_name "experiment_name" \
    --checkepoch 20 \
    --test_size 640
```

### 2. 평가 메트릭
- **Hit Rate**: 검출된 텍스트 수 / 실제 텍스트 수
- **Precision**: 정확하게 검출된 텍스트 비율
- **Recall**: 실제 텍스트 중 검출된 비율
- **F1-Score (H-mean)**: Precision과 Recall의 조화평균
- **IoU**: Intersection over Union 기반 정확도

### 3. 평가 과정
1. 모델을 통한 텍스트 경계 예측
2. 예측된 다각형과 Ground Truth 비교
3. IoU 임계값(일반적으로 0.5) 기준으로 정확도 계산

## 추론 (Inference)

### 1. 단일 이미지 추론
```bash
python inference.py \
    --image_path "path/to/image.jpg" \
    --checkepoch 20 \
    --exp_name "experiment_name"
```

### 2. 추론 과정
1. 이미지 전처리 및 정규화
2. 모델을 통한 특징 추출
3. Evolution 모듈로 경계 정제 (1회 반복)
4. 후처리를 통한 최종 다각형 생성

### 3. 결과 저장
- 검출된 텍스트 영역을 다각형 좌표로 저장
- 텍스트 파일 형식: `x1,y1,x2,y2,...,xn,yn`

## 주요 유틸리티

### 1. 시각화
- `util/visualize.py`: 검출 결과 시각화
- `visualize_detection()`: 검출된 텍스트 영역 표시
- `visualize_network_output()`: 네트워크 중간 출력 시각화

### 2. 데이터 증강
- `util/augmentation.py`: 다양한 데이터 증강 기법
- 회전, 크기 조정, 색상 변환, 원근 변환 등

### 3. IoU 계산
- `util/IoU.py`: 예측과 Ground Truth 간 IoU 계산
- 평가 메트릭 계산에 사용

## 모니터링 및 로깅

### 1. WandB 연동
```python
# 환경 변수 설정
export WANDB_API_KEY="your_api_key"
export WANDB_ENTITY="your_entity"

# 학습 시 자동 로깅
- 손실 함수들 (7가지)
- 학습률
- 평가 메트릭
- 모델 체크포인트
```

### 2. 로그 파일
- `output/{exp_name}/train_log.txt`: 학습 과정 로그
- `output/{exp_name}/config.txt`: 사용된 설정 저장

## 문제 해결 (Troubleshooting)

### 1. 메모리 관련
- 배치 크기 조정: `--batch_size`
- Gradient Accumulation 사용: `config.accumulation > 0`
- 메모리 정리: 코드 내 자동 가비지 컬렉션 구현

### 2. 데이터 관련
- 데이터셋 경로 확인
- JSON 형식 검증
- annotation 개수 확인

### 3. 모델 관련
- 체크포인트 경로 확인
- 백본 아키텍처 일치 여부
- CUDA 설정 확인

## 확장 및 개선 방향

### 1. 새로운 데이터셋 추가
1. `dataset/open_data/` 에 새 데이터셋 클래스 추가
2. `concat_datasets.py`에 등록
3. 데이터 로더 구현

### 2. 새로운 백본 추가
1. `network/layers/`에 새 백본 구현
2. `model_block.py`의 FPN과 연결
3. 설정 파일에 추가

### 3. 손실 함수 개선
- `network/loss.py`에서 새로운 손실 함수 추가
- 가중치 조정을 통한 균형 개선