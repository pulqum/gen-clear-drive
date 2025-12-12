# gen-clear-drive

**CycleGAN + YOLO11 기반 야간 주행 이미지 개선 파이프라인** (제작중)

야간 주행 이미지를 주간 이미지로 변환하여 객체 탐지(YOLO) 성능을 개선하는 연구 프로젝트입니다.

## 프로젝트 개요

- **CycleGAN**: 야간(Night) → 주간(Day) 이미지 도메인 변환
- **YOLO11**: 변환된 이미지에서 객체 탐지 성능 평가
- **BDD100K 데이터셋**: 날씨(clear/adverse) × 시간대(day/night) 조합

## 주요 기능

### 1. 데이터 준비 (`tools/`)
- BDD100K JSON → YOLO 형식 변환
- CycleGAN 학습용 데이터셋 구성

### 2. CycleGAN 학습 (`pytorch-CycleGAN-and-pix2pix/`)
- Night → Day 도메인 변환 모델 학습
- ResNet 9blocks 생성기 사용

### 3. 평가 실행
- `run.py`: 배치 실험 (랜덤 샘플링 → 변환 → 평가)
- `realtime_pipeline.py`: 실시간 추론 (단일 이미지/비디오)

## 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install -r requirements.txt

# CycleGAN 저장소 클론 (별도 설치 필요)
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
```

### 2. 데이터셋 준비

**데이터셋은 용량 문제로 저장소에 포함되지 않습니다.**

1. [BDD100K 데이터셋](https://bdd-data.berkeley.edu/) 다운로드
2. `tools/bdd_time_weather_to_yolo.py`로 YOLO 형식 변환
3. `datasets/yolo_bdd100k/` 아래 배치

예상 구조:
```
datasets/yolo_bdd100k/
  ├── clear_daytime/images/test/*.jpg
  ├── clear_daytime/labels/test/*.txt
  ├── clear_night/images/test/*.jpg
  └── clear_night/labels/test/*.txt
```

### 3. 사전 학습 모델 다운로드

```bash
# YOLO11 모델 (Ultralytics에서 자동 다운로드)
# 또는 수동 다운로드: https://github.com/ultralytics/assets/releases

# CycleGAN 체크포인트 (직접 학습 필요 또는 공유 링크 사용)
# pytorch-CycleGAN-and-pix2pix/checkpoints/<name>/latest_net_G_B.pth
```

### 4. 실험 실행

```bash
# 배치 평가 (10,000장 전체 평가, 1280x720 해상도)
python compare_models.py --direction night2day --n_samples 10000 --device 0

# 이미 생성된 이미지가 있다면 생성 건너뛰기 (평가만 수행)
python compare_models.py --direction night2day --n_samples 10000 --device 0 --skip_gen
```

## 📊 실험 결과 (최신)

| Model | mAP50 | Precision | Recall | 비고 |
|-------|-------|-----------|--------|------|
| **Original (Night)** | 0.3802 | 0.5266 | 0.3354 | 야간 원본 (Baseline) |
| **CycleGAN (Baseline)** | 0.2245 | 0.3512 | 0.2015 | 순수 CycleGAN (논문 설정) |
| **Ours (CycleGAN+YOLO)** | **0.3599** | **0.4812** | **0.3105** | YOLO Loss 적용 + Ensemble |

> **참고**: 위 결과는 10,000장 전체 평가 기준이며, Ours 모델은 Ensemble(Original + Fake Day) 적용 결과입니다.

## 핵심 설정 및 트러블슈팅

### 1. 해상도 및 비율 문제 (Squashing)
- **문제**: CycleGAN 학습 시 16:9 이미지를 1:1(256x256)로 강제 리사이즈하여 학습했기 때문에, 생성된 이미지의 객체가 위아래로 길쭉하게 찌그러지는 현상 발생.
- **해결 (평가 시)**: `compare_models.py`에서 `LOAD_SIZE=1280`으로 설정하여 강제로 원본 비율(16:9)을 유지하며 변환하도록 수정함.
- **해결 (재학습 시)**: `TRAIN.bat`에서 `LOAD_SIZE=512`로 설정하여 비율 왜곡을 최소화해야 함.

### 2. CycleGAN 전처리 옵션
- **Baseline (순수 CycleGAN)**: `--preprocess resize_and_crop` (논문 표준, 화질 우선)
- **Ours (YOLO Loss)**: `--preprocess scale_width` (전체 이미지 사용, 라벨 매칭 우선)

### 3. 용량 관리
- 평가 시 생성되는 `_real.png` (원본 복사본) 파일은 용량 낭비이므로, 스크립트가 자동으로 삭제하도록 설정되어 있습니다.
- 생성된 이미지는 PNG 포맷(무손실)을 유지하여 YOLO 인식률 저하를 방지합니다.

**비권장: `resize_and_crop`**
```bash
python run.py --use_crop  # 256×256 크롭 → 5배 업스케일 (품질 저하)
```

**성능 비교:**
- `scale_width`: mAP50 = 0.135 ✅
- `resize_and_crop`: mAP50 = 0.008 ❌

자세한 내용: `.github/copilot-instructions.md` 참조

## 프로젝트 구조

```
gen-clear-drive/
├── run.py                      # 메인 실험 스크립트
├── realtime_pipeline.py        # 실시간 추론
├── requirements.txt            # Python 의존성
├── tools/                      # 데이터 준비 스크립트
├── pytorch-CycleGAN-and-pix2pix/  # CycleGAN 서브모듈
├── datasets/                   # 데이터셋 (Git에서 제외)
├── experiments/                # 실험 결과 (Git에서 제외)
└── .github/
    └── copilot-instructions.md # AI 에이전트 가이드
```

## 고급 사용법

### CycleGAN 학습

```bash
cd pytorch-CycleGAN-and-pix2pix

python train.py \
  --dataroot ./datasets/clear_day2night \
  --name clear_d2n_256_e200_k10k \
  --model cycle_gan \
  --netG resnet_9blocks \
  --norm instance \
  --no_dropout \
  --load_size 286 \
  --crop_size 256 \
  --n_epochs 100 \
  --n_epochs_decay 100
```

### 야간 감지 휴리스틱 조정

```python
# realtime_pipeline.py
def is_night(img_bgr, v_thresh=55.0, dark_ratio_thresh=0.35):
    # HSV V채널 기반 단순 규칙
    # 필요 시 임계값 조정
```

## 트러블슈팅

### 1. YOLO mAP가 0
- `data.yaml`의 `test:` 경로가 `images` 폴더를 가리키는지 확인
- 병렬 `labels` 폴더 존재 확인

### 2. CycleGAN 결과 이상
- `--norm instance --no_dropout` 플래그 확인
- 체크포인트 에폭 번호 확인

### 3. 메모리 부족
- YOLO `--imgsz` 줄이기 (1280 → 640)
- CycleGAN `--batch_size 1`

## 참고 자료

- [CycleGAN 원본 저장소](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [Ultralytics YOLO 문서](https://docs.ultralytics.com/)
- [BDD100K 데이터셋](https://bdd-data.berkeley.edu/)

## 라이선스

이 프로젝트는 연구 목적으로 작성되었습니다.

- CycleGAN 코드: [BSD License](pytorch-CycleGAN-and-pix2pix/LICENSE)
- YOLO11: [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- BDD100K: [Berkeley DeepDrive License](https://bdd-data.berkeley.edu/)

## 기여

이슈나 개선 사항은 GitHub Issues를 통해 제안해주세요.

## 연락처

프로젝트 관련 문의: [GitHub Issues](../../issues)
