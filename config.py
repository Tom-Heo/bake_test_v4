import os

class Config:
    # -------------------------------------------------------------------------
    # [Path Settings]
    # -------------------------------------------------------------------------
    # 데이터셋 저장/로드 루트
    DATA_DIR = "dataset"
    DIV2K_TRAIN_ROOT = os.path.join(DATA_DIR, "DIV2K_train_HR")
    DIV2K_VALID_ROOT = os.path.join(DATA_DIR, "DIV2K_valid_HR")

    # 체크포인트 및 로그 저장 경로
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    RESULT_DIR = "results"

    # 마지막 학습 상태 자동 로드 파일명
    LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "last.pth")

    # -------------------------------------------------------------------------
    # [Data Settings]
    # -------------------------------------------------------------------------
    # v4 모델의 Body 연산량과 96ch Loss 메모리 부하를 고려하여 안정적인 1로 설정
    BATCH_SIZE = 1
    NUM_WORKERS = 4  # 데이터 로더 워커 수

    # 데이터 손상(Degradation) 파라미터
    # 3-bit: 극단적인 양자화 노이즈 테스트 (AI가 구조적 특징을 배우도록 유도)
    BIT_DEPTH_INPUT = 3  
    CHROMA_SUBSAMPLE = True  # 4:2:0 Subsampling (Oklab a, b channel)

    # -------------------------------------------------------------------------
    # [Model Settings]
    # -------------------------------------------------------------------------
    INTERNAL_DIM = 30  # BakeNet 내부 연산 차원 (Internal Baking)
    EMA_DECAY = 0.999  # EMA 감쇠율

    # -------------------------------------------------------------------------
    # [Training Settings]
    # -------------------------------------------------------------------------
    TOTAL_EPOCHS = 10000  # 총 학습 에폭

    # Optimizer (AdamW) - v3의 안정적인 설정 유지
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6  # 디테일 보존을 위해 약한 규제 적용

    # Scheduler (ExponentialLR)
    SCHEDULER_GAMMA = 0.999996  # 아주 천천히 떨어지는 학습률

    # 주기 설정
    LOG_INTERVAL_STEPS = 50  # 50 스텝마다 로그 출력
    VALID_INTERVAL_EPOCHS = 5  # 5 에폭마다 검증 및 체크포인트 저장

    # -------------------------------------------------------------------------
    # [Hardware Settings]
    # -------------------------------------------------------------------------
    DEVICE = "cuda"  # CUDA 사용 필수
    USE_AMP = False  # 정밀도 유지를 위해 AMP(Mixed Precision) 미사용

    @classmethod
    def create_directories(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.RESULT_DIR, exist_ok=True)