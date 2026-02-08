import os
import sys
import random
import logging
import torch
import numpy as np
import shutil


class ModelEMA:
    """
    [Exponential Moving Average]
    학습 중 파라미터의 이동 평균을 유지하여,
    최종 추론 시 더 안정적이고 일반화된 성능을 제공합니다.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register(model)

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                # new_avg = (1 - decay) * current + decay * old_avg
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        """검증 전 호출: EMA 가중치를 모델에 덮어씌움"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()  # 원본 백업
                param.data.copy_(self.shadow[name])  # EMA 적용

    def restore(self, model):
        """검증 후 호출: 원본 가중치 복구"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup = {}


def seed_everything(seed=42):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # cudnn.benchmark = True와 deterministic은 충돌할 수 있으나,
    # 시드 고정은 최소한의 안전장치입니다.
    # torch.backends.cudnn.deterministic = True (속도를 위해 생략 가능)


def get_logger(log_dir):
    """콘솔 + 파일 로거"""
    logger = logging.getLogger("BakeTrain")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File
    file_path = os.path.join(log_dir, "train.log")
    file_handler = logging.FileHandler(file_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def quantize_validation(tensor, bit_depth=3):
    """
    [Validation Only]
    검증 시 성능 측정을 위해 고정된 3-bit 양자화를 적용
    (디더링 없이 순수 양자화로 성능 측정)
    """
    steps = (2**bit_depth) - 1
    quantized = torch.round(tensor * steps) / steps
    return quantized.clamp(0.0, 1.0)


def compute_delta_e(pred_oklabp, target_oklabp):
    """
    [Delta E Metric]
    OklabP (Processing Scale) -> Standard Oklab -> Euclidean Distance

    OklabP: Lp [-1, 1], ap/bp [Large]
    Std Oklab: L [0, 1], a/b [Small]

    Conversion:
      L = (Lp + 1) / 2
      a = ap / 2
      b = bp / 2
    """
    # 1. Re-scale to Standard Oklab
    L_pred = (pred_oklabp[:, 0] + 1.0) * 0.5
    a_pred = pred_oklabp[:, 1] * 0.5
    b_pred = pred_oklabp[:, 2] * 0.5

    L_tgt = (target_oklabp[:, 0] + 1.0) * 0.5
    a_tgt = target_oklabp[:, 1] * 0.5
    b_tgt = target_oklabp[:, 2] * 0.5

    # 2. Euclidean Distance (Perceptual Difference)
    delta_L = L_pred - L_tgt
    delta_a = a_pred - a_tgt
    delta_b = b_pred - b_tgt

    # Delta E = sqrt(dL^2 + da^2 + db^2)
    delta_e = torch.sqrt(delta_L**2 + delta_a**2 + delta_b**2 + 1e-8)

    # Batch Mean
    return delta_e.mean()


def save_checkpoint(
    config, epoch, model, model_ema, optimizer, scheduler, is_best=False
):
    """Last & Best 저장 전략"""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "ema_shadow": model_ema.shadow,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    # Always save 'last.pth' (Overwrite)
    last_path = config.LAST_CKPT_PATH
    torch.save(state, last_path)

    # If best, copy 'last.pth' to 'best.pth'
    if is_best:
        best_path = os.path.join(config.CHECKPOINT_DIR, "best.pth")
        shutil.copyfile(last_path, best_path)
