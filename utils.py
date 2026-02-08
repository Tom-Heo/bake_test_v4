import os
import sys
import random
import logging
import torch
import numpy as np
import shutil
import requests
from zipfile import ZipFile
from tqdm import tqdm


class ModelEMA:
    """Exponential Moving Average"""

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
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup = {}


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_logger(log_dir, log_filename="train.log"):
    """
    콘솔 + 파일 로거
    - log_filename: train.log 또는 inference.log 등으로 변경 가능
    """
    # 로거 이름을 파일명에 따라 다르게 주어 중복 방지
    logger_name = f"Bake_{os.path.splitext(log_filename)[0]}"
    logger = logging.getLogger(logger_name)
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
    os.makedirs(log_dir, exist_ok=True)  # 폴더가 없으면 생성
    file_path = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(file_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def quantize_validation(tensor, bit_depth=3):
    steps = (2**bit_depth) - 1
    quantized = torch.round(tensor * steps) / steps
    return quantized.clamp(0.0, 1.0)


def compute_delta_e(pred_oklabp, target_oklabp):
    L_pred = (pred_oklabp[:, 0] + 1.0) * 0.5
    a_pred = pred_oklabp[:, 1] * 0.5
    b_pred = pred_oklabp[:, 2] * 0.5

    L_tgt = (target_oklabp[:, 0] + 1.0) * 0.5
    a_tgt = target_oklabp[:, 1] * 0.5
    b_tgt = target_oklabp[:, 2] * 0.5

    delta_L = L_pred - L_tgt
    delta_a = a_pred - a_tgt
    delta_b = b_pred - b_tgt

    delta_e = torch.sqrt(delta_L**2 + delta_a**2 + delta_b**2 + 1e-8)
    return delta_e.mean()


def save_checkpoint(
    config, epoch, model, model_ema, optimizer, scheduler, is_best=False
):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "ema_shadow": model_ema.shadow,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    last_path = config.LAST_CKPT_PATH
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(config.CHECKPOINT_DIR, "best.pth")
        shutil.copyfile(last_path, best_path)


def download_file(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    with open(save_path, "wb") as file, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)


def prepare_div2k_dataset(config, logger):
    urls = {
        "train": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "valid": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    }
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if not os.path.exists(config.DIV2K_TRAIN_ROOT):
        logger.info("DIV2K Train dataset not found. Downloading...")
        zip_path = os.path.join(config.DATA_DIR, "DIV2K_train_HR.zip")
        try:
            download_file(urls["train"], zip_path)
            logger.info("Extracting Train Set...")
            with ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(config.DATA_DIR)
            os.remove(zip_path)
        except Exception as e:
            logger.error(f"Failed to download Train set: {e}")

    if not os.path.exists(config.DIV2K_VALID_ROOT):
        logger.info("DIV2K Valid dataset not found. Downloading...")
        zip_path = os.path.join(config.DATA_DIR, "DIV2K_valid_HR.zip")
        try:
            download_file(urls["valid"], zip_path)
            logger.info("Extracting Valid Set...")
            with ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(config.DATA_DIR)
            os.remove(zip_path)
        except Exception as e:
            logger.error(f"Failed to download Valid set: {e}")
