import os
import sys
import random
import logging
import math
import torch
import torch.nn.functional as F
import numpy as np
import shutil
import requests
from zipfile import ZipFile
from tqdm import tqdm


# -----------------------------------------------------------------------------
# [GPU Accelerated Degradation Utils]
# -----------------------------------------------------------------------------
_JPEG_CACHE = {}


def _get_dct_matrix(N=8):
    n = torch.arange(N).float()
    k = torch.arange(N).float()
    dct = torch.cos((math.pi / N) * (n + 0.5) * k.unsqueeze(1))
    dct[0] *= 1.0 / math.sqrt(2.0)
    dct *= math.sqrt(2.0 / N)
    return dct


def _rgb_to_ycbcr(x):
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5
    return torch.stack([y, cb, cr], dim=1)


def _ycbcr_to_rgb(x):
    y, cb, cr = x[:, 0], x[:, 1], x[:, 2]
    cb, cr = cb - 0.5, cr - 0.5
    r = y + 1.402 * cr
    g = y - 0.34414 * cb - 0.71414 * cr
    b = y + 1.772 * cb
    return torch.stack([r, g, b], dim=1)


def quantize_validation(
    tensor: torch.Tensor, bit_depth: int = 4, jpeg_quality: int = 40
) -> torch.Tensor:
    """
    [Validation Degradation]
    Input: (B, 3, H, W) Tensor (0~1)

    1. Apply Bit-depth Reduction (Quantization)
    2. Apply JPEG Compression (Optional, if jpeg_quality is set)
       - Includes 4:2:0 Chroma Subsampling
       - Includes 8x8 Block DCT & Quantization
    """
    device = tensor.device

    # 1. Bit-depth Quantization
    if bit_depth is not None:
        steps = (2**bit_depth) - 1
        tensor = torch.round(tensor * steps) / steps

    # 2. JPEG Compression (GPU Accelerated)
    if jpeg_quality is not None:
        B, C, H, W = tensor.shape

        # --- Caching Mechanism (Generate matrices only once per device) ---
        if device not in _JPEG_CACHE:
            dct_m = _get_dct_matrix(8).to(device)
            idct_m = dct_m.t()
            y_table = torch.tensor(
                [
                    [16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99],
                ],
                dtype=torch.float32,
                device=device,
            )
            _JPEG_CACHE[device] = (dct_m, idct_m, y_table)

        dct_matrix, idct_matrix, y_table = _JPEG_CACHE[device]

        # --- Pre-processing ---
        # Pad to multiple of 8
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        x = (
            F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
            if (pad_h or pad_w)
            else tensor
        )

        # RGB -> YCbCr
        x_yuv = _rgb_to_ycbcr(x)

        # --- 4:2:0 Chroma Subsampling ---
        y = x_yuv[:, 0:1, :, :]
        cbcr = x_yuv[:, 1:3, :, :]
        # Downsample & Upsample (Simulate information loss)
        cbcr = F.avg_pool2d(cbcr, 2, 2)
        cbcr = F.interpolate(
            cbcr, size=(x_yuv.shape[2], x_yuv.shape[3]), mode="nearest"
        )
        x_yuv = torch.cat([y, cbcr], dim=1)

        # --- DCT & Quantization ---
        # Unfold to 8x8 blocks
        patches = x_yuv.unfold(2, 8, 8).unfold(3, 8, 8)

        # DCT
        dct_patches = torch.einsum(
            "ij,bcxyjk,kl->bcxyil", dct_matrix, patches, idct_matrix
        )

        # Quantize Coefficients
        scale = 50.0 / jpeg_quality if jpeg_quality < 50 else 2.0 - jpeg_quality * 0.02
        q_table = y_table.view(1, 1, 1, 1, 8, 8) * scale
        dct_quant = torch.round(dct_patches / (q_table + 1e-5)) * (q_table + 1e-5)

        # IDCT
        rec_patches = torch.einsum(
            "ij,bcxyjk,kl->bcxyil", idct_matrix, dct_quant, dct_matrix
        )

        # --- Post-processing ---
        # Fold back (Reshape)
        rec_yuv = rec_patches.permute(0, 1, 2, 4, 3, 5).reshape(
            B, 3, x.shape[2], x.shape[3]
        )

        # YCbCr -> RGB
        out = _ycbcr_to_rgb(rec_yuv)

        # Unpad
        tensor = out[:, :, :H, :W]

    return tensor.clamp(0.0, 1.0)


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


# -----------------------------------------------------------------------------
# [Standard Utils]
# -----------------------------------------------------------------------------
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
    """
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
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(file_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


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
