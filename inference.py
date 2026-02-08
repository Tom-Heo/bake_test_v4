import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import imageio.v3 as iio

# Local Modules
from config import Config
from core.net import BakeNet
from core.palette import Palette
from utils import get_logger  # 로거 임포트


def auto_detect_bit_depth(img_np):
    max_val = img_np.max()
    if max_val <= 1023:
        return 10
    elif max_val <= 4095:
        return 12
    elif max_val <= 16383:
        return 14
    else:
        return 16


def load_image_to_tensor(path, device, logger, input_bit_depth=None):
    try:
        img_np = iio.imread(path)
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return None, None

    if img_np.ndim == 3 and img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    img_float = img_np.astype(np.float32)
    detected_depth = 8

    if img_np.dtype == np.uint16:
        if input_bit_depth is not None:
            depth = input_bit_depth
            normalization_scale = (2**depth) - 1
            detected_depth = depth
            # logger.info는 루프에서 너무 많이 찍힐 수 있으므로 상위에서 처리
        else:
            depth = auto_detect_bit_depth(img_np)
            normalization_scale = (2**depth) - 1
            detected_depth = depth

        img_float = img_float / normalization_scale

    elif img_np.dtype == np.uint8:
        img_float = img_float / 255.0
        detected_depth = 8

    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device).clamp(0.0, 1.0), detected_depth


def save_tensor_to_16bit_png(tensor, path):
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    img_np = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img_uint16 = (img_np * 65535.0).astype(np.uint16)
    iio.imwrite(path, img_uint16)


def pad_image(tensor):
    _, _, h, w = tensor.shape
    pad_h = 1 if (h % 2 != 0) else 0
    pad_w = 1 if (w % 2 != 0) else 0
    if pad_h + pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, (h, w)


def unpad_image(tensor, original_size):
    h, w = original_size
    return tensor[:, :, :h, :w]


def inference(args):
    # 1. Setup Logger
    Config.create_directories()  # logs 폴더 생성 보장
    logger = get_logger(Config.LOG_DIR, "inference.log")  # inference.log 사용

    device = (
        torch.device(Config.DEVICE)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f"Start Inference on {device}")

    # 2. Model
    logger.info("Initializing BakeNet...")
    model = BakeNet(dim=Config.INTERNAL_DIM).to(device)

    to_oklabp = Palette.sRGBtoOklabP().to(device)
    to_rgb = Palette.OklabPtosRGB().to(device)

    # 3. Checkpoint
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found at {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "ema_shadow" in checkpoint:
        logger.info("Loading EMA weights (Preferred)...")
        model.load_state_dict(checkpoint["ema_shadow"])
    else:
        logger.info("Loading standard weights...")
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4. Process Images
    if os.path.isdir(args.input):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dpx")
        image_paths = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith(exts)
        ]
        image_paths.sort()
        save_dir = os.path.join(Config.RESULT_DIR, "inference_batch")
    else:
        image_paths = [args.input]
        save_dir = os.path.join(Config.RESULT_DIR, "inference_single")

    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Found {len(image_paths)} images. Saving to {save_dir}")

    # 시퀀스 경고
    if len(image_paths) > 1 and args.bit_depth is None:
        logger.warning(
            "Running in AUTO-DETECT mode for a sequence. FLICKERING may occur if dark frames are misdetected. Consider using --bit_depth."
        )

    # 5. Inference Loop
    for i, img_path in enumerate(image_paths):
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        try:
            # A. Load
            input_tensor, detected_depth = load_image_to_tensor(
                img_path, device, logger, args.bit_depth
            )
            if input_tensor is None:
                continue

            # B. Pad
            input_padded, org_size = pad_image(input_tensor)

            # C. Inference
            with torch.no_grad():
                input_oklabp = to_oklabp(input_padded)
                output_oklabp = model(input_oklabp)
                output_rgb = to_rgb(output_oklabp)

            # D. Unpad
            output_rgb = unpad_image(output_rgb, org_size)

            # E. Save Result
            save_path = os.path.join(save_dir, f"{img_name}_bake.png")
            save_tensor_to_16bit_png(output_rgb, save_path)

            # F. Save Comparison
            input_unpadded = unpad_image(input_tensor, org_size)
            combined = torch.cat([input_unpadded, output_rgb], dim=3)
            save_tensor_to_16bit_png(
                combined, os.path.join(save_dir, f"{img_name}_comp.png")
            )

            logger.info(
                f"[{i+1}/{len(image_paths)}] Processed {img_name} | {detected_depth}-bit Mode"
            )

        except Exception as e:
            logger.error(f"Failed to process {img_name}: {e}")
            torch.cuda.empty_cache()

    logger.info("Inference Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bake v4 Inference")
    parser.add_argument(
        "--input", type=str, required=True, help="Input file or directory"
    )
    parser.add_argument("--checkpoint", type=str, default=Config.LAST_CKPT_PATH)
    parser.add_argument(
        "--bit_depth",
        type=int,
        default=None,
        choices=[8, 10, 12, 14, 16],
        help="Force specific bit depth.",
    )
    args = parser.parse_args()
    inference(args)
