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


def auto_detect_bit_depth(img_np):
    """
    [Smart Bit-Depth Detection]
    픽셀 최댓값을 분석하여 비트 심도를 추정합니다.
    주의: 아주 어두운 12-bit 프레임은 10-bit로 오인될 수 있습니다.
    """
    max_val = img_np.max()

    if max_val <= 1023:
        return 10
    elif max_val <= 4095:
        return 12
    elif max_val <= 16383:
        return 14
    else:
        return 16


def load_image_to_tensor(path, device, input_bit_depth=None):
    """
    이미지 로드 및 정규화
    """
    try:
        # imageio는 다양한 포맷(TIFF, PNG, DPX 등)을 지원함
        img_np = iio.imread(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None, None

    # Alpha Channel 제거
    if img_np.ndim == 3 and img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    img_float = img_np.astype(np.float32)
    detected_depth = 8  # 기본값

    # 정규화 로직
    if img_np.dtype == np.uint16:
        # 16-bit Container
        if input_bit_depth is not None:
            # [Manual] 사용자가 지정 (가장 안전함)
            depth = input_bit_depth
            normalization_scale = (2**depth) - 1
            detected_depth = depth
        else:
            # [Auto] 픽셀 분포로 추정 (편리하지만 암부 플리커링 위험 있음)
            depth = auto_detect_bit_depth(img_np)
            normalization_scale = (2**depth) - 1
            detected_depth = depth

        img_float = img_float / normalization_scale

    elif img_np.dtype == np.uint8:
        # 8-bit
        img_float = img_float / 255.0
        detected_depth = 8

    # Float(EXR) 등은 그대로 둠

    # Tensor 변환 (HWC -> CHW)
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)

    # 1.0을 넘지 않도록 Clamp (Auto 감지 오차 방지)
    return tensor.to(device).clamp(0.0, 1.0), detected_depth


def save_tensor_to_16bit_png(tensor, path):
    """Bake Output: Always 16-bit PNG"""
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    img_np = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img_uint16 = (img_np * 65535.0).astype(np.uint16)
    iio.imwrite(path, img_uint16)


def pad_image(tensor):
    """홀수 해상도 -> 짝수 패딩 (Reflect)"""
    _, _, h, w = tensor.shape
    pad_h = 1 if (h % 2 != 0) else 0
    pad_w = 1 if (w % 2 != 0) else 0
    if pad_h + pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, (h, w)


def unpad_image(tensor, original_size):
    """패딩 제거"""
    h, w = original_size
    return tensor[:, :, :h, :w]


def inference(args):
    # 1. Setup
    device = (
        torch.device(Config.DEVICE)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # 2. Model
    print("Initializing BakeNet...")
    model = BakeNet(dim=Config.INTERNAL_DIM).to(device)

    # Converters
    to_oklabp = Palette.sRGBtoOklabP().to(device)
    to_rgb = Palette.OklabPtosRGB().to(device)

    # 3. Checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "ema_shadow" in checkpoint:
        print("Loading EMA weights (Preferred)...")
        model.load_state_dict(checkpoint["ema_shadow"])
    else:
        print("Loading standard weights...")
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4. Prepare Files
    if os.path.isdir(args.input):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dpx")
        image_paths = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith(exts)
        ]
        image_paths.sort()  # 시퀀스 순서 보장
        save_dir = os.path.join(Config.RESULT_DIR, "inference_batch")
    else:
        image_paths = [args.input]
        save_dir = os.path.join(Config.RESULT_DIR, "inference_single")

    os.makedirs(save_dir, exist_ok=True)
    print(f"Found {len(image_paths)} images.")

    # 시퀀스 경고 메시지
    if len(image_paths) > 1 and args.bit_depth is None:
        print("\n" + "=" * 60)
        print(" [WARNING] Running in AUTO-DETECT mode for a sequence.")
        print(" If dark frames are detected as 10-bit but represent 12-bit,")
        print(" FLICKERING may occur. Consider using --bit_depth 12.")
        print("=" * 60 + "\n")

    # 5. Inference Loop
    for i, img_path in enumerate(image_paths):
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # A. Load
        input_tensor, detected_depth = load_image_to_tensor(
            img_path, device, args.bit_depth
        )
        if input_tensor is None:
            continue

        print(
            f"[{i+1}/{len(image_paths)}] {img_name} | {detected_depth}-bit Mode",
            end=" ... ",
        )

        try:
            # B. Pad
            input_padded, org_size = pad_image(input_tensor)

            # C. Inference
            with torch.no_grad():
                input_oklabp = to_oklabp(input_padded)
                output_oklabp = model(input_oklabp)
                output_rgb = to_rgb(output_oklabp)

            # D. Unpad
            output_rgb = unpad_image(output_rgb, org_size)

            # E. Save Result (16-bit)
            save_path = os.path.join(save_dir, f"{img_name}_bake.png")
            save_tensor_to_16bit_png(output_rgb, save_path)

            # F. Save Comparison (16-bit Side-by-Side)
            input_unpadded = unpad_image(input_tensor, org_size)
            combined = torch.cat([input_unpadded, output_rgb], dim=3)
            save_tensor_to_16bit_png(
                combined, os.path.join(save_dir, f"{img_name}_comp.png")
            )

            print("Done.")

        except Exception as e:
            print(f"ERROR: {e}")
            torch.cuda.empty_cache()

    print("\nInference Finished.")


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
        help="Force specific bit depth. Recommended for video sequences.",
    )
    args = parser.parse_args()
    inference(args)
