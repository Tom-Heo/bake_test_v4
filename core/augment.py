import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math


class BakeAugment(nn.Module):
    """
    [Bake GPU Darkroom]
    GPU-Accelerated Degradation Pipeline
    """

    def __init__(self):
        super().__init__()

        # JPEG Simulation: DCT Matrix & Quantization Table
        self.register_buffer("dct_matrix", self._get_dct_matrix(8))
        self.register_buffer("idct_matrix", self._get_dct_matrix(8).t())

        # Standard JPEG Luminance Quantization Table
        self.register_buffer(
            "y_table",
            torch.tensor(
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
            ),
        )

    def _get_dct_matrix(self, N=8):
        n = torch.arange(N).float()
        k = torch.arange(N).float()
        dct = torch.cos((math.pi / N) * (n + 0.5) * k.unsqueeze(1))
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / N)
        return dct

    def _rgb_to_ycbcr(self, x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5
        return torch.stack([y, cb, cr], dim=1)

    def _ycbcr_to_rgb(self, x):
        y, cb, cr = x[:, 0], x[:, 1], x[:, 2]
        cb, cr = cb - 0.5, cr - 0.5
        r = y + 1.402 * cr
        g = y - 0.34414 * cb - 0.71414 * cr
        b = y + 1.772 * cb
        return torch.stack([r, g, b], dim=1)

    def apply_jpeg(self, x, quality_range=(10, 20)):
        """Differentiable JPEG Artifact Simulator"""
        B, C, H, W = x.shape

        # 1. Pad to multiple of 8
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # 2. RGB -> YCbCr
        x_yuv = self._rgb_to_ycbcr(x)

        # 3. 4:2:0 Chroma Subsampling Simulation
        # ==========================================
        # Y(0번 채널)는 건드리지 않고, Cb/Cr(1,2번 채널)만 해상도를 반으로 줄였다가 억지로 늘립니다.

        y = x_yuv[:, 0:1, :, :]
        cbcr = x_yuv[:, 1:3, :, :]

        # Downsample (Average Pooling으로 정보 손실 유도)
        cbcr_down = F.avg_pool2d(cbcr, kernel_size=2, stride=2)

        # Upsample (Nearest or Bilinear로 깍두기 혹은 번짐 현상 유도)
        # JPEG의 블록 현상을 강조하려면 'nearest', 부드러운 번짐을 원하면 'bilinear'
        cbcr_up = F.interpolate(
            cbcr_down, size=(x_yuv.shape[2], x_yuv.shape[3]), mode="nearest"
        )

        # 다시 합치기 (이제 색상 정보는 1/4만 남았습니다)
        x_yuv = torch.cat([y, cbcr_up], dim=1)
        # ==========================================

        # 4. Block Process (DCT)
        # Unfold: (B, 3, H, W) -> (B, 3, H/8, W/8, 8, 8)
        patches = x_yuv.unfold(2, 8, 8).unfold(3, 8, 8)

        # DCT Transform: D * P * D^T
        dct_patches = torch.einsum(
            "ij,bcxyjk,kl->bcxyil", self.dct_matrix, patches, self.idct_matrix
        )

        # 5. Quantization
        quality = random.uniform(*quality_range)
        scale = 50.0 / quality if quality < 50 else 2.0 - quality * 0.02

        # Apply Y-Table to all channels (Simplified for speed) or just Y
        # Here we apply to all for stronger artifact simulation
        q_table = self.y_table.view(1, 1, 1, 1, 8, 8).to(x.device) * scale
        dct_quant = torch.round(dct_patches / (q_table + 1e-5)) * (q_table + 1e-5)

        # 6. IDCT Transform
        rec_patches = torch.einsum(
            "ij,bcxyjk,kl->bcxyil", self.idct_matrix, dct_quant, self.dct_matrix
        )

        # 7. Reconstruct
        # Permute & Reshape back to image
        rec_yuv = rec_patches.permute(0, 1, 2, 4, 3, 5).reshape(
            B, 3, x_yuv.shape[2] + pad_h, x_yuv.shape[3] + pad_w
        )  # Using padded size logic from shape
        rec_yuv = rec_yuv[
            :, :, : x_yuv.shape[2], : x_yuv.shape[3]
        ]  # Ensure strict shape match before resize if needed, actually simple reshape is fine if dimensions match

        # 8. YCbCr -> RGB
        out = self._ycbcr_to_rgb(rec_yuv)

        return out[:, :, :H, :W].clamp(0, 1)

    def forward(self, x):
        """
        Input: (B, 3, H, W) Clean Tensor
        Returns: (Input_Degraded, Target_Augmented)
        """
        # --- [Geometric Augmentation] ---
        # Flip & Rotate (GPU)
        if random.random() < 0.5:
            x = torch.flip(x, [3])  # H-Flip
        if random.random() < 0.5:
            x = torch.flip(x, [2])  # V-Flip
        if random.random() < 0.5:
            x = torch.rot90(x, 1, [2, 3])

        target = x.clone()
        input_t = x.clone()

        # --- [Degradation Pipeline] ---

        # 1. Quantization (Bit-depth Reduction)
        bit_depth = random.choice([4, 5])
        steps = (2**bit_depth) - 1

        # Dithering (Noise Injection before quant)
        if random.random() < 0.9:
            noise = (torch.rand_like(input_t) - 0.5) / steps
            input_t = input_t + noise

        input_t = torch.round(input_t * steps) / steps

        # 2. JPEG Compression (90% Chance)
        if random.random() < 0.9:
            input_t = self.apply_jpeg(input_t)

        # 3. Gaussian Noise (Texture)
        if random.random() < 0.9:
            sigma = random.uniform(0.02, 0.04)
            input_t = input_t + torch.randn_like(input_t) * sigma

        input_t = input_t.clamp(0, 1)

        # --- [Color Augmentation] ---
        # 1. RGB Shift (White Balance)
        if random.random() < 0.9:
            shift = torch.randn(1, 3, 1, 1, device=x.device) * 0.05
            input_t = input_t + shift
            target = target + shift

        # 2. Gamma (Curve)
        if random.random() < 0.9:
            gamma = random.uniform(0.7, 1.3)
            input_t = input_t.clamp(1e-8, 1.0) ** gamma
            target = target.clamp(1e-8, 1.0) ** gamma

        return input_t.clamp(0, 1), target.clamp(0, 1)
