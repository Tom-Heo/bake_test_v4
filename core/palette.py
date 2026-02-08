import torch
import torch.nn as nn


class Palette:
    class sRGBtoOklabP(nn.Module):
        """
        [Color Space Converter]
        Standard sRGB (0~1) -> OklabP (Lp, ap, bp)

        * OklabP (Processing Scale): [-1, 1] Normalized
        """

        def __init__(self):
            super().__init__()
            self.epsilon = 1e-8  # 수치 안정성을 위한 작은 값

        @staticmethod
        def srgb_to_lsrgb(srgb: torch.Tensor) -> torch.Tensor:
            """Linearize sRGB (Gamma Decoding)"""
            return torch.where(
                srgb <= 0.04045,
                srgb / 12.92,
                ((srgb + 0.055) / 1.055) ** 2.4,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Input: (B, 3, H, W) in [0, 1]
            if x.dim() != 4 or x.size(1) != 3:
                raise ValueError(f"Expected (B,3,H,W), got {tuple(x.shape)}")

            # 1. sRGB -> Linear sRGB
            x_perm = x.permute(0, 2, 3, 1)  # NHWC
            srgb = x_perm.clamp(0.0, 1.0)
            lsrgb = self.srgb_to_lsrgb(srgb)

            r, g, b = lsrgb[..., 0], lsrgb[..., 1], lsrgb[..., 2]

            # 2. Linear sRGB -> LMS
            l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
            m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
            s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

            # 3. Non-linearity (Cube Root) with Epsilon for Stability
            # 0 근처에서 기울기가 무한대로 발산하는 것을 방지 (NaN 방지)
            l_ = torch.sign(l) * (torch.abs(l) + self.epsilon).pow(1.0 / 3.0)
            m_ = torch.sign(m) * (torch.abs(m) + self.epsilon).pow(1.0 / 3.0)
            s_ = torch.sign(s) * (torch.abs(s) + self.epsilon).pow(1.0 / 3.0)

            # 4. LMS -> Standard Oklab
            L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
            a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
            b_ = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

            # 5. Scale to OklabP
            Lp = (2.0 * L) - 1.0
            ap = 2.0 * a
            bp = 2.0 * b_

            out = torch.stack([Lp, ap, bp], dim=1)  # (B, 3, H, W)
            return out

    class OklabPtosRGB(nn.Module):
        """
        [Color Space Converter]
        OklabP (Lp, ap, bp) -> Standard sRGB (0~1)
        """

        def __init__(self):
            super().__init__()

        @staticmethod
        def lsrgb_to_srgb(lsrgb: torch.Tensor) -> torch.Tensor:
            """Gamma Encoding (Linear -> sRGB)"""
            threshold = 0.0031308
            return torch.where(
                lsrgb <= threshold,
                12.92 * lsrgb,
                1.055 * torch.clamp(lsrgb, min=0.0) ** (1.0 / 2.4) - 0.055,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 4 or x.size(1) != 3:
                raise ValueError(f"Expected (B,3,H,W), got {tuple(x.shape)}")

            # 1. OklabP -> Standard Oklab
            Lp, ap, bp = x[:, 0], x[:, 1], x[:, 2]
            L = (Lp + 1.0) * 0.5
            a = ap * 0.5
            b = bp * 0.5

            # 2. Oklab -> LMS
            l_ = L + 0.3963377774 * a + 0.2158037573 * b
            m_ = L - 0.1055613458 * a - 0.0638541728 * b
            s_ = L - 0.0894841775 * a - 1.2914855480 * b

            # 3. LMS -> Linear LMS (Cube)
            # 여기서는 Gradient 폭발 위험이 없으므로 epsilon 불필요
            l = l_**3
            m = m_**3
            s = s_**3

            # 4. Linear LMS -> Linear sRGB
            r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
            g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
            b_rgb = 0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

            rgb_lin = torch.stack([r, g, b_rgb], dim=1)

            # 5. Linear sRGB -> sRGB
            srgb = self.lsrgb_to_srgb(rgb_lin)

            return srgb.clamp(0.0, 1.0)
