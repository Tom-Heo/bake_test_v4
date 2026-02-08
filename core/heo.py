from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt


class Heo:
    class HeLU(nn.Module):
        """
        원본 HeLU: last-dim 기반 (..., dim) 입력용
        """

        def __init__(self, dim: int):
            super().__init__()

            self.alpha = nn.Parameter(torch.full((dim,), 0.9))
            self.beta = nn.Parameter(torch.full((dim,), -0.9))
            self.redweight = nn.Parameter(torch.empty(dim).normal_(mean=0.0, std=0.45))
            self.blueweight = nn.Parameter(torch.empty(dim).normal_(mean=0.0, std=0.45))
            self.redgelu = nn.GELU()
            self.bluegelu = nn.GELU()

        def forward(self, x: torch.Tensor):
            raw = x

            rgx = self.redgelu(x)
            bgx = -1.0 * self.bluegelu(-x)

            red = torch.tanh(sqrt(3.0) * self.redweight) + 1.0
            blue = torch.tanh(sqrt(3.0) * self.blueweight) + 1.0
            redx = rgx * red
            bluex = bgx * blue
            x = redx + bluex
            alpha = torch.tanh(sqrt(3.0) * self.alpha) + 1.0
            beta = torch.tanh(sqrt(3.0) * self.beta) + 1.0
            y = (alpha * x + beta * raw) / 2
            return y

    class HeLU2d(nn.Module):
        """
        입력: (N,C,H,W)
        """

        def __init__(self, channels: int):
            super().__init__()
            c = int(channels)
            self.channels = c

            # 원본 HeLU와 같은 파라미터 의미(채널별)
            self.alpha = nn.Parameter(torch.full((c,), 0.9))
            self.beta = nn.Parameter(torch.full((c,), -0.9))
            self.redweight = nn.Parameter(torch.empty(c).normal_(mean=0.0, std=0.45))
            self.blueweight = nn.Parameter(torch.empty(c).normal_(mean=0.0, std=0.45))

            self.redgelu = nn.GELU()
            self.bluegelu = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 4:
                raise ValueError(
                    f"HeLU2d expects NCHW 4D tensor, got shape={tuple(x.shape)}"
                )
            if x.size(1) != self.channels:
                raise ValueError(
                    f"HeLU2d channels mismatch: expected C={self.channels}, got C={x.size(1)}"
                )

            raw = x

            rgx = self.redgelu(x)
            bgx = -1.0 * self.bluegelu(-x)

            # (C,) -> (1,C,1,1) broadcasting
            red = (torch.tanh(sqrt(3.0) * self.redweight) + 1.0).view(1, -1, 1, 1)
            blue = (torch.tanh(sqrt(3.0) * self.blueweight) + 1.0).view(1, -1, 1, 1)
            x = rgx * red + bgx * blue

            alpha = (torch.tanh(sqrt(3.0) * self.alpha) + 1.0).view(1, -1, 1, 1)
            beta = (torch.tanh(sqrt(3.0) * self.beta) + 1.0).view(1, -1, 1, 1)
            y = (alpha * x + beta * raw) / 2
            return y

    class HeoGate(nn.Module):
        def __init__(self, dim: int):
            super().__init__()

            self.alpha = nn.Parameter(torch.full((dim,), 0.9))
            self.beta = nn.Parameter(torch.full((dim,), -0.9))

        def forward(self, x: torch.Tensor, raw: torch.Tensor):
            alpha = torch.tanh(sqrt(3.0) * self.alpha) + 1.0
            beta = torch.tanh(sqrt(3.0) * self.beta) + 1.0
            return (alpha * x + beta * raw) / 2

    class HeoGate2d(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            c = int(channels)
            self.channels = c

            self.alpha = nn.Parameter(torch.full((c,), 0.9))
            self.beta = nn.Parameter(torch.full((c,), -0.9))

        def forward(self, x: torch.Tensor, raw: torch.Tensor):
            if x.dim() != 4 or x.size(1) != self.channels:
                raise ValueError(
                    f"HeoGate2d channels mismatch: expected C={self.channels}, got C={x.size(1)}"
                )

            alpha = (torch.tanh(sqrt(3.0) * self.alpha) + 1.0).view(1, -1, 1, 1)
            beta = (torch.tanh(sqrt(3.0) * self.beta) + 1.0).view(1, -1, 1, 1)
            return (alpha * x + beta * raw) / 2

    class NeMO33(nn.Module):
        def __init__(self, dim: int):
            super().__init__()

            self.conv0 = nn.Conv2d(dim, dim, 3, 1, 1)
            self.conv1 = nn.Conv2d(dim, dim, 1, 1)
            self.conv2 = nn.Conv2d(dim, dim, 1, 1)

            self.act0 = Heo.HeLU2d(dim)
            self.act1 = Heo.HeLU2d(dim)
            self.act2 = Heo.HeLU2d(dim)

            self.gate0 = Heo.HeoGate2d(dim)
            self.gate1 = Heo.HeoGate2d(dim)
            self.gate2 = Heo.HeoGate2d(dim)

        def forward(self, x: torch.Tensor):

            raw = x
            x = self.conv0(x)
            x = self.act0(x)
            x = self.gate0(x, raw)

            x0 = x
            x = self.conv1(x)
            x = self.act1(x)
            x = self.gate1(x, x0)

            x1 = x
            x = self.conv2(x)
            x = self.act2(x)
            x = self.gate2(x, x1)

            return x

    class SharpLoss(nn.Module):
        def __init__(self, epsilon=0.001):
            super().__init__()
            self.epsilon = epsilon
            self.epsilon_char = 1e-8

        def forward(self, pred, target):
            pred = pred.float()  # FP32 강제
            target = target.float()

            diff = pred - target
            charbonnier = torch.sqrt(diff**2 + self.epsilon_char**2)

            loss = torch.log(1 + charbonnier / self.epsilon)

            return loss.mean()
