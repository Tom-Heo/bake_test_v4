import torch
import torch.nn as nn
from core.bake import OklabPtoBakedLossColor
from core.heo import Heo


class BakeLoss(nn.Module):
    """
    [BakeLoss v4]
    Hyper-Baked Robust Loss (Log-Charbonnier)

    Architecture:
    - Projection: core.bake.OklabPtoBakedLossColor (3ch -> 96ch)
    - Metric: core.heo.Heo.SharpLoss (Log-Charbonnier)
    """

    def __init__(self):
        super().__init__()

        # 1. Loss Projector (Frozen)
        # 3ch -> 96ch (Includes 8x original signal + 72x Multi-scale RBF)
        self.projector = OklabPtoBakedLossColor()

        # Projector는 학습되지 않는 고정된 '자(Ruler)'입니다.
        for param in self.projector.parameters():
            param.requires_grad = False

        # 2. Loss Calculator
        # heo.py에 정의된 Robust Loss 활용 (epsilon=1e-3)
        self.criterion = Heo.SharpLoss(epsilon=1e-3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 3, H, W) - Normalized OklabP [-1, 1]
        """

        # A. Projection to Hyper-Baked Space (96ch)
        # Prediction은 Gradient가 흘러야 함
        pred_baked = self.projector(pred)

        # Target은 고정된 정답이므로 Gradient 차단 (VRAM 절약)
        with torch.no_grad():
            target_baked = self.projector(target)

        # B. Calculate SharpLoss
        # 96채널 전체에 대해 Robust Loss 계산
        loss = self.criterion(pred_baked, target_baked)

        return loss
