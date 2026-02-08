from __future__ import annotations

import torch
import torch.nn as nn
from .heo import Heo


class OklabPtoBakedBaseColor(nn.Module):
    """
    [Model Stem]
    3-channel OklabP -> 30-channel BakedBase
    """

    def __init__(self):
        super().__init__()

        # Macbeth ColorChecker 24 Reference Points (Normalized OklabP Scale)
        MACBETH_REFS = [
            [-0.0597942, 0.0715682, 0.0689456],
            [0.4183730, 0.0844511, 0.0862138],
            [0.1485260, -0.0260145, -0.1197130],
            [0.0060500, -0.0867117, 0.1012940],
            [0.2456150, 0.0471529, -0.1390040],
            [0.4774960, -0.1772700, 0.0070673],
            [0.3507240, 0.1477280, 0.2429970],
            [-0.0020556, 0.0205867, -0.2380310],
            [0.1933670, 0.2535120, 0.0753901],
            [-0.1681210, 0.1244790, -0.1221760],
            [0.4962120, -0.1630230, 0.2584340],
            [0.5099230, 0.0561745, 0.2807750],
            [-0.1797940, 0.0282334, -0.2856320],
            [0.1974370, -0.2180970, 0.1560710],
            [0.0268793, 0.2907220, 0.1168620],
            [0.6649640, -0.0396422, 0.3292740],
            [0.1965930, 0.2874640, -0.0811799],
            [0.1415920, -0.1571350, -0.1312720],
            [0.9277470, -0.0007701, 0.0023781],
            [0.6655870, -0.0000168, -0.0001434],
            [0.4115140, -0.0000142, -0.0001216],
            [0.1583850, -0.0008645, 0.0027884],
            [-0.1009340, -0.0000091, -0.0000774],
            [-0.3499260, -0.0000066, -0.0000560],
        ]

        self.register_buffer(
            "macbeth_refs",
            torch.tensor(MACBETH_REFS, dtype=torch.float32).view(1, 24, 3, 1, 1),
        )

        self.coeffs = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Part 1: Signal Replication (3 * 2 = 6 Channels)
        part1 = x.repeat(1, 2, 1, 1)

        # Part 2: RBF Global Context (24 Channels)
        diff = x.unsqueeze(1) - self.macbeth_refs
        dist_sq = (diff**2).sum(dim=2)
        gamma = 1.0 / (2.0 * (self.coeffs**2))
        part2 = torch.exp(-dist_sq * gamma)

        return torch.cat([part1, part2], dim=1)


class BakedBaseColortoColorEmbedding(nn.Module):
    """
    [Context Module]
    30ch BakedBase -> 30ch Context Embedding
    Dense Gating 구조로 업그레이드 (성능 강화)
    """

    def __init__(self, dim=30):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1, 1)
        self.conv3 = nn.Conv2d(dim, dim, 1, 1)

        self.act1 = Heo.HeLU2d(dim)
        self.act2 = Heo.HeLU2d(dim)
        self.act3 = Heo.HeLU2d(dim)

        self.gate1 = Heo.HeoGate2d(dim)
        self.gate2 = Heo.HeoGate2d(dim)
        self.gate3 = Heo.HeoGate2d(dim)

    def forward(self, baked_base: torch.Tensor) -> torch.Tensor:
        """
        Input: baked_base (B, 30, H, W)
        """
        x1 = self.conv1(baked_base)
        x1 = self.act1(x1)
        x1 = self.gate1(x1, baked_base)

        x2 = self.conv2(x1)
        x2 = self.act2(x2)
        x2 = self.gate2(x2, x1)

        x3 = self.conv3(x2)
        x3 = self.act3(x3)
        return self.gate3(x3, x2)


class BakedBaseColortoPerceptualOklabP(nn.Module):
    """
    [Model Head]
    30ch BakedBase Feature -> 3ch OklabP Output
    Embedding과 동일한 3-Stage Dense Gating 구조 적용 후 압축 (성능 강화)
    """

    def __init__(self, in_dim=30, out_dim=3):
        super().__init__()

        # Stage 1: Refinement (30->30)
        self.conv1 = nn.Conv2d(in_dim, in_dim, 1, 1)
        self.act1 = Heo.HeLU2d(in_dim)
        self.gate1 = Heo.HeoGate2d(in_dim)

        # Stage 2: Deep Mixing (30->30)
        self.conv2 = nn.Conv2d(in_dim, in_dim, 1, 1)
        self.act2 = Heo.HeLU2d(in_dim)
        self.gate2 = Heo.HeoGate2d(in_dim)

        # Stage 3: Final Projection (30->3)
        # 마지막은 Gate나 Act 없이 순수 선형 변환으로 값을 매핑
        self.head = nn.Conv2d(in_dim, out_dim, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = x  # Residual Source

        # Stage 1
        x1 = self.conv1(x)
        x1 = self.act1(x1)
        x1 = self.gate1(x1, raw)

        # Stage 2
        x2 = self.conv2(x1)
        x2 = self.act2(x2)
        x2 = self.gate2(x2, x1)

        # Stage 3 (Projection)
        out = self.head(x2)

        return out


class OklabPtoBakedLossColor(nn.Module):
    """
    [Loss Projector]
    3ch OklabP -> 96ch Hyper-Baked (Evaluation Only)
    """

    def __init__(self):
        super().__init__()

        MACBETH_REFS = [
            [-0.0597942, 0.0715682, 0.0689456],
            [0.4183730, 0.0844511, 0.0862138],
            [0.1485260, -0.0260145, -0.1197130],
            [0.0060500, -0.0867117, 0.1012940],
            [0.2456150, 0.0471529, -0.1390040],
            [0.4774960, -0.1772700, 0.0070673],
            [0.3507240, 0.1477280, 0.2429970],
            [-0.0020556, 0.0205867, -0.2380310],
            [0.1933670, 0.2535120, 0.0753901],
            [-0.1681210, 0.1244790, -0.1221760],
            [0.4962120, -0.1630230, 0.2584340],
            [0.5099230, 0.0561745, 0.2807750],
            [-0.1797940, 0.0282334, -0.2856320],
            [0.1974370, -0.2180970, 0.1560710],
            [0.0268793, 0.2907220, 0.1168620],
            [0.6649640, -0.0396422, 0.3292740],
            [0.1965930, 0.2874640, -0.0811799],
            [0.1415920, -0.1571350, -0.1312720],
            [0.9277470, -0.0007701, 0.0023781],
            [0.6655870, -0.0000168, -0.0001434],
            [0.4115140, -0.0000142, -0.0001216],
            [0.1583850, -0.0008645, 0.0027884],
            [-0.1009340, -0.0000091, -0.0000774],
            [-0.3499260, -0.0000066, -0.0000560],
        ]

        self.register_buffer(
            "macbeth_refs",
            torch.tensor(MACBETH_REFS, dtype=torch.float32).view(1, 24, 3, 1, 1),
        )

        self.coeffs = [0.1, 0.4, 1.6]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        part1 = x.repeat(1, 8, 1, 1)

        diff = x.unsqueeze(1) - self.macbeth_refs
        dist_sq = (diff**2).sum(dim=2)

        rbf_features = []
        for sigma in self.coeffs:
            gamma = 1.0 / (2.0 * (sigma**2))
            rbf_features.append(torch.exp(-dist_sq * gamma))

        part2 = torch.cat(rbf_features, dim=1)
        return torch.cat([part1, part2], dim=1)
