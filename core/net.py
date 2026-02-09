from __future__ import annotations

import torch
import torch.nn as nn
from .heo import Heo
from .bake import (
    OklabPtoBakedBaseColor,
    BakedBaseColortoColorEmbedding,
    BakedBaseColortoPerceptualOklabP,
)


class BakeNet(nn.Module):
    def __init__(self, depth=50, dim=30):
        super().__init__()
        self.depth = depth
        self.dim = dim

        # 1. Stem (The Anchor)
        # 3ch -> 30ch Internal Base
        self.stem = OklabPtoBakedBaseColor()

        # 2. Components for the Body (The Loop)

        # A. Embedding Modules (50개)
        # 매 층마다 원본(Base)에서 문맥을 추출
        self.embedding_modules = nn.ModuleList(
            [BakedBaseColortoColorEmbedding(dim) for _ in range(depth)]
        )

        # B. Context Gates (50개) - [수정됨] 0번 블록 포함
        # 추출된 문맥(Embedding)과 주 흐름(Stream)을 혼합
        self.context_gates = nn.ModuleList([Heo.HeoGate2d(dim) for _ in range(depth)])

        # C. NeMO Blocks (50개)
        # 국소적 디테일 복원
        self.nemo_modules = nn.ModuleList([Heo.NeMO33(dim) for _ in range(depth)])

        # D. Residual Gates (50개)
        # 연산 전후의 차이를 학습 (Gradient Flow 보장)
        self.residual_gates = nn.ModuleList([Heo.HeoGate2d(dim) for _ in range(depth)])

        # 3. Head (The Finish)
        # 30ch -> 3ch Output
        self.head = BakedBaseColortoPerceptualOklabP(in_dim=dim, out_dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, 3, H, W) - OklabP
        Output: (B, 3, H, W) - Refined OklabP
        """
        # A. Create the Base (Immutable Reference)
        base = self.stem(x)

        # B. Initialization (0-th Block)
        # 1. 첫 번째 문맥 추출
        emb0 = self.embedding_modules[0](base)

        # 2. 첫 번째 Gating: [Embedding vs Base]
        # 지시사항: "context_gates[0](emb0, base)"
        # 임베딩(emb0)을 주체로 하되, 원본(base)을 참조하여 초기 신호를 형성
        feat = self.context_gates[0](emb0, base)

        # 3. NeMO Processing
        out0 = self.nemo_modules[0](feat)

        # 4. Residual Connection
        # NeMO 출력과 입력(feat)을 결합하여 흐름을 시작
        feat = self.residual_gates[0](out0, feat)

        # C. The Loop (1 ~ 49 Block)
        for i in range(1, self.depth):
            # 1. Context Extraction
            emb = self.embedding_modules[i](base)

            # 2. Context Injection: [Previous Feature vs New Embedding]
            # 이전 흐름(feat)에 새로운 문맥(emb)을 주입
            gated_input = self.context_gates[i](feat, emb)

            # 3. NeMO Processing
            nemo_out = self.nemo_modules[i](gated_input)

            # 4. Residual Connection
            feat = self.residual_gates[i](nemo_out, gated_input)

        # D. Head & Global Residual
        return x + 0.2 * self.head(feat)
