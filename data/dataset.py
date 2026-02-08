from __future__ import annotations

import os
import random
import io
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image


class DIV2KDataset(Dataset):
    def __init__(self, config, is_train=True):
        super().__init__()
        self.config = config
        self.is_train = is_train

        # 데이터셋 경로 설정
        if is_train:
            self.root_dir = config.DIV2K_TRAIN_ROOT
        else:
            self.root_dir = config.DIV2K_VALID_ROOT

        # 이미지 리스트 로드 및 필터링 (흑백 제거)
        self.image_files = self._scan_and_filter_files()

    def _scan_and_filter_files(self) -> list[str]:
        """
        디렉토리를 스캔하고, 흑백(Grayscale) 이미지는 리스트에서 제외합니다.
        Bake는 '색(Color)'을 복원하는 것이 주 목적이기 때문에,
        색 정보가 없는 이미지는 학습에 노이즈가 될 수 있습니다.
        """
        files = sorted(
            [
                f
                for f in os.listdir(self.root_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        valid_files = []
        # 파일 헤더만 빠르게 읽어서 모드 확인
        for f in files:
            path = os.path.join(self.root_dir, f)
            try:
                with Image.open(path) as img:
                    if img.mode == "RGB":
                        valid_files.append(f)
            except:
                continue

        print(
            f"[{'Train' if self.is_train else 'Valid'}] Loaded {len(valid_files)} RGB images from {self.root_dir}"
        )
        return valid_files

    def __len__(self):
        return len(self.image_files)

    # -------------------------------------------------------------------------
    # [Degradation Methods] - Applied to INPUT Only
    # -------------------------------------------------------------------------
    def _apply_quantization(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        [Hardcore Quantization]
        - Bit Depth: 2, 3, 4, 5 bit 중 랜덤 선택 (Robustness 강화)
        - Dithering: 50% 확률로 노이즈 패턴 주입 (Reality 반영)
        """
        # 1. 비트 심도 랜덤 결정
        bit_depth = random.choice([2, 3, 4, 5])
        steps = (2**bit_depth) - 1

        # 2. 디더링 적용 (50%)
        # 계단 현상을 노이즈로 흩뿌려서, 모델이 '매끈한 계단'과 '거친 패턴'을 모두 학습
        if random.random() < 0.5:
            # -0.5 ~ +0.5 step 크기의 Uniform Noise
            noise = (torch.rand_like(tensor) - 0.5) / steps
            tensor = tensor + noise

        # 3. 양자화 수행
        quantized = torch.round(tensor * steps) / steps

        return torch.clamp(quantized, 0.0, 1.0)

    def _apply_jpeg(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        [JPEG Compression]
        - Quality: 30(Low) ~ 95(High) 랜덤
        - 8x8 블록 아티팩트(깍두기) 생성
        """
        # Tensor -> PIL
        img_pil = TF.to_pil_image(tensor)

        # In-memory Buffer를 이용한 압축/해제 시뮬레이션
        buffer = io.BytesIO()
        quality = random.randint(30, 95)
        img_pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        # 다시 Tensor로 복원
        img_jpeg = Image.open(buffer)
        return TF.to_tensor(img_jpeg)

    def _apply_gaussian_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        [Gaussian Noise]
        - 플라스틱 질감 방지 및 리얼리티 확보
        - 아주 미세한 노이즈를 섞어 디테일의 기준을 세움
        """
        sigma = random.uniform(0.005, 0.02)
        noise = torch.randn_like(tensor) * sigma
        return torch.clamp(tensor + noise, 0.0, 1.0)

    # -------------------------------------------------------------------------
    # [Augmentation Methods] - Applied to INPUT & TARGET Identically
    # -------------------------------------------------------------------------
    def _get_geometric_params(self, size: tuple[int, int]):
        """
        Flip, Rotate, Crop 파라미터를 미리 생성하여
        Input과 Target에 동일하게 적용되도록 함.
        """
        w, h = size
        crop_size = 512

        # 1. Random Crop 좌표 계산
        # 이미지가 512보다 작을 경우 (0,0)에서 시작 (이후 Pad 처리)
        if h <= crop_size or w <= crop_size:
            top, left = 0, 0
        else:
            top = random.randint(0, h - crop_size)
            left = random.randint(0, w - crop_size)

        # 2. Random Flip & Rotate
        h_flip = random.random() < 0.5
        v_flip = random.random() < 0.5
        rotate = random.choice([0, 90, 180, 270])

        return top, left, h_flip, v_flip, rotate

    def _apply_geometric(self, img: Image.Image, params) -> Image.Image:
        top, left, h_flip, v_flip, rotate = params
        crop_size = 512

        # 1. Pad if needed (Reflect)
        w, h = img.size
        if w < crop_size or h < crop_size:
            pad_w = max(crop_size - w, 0)
            pad_h = max(crop_size - h, 0)
            img = TF.pad(img, (0, 0, pad_w, pad_h), padding_mode="reflect")

        # 2. Crop
        img = TF.crop(img, top, left, crop_size, crop_size)

        # 3. Flip & Rotate
        if h_flip:
            img = TF.hflip(img)
        if v_flip:
            img = TF.vflip(img)
        if rotate != 0:
            img = TF.rotate(img, rotate)

        return img

    def _apply_color_aug(
        self, input_t: torch.Tensor, target_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        [Color Augmentation Pipeline]
        Logic: 열화된 아티팩트(계단, 깍두기)를 색공간 상에서 비틀어버림.
        Order: RGB Shift -> Saturation/Hue -> Gamma
        Rationale: 물리적 ISP 파이프라인(Sensor Bias -> Color Grading -> Tone Curve) 모사
        """

        # 1. RGB Shift (White Balance / Sensor Bias)
        # 가장 기초적인 선형 왜곡. 섀도우(Shadow) 영역에도 색을 입힘.
        if random.random() < 0.5:
            r_shift = random.uniform(-0.05, 0.05)
            g_shift = random.uniform(-0.05, 0.05)
            b_shift = random.uniform(-0.05, 0.05)

            shift = torch.tensor([r_shift, g_shift, b_shift]).view(3, 1, 1)
            input_t = input_t + shift
            target_t = target_t + shift

            # 중간 Clamp (다음 단계인 Sat/Hue가 0~1 범위를 기대함)
            input_t = torch.clamp(input_t, 0.0, 1.0)
            target_t = torch.clamp(target_t, 0.0, 1.0)

        # 2. Saturation & Hue (Color Grading Simulation)
        # Shift로 인해 생긴 색조(Tint)까지 포함하여 비틂

        # Saturation
        if random.random() < 0.5:
            sat_factor = random.uniform(0.5, 1.5)
            input_t = TF.adjust_saturation(input_t, sat_factor)
            target_t = TF.adjust_saturation(target_t, sat_factor)

        # Hue (+/- 18도 회전)
        if random.random() < 0.3:
            hue_factor = random.uniform(-0.05, 0.05)
            input_t = TF.adjust_hue(input_t, hue_factor)
            target_t = TF.adjust_hue(target_t, hue_factor)

        # 3. Random Gamma (Tone Curve / Display Encoding)
        # 최후의 비선형 변환. 밴딩의 간격과 위치를 이동시킴.
        if random.random() < 0.8:
            gamma = random.uniform(0.8, 1.2)
            # Gamma 연산 전 안전장치 (0이 들어가면 미분 불가능할 수 있으므로 epsilon 추가)
            input_t = torch.clamp(input_t, 1e-8, 1.0) ** gamma
            target_t = torch.clamp(target_t, 1e-8, 1.0) ** gamma

        return torch.clamp(input_t, 0.0, 1.0), torch.clamp(target_t, 0.0, 1.0)

    def _make_even_size(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reflect Padding을 사용하여 짝수 해상도로 맞춤 (Valid용)"""
        _, h, w = tensor.shape
        pad_h = 1 if (h % 2 != 0) else 0
        pad_w = 1 if (w % 2 != 0) else 0

        if pad_h + pad_w > 0:
            tensor = TF.pad(tensor, (0, 0, pad_w, pad_h), mode="reflect")
        return tensor

    def __getitem__(self, idx: int):
        # 1. Load Image (RGB)
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.is_train:
            # --- [Train Pipeline] ---

            # A. Geometric Augmentation (PIL Level)
            # Input과 Target이 동일한 물리적 변형을 겪어야 함
            geo_params = self._get_geometric_params(img.size)
            img_aug = self._apply_geometric(img, geo_params)

            # To Tensor (0~1)
            target_rgb = TF.to_tensor(img_aug)
            input_rgb = target_rgb.clone()

            # B. Degradation (Input Only)
            # 순서: Quantization -> JPEG -> Noise
            # (구조적 아티팩트를 먼저 만들고, 그 위에 노이즈를 덮음)

            # 1. Quantization (w/ Dithering)
            input_rgb = self._apply_quantization(input_rgb)

            # 2. JPEG Compression (50% 확률)
            if random.random() < 0.5:
                input_rgb = self._apply_jpeg(input_rgb)

            # 3. Gaussian Noise (50% 확률)
            if random.random() < 0.5:
                input_rgb = self._apply_gaussian_noise(input_rgb)

            # C. Color Augmentation (Both Input & Target)
            # 열화된 상태에서 색공간을 비틀어, 특정 색이 아닌 '구조적 결함'을 학습하게 함
            input_rgb, target_rgb = self._apply_color_aug(input_rgb, target_rgb)

            return input_rgb, target_rgb

        else:
            # --- [Valid Pipeline] ---
            # 원본 해상도 유지
            target_rgb = TF.to_tensor(img)

            # 짝수 해상도 보정 (모델 통과를 위해)
            target_rgb = self._make_even_size(target_rgb)

            # 검증 시 Input은 깨끗한 원본을 그대로 반환
            # (실제 평가를 위한 열화는 Train/Eval Loop에서 고정된 세팅으로 적용)
            input_rgb = target_rgb.clone()

            return input_rgb, target_rgb
