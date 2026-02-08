from __future__ import annotations

import os
import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image


class DIV2KDataset(Dataset):
    """
    [Lightweight Dataset Loader]
    - CPU Heavy Task 제거 (JPEG, Noise 등은 GPU Augmentor로 이관)
    - 역할: Image Read -> Random Crop (Train) -> ToTensor
    """

    def __init__(self, config, is_train=True):
        super().__init__()
        self.config = config
        self.is_train = is_train

        # 데이터셋 경로 설정
        if is_train:
            self.root_dir = config.DIV2K_TRAIN_ROOT
        else:
            self.root_dir = config.DIV2K_VALID_ROOT

        # 이미지 리스트 로드
        self.image_files = self._scan_files()

    def _scan_files(self) -> list[str]:
        """이미지 파일 목록 스캔"""
        files = sorted(
            [
                f
                for f in os.listdir(self.root_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        print(
            f"[{'Train' if self.is_train else 'Valid'}] Loaded {len(files)} images from {self.root_dir}"
        )
        return files

    def __len__(self):
        return len(self.image_files)

    def _get_random_crop(self, img: Image.Image, size=512) -> Image.Image:
        """
        [CPU-Side Crop]
        DataLoader가 배치를 구성하려면 모든 텐서의 크기가 같아야 하므로,
        여기서 1차적으로 512x512 크기로 잘라서 보냅니다.
        """
        w, h = img.size

        # 이미지가 Crop Size보다 작으면 Reflect Padding
        if w < size or h < size:
            pad_w = max(size - w, 0)
            pad_h = max(size - h, 0)
            img = TF.pad(img, (0, 0, pad_w, pad_h), padding_mode="reflect")
            w, h = img.size

        i = random.randint(0, h - size)
        j = random.randint(0, w - size)

        return TF.crop(img, i, j, size, size)

    def __getitem__(self, idx: int):
        # 1. Load Image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # PIL Image Load (RGB)
        img = Image.open(img_path).convert("RGB")

        if self.is_train:
            # [Train Mode]
            # Random Crop 수행 -> Tensor 반환
            # (복잡한 열화 및 증강은 GPU의 augment.py에서 수행)
            img_crop = self._get_random_crop(img, size=512)
            return TF.to_tensor(img_crop)

        else:
            # [Valid Mode]
            # 원본 해상도 유지 (배치 크기 1)
            # 모델 통과를 위해 짝수 해상도로만 보정
            tensor = TF.to_tensor(img)
            _, h, w = tensor.shape

            pad_h = 1 if (h % 2 != 0) else 0
            pad_w = 1 if (w % 2 != 0) else 0

            if pad_h + pad_w > 0:
                tensor = TF.pad(tensor, (0, 0, pad_w, pad_h), padding_mode="reflect")

            return tensor
