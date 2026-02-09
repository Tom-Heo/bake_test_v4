import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ExponentialLR

# Local Modules
from config import Config
from data.dataset import DIV2KDataset
from core.net import BakeNet
from core.loss import BakeLoss
from core.palette import Palette
from core.augment import BakeAugment  # [NEW] GPU Augmentation Module
from utils import (
    seed_everything,
    get_logger,
    ModelEMA,
    save_checkpoint,
    compute_delta_e,
    quantize_validation,
    prepare_div2k_dataset,
)


def train(args):
    # 1. Setup
    seed_everything(42)
    Config.create_directories()

    logger = get_logger(Config.LOG_DIR)
    device = torch.device(Config.DEVICE)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("Cudnn Benchmark Enabled.")

    logger.info(f"Initialize Bake Training on {device}")

    # Dataset Download Check
    prepare_div2k_dataset(Config, logger)

    # 2. Dataset & Loader
    # dataset.py 수정으로 인해 이제 CPU 부하가 매우 적습니다.
    logger.info("Loading Datasets...")
    train_dataset = DIV2KDataset(Config, is_train=True)
    valid_dataset = DIV2KDataset(Config, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,  # Config에서 4~8로 늘리는 것을 권장
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model & Components
    logger.info(f"Initializing BakeNet (Dim: {Config.INTERNAL_DIM})...")

    # [NEW] GPU Augmentor Initialize
    # 학습 루프 내에서 고속 열화를 담당합니다.
    augmentor = BakeAugment().to(device)

    model = BakeNet(dim=Config.INTERNAL_DIM).to(device)
    criterion = BakeLoss().to(device)

    to_oklabp = Palette.sRGBtoOklabP().to(device)
    to_rgb = Palette.OklabPtosRGB().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler Setup
    warmup_epochs = 10
    total_warmup_iters = warmup_epochs * len(train_loader)

    scheduler_warmup = LinearLR(
        optimizer, start_factor=0.01, total_iters=total_warmup_iters
    )
    scheduler_decay = ExponentialLR(optimizer, gamma=Config.SCHEDULER_GAMMA)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_decay],
        milestones=[total_warmup_iters],
    )

    model_ema = ModelEMA(model, decay=Config.EMA_DECAY)

    # -------------------------------------------------------------------------
    # [Resume Logic Control]
    # -------------------------------------------------------------------------
    start_epoch = 1
    best_delta_e = 999.0
    ckpt_path = Config.LAST_CKPT_PATH
    should_load = False

    if args.restart:
        logger.info("🚫 Flag [--restart] detected. Ignoring existing checkpoints.")
        should_load = False
    elif args.resume:
        logger.info("🔄 Flag [--resume] detected. Attempting to load checkpoint.")
        if os.path.exists(ckpt_path):
            should_load = True
        else:
            logger.warning(
                f"⚠️ Checkpoint not found at {ckpt_path}. Starting from scratch."
            )
    else:
        # Default: Auto-resume if exists
        if os.path.exists(ckpt_path):
            logger.info("💾 Checkpoint found. Auto-resuming...")
            should_load = True
        else:
            logger.info("✨ No checkpoint found. Starting fresh training...")

    # Load Checkpoint if needed
    if should_load and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        if "ema_shadow" in ckpt:
            model_ema.shadow = ckpt["ema_shadow"]

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1

        logger.info(f"✅ Successfully resumed from Epoch {start_epoch-1}.")

    # 4. Training Loop
    logger.info(">>> Start Training Loop (GPU Accelerated) <<<")

    for epoch in range(start_epoch, Config.TOTAL_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        # [Modified] clean_batch만 받습니다. (dataset.py가 Clean Tensor만 반환)
        for step, clean_batch in enumerate(train_loader, 1):

            # 1. Move to GPU immediately
            clean_batch = clean_batch.to(device)

            # 2. On-the-fly Degradation & Augmentation
            # GPU 상에서 즉시 열화 이미지를 생성합니다.
            with torch.no_grad():
                input_rgb, target_rgb = augmentor(clean_batch)

            # 3. Convert to OklabP
            input_oklabp = to_oklabp(input_rgb)
            target_oklabp = to_oklabp(target_rgb)

            # 4. Forward
            pred_oklabp = model(input_oklabp)

            # 5. Loss
            loss = criterion(pred_oklabp, target_oklabp)

            # Safety
            if torch.isnan(loss):
                logger.error(
                    f"[CRITICAL] NaN Loss at Epoch {epoch} Step {step}. Skipping."
                )
                optimizer.zero_grad()
                continue

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update
            model_ema.update(model)
            scheduler.step()

            epoch_loss += loss.item()

            if step % Config.LOG_INTERVAL_STEPS == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch [{epoch}/{Config.TOTAL_EPOCHS}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f} "
                    f"LR: {current_lr:.2e}"
                )

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"==> Epoch {epoch} Finished. Avg Loss: {avg_loss:.6f}")

        # 5. Validation Loop
        if epoch % Config.VALID_INTERVAL_EPOCHS == 0:
            logger.info(f"==> Validating at Epoch {epoch}...")

            model_ema.apply_shadow(model)
            model.eval()

            total_delta_e = 0.0
            num_samples = 0
            vis_input, vis_pred, vis_target = None, None, None

            with torch.no_grad():
                # [Modified] Validation Loader도 Clean Tensor 하나만 반환합니다.
                for v_step, v_tgt_rgb in enumerate(valid_loader):
                    v_tgt_rgb = v_tgt_rgb.to(device)

                    # Validation은 고정된 4-bit 열화를 적용 (일관된 평가를 위해)
                    v_clean_rgb = v_tgt_rgb  # 원본
                    v_in_rgb = quantize_validation(v_clean_rgb, bit_depth=4)

                    v_in_oklabp = to_oklabp(v_in_rgb)
                    v_tgt_oklabp = to_oklabp(v_tgt_rgb)

                    v_pred_oklabp = model(v_in_oklabp)

                    batch_delta_e = compute_delta_e(v_pred_oklabp, v_tgt_oklabp)
                    total_delta_e += batch_delta_e.item()
                    num_samples += 1

                    if v_step == 0:
                        v_pred_rgb = to_rgb(v_pred_oklabp)
                        vis_input = v_in_rgb
                        vis_target = v_tgt_rgb
                        vis_pred = v_pred_rgb.clamp(0, 1)

            avg_delta_e = total_delta_e / num_samples
            logger.info(
                f"==> Validation Result - Delta E: {avg_delta_e:.4f} (Lower is Better)"
            )

            is_best = avg_delta_e < best_delta_e
            if is_best:
                best_delta_e = avg_delta_e
                logger.info(f"🏆 New Best Record! Delta E: {best_delta_e:.4f}")

            save_checkpoint(
                Config, epoch, model, model_ema, optimizer, scheduler, is_best
            )

            if vis_input is not None:
                combined = torch.cat([vis_input, vis_pred, vis_target], dim=3)
                save_path = os.path.join(
                    Config.RESULT_DIR, f"valid_epoch_{epoch:05d}.png"
                )
                save_image(combined, save_path)
                logger.info(f"Visualization saved.")

            model_ema.restore(model)

    logger.info("Training Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bake v4 Training Script")

    # 상호 배타적 옵션 그룹 (둘 중 하나만 사용 가능)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume", action="store_true", help="Force resume from the last checkpoint."
    )
    group.add_argument(
        "--restart",
        action="store_true",
        help="Ignore existing checkpoints and start from Epoch 1.",
    )

    args = parser.parse_args()
    train(args)
