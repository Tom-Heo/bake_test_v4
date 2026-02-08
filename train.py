import os
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
from utils import (
    seed_everything,
    get_logger,
    ModelEMA,
    save_checkpoint,
    compute_delta_e,
    quantize_validation,
    prepare_div2k_dataset,  # <--- 추가됨
)


def train():
    # 1. Setup
    seed_everything(42)
    Config.create_directories()
    logger = get_logger(Config.LOG_DIR)
    device = torch.device(Config.DEVICE)

    # Cudnn Benchmark (속도 최적화)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("Cudnn Benchmark Enabled.")

    logger.info(f"Initialize Bake Training on {device}")

    prepare_div2k_dataset(Config, logger)

    # 2. Dataset & Loader
    train_dataset = DIV2KDataset(Config, is_train=True)
    valid_dataset = DIV2KDataset(Config, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,  # 검증은 1장씩
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model & Components
    model = BakeNet(dim=Config.INTERNAL_DIM).to(device)
    criterion = BakeLoss().to(device)

    # Converters (Palette)
    to_oklabp = Palette.sRGBtoOklabP().to(device)
    to_rgb = Palette.OklabPtosRGB().to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Warmup (5 epochs) -> Exponential Decay
    # Warmup은 Step 단위로 계산되어야 정확함
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

    # EMA
    model_ema = ModelEMA(model, decay=Config.EMA_DECAY)

    # Resume Check (Last ckpt)
    start_epoch = 1
    best_delta_e = 999.0  # Lower is better

    if os.path.exists(Config.LAST_CKPT_PATH):
        logger.info(f"Resuming from {Config.LAST_CKPT_PATH}")
        ckpt = torch.load(Config.LAST_CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "ema_shadow" in ckpt:
            model_ema.shadow = ckpt["ema_shadow"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1

    # 4. Training Loop
    logger.info("Start Training...")

    for epoch in range(start_epoch, Config.TOTAL_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for step, (input_rgb, target_rgb) in enumerate(train_loader, 1):
            # A. Upload to GPU
            input_rgb = input_rgb.to(device)  # (B, 3, H, W)
            target_rgb = target_rgb.to(device)

            # B. Convert to OklabP (On-the-fly)
            # Dataset이 RGB를 줬으므로, 모델 입력 직전에 변환
            input_oklabp = to_oklabp(input_rgb)
            target_oklabp = to_oklabp(target_rgb)  # Loss Target

            # C. Forward
            pred_oklabp = model(input_oklabp)

            # D. Loss Calculation (Includes 96ch Projection)
            loss = criterion(pred_oklabp, target_oklabp)

            # NaN Check (Safety)
            if torch.isnan(loss):
                logger.error(
                    f"NaN Loss detected at Epoch {epoch} Step {step}. Skipping..."
                )
                optimizer.zero_grad()
                continue

            # E. Backward (No Clipping)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # F. Update EMA & Scheduler
            model_ema.update(model)
            scheduler.step()  # Per-step scheduler

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
        logger.info(f"==> Epoch {epoch} Avg Loss: {avg_loss:.6f}")

        # 5. Validation Loop
        if epoch % Config.VALID_INTERVAL_EPOCHS == 0:
            logger.info(f"==> Validating at Epoch {epoch}...")

            model_ema.apply_shadow(model)  # Use EMA weights
            model.eval()

            total_delta_e = 0.0
            num_samples = 0

            # Visualization buffer
            vis_input, vis_pred, vis_target = None, None, None

            with torch.no_grad():
                for v_step, (v_clean_rgb, v_tgt_rgb) in enumerate(valid_loader):
                    v_clean_rgb = v_clean_rgb.to(device)
                    v_tgt_rgb = v_tgt_rgb.to(device)

                    # [Validation Degradation]
                    # 검증 시에는 깨끗한 원본을 3-bit로 강제 양자화하여 성능 테스트
                    v_in_rgb = quantize_validation(v_clean_rgb, bit_depth=3)

                    # 1. Convert
                    v_in_oklabp = to_oklabp(v_in_rgb)
                    v_tgt_oklabp = to_oklabp(v_tgt_rgb)  # Metric Target

                    # 2. Inference
                    v_pred_oklabp = model(v_in_oklabp)

                    # 3. Metric: Delta E (in Oklab Domain)
                    batch_delta_e = compute_delta_e(v_pred_oklabp, v_tgt_oklabp)
                    total_delta_e += batch_delta_e.item()
                    num_samples += 1

                    # 4. Save Visualization (First batch only)
                    if v_step == 0:
                        # Convert Pred back to RGB for viewing
                        v_pred_rgb = to_rgb(v_pred_oklabp)
                        vis_input = v_in_rgb  # 망가진 입력
                        vis_target = v_tgt_rgb  # 정답
                        vis_pred = v_pred_rgb.clamp(0, 1)

            avg_delta_e = total_delta_e / num_samples
            logger.info(f"==> Valid Delta E: {avg_delta_e:.4f} (Lower is Better)")

            # Check Best
            is_best = avg_delta_e < best_delta_e
            if is_best:
                best_delta_e = avg_delta_e
                logger.info(f"==> New Best Record! Delta E: {best_delta_e:.4f}")

            # Save Checkpoint (Last + Best)
