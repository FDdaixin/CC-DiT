import os
import cv2
import time
import json
import math
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator

from sklearn.metrics import accuracy_score, precision_score, recall_score

# your project imports
from ColdSeg.datasets import Datasets
from ColdSeg.SegDiffusion import Unet, MedSegDiff
from ColdSeg.SegDiffusion import create_lr_scheduler

from data_augmentation import GammaCLAHE
from utils_sr.tile_utils import tile_image, merge_tiles


def fmt_w(x: float) -> str:
    """format weight for filename: 0.25 -> 0p25"""
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def run_one_experiment(
    *,
    root: str,
    dim: int,
    epochs: int,
    img_size: int,
    batch_size: int,
    time_steps: int,
    overlap: float,
    save_epoch: int,
    save_every: int,
    learning_rate: float,
    weight_decay: float,
    adam_beta_1: float,
    adam_beta_2: float,
    adam_epsilon: float,
    mask_channels: int,
    input_img_channels: int,
    self_condition: bool,
    loss_alpha: float,
    loss_beta: float,
    loss_gamma: float,
    exp_dir: Path,
    val_every: int = 200,
):
    """
    One full train run with fixed (alpha, beta, gamma).
    """

    exp_dir.mkdir(parents=True, exist_ok=True)
    val_dir = exp_dir / "val_picture"
    val_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # save config for reproducibility
    cfg = dict(
        root=root,
        dim=dim,
        epochs=epochs,
        img_size=img_size,
        batch_size=batch_size,
        time_steps=time_steps,
        overlap=overlap,
        save_epoch=save_epoch,
        save_every=save_every,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=[adam_beta_1, adam_beta_2],
        eps=adam_epsilon,
        loss_alpha=loss_alpha,
        loss_beta=loss_beta,
        loss_gamma=loss_gamma,
    )
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # Accelerator must be created per run to avoid mixed logs
    accelerator = Accelerator(
        gradient_accumulation_steps=16,
        mixed_precision="fp16",
        log_with=["tensorboard"],
        project_dir=str(logs_dir),
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            "ColdSegDiffusion"
        )
    accelerator.print(f"[RUN] dir={exp_dir}")
    accelerator.print(f"[LOSS] alpha={loss_alpha}, beta={loss_beta}, gamma={loss_gamma}")

    # Build model
    model = Unet(
        dim=dim,
        image_size=img_size,
        dim_mult=(1, 2, 4, 8),
        mask_channels=mask_channels,
        input_img_channels=input_img_channels,
        self_condition=self_condition
    )

    # Transforms
    image_transform = transforms.Compose([
        GammaCLAHE(gamma=1.5, clip_limit=2.0),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])

    # Dataset
    dataset_train = Datasets(
        root, split="train",
        image_transform=image_transform,
        mask_transform=mask_transform,
        augment=True,
        crop_size=img_size,
        overlap=overlap,
        train_ratio=0.7,
        val_ratio=0.1
    )
    dataset_valid = Datasets(
        root, split="val",
        image_transform=image_transform,
        mask_transform=mask_transform,
        augment=False
    )

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=1, shuffle=False)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(adam_beta_1, adam_beta_2),
        weight_decay=weight_decay,
        eps=adam_epsilon
    )

    # Prepare
    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )

    # Diffusion (IMPORTANT: your MedSegDiff must accept these args)
    diffusion = MedSegDiff(
        model,
        time_steps=time_steps,
        objective="predict_x0",
        loss_alpha=loss_alpha,
        loss_beta=loss_beta,
        loss_gamma=loss_gamma
    ).to(accelerator.device)

    # Scheduler
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs, warmup=True)

    # Training
    train_losses = []
    epoch_times = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()

        total_loss = 0.0
        total_samples = 0

        for (train_img, train_mask) in tqdm(train_loader, disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                loss = diffusion(train_mask, train_img, epoch, epochs)
                total_loss += loss.item()
                total_samples += 1

                loss = loss / accelerator.gradient_accumulation_steps
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

        epoch_train_loss = total_loss / max(1, total_samples)
        train_losses.append(epoch_train_loss)

        lr = optimizer.param_groups[0]["lr"]
        accelerator.log({"train_loss": epoch_train_loss, "lr": lr}, step=epoch)

        accelerator.print(f"Epoch {epoch+1}/{epochs} | train_loss={epoch_train_loss:.6f} | lr={lr:.3e}")

        # ===== validation =====
        if (epoch % val_every == 0):
            model.eval()
            for idx, (images, masks, _) in enumerate(tqdm(valid_loader, disable=not accelerator.is_main_process)):
                large_image = images.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H,W,C]
                original_h, original_w = large_image.shape[:2]

                tiles, _, pad_shape = tile_image(large_image, tile_size=img_size, overlap=overlap)

                predicted_tiles = []
                for tile in tiles:
                    tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().to(accelerator.device)
                    with torch.no_grad():
                        pred_tile = diffusion.sample(tile_tensor).squeeze().cpu().numpy()
                    predicted_tiles.append(pred_tile)

                merged_pred = merge_tiles(predicted_tiles, pad_shape, (original_h, original_w),
                                          tile_size=img_size, overlap=overlap)
                merged_pred_binary = (merged_pred > 0.5).astype(np.uint8)

                prob_uint8 = (merged_pred * 255).astype(np.uint8)
                cv2.imwrite(str(val_dir / f"epoch{epoch}_img{idx}_prob.png"), prob_uint8)
                cv2.imwrite(str(val_dir / f"epoch{epoch}_img{idx}_binary.png"), merged_pred_binary * 255)

                mask_np = masks.squeeze().cpu().numpy().astype(np.uint8)
                predict_flat = merged_pred_binary.flatten()
                mask_flat = mask_np.flatten()

                acc = accuracy_score(mask_flat, predict_flat)
                precision = precision_score(mask_flat, predict_flat, zero_division=0)
                sensitivity = recall_score(mask_flat, predict_flat, zero_division=0)
                specificity = recall_score(mask_flat, predict_flat, pos_label=0, zero_division=0)

                accelerator.log({
                    "acc": acc,
                    "precision": precision,
                    "sensitivity": sensitivity,
                    "specificity": specificity
                }, step=epoch)

                accelerator.print(
                    f"[VAL] epoch={epoch} img={idx} "
                    f"acc={acc:.4f} prec={precision:.4f} se={sensitivity:.4f} sp={specificity:.4f}"
                )

        # ===== save checkpoint =====
        if epoch >= save_epoch and (epoch % save_every == 0):
            ckpt_name = f"epoch{epoch}_a{fmt_w(loss_alpha)}_b{fmt_w(loss_beta)}_g{fmt_w(loss_gamma)}.pt"
            ckpt_path = exp_dir / ckpt_name
            torch.save({
                "epoch": epoch,
                "model_state_dict": diffusion.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_alpha": loss_alpha,
                "loss_beta": loss_beta,
                "loss_gamma": loss_gamma,
            }, ckpt_path)

        # time stats
        epoch_t = time.time() - epoch_start
        epoch_times.append(epoch_t)

    total_time = time.time() - start_time
    accelerator.print(f"[DONE] total_minutes={total_time/60:.2f}")

    # plot loss curve
    try:
        plt.figure()
        plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(str(exp_dir / "train_loss_curve.png"))
        plt.close()
    except Exception as e:
        accelerator.print(f"[WARN] loss plot failed: {e}")

    accelerator.end_training()


if __name__ == "__main__":
# ===================== basic hyperparams =====================
    dim = 64
    epochs = 501
    img_size = 512
    batch_size = 1
    time_steps = 50
    overlap = 0.75

    save_epoch = 400
    save_every = 20
    val_every = 200

    adam_beta_1 = 0.950
    adam_beta_2 = 0.999
    adam_epsilon = 1e-8

    weight_decay = 1e-6
    learning_rate = 2e-5

    mask_channels = 1
    input_img_channels = 1
    self_condition = False

    root = "../retinal_vascular/STARE"

# ===================== sweep config =====================

    
    
    grid = [0, 0.25, 0.5, 0.75, 1.0]

    # choose which param to sweep: "alpha" / "beta" / "gamma"
    sweep_param = "beta"

    # default fixed values (your baseline)
    alpha_fixed = 0.5
    beta_fixed = 0.5
    gamma_fixed = 1.0

    out_base = Path(f"output/STARE/sweep_{sweep_param}")
    out_base.mkdir(parents=True, exist_ok=True)

    for v in grid:
        if sweep_param == "alpha":
            a, b, g = v, beta_fixed, gamma_fixed
        elif sweep_param == "beta":
            a, b, g = alpha_fixed, v, gamma_fixed
        elif sweep_param == "gamma":
            a, b, g = alpha_fixed, beta_fixed, v
        else:
            raise ValueError("sweep_param must be one of: alpha/beta/gamma")

        exp_name = f"a{fmt_w(a)}_b{fmt_w(b)}_g{fmt_w(g)}"
        exp_dir = out_base / exp_name

        run_one_experiment(
            root=root,
            dim=dim,
            epochs=epochs,
            img_size=img_size,
            batch_size=batch_size,
            time_steps=time_steps,
            overlap=overlap,
            save_epoch=save_epoch,
            save_every=save_every,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta_1=adam_beta_1,
            adam_beta_2=adam_beta_2,
            adam_epsilon=adam_epsilon,
            mask_channels=mask_channels,
            input_img_channels=input_img_channels,
            self_condition=self_condition,
            loss_alpha=a,
            loss_beta=b,
            loss_gamma=g,
            exp_dir=exp_dir,
            val_every=val_every
        )