import os
import math
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from torch.utils.data import DataLoader

from ColdSeg.datasets import Datasets
from ColdSeg.SegDiffusion import Unet, MedSegDiff
from data_augmentation import GammaCLAHE

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from matplotlib import pyplot as plt

from ColdSeg.cldice import cldice_hard, dice_coefficient
from utils_sr.tile_utils import tile_image, merge_tiles


def ensure_uint8_gray(img01: np.ndarray) -> np.ndarray:
    """[0,1] float -> uint8 grayscale"""
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255).astype(np.uint8)


if __name__ == "__main__":

    # ===================== Hyper Parameter Setting =====================
    dim = 64
    img_size = 512
    batch_size = 1
    time_steps = 50
    overlap = 0.75

    mask_channels = 1
    input_img_channels = 1
    self_condition = False

    # ===================== Dataset / Checkpoint =====================
    root = r"../retinal_vascular/STARE"  # 你要测哪个数据集，就改这里
    load_model_from = r"output/STARE/sweep_beta/a0p5_b0p5_g1/epoch480_a0p5_b0p5_g1.pt"  # 改成你的权重

    ckpt_path = Path(load_model_from)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ========= 输出目录：严格放到对应 exp_dir 下，并用 ckpt 名命名 =========
    exp_dir = ckpt_path.parent  # e.g. output/STARE/sweep_beta/a0p5_b0_g1
    ckpt_stem = ckpt_path.stem  # e.g. epoch400_a0p5_b0_g1

    inference_dir = exp_dir / "test" / ckpt_stem
    inference_dir.mkdir(parents=True, exist_ok=True)

    # 指标保存文件（也用 ckpt 命名）
    metrics_file = inference_dir / f"{ckpt_stem}_test_metrics.txt"
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write("Img\tAccuracy\tPrecision\tSensitivity\tSpecificity\tF1\tDice\tclDice\tAUC\n")

    # ===================== Accelerator =====================
    accelerator = Accelerator(mixed_precision="no")
    device = accelerator.device

    # ===================== Model =====================
    model = Unet(
        dim=dim,
        image_size=img_size,
        dim_mult=(1, 2, 4, 8),
        mask_channels=mask_channels,
        input_img_channels=input_img_channels,
        self_condition=self_condition
    ).to(device)

    # ===================== Diffusion (对齐训练：objective="predict_x0") =====================
    diffusion = MedSegDiff(
        model,
        time_steps=time_steps,
        objective="predict_x0",
        # 推理不需要 loss_*，但若你的 MedSegDiff __init__ 要求这些参数，也可以加上：
        # loss_alpha=0.5, loss_beta=0.5, loss_gamma=1.0
    ).to(device)

    # ===================== Load weights =====================
    save_dict = torch.load(str(ckpt_path), map_location="cpu")
    if "model_state_dict" not in save_dict:
        raise KeyError("Checkpoint missing key: model_state_dict")

    diffusion.model.load_state_dict(save_dict["model_state_dict"], strict=True)
    diffusion.model.eval()

    # ===================== Transforms (对齐训练) =====================
    image_transform = transforms.Compose([
        GammaCLAHE(gamma=1.5, clip_limit=2.0),
        transforms.ToTensor(),
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])

    # ===================== Data =====================
    dataset_test = Datasets(
        root, split="test",
        image_transform=image_transform,
        mask_transform=mask_transform,
        augment=False
    )
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # ===================== Eval Loop =====================
    all_metrics = []

    for idx, (images, masks, _) in enumerate(tqdm(test_loader, desc="Processing images")):
        # images: [1, C, H, W]
        # masks:  [1, 1, H, W] (or similar)

        # 转 numpy（确保在 CPU）
        large_image = images.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        original_h, original_w = large_image.shape[:2]

        # 1) tiling
        tiles, _, pad_shape = tile_image(
            large_image,
            tile_size=img_size,
            overlap=overlap
        )

        # 2) tile-wise sampling
        predicted_tiles = []
        for tile in tqdm(tiles, desc=f"Tiles img {idx}", leave=False):
            tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                pred_tile = diffusion.sample(tile_tensor).squeeze().detach().cpu().numpy()  # [H,W] or [H,W,?]
            predicted_tiles.append(pred_tile)

        # 3) merge
        merged_pred = merge_tiles(
            predicted_tiles,
            pad_shape,
            (original_h, original_w),
            tile_size=img_size,
            overlap=overlap
        )
        merged_pred = np.clip(merged_pred, 0.0, 1.0)
        merged_pred_binary = (merged_pred > 0.5).astype(np.uint8)

        # ===================== Save outputs (用 ckpt_stem 做前缀) =====================
        prefix = f"{ckpt_stem}_img{idx}"

        # 原图：large_image 是 [0,1] 的灰度 (H,W,1)
        orig_u8 = ensure_uint8_gray(large_image[..., 0] if large_image.shape[-1] == 1 else large_image[..., 0])
        cv2.imwrite(str(inference_dir / f"{prefix}_original.png"), orig_u8)

        # 概率图 / 二值图
        prob_u8 = ensure_uint8_gray(merged_pred)
        cv2.imwrite(str(inference_dir / f"{prefix}_probability.png"), prob_u8)
        cv2.imwrite(str(inference_dir / f"{prefix}_prediction.png"), merged_pred_binary * 255)

        # GT mask
        mask_np = masks.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)  # [H,W] 0/1
        cv2.imwrite(str(inference_dir / f"{prefix}_mask.png"), mask_np * 255)

        # overlay（灰度转 RGB，再叠红色）
        overlay = cv2.cvtColor(orig_u8, cv2.COLOR_GRAY2BGR)
        overlay[merged_pred_binary == 1] = (0, 0, 255)  # BGR: red
        cv2.imwrite(str(inference_dir / f"{prefix}_overlay.png"), overlay)

        # ===================== Metrics =====================
        mask_flat = mask_np.flatten().astype(np.uint8)
        pred_flat = merged_pred_binary.flatten().astype(np.uint8)
        prob_flat = merged_pred.flatten().astype(np.float32)

        acc = accuracy_score(mask_flat, pred_flat)
        prec = precision_score(mask_flat, pred_flat, zero_division=0)
        se = recall_score(mask_flat, pred_flat, zero_division=0)
        sp = recall_score(mask_flat, pred_flat, pos_label=0, zero_division=0)
        f1 = f1_score(mask_flat, pred_flat, zero_division=0)
        dice = dice_coefficient(mask_flat, pred_flat)
        cldice = cldice_hard(mask_np, merged_pred_binary)

        try:
            auc = roc_auc_score(mask_flat, prob_flat)
        except ValueError:
            auc = 0.0

        # ROC（只画前5张）
        if idx < 5:
            fpr, tpr, _ = roc_curve(mask_flat, prob_flat)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {prefix}")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(str(inference_dir / f"{prefix}_roc_curve.png"))
            plt.close()

        all_metrics.append((acc, prec, se, sp, f1, dice, cldice, auc))

        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(
                f"{idx}\t{acc:.4f}\t{prec:.4f}\t{se:.4f}\t{sp:.4f}\t"
                f"{f1:.4f}\t{dice:.4f}\t{cldice:.4f}\t{auc:.4f}\n"
            )

        print(
            f"[{prefix}] ACC={acc:.4f} PRE={prec:.4f} SE={se:.4f} SP={sp:.4f} "
            f"F1={f1:.4f} DICE={dice:.4f} clDICE={cldice:.4f} AUC={auc:.4f}"
        )

    # ===================== Average =====================
    avg = np.mean(np.array(all_metrics), axis=0)
    print("\nAverage Metrics:")
    print(f"Accuracy:     {avg[0]:.4f}")
    print(f"Precision:    {avg[1]:.4f}")
    print(f"Sensitivity:  {avg[2]:.4f}")
    print(f"Specificity:  {avg[3]:.4f}")
    print(f"F1 Score:     {avg[4]:.4f}")
    print(f"Dice:         {avg[5]:.4f}")
    print(f"clDice:       {avg[6]:.4f}")
    print(f"AUC:          {avg[7]:.4f}")

    with open(metrics_file, "a", encoding="utf-8") as f:
        f.write("\nAverage Metrics:\t")
        f.write("\t".join([f"{x:.4f}" for x in avg.tolist()]) + "\n")

    print(f"\nSaved everything to: {inference_dir}")
    print(f"Metrics file: {metrics_file}")