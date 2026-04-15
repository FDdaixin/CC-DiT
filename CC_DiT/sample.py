import os
import math
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms

from tqdm import tqdm
from accelerate import Accelerator
from model.datasets import Datasets
from torch.utils.data import DataLoader
from model.SegDiffusion import Unet, MedSegDiff
from data_augmentation import GammaCLAHE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model.cldice import cldice_hard, dice_coefficient


def tile_image(image, tile_size=512, overlap=0.75):
    stride = int(tile_size * (1 - overlap))
    H, W, C = image.shape

    n_h = math.ceil((H - tile_size) / stride) if H > tile_size else 0
    H_pad = tile_size + n_h * stride
    if H_pad < H:
        H_pad = tile_size
        n_h = 0

    n_w = math.ceil((W - tile_size) / stride) if W > tile_size else 0
    W_pad = tile_size + n_w * stride
    if W_pad < W:
        W_pad = tile_size
        n_w = 0

    pad_h = H_pad - H
    pad_w = W_pad - W

    image_padded = np.pad(
        image,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode='constant',
        constant_values=0
    )

    tiles = []
    coords = []
    for y in range(0, H_pad - tile_size + 1, stride):
        for x in range(0, W_pad - tile_size + 1, stride):
            tile = image_padded[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)
            coords.append((x, y, x + tile_size, y + tile_size))

    return tiles, coords, (pad_h, pad_w)


def merge_tiles(tiles, pad_shape, orig_shape, tile_size=512, overlap=0.75):
    stride = int(tile_size * (1 - overlap))
    orig_h, orig_w = orig_shape
    pad_h, pad_w = pad_shape
    H_pad = orig_h + pad_h
    W_pad = orig_w + pad_w

    merged = np.zeros((H_pad, W_pad), dtype=np.float32)
    count = np.zeros((H_pad, W_pad), dtype=np.float32)

    xx = np.linspace(-1, 1, tile_size)
    yy = np.linspace(-1, 1, tile_size)
    xv, yv = np.meshgrid(xx, yy, indexing='xy')
    sigma = 0.4
    gauss = np.exp(-(xv**2 + yv**2) / (2 * sigma**2))
    gauss = gauss / np.max(gauss)

    idx = 0
    for y in range(0, H_pad - tile_size + 1, stride):
        for x in range(0, W_pad - tile_size + 1, stride):
            tile = tiles[idx]
            merged[y:y + tile_size, x:x + tile_size] += gauss * tile
            count[y:y + tile_size, x:x + tile_size] += gauss
            idx += 1

    count[count < 1e-6] = 1.0
    merged = merged / count
    merged = merged[:orig_h, :orig_w]
    return merged


if __name__ == '__main__':
    dim = 64
    img_size = 512
    batch_size = 1
    time_steps = 50
    mask_channels = 1
    self_condition = False
    save_uncertainty = False
    input_img_channels = 1

    root = r"../retinal_vascular/STARE"
    load_model_from = r"output/STARE/01_1000epoch_vitpos/epoch1000.pt"

    inference_dir = "./output/STARE/1000epoch_vitpos/test/epoch1000"
    os.makedirs(inference_dir, exist_ok=True)

    metrics_file = os.path.join(inference_dir, "test_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("Img\tAccuracy\tPrecision\tSensitivity\tSpecificity\tF1\tDice\tclDice\n")

    accelerator = Accelerator(mixed_precision="no")

    model = Unet(
        dim=dim,
        image_size=img_size,
        dim_mult=(1, 2, 4, 8),
        mask_channels=mask_channels,
        input_img_channels=input_img_channels,
        self_condition=self_condition
    )

    image_transform = transforms.Compose([
        GammaCLAHE(gamma=1.5, clip_limit=2.0),
        transforms.ToTensor(),
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])

    dataset_test = Datasets(
        root,
        split="test",
        image_transform=image_transform,
        mask_transform=mask_transform,
        augment=False
    )

    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    diffusion = MedSegDiff(model, time_steps=time_steps).to(accelerator.device)

    if load_model_from is not None:
        save_dict = torch.load(load_model_from)
        new_state_dict = {}
        for k, v in save_dict['model_state_dict'].items():
            new_state_dict[k] = v
        diffusion.model.load_state_dict(new_state_dict)

    diffusion.model.eval()

    all_metrics = []

    for idx, (images, masks, _) in enumerate(tqdm(test_loader, desc="Processing images")):
        large_image = images.squeeze(0).permute(1, 2, 0).numpy()
        original_h, original_w = large_image.shape[:2]

        tiles, coords, pad_shape = tile_image(large_image, tile_size=img_size, overlap=0.75)

        predicted_tiles = []
        for tile in tqdm(tiles, desc=f"Processing tiles for image {idx}", leave=False):
            tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().to(accelerator.device)
            with torch.no_grad():
                pred_tile = diffusion.sample(tile_tensor).squeeze().cpu().numpy()
            predicted_tiles.append(pred_tile)

        merged_pred = merge_tiles(
            predicted_tiles,
            pad_shape,
            (original_h, original_w),
            tile_size=img_size,
            overlap=0.75
        )

        merged_pred_binary = (merged_pred > 0.5).astype(np.uint8)

        orig_image = (large_image * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(inference_dir, f"img_{idx}_original.png"), orig_image)

        cv2.imwrite(os.path.join(inference_dir, f"img_{idx}_prediction.png"), merged_pred_binary * 255)

        mask_np = masks.squeeze(0).numpy().squeeze()
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(inference_dir, f"img_{idx}_mask.png"), mask_uint8)

        overlay = orig_image.copy()
        if orig_image.shape[-1] == 1:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        overlay[merged_pred_binary == 1, :] = [255, 0, 0]
        cv2.imwrite(os.path.join(inference_dir, f"img_{idx}_overlay.png"), overlay)

        mask_flat = mask_np.flatten().astype(np.uint8)
        pred_flat = merged_pred_binary.flatten()

        acc = accuracy_score(mask_flat, pred_flat)
        precision = precision_score(mask_flat, pred_flat, zero_division=0)
        sensitivity = recall_score(mask_flat, pred_flat, zero_division=0)
        specificity = recall_score(mask_flat, pred_flat, pos_label=0, zero_division=0)
        f1 = f1_score(mask_flat, pred_flat, zero_division=0)
        dice = dice_coefficient(mask_flat, pred_flat)
        cldice = cldice_hard(mask_np, merged_pred_binary)

        metrics = (acc, precision, sensitivity, specificity, f1, dice, cldice)
        all_metrics.append(metrics)

        with open(metrics_file, 'a') as f:
            f.write(
                f"{idx}\t{acc:.4f}\t{precision:.4f}\t{sensitivity:.4f}\t"
                f"{specificity:.4f}\t{f1:.4f}\t{dice:.4f}\t{cldice:.4f}\n"
            )

        print(
            f"Image {idx} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, "
            f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, "
            f"F1: {f1:.4f}, Dice: {dice:.4f}, clDice: {cldice:.4f}"
        )

    avg_metrics = np.mean(all_metrics, axis=0)
    print("\nAverage Metrics:")
    print(f"Accuracy: {avg_metrics[0]:.4f}")
    print(f"Precision: {avg_metrics[1]:.4f}")
    print(f"Sensitivity: {avg_metrics[2]:.4f}")
    print(f"Specificity: {avg_metrics[3]:.4f}")
    print(f"F1 Score: {avg_metrics[4]:.4f}")
    print(f"Dice: {avg_metrics[5]:.4f}")
    print(f"clDice: {avg_metrics[6]:.4f}")

    with open(metrics_file, 'a') as f:
        f.write("\nAverage Metrics:\t")
        f.write(f"Accuracy: {avg_metrics[0]:.4f}\t")
        f.write(f"Precision: {avg_metrics[1]:.4f}\t")
        f.write(f"Sensitivity: {avg_metrics[2]:.4f}\t")
        f.write(f"Specificity: {avg_metrics[3]:.4f}\t")
        f.write(f"F1 Score: {avg_metrics[4]:.4f}\t")
        f.write(f"Dice: {avg_metrics[5]:.4f}\t")
        f.write(f"clDice: {avg_metrics[6]:.4f}\t")