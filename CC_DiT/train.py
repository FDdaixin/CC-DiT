import os
import cv2
import torch
import math
import wandb
import random
import datetime
import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from tqdm import tqdm
from torch.optim import AdamW
from accelerate import Accelerator
from model.datasets import Datasets
from torch.utils.data import DataLoader
from model.SegDiffusion import Unet, MedSegDiff
from model.SegDiffusion import create_lr_scheduler
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torchvision.utils as vutils
from data_augmentation import GammaCLAHE


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


def merge_tiles(tiles, pad_shape, orig_shape, tile_size=512, overlap=0.2):
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

    count = np.where(count < 1e-6, 1.0, count)
    merged = merged[:orig_h, :orig_w] / count[:orig_h, :orig_w]
    return merged


if __name__ == '__main__':
    dim = 64
    epochs = 101
    img_size = 512
    batch_size = 1
    save_epoch = 90
    save_every = 5
    time_steps = 50

    adam_beta_1 = 0.950
    adam_beta_2 = 0.999
    adam_epsilon = 1e-8

    weight_decay = 1e-6
    learning_rate = 2e-5
    mask_channels = 1
    self_condition = False
    input_img_channels = 1
    load_model_from = None

    weight_acc = 0.1
    weight_precision = 0.2
    weight_sensitivity = 0.5
    weight_specificity = 0.2

    save_dir = "output/DRIVE/100epoch**1"
    os.makedirs(save_dir, exist_ok=True)
    val_dir = os.path.join(save_dir, f"val_picture")
    os.makedirs(val_dir, exist_ok=True)

    logging_dir = "output/logs"
    os.makedirs(logging_dir, exist_ok=True)

    root = "../retinal_vascular/DRIVE"

    accelerator = Accelerator(
        gradient_accumulation_steps=16,
        mixed_precision="fp16",
        log_with=["tensorboard"],
        project_dir="logs"
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("CC_DiT")

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

    dataset_train = Datasets(
        root,
        split="train",
        image_transform=image_transform,
        mask_transform=mask_transform,
        augment=True,
        crop_size=img_size,
        overlap=0.75,
        train_ratio=0.7,
        val_ratio=0.1
    )
    dataset_valid = Datasets(
        root,
        split="val",
        image_transform=image_transform,
        mask_transform=mask_transform,
        augment=False
    )

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(adam_beta_1, adam_beta_2),
        weight_decay=weight_decay,
        eps=adam_epsilon
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        [model, optimizer, train_loader, valid_loader]
    )

    diffusion = MedSegDiff(model, time_steps=time_steps, objective='predict_x0').to(accelerator.device)

    start_time = time.time()
    epoch_times = []

    if load_model_from is not None:
        save_dict = torch.load(load_model_from)
        diffusion.model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        accelerator.print(f'Loaded from {load_model_from}')

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs, warmup=True)

    train_losses = []
    best_score = 0.0
    best_metrics = {'acc': 0, 'precision': 0, 'sensitivity': 0, 'specificity': 0}

    for epoch in range(epochs):
        running_train_loss = 0.0
        running_valid_loss = 0.0
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        epoch_start_time = time.time()

        train_img = None
        valid_img = None
        train_mask = None
        valid_mask = None

        total_loss = 0.0
        total_samples = 0
        model.train()
        for (train_img, train_mask) in tqdm(train_loader):
            with accelerator.accumulate(model):
                train_loss = diffusion(train_mask, train_img, epoch, epochs)
                total_loss += train_loss.item()
                total_samples += 1
                train_loss = train_loss / accelerator.gradient_accumulation_steps
                accelerator.backward(train_loss)

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                lr = optimizer.param_groups[0]["lr"]

        epoch_train_loss = total_loss / total_samples
        train_losses.append(epoch_train_loss)
        print('Train Loss : {:.7f}'.format(epoch_train_loss))

        accelerator.log(
            {'train_loss': epoch_train_loss, 'lr': lr},
            step=epoch
        )

        if epoch >= 0 and epoch % 200 == 0:
            model.eval()
            valid_loss = 0.0
            for idx, (images, masks, _) in enumerate(tqdm(valid_loader)):
                large_image = images.squeeze(0).permute(1, 2, 0).numpy()
                original_h, original_w = large_image.shape[:2]

                tiles, _, pad_shape = tile_image(large_image, tile_size=512, overlap=0.75)

                predicted_tiles = []
                for tile in tiles:
                    tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().to(accelerator.device)
                    with torch.no_grad():
                        pred_tile = diffusion.sample(tile_tensor).squeeze().cpu().numpy()
                    predicted_tiles.append(pred_tile)

                print("pad_shape:", pad_shape)
                merged_pred = merge_tiles(
                    predicted_tiles,
                    pad_shape,
                    (original_h, original_w),
                    tile_size=512,
                    overlap=0.75
                )
                merged_pred_binary = (merged_pred > 0.5).astype(np.uint8)

                prob_uint8 = (merged_pred * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(val_dir, f"epoch{epoch}_img{idx}_binary.png"), merged_pred_binary * 255)

                mask_np = masks.squeeze().cpu().numpy().astype(np.uint8)

                predict_flat = merged_pred_binary.flatten()
                mask_flat = mask_np.flatten()

                acc = accuracy_score(mask_flat, predict_flat)
                precision = precision_score(mask_flat, predict_flat, zero_division=0)
                sensitivity = recall_score(mask_flat, predict_flat, zero_division=0)
                specificity = recall_score(mask_flat, predict_flat, pos_label=0, zero_division=0)

                print(
                    f"Metrics - Accuracy: {acc:.4f}, Precision: {precision:.4f}, "
                    f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}"
                )

                accelerator.log({
                    "acc": acc,
                    "precision": precision,
                    "sensitivity": sensitivity,
                    "specificity": specificity
                }, step=epoch)

        if epoch >= save_epoch and epoch % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, f'epoch{epoch}.pt'))

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        remaining_epochs = epochs - (epoch + 1)
        recent_epochs = min(5, len(epoch_times))
        avg_epoch_time = sum(epoch_times[-recent_epochs:]) / recent_epochs if epoch_times else epoch_time
        remaining_time = avg_epoch_time * remaining_epochs
        print(f"Estimated remaining time: {time.strftime('%H:%M:%S', time.gmtime(remaining_time))}")

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total training time: {total_time / 60:.2f} minutes")

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'train_loss_curve.png'))
    plt.close()

    print("best_metrics", best_metrics)
    print(f"Total training time: {total_time / 60:.2f} minutes")