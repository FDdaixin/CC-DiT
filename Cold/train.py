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

from tqdm import tqdm  # Progress bar
from torch.optim import AdamW
from accelerate import Accelerator  # 加速器
from ColdSeg.datasets import Datasets
from torch.utils.data import DataLoader
from ColdSeg.SegDiffusion import Unet, MedSegDiff
from ColdSeg.SegDiffusion import create_lr_scheduler
from ColdSeg.dataset import DRIVE_DataSet
import time

from sklearn.metrics import accuracy_score, precision_score, recall_score
import torchvision.utils as vutils

# data_augmentation
from data_augmentation import GammaCLAHE

from utils_sr.tile_utils import tile_image, merge_tiles



if __name__ == '__main__':
    # Hyper Parameter Setting
    dim =  64 
    epochs = 1001 
    img_size = 512  
    batch_size = 1 
    save_epoch = 900 # 大于X，开始保存模型
    save_every = 20  # 迭代 X 次，保存模型
    time_steps = 50  

    adam_beta_1 = 0.950  # Adam Parameter 1
    adam_beta_2 = 0.999  # Adam Parameter 2
    adam_epsilon = 1e-8  # eps value

    weight_decay = 1e-6  # regularization factor 正则化因子
    learning_rate = 2e-5  # learning rate
    mask_channels = 1  # Number of mask channels
    self_condition = False  # self-conditional input
    input_img_channels = 1  # Number of input image channels
    load_model_from = None  


    # 创建保存训练结果
    save_dir = "output/CHASEDB/1000epoch**1"
    os.makedirs(save_dir, exist_ok=True)
    val_dir = os.path.join(save_dir, f"val_picture")
    os.makedirs(val_dir, exist_ok=True)

    # 保存训练日志
    logging_dir = "output/logs"
    os.makedirs(logging_dir, exist_ok=True)

    root = "../retinal_vascular/CHASEDB1"

    # 使用油门踏板加速训练过程（梯度累积步骤、混合精度、记录位置）
    accelerator = Accelerator(gradient_accumulation_steps=16,
                              mixed_precision="fp16",
                              log_with=["tensorboard"],
                              project_dir="logs")
    # ---------------------------wandb--------------------------
    if accelerator.is_main_process:
        accelerator.init_trackers("ColdSegDiffusion")
    print("===========wandb开启============")

    model = Unet(dim=dim, image_size=img_size,
                 dim_mult=(1, 2, 4, 8), mask_channels=mask_channels,
                 input_img_channels=input_img_channels, self_condition=self_condition)

    # Data Augmentation
    image_transform = transforms.Compose([
        GammaCLAHE(gamma=1.5, clip_limit=2.0),
        transforms.ToTensor(),
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])
    
    # Data Processing
    dataset_train = Datasets(root, split="train", image_transform=image_transform, mask_transform=mask_transform, augment=True,crop_size=img_size,overlap=0.5, train_ratio=0.7, val_ratio=0.1)
    dataset_valid = Datasets(root, split="val", image_transform=image_transform, mask_transform=mask_transform, augment=False)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    # 初始化优化器 Initializing the Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(adam_beta_1, adam_beta_2), weight_decay=weight_decay, eps=adam_epsilon)

    # 数据加速和混合精度设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, train_loader, valid_loader = accelerator.prepare([model, optimizer, train_loader, valid_loader])

    diffusion = MedSegDiff(model, time_steps=time_steps, objective='predict_x0').to(accelerator.device)

    # 记录训练开始时间
    start_time = time.time()
    epoch_times = []

    # 加载预训练模型
    if load_model_from is not None:
        save_dict = torch.load(load_model_from)
        diffusion.model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        accelerator.print(f'Loaded from {load_model_from}')

    # 创建学习率更新策略，在本例中为每个步骤一次（而不是每个纪元）
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs, warmup=True)

    # 训练流程
    train_losses = []  #记录每个 epoch 的训练损失
    for epoch in range(epochs):
        # 记录损失
        running_train_loss = 0.0
        running_valid_loss = 0.0
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        # 记录当前 epoch 开始时间
        epoch_start_time = time.time()

        # Initialize images and masks
        # 初始化
        train_img = None
        valid_img = None
        train_mask = None
        valid_mask = None
# =================================================train================================================
        total_loss = 0.0
        total_samples = 0
        model.train()
        for (train_img, train_mask) in tqdm(train_loader):
            # 梯度累计通过将多个小批量的梯度累积起来，然后一次性更新模型参数，实现更大批量的效果
            with accelerator.accumulate(model):
                train_loss = diffusion(train_mask, train_img, epoch, epochs)
                total_loss += train_loss.item()
                total_samples += 1
                # 计算平均损失
                train_loss = train_loss / accelerator.gradient_accumulation_steps
                # 反向传播
                accelerator.backward(train_loss)

                # —— 只有当累计了16次，或者到达最后一个批次时，才真正更新参数 ——
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                lr = optimizer.param_groups[0]["lr"]

        epoch_train_loss = total_loss / total_samples
        train_losses.append(epoch_train_loss)  # ✅ 新增：添加到列表中
        print('Train Loss : {:.7f}'.format(epoch_train_loss))

        accelerator.log(
            {'train_loss': epoch_train_loss, 'lr': lr},
            step=epoch
        )
# =================================================train================================================        
                
        # 模型验证过程
        if epoch >= 0 and epoch % 200 == 0:
            model.eval()
            valid_loss = 0.0 
            for idx, (images, masks, _) in enumerate(tqdm(valid_loader)):
                # 假设 batch_size=1，直接取第一个样本
                large_image = images.squeeze(0).permute(1, 2, 0).numpy()  # [H, W, C]
                original_h, original_w = large_image.shape[:2]

                # Step 1: 分块预处理
                tiles, _,pad_shape, = tile_image(large_image, tile_size=512, overlap=0.5)  # 修改为实际输入尺寸512

                # Step 2: 逐块预测
                predicted_tiles = []
                for tile in tiles:
                    # 转换为模型输入格式 [B, C, H, W]
                    tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().to(accelerator.device)
                    with torch.no_grad():
                        pred_tile = diffusion.sample(tile_tensor).squeeze().cpu().numpy()  # [H, W]
                    predicted_tiles.append(pred_tile)

                # Step 3: 融合结果
                print("pad_shape:", pad_shape)
                merged_pred = merge_tiles(predicted_tiles, pad_shape, (original_h, original_w), tile_size=512, overlap=0.5)
                merged_pred_binary = (merged_pred > 0.5).astype(np.uint8)  # 与原有阈值一致
                
                # 保存概率图和二值图时，加上 idx 保证每张都不同
                prob_uint8 = (merged_pred * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(val_dir, f"epoch{epoch}_img{idx}_prob.png"), prob_uint8)
                cv2.imwrite(os.path.join(val_dir, f"epoch{epoch}_img{idx}_binary.png"), merged_pred_binary * 255)

                # --- 指标计算与结果保存 ---
                mask_np = masks.squeeze().cpu().numpy().astype(np.uint8)

                # 展平计算指标
                predict_flat = merged_pred_binary.flatten()
                mask_flat = mask_np.flatten()

                acc = accuracy_score(mask_flat, predict_flat)
                precision = precision_score(mask_flat, predict_flat, zero_division=0)
                sensitivity = recall_score(mask_flat, predict_flat, zero_division=0)
                specificity = recall_score(mask_flat, predict_flat, pos_label=0, zero_division=0)

                print(f"Metrics - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

                accelerator.log({
                    "acc": acc,
                    "precision": precision,
                    "sensitivity": sensitivity,
                    "specificity": specificity
                }, step=epoch)
                
        # =============================保存模型参数=============================    
        # --------------每50epoch保存模型参数----------------
        if epoch >= save_epoch and epoch % save_every==0:
            torch.save({'epoch': epoch, 'model_state_dict': diffusion.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, os.path.join(save_dir, f'epoch{epoch}.pt'))
        # --------------每50epoch保存模型参数----------------
        
        # =============================综合得分保存模型参数=========================           

        # 并计算时间
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        remaining_epochs = epochs - (epoch + 1)
        recent_epochs = min(5, len(epoch_times))
        avg_epoch_time = sum(epoch_times[-recent_epochs:]) / recent_epochs if epoch_times else epoch_time
        remaining_time = avg_epoch_time * remaining_epochs
        print(f"Estimated remaining time: {time.strftime('%H:%M:%S', time.gmtime(remaining_time))}")
    # 记录训练结束时间，并计算总训练时间
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total training time: {total_time / 60:.2f} minutes")
    
    
    # 训练完后绘制 loss 曲线
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'train_loss_curve.png'))
    plt.close()
    
    print(f"Total training time: {total_time / 60:.2f} minutes")