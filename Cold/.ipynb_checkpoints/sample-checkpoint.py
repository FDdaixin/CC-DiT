import os
import math
import torch
import numpy as np
import cv2
import skimage.io as io
import torchvision.transforms as transforms
from tqdm import tqdm
from accelerate import Accelerator
from ColdSeg.datasets import Datasets
from torch.utils.data import DataLoader
from ColdSeg.SegDiffusion import Unet, MedSegDiff
from data_augmentation import GammaCLAHE
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from matplotlib import pyplot as plt
from ColdSeg.cldice import cldice_hard, dice_coefficient #cldice 指标

from utils_sr.tile_utils import tile_image, merge_tiles
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops


if __name__ == '__main__':

    # Hyper Parameter Setting
    dim = 64  # Foundation dimensions of the UNet network
    img_size = 512  # Input Image Size, or 128
    batch_size = 1  # Batch Size
    time_steps = 50  # Time step - number of noise stacks
    mask_channels = 1  # Number of mask channels, will be
    self_condition = False  # self-conditional input
    save_uncertainty = False  # Preservation of uncertainty
    input_img_channels = 1  # Number of input image channels

    # 加载路径
    root = r"../retinal_vascular/CHASEDB1"  # autodl-tmp/DRIVE/
    load_model_from = r"output/CHASEDB/1000epoch**1/epoch940.pt"

    # 推理过程存储路径
    inference_dir = "./output/CHASEDB/1000epoch**1/test/epoch940"
    os.makedirs(inference_dir, exist_ok=True)
    
    # 指标保存文件
    metrics_file = os.path.join(inference_dir, "test_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("Img\tAccuracy\tPrecision\tSensitivity\tSpecificity\tF1\tDice\tclDice\tAUC\n")

    # 使用油门加速训练过程（混合精度）
    accelerator = Accelerator(mixed_precision="no")

    # 建立UNet
    model = Unet(dim=dim, image_size=img_size,dim_mult=(1, 2, 4, 8), mask_channels=mask_channels, input_img_channels=input_img_channels, self_condition=self_condition)

    # 数据增强
    image_transform = transforms.Compose([
        GammaCLAHE(gamma=1.5, clip_limit=2.0),
        transforms.ToTensor(),
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())  # 强制二值化为 0/1
    ])

    # 数据加载
    dataset_test = Datasets(root, split="test", image_transform=image_transform, mask_transform=mask_transform, augment=False)
    
    # 数据处理
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # 扩散模型
    diffusion = MedSegDiff(model, time_steps=time_steps).to(accelerator.device)

    # 读取保存的模型参数
    if load_model_from is not None:
        save_dict = torch.load(load_model_from)
        new_state_dict = {}
        for k, v in save_dict['model_state_dict'].items():
            new_state_dict[k] = v
        diffusion.model.load_state_dict(new_state_dict)

    diffusion.model.eval()
    
    # 初始化指标列表
    all_metrics = []
    
    for idx, (images, masks, _) in enumerate(tqdm(test_loader, desc="Processing images")):
        # 假设 batch_size=1，直接取第一个样本
        large_image = images.squeeze(0).permute(1, 2, 0).numpy()  # [H, W, C]
        original_h, original_w = large_image.shape[:2]

        # Step 1: 分块预处理
        tiles, coords, pad_shape = tile_image(large_image, tile_size=img_size, overlap=0.75)

        # Step 2: 逐块预测
        predicted_tiles = []
        for tile in tqdm(tiles, desc=f"Processing tiles for image {idx}", leave=False):
            # 转换为模型输入格式 [B, C, H, W]
            tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().to(accelerator.device)
            with torch.no_grad():
                pred_tile = diffusion.sample(tile_tensor).squeeze().cpu().numpy()  # [H, W]
            predicted_tiles.append(pred_tile)

        # Step 3: 融合结果
        merged_pred = merge_tiles(predicted_tiles, pad_shape, (original_h, original_w), 
                                 tile_size=img_size, overlap=0.75)
        # merged_pred[merged_pred < 0.005] = 0
        # 计算合并后预测结果的最大值和最小值
#         max_value = np.max(merged_pred)
#         min_value = np.min(merged_pred)

#         print(f"合并预测结果的最大值: {max_value}")
#         print(f"合并预测结果的最小值: {min_value}")
        # 原始二值化
        merged_pred_binary = (merged_pred > 0.5).astype(np.uint8)

        # 原始图像
        orig_image = (large_image * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(inference_dir, f"img_{idx}_original.png"), orig_image)
        
        # 概率图
        prob_map = (merged_pred * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(inference_dir, f"img_{idx}_probability.png"), prob_map)
        
        # 二值预测图
        cv2.imwrite(os.path.join(inference_dir, f"img_{idx}_prediction.png"), merged_pred_binary * 255)
        
        # 真实掩码
        mask_np = masks.squeeze(0).numpy().squeeze()  # [H, W]
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(inference_dir, f"img_{idx}_mask.png"), mask_uint8)
        
        # 叠加可视化
        overlay = orig_image.copy()
        if orig_image.shape[-1] == 1:  # 如果是灰度图，转换为RGB
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        overlay[merged_pred_binary == 1, :] = [255, 0, 0]  # 红色表示预测血管
        cv2.imwrite(os.path.join(inference_dir, f"img_{idx}_overlay.png"), overlay)
        
        # Step 5: 计算指标
        mask_flat = mask_np.flatten().astype(np.uint8)
        pred_flat = merged_pred_binary.flatten()
        prob_flat = merged_pred.flatten()  # 概率图（0~1）
        
        acc = accuracy_score(mask_flat, pred_flat)
        precision = precision_score(mask_flat, pred_flat, zero_division=0)
        sensitivity = recall_score(mask_flat, pred_flat, zero_division=0)
        specificity = recall_score(mask_flat, pred_flat, pos_label=0, zero_division=0)
        f1 = f1_score(mask_flat, pred_flat, zero_division=0)
        dice = dice_coefficient(mask_flat, pred_flat)
        cldice = cldice_hard(mask_np, merged_pred_binary)
        try:
            auc = roc_auc_score(mask_flat, prob_flat)
        except ValueError:
            auc = 0.0  # 如果样本只有一个类别，AUC 无法计算
        
        # ROC 曲线（只画前5张）
        if idx < 5:
            fpr, tpr, _ = roc_curve(mask_flat, prob_flat)
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Image {idx}')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(os.path.join(inference_dir, f"img_{idx}_roc_curve.png"))
            plt.close()

        # 保存指标到列表和文件
        metrics = (acc, precision, sensitivity, specificity, f1, dice, cldice, auc)
        all_metrics.append(metrics)

        with open(metrics_file, 'a') as f:
            f.write(
                f"{idx}\t{acc:.4f}\t{precision:.4f}\t{sensitivity:.4f}\t"
                f"{specificity:.4f}\t{f1:.4f}\t{dice:.4f}\t{cldice:.4f}\t{auc:.4f}\n"
            )

        print(
            f"Image {idx} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, "
            f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, "
            f"F1: {f1:.4f}, Dice: {dice:.4f}, clDice: {cldice:.4f}, AUC: {auc:.4f}"
        )
    # 计算平均指标
    avg_metrics = np.mean(all_metrics, axis=0)
    print("\nAverage Metrics:")
    print(f"Accuracy: {avg_metrics[0]:.4f}")
    print(f"Precision: {avg_metrics[1]:.4f}")
    print(f"Sensitivity: {avg_metrics[2]:.4f}")
    print(f"Specificity: {avg_metrics[3]:.4f}")
    print(f"F1 Score: {avg_metrics[4]:.4f}")
    print(f"Dice: {avg_metrics[5]:.4f}")
    print(f"clDice: {avg_metrics[6]:.4f}")
    print(f"AUC: {avg_metrics[7]:.4f}")

    with open(metrics_file, 'a') as f:
        f.write("\nAverage Metrics:\t")
        f.write(f"{avg_metrics[0]:.4f}\t")
        f.write(f"{avg_metrics[1]:.4f}\t")
        f.write(f"{avg_metrics[2]:.4f}\t")
        f.write(f"{avg_metrics[3]:.4f}\t")
        f.write(f"{avg_metrics[4]:.4f}\t")
        f.write(f"{avg_metrics[5]:.4f}\t")
        f.write(f"{avg_metrics[6]:.4f}\t")
        f.write(f"{avg_metrics[7]:.4f}\t")
