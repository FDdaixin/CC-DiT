import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms.functional as TF
import random
import numpy as np
from skimage import io
from skimage.util import view_as_windows
from torchvision import transforms
import math


def tile_image(image, tile_size=512, overlap=0.75):
    """
    用于 DRIVE 这类约 560×580 左右的图像：
    - tile_size: 512
    - overlap: 0.75
    本函数先计算最小 pad_h、pad_w，使得 (H_pad - 512)%stride == 0, (W_pad - 512)%stride == 0，
    然后做局部 pad→滑窗切块→返回所有 tiles 及它们在 padded 图上的坐标。
    """
    # 计算 stride
    stride = int(tile_size * (1 - overlap))  # 512 * 0.25 = 128
    
    H, W, C = image.shape
    
    # 1) 计算最小的 H_pad、W_pad，使 (H_pad - tile_size) 能被 stride 整除
    #    令 n_h = ceil((H - tile_size)/stride)，则 H_pad = tile_size + n_h*stride
    n_h = math.ceil((H - tile_size) / stride) if H > tile_size else 0
    H_pad = tile_size + n_h * stride
    if H_pad < H:
        # 如果原本 H < tile_size，直接让 H_pad = tile_size
        H_pad = tile_size
        n_h = 0
    
    n_w = math.ceil((W - tile_size) / stride) if W > tile_size else 0
    W_pad = tile_size + n_w * stride
    if W_pad < W:
        W_pad = tile_size
        n_w = 0
    
    # 2) 计算各个方向需要 pad 的像素
    pad_h = H_pad - H
    pad_w = W_pad - W
    
    # 这里为了简单，就在底部/右侧一次性 pad，或也可对称地上下、左右各分一半pad
    image_padded = np.pad(image,
                          ((0, pad_h), (0, pad_w), (0, 0)),
                          mode='constant', constant_values=0)
    
    # 3) 滑窗切块
    tiles = []
    coords = []
    # sliding from y=0 到 y = H_pad - tile_size，步长 = stride
    for y in range(0, H_pad - tile_size + 1, stride):
        for x in range(0, W_pad - tile_size + 1, stride):
            tile = image_padded[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)
            # coords 存的是 (x1, y1, x2, y2)，但这里 x2 = x1+512, y2 = y1+512
            coords.append((x, y, x + tile_size, y + tile_size))
    
    return tiles, coords, (pad_h, pad_w)


def merge_tiles(tiles, pad_shape, orig_shape, tile_size=512, overlap=0.2):
    """
    将预测后的多个 tile 融合回原始图像尺寸。
    tiles: 列表，每个元素为 (tile_size, tile_size) 的概率矩阵（float）
    pad_shape: (pad_h, pad_w)，tile_image 填充时的值
    orig_shape: (orig_h, orig_w)，原图未填充时的大小
    返回：
      merged: 形状 (orig_h, orig_w) 的融合后概率图
    """
    stride = int(tile_size * (1 - overlap))
    orig_h, orig_w = orig_shape
    pad_h, pad_w = pad_shape
    H_pad = orig_h + pad_h
    W_pad = orig_w + pad_w

    merged = np.zeros((H_pad, W_pad), dtype=np.float32)
    count  = np.zeros((H_pad, W_pad), dtype=np.float32)

    # 生成高斯窗进行加权
    xx = np.linspace(-1, 1, tile_size)
    yy = np.linspace(-1, 1, tile_size)
    xv, yv = np.meshgrid(xx, yy, indexing='xy')
    sigma = 0.4
    gauss = np.exp(- (xv**2 + yv**2) / (2 * sigma**2))
    gauss = gauss / np.max(gauss)

    idx = 0
    for y in range(0, H_pad - tile_size + 1, stride):
        for x in range(0, W_pad - tile_size + 1, stride):
            tile = tiles[idx]
            merged[y:y + tile_size, x:x + tile_size] += gauss * tile
            count [y:y + tile_size, x:x + tile_size] += gauss
            idx += 1

    count = np.where(count < 1e-6, 1.0, count)
    merged = merged[:orig_h, :orig_w] / count[:orig_h, :orig_w]
    return merged
        
class Datasets(Dataset):
    def __init__(self, root_dir, split='train', image_transform=None, mask_transform=None,
                 augment=False, crop_size=512, overlap=0.2, train_ratio=0.7, val_ratio=0.1, seed=42):
        """
        root_dir: 数据集根目录，包含 images/ 和 labels/ 子目录
        split: 'train'/'val'/'test'
        image_transform: 对 PIL.Image 的变换
        mask_transform: 对 PIL.Image 的变换
        augment: 是否对训练时的 tile 进行随机增强
        crop_size: tile_size，大概 512
        overlap: 重叠比例，0~1 之间
        train_ratio, val_ratio: 划分比例
        seed: 随机种子
        """
        self.root_dir = root_dir
        self.split = split
        self.augment = augment
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.crop_size = crop_size
        self.overlap = overlap

        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'labels')
        image_files = sorted(os.listdir(self.image_dir))
        mask_files = sorted(os.listdir(self.mask_dir))

        assert len(image_files) == len(mask_files), "图像和掩膜数量不一致"
        for img, mask in zip(image_files, mask_files):
            assert img.split('.')[0] == mask.split('.')[0], f"文件不匹配：{img} vs {mask}"

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        indices = np.random.permutation(len(image_files))
        split1 = int(len(indices) * train_ratio)
        split2 = split1 + int(len(indices) * val_ratio)

        if split == 'train':
            self.indices = indices[:split1]
        elif split == 'val':
            self.indices = indices[split1:split2]
        else:  # 'test'
            self.indices = indices[split2:]

        self.image_files = [image_files[i] for i in self.indices]
        self.mask_files = [mask_files[i] for i in self.indices]

        if self.split == 'train':
            self.train_tiles = []
            self._generate_train_tiles()

    def _generate_train_tiles(self):
        """
        在训练时，将每张图像按重叠策略裁成若干个 tile，
        并把 (image_index, tile_index) 存到 self.train_tiles。
        """
        for idx in range(len(self.image_files)):
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

            image = Image.open(img_path).convert('RGB')
    
            image_np = np.array(image) 
            tiles, _, _ = tile_image(image_np, tile_size=self.crop_size, overlap=self.overlap)
            for tile_idx in range(len(tiles)):
                self.train_tiles.append((idx, tile_idx))

    def __len__(self):
        if self.split == 'train':
            return len(self.train_tiles)
        return len(self.image_files)

    def __getitem__(self, idx):
        if self.split == 'train':
            # 训练阶段返回单个 tile
            img_idx, tile_idx = self.train_tiles[idx]
            img_path = os.path.join(self.image_dir, self.image_files[img_idx])
            mask_path = os.path.join(self.mask_dir, self.mask_files[img_idx])

            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            # 先按整体应用 transform
            image = self.image_transform(image)      # Tensor [C, H, W]
            mask = self.mask_transform(mask)         # Tensor [1, H, W]
            mask = (mask > 0.5).float()

            image_np = image.permute(1, 2, 0).numpy()    # [H, W, C]
            mask_np = mask.squeeze(0).numpy()            # [H, W]

            img_tiles, _, _ = tile_image(image_np, tile_size=self.crop_size, overlap=self.overlap)
            mask_tiles, _, _ = tile_image(mask_np[..., None], tile_size=self.crop_size, overlap=self.overlap)

            tile_img_np = img_tiles[tile_idx]               # [tile_size, tile_size, C]
            tile_mask_np = mask_tiles[tile_idx][:, :, 0]     # [tile_size, tile_size]

            tile_img = torch.from_numpy(tile_img_np).permute(2, 0, 1).float()
            tile_mask = torch.from_numpy(tile_mask_np).unsqueeze(0).float()

            if self.augment:
                if random.random() > 0.5:
                    tile_img = TF.hflip(tile_img)
                    tile_mask = TF.hflip(tile_mask)
                if random.random() > 0.7:
                    tile_img = TF.vflip(tile_img)
                    tile_mask = TF.vflip(tile_mask)
                if random.random() > 0.5:
                    angle = random.uniform(-15, 15)
                    tile_img = TF.rotate(tile_img, angle)
                    tile_mask = TF.rotate(tile_mask, angle)

            return tile_img, tile_mask

        else:
            # 验证/测试阶段，返回整张图 + 文件名
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            image = self.image_transform(image)    # Tensor [C, H, W]
            mask = self.mask_transform(mask)       # Tensor [1, H, W]
            mask = (mask > 0.5).float()

            # 返回图像、mask 和 文件名（不含后缀）
            filename = os.path.splitext(self.image_files[idx])[0]
            return image, mask, filename
        



