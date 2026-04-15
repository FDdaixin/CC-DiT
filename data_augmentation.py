import cv2
import numpy as np
from PIL import Image

class GammaCLAHE(object):
    def __init__(self, gamma=1.0, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.gamma = gamma
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        # 将PIL图像转换为NumPy数组（RGB格式）
        img_np = np.array(img)

        # 转换为灰度图（使用你的公式）
        gray_np = 0.287 * img_np[:, :, 0] + 0.611 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]
        gray_np = gray_np.astype(np.uint8)

        # 伽马校正
        gamma_corrected = ((gray_np / 255.0) ** self.gamma) * 255
        gamma_corrected = gamma_corrected.astype(np.uint8)

        # CLAHE增强
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        clahe_img = clahe.apply(gamma_corrected)

        # 将单通道转回RGB三通道（复制灰度通道）
        clahe_rgb = np.stack([clahe_img] * 3, axis=2)

        # 转回PIL图像
        return Image.fromarray(clahe_img)