# tile_utils.py

import math
import numpy as np


def tile_image(image, tile_size=512, overlap=0.5):
    """
    将输入图像按滑窗裁剪成多个 tile，并返回：
      - tiles: list[np.ndarray], 每个形状为 (tile_size, tile_size, C)
      - coords: list[(x1, y1, x2, y2)]
      - (pad_h, pad_w): 为了满足滑窗步长而在底部/右侧填充的像素数
    """
    stride = int(tile_size * (1 - overlap))  # 例如 512 * 0.25 = 128
    H, W, C = image.shape

    # 计算 padding 后的尺寸
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

    # 底部和右侧补零
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
    """
    将多个 tile 的预测结果融合回原图尺寸：
      tiles: list[np.ndarray], 每个 (tile_size, tile_size) 概率图
      pad_shape: (pad_h, pad_w)
      orig_shape: (orig_h, orig_w)
    返回：
      merged: (orig_h, orig_w) 概率图
    """
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
    gauss = np.exp(-(xv ** 2 + yv ** 2) / (2 * sigma **2))
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
