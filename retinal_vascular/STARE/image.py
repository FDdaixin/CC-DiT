import os
import cv2
from tqdm import tqdm

# 设置路径
input_image_dir = r"./images"     # 原始图像
input_mask_dir  = r"./labels"     # 掩码图像
output_image_dir = r"./converted/images_png"              # 输出图像路径
output_mask_dir  = r"./converted/masks_png"               # 输出掩码路径

# 创建输出文件夹
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# 获取所有图像名
image_names = sorted([f for f in os.listdir(input_image_dir) if f.endswith(".ppm")])

for name in tqdm(image_names, desc="Converting STARE images"):
    base_name = os.path.splitext(name)[0]  # 例如 im0001

    # ---------- 原图 ----------
    image_path = os.path.join(input_image_dir, name)
    image = cv2.imread(image_path)  # 读取为 BGR
    if image is None:
        print(f"[Warning] Cannot read image: {image_path}")
        continue
    out_image_path = os.path.join(output_image_dir, base_name + ".png")
    cv2.imwrite(out_image_path, image)

    # ---------- 掩码 ----------
    # 假设掩码名为 im0001.ah.ppm 或 im0001_mask.png，请根据实际情况调整
    possible_mask_names = [
        base_name + ".ah.ppm", base_name + "_ah.ppm", base_name + "_mask.png", base_name + ".png"
    ]
    found_mask = False
    for mask_name in possible_mask_names:
        mask_path = os.path.join(input_mask_dir, mask_name)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                out_mask_path = os.path.join(output_mask_dir, base_name + "_mask.png")
                cv2.imwrite(out_mask_path, mask)
                found_mask = True
            break
    if not found_mask:
        print(f"[Warning] No mask found for {base_name}")
