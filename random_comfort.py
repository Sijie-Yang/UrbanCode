import os
import random
import numpy as np
from urbancode.svi import comfort

# 图片和模型路径
IMAGE_DIR = "urbancode/tests/data/svi_images"
MODEL_PATH = "urbancode/tests/data/perception_models/best_model.pth"

# 获取所有图片文件
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print("没有找到图片文件！")
    exit(1)

# 随机选择一张图片
random_image = random.choice(image_files)
image_path = os.path.join(IMAGE_DIR, random_image)

print(f"随机选择的图片: {random_image}")
print(f"图片路径: {image_path}")
print(f"模型路径: {MODEL_PATH}")

# 创建一些初始特征（19个特征）
initial_features = np.zeros(52)

# 预测舒适度分数
try:
    score = comfort(
        img_path=image_path,
        model_path=MODEL_PATH,
        initial_features=initial_features
    )
    print(f"舒适度分数: {score:.4f}")
except Exception as e:
    print(f"预测失败: {e}") 