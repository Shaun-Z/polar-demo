import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import cv2
import math

# 设置计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 保持 CIFAR-10 原始大小
    transforms.ToTensor()
])

# 加载测试集
dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
raw_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=None)
classes = dataset.classes  # CIFAR-10 类别名称

# 加载 ResNet-18
model = resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 10)  # 适配 CIFAR-10
model = model.to(device)
model.eval()

# 图像分块（像素块特征）
def extract_patch_features(image, patch_size=4):
    """将图像拆分为多个小块（patch），并转换为二值特征"""
    H, W, C = image.shape
    assert H == W == 32, "只适用于 32x32 CIFAR-10 图像"
    
    # 转换为灰度图（平均三通道）
    gray = image.mean(axis=2)

    h, w = patch_size, patch_size
    features = []

    for i in range(0, H, h):
        for j in range(0, W, w):
            patch = gray[i:i+h, j:j+w]
            patch_mean = patch.mean()
            feature = 1 if patch_mean > 0.5 else 0  # 二值化
            features.append(feature)

    return np.array(features, dtype=int)

# 极化变换
def polar_transform(bits):
    """对特征进行极化变换"""
    bits = bits.copy().astype(int)
    N = bits.shape[0]
    assert (N & (N-1)) == 0, "特征数量必须是2的幂（如 16, 32, 64）"

    step = 1
    while step < N:
        for i in range(0, N, 2*step):
            for j in range(i, i+step):
                bits[j] = bits[j] ^ bits[j+step]
        step *= 2
    return bits

# 计算特征重要性（基于 Shannon 熵）
def estimate_feature_capacity(dataset, patch_size=4):
    """计算特征的重要性（基于 Shannon 熵）"""
    sample_img, _ = dataset[0]
    img_arr = np.array(sample_img) / 255.0
    u = extract_patch_features(img_arr, patch_size=patch_size)
    N = len(u)

    count_ones = np.zeros(N, dtype=int)
    total = 0

    for idx in range(len(dataset)):
        img, _ = dataset[idx]
        img = np.array(img) / 255.0
        u = extract_patch_features(img, patch_size=patch_size)
        x = polar_transform(u)
        count_ones += x
        total += 1

    capacities = []
    for k in range(len(count_ones)):
        p = count_ones[k] / total
        entropy = 0 if p == 0 or p == 1 else -(p * math.log2(p) + (1-p) * math.log2(1-p))
        capacities.append(entropy)

    return np.array(capacities)

# 归一化特征重要性
def normalize_scores(scores):
    """归一化特征重要性分数到 [0, 1]"""
    min_val, max_val = scores.min(), scores.max()
    return (scores - min_val) / (max_val - min_val + 1e-10)

# 在原图上可视化特征重要性
def visualize_feature_importance(image, scores, patch_size=4):
    """将特征重要性分数可视化叠加到原图上"""
    image = np.array(image) / 255.0  # 归一化
    scores = scores.reshape(32 // patch_size, 32 // patch_size)  # 变换为 8x8
    
    heatmap = cv2.resize(scores, (32, 32), interpolation=cv2.INTER_LINEAR)  # 放大到原图大小
    heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)  # 伪彩色处理
    overlay = (1.0 * image + 0.0 * heatmap / 255.0)  # 叠加原图和热图
    
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Feature Importance Overlay (Polar Code)")
    plt.show()

# 运行完整流程
img_pil, label = raw_dataset[1]
img_arr = np.array(img_pil)

# 提取特征 & 计算重要性
patch_feats = extract_patch_features(img_arr, patch_size=4)
polar_feats = polar_transform(patch_feats)
feature_capacities = estimate_feature_capacity(raw_dataset, patch_size=4)
normalized_capacities = normalize_scores(feature_capacities)

# 可视化特征重要性
visualize_feature_importance(img_arr, normalized_capacities, patch_size=4)