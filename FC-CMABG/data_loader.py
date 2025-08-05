# 1_data_loader.py
# 职责：加载高光谱数据，提取数据立方体样本，并使用PyTorch的Dataset和DataLoader进行封装。

import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

def load_hsi_data(data_path, gt_path):
    """
    加载高光谱数据和对应的地面真实标签。
    参数:
        data_path (str): .mat格式的高光谱数据文件路径。
        gt_path (str): .mat格式的地面真实标签文件路径。
    返回:
        data (numpy.ndarray): 高光谱数据立方体。
        gt (numpy.ndarray): 地面真实标签二维矩阵。
    """
    data = sio.loadmat(data_path)
    gt = sio.loadmat(gt_path)
    
    # 从加载的字典中提取数据数组
    data_key = list(data.keys())[-1]
    gt_key = list(gt.keys())[-1]
    
    return data[data_key], gt[gt_key]

def create_patches(data, gt, patch_size=13):
    """
    从高光谱数据中为每个带标签的像素提取空间邻域块（patch）。
    参数:
        data (numpy.ndarray): HSI数据, 形状为 (H, W, C)。
        gt (numpy.ndarray): 地面真实数据, 形状为 (H, W)。
        patch_size (int): 邻域块的边长 (应为奇数)。
    返回:
        patches (list): 数据块列表。
        labels (list): 对应的标签列表。
    """
    h, w, c = data.shape
    padding = patch_size // 2
    
    # 对数据进行镜像填充，以处理边界像素
    padded_data = np.pad(data, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    
    patches = []
    labels = []
    
    # 遍历所有像素
    for i in range(h):
        for j in range(w):
            label = gt[i, j]
            # 忽略背景像素 (标签为0)
            if label != 0:
                # 提取数据块
                patch = padded_data[i:i + patch_size, j:j + patch_size, :]
                patches.append(patch)
                labels.append(label)
                
    return np.array(patches), np.array(labels)

class HSIDataset(Dataset):
    """高光谱数据集的PyTorch Dataset封装。"""
    def __init__(self, patches, labels):
        """
        初始化数据集。
        参数:
            patches (numpy.ndarray): 数据块数组, 形状 (N, H, W, C)。
            labels (numpy.ndarray): 标签数组, 形状 (N,)。
        """
        self.patches = patches
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        获取单个样本。
        返回:
            patch (torch.Tensor): 形状 (C, H, W)。
            label (torch.Tensor): 标签值。
        """
        patch = self.patches[idx]
        label = self.labels[idx]
        
        # 转换为PyTorch张量并调整维度以匹配 (C, H, W) 格式
        patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return patch_tensor, label_tensor

if __name__ == '__main__':
    # --- 这是一个测试该模块功能的示例 ---
    DATA_PATH = r'C:\Project\GIthub_Project\HSI_data\Pavia_University\PaviaU.mat'
    GT_PATH = r'C:\Project\GIthub_Project\HSI_data\Pavia_University\PaviaU_gt.mat'
    
    print("正在加载数据...")
    hsi_data, gt_data = load_hsi_data(DATA_PATH, GT_PATH)
    print(f"数据形状: {hsi_data.shape}, 标签形状: {gt_data.shape}")

    print("正在创建数据块...")
    patches, labels = create_patches(hsi_data, gt_data, patch_size=13)
    print(f"创建的数据块数量: {len(patches)}, 数据块形状: {patches[0].shape}, 标签数量: {len(labels)}")
    
    dataset = HSIDataset(patches, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 取一个批次的数据进行检查
    sample_batch, label_batch = next(iter(dataloader))
    print(f"一个批次的数据块形状: {sample_batch.shape}")
    print(f"一个批次的标签形状: {label_batch.shape}")