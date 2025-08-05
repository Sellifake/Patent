# 1_data_loader.py
# 职责：加载高光谱数据，构建数据立方体，并实现分层随机抽样来划分数据集。

import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

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
    
    data_key = list(data.keys())[-1]
    gt_key = list(gt.keys())[-1]
    
    return data[data_key], gt[gt_key]

def stratified_split(gt, test_ratio=0.9):
    """
    根据地面真实标签进行分层随机抽样，返回训练和测试样本的索引。
    专利S2步骤的实现。
    参数:
        gt (numpy.ndarray): 地面真实标签, 形状为 (H, W)。
        test_ratio (float): 测试集所占的比例。
    返回:
        train_indices (list): 训练样本的坐标元组 (y, x) 列表。
        test_indices (list): 测试样本的坐标元组 (y, x) 列表。
    """
    h, w = gt.shape
    pixels, labels = [], []
    # 收集所有带标签的像素及其坐标
    for i in range(h):
        for j in range(w):
            if gt[i, j] != 0: # 忽略背景
                pixels.append((i, j))
                labels.append(gt[i, j])

    pixels = np.array(pixels)
    labels = np.array(labels)

    # 使用sklearn进行分层抽样
    try:
        train_indices_flat, test_indices_flat, _, _ = train_test_split(
            range(len(labels)), labels, test_size=test_ratio, random_state=42, stratify=labels
        )
    except ValueError:
        # 如果某个类别样本太少，无法分层，则进行普通随机抽样
        train_indices_flat, test_indices_flat = train_test_split(
            range(len(labels)), test_size=test_ratio, random_state=42
        )

    train_indices = pixels[train_indices_flat]
    test_indices = pixels[test_indices_flat]

    return train_indices.tolist(), test_indices.tolist()

class HSIDataset(Dataset):
    """
    高光谱数据集的PyTorch Dataset封装。
    它在被调用时才从完整数据中动态提取数据立方体，以节省内存。
    """
    def __init__(self, data, gt, indices, patch_size=15):
        """
        初始化数据集。
        参数:
            data (numpy.ndarray): 完整的高光谱数据, 形状 (H, W, C)。
            gt (numpy.ndarray): 完整的地面真实数据, 形状 (H, W)。
            indices (list): 数据集包含的样本坐标 (y, x) 列表。
            patch_size (int): 邻域块的边长 (应为奇数)。
        """
        self.data = data
        self.gt = gt
        self.indices = indices
        self.patch_size = patch_size
        self.padding = patch_size // 2
        # 对数据进行镜像填充
        self.padded_data = np.pad(data, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='reflect')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        获取单个样本。
        """
        y, x = self.indices[idx]
        
        # 提取数据块
        patch = self.padded_data[y:y + self.patch_size, x:x + self.patch_size, :]
        label = self.gt[y, x]
        
        # 转换为PyTorch张量并调整维度以匹配 (C, H, W) 格式
        patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1)
        # 标签从1开始，为适应CrossEntropyLoss，减1使其从0开始
        label_tensor = torch.tensor(label - 1, dtype=torch.long)
        
        return patch_tensor, label_tensor