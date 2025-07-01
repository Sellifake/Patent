# utils/data_loader.py
import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm

# 数据集URL
DATA_URL = "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat"
GT_URL = "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat"
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "PaviaU.mat")
GT_PATH = os.path.join(DATA_DIR, "PaviaU_gt.mat")

def download_file(url, path):
    """带进度条的文件下载函数"""
    if os.path.exists(path):
        print(f"{os.path.basename(path)} 已存在，跳过下载。")
        return
    print(f"正在下载 {os.path.basename(path)}...")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("下载错误，文件可能不完整。")

def load_pavia_university():
    """
    加载Pavia University数据集。如果本地不存在，则自动下载。
    返回:
        data (np.array): 高光谱数据立方体 (高, 宽, 波段数)
        gt (np.array): 真实地物标签 (高, 宽)
    """
    download_file(DATA_URL, DATA_PATH)
    download_file(GT_URL, GT_PATH)
    
    data = loadmat(DATA_PATH)['paviaU']
    gt = loadmat(GT_PATH)['paviaU_gt']
    
    return data, gt

def apply_pca(X, num_components):
    """
    应用PCA降维
    参数:
        X: 原始数据，形状 (样本数, 特征数)
        num_components: 降维后的主成分数量
    返回:
        X_pca: 降维后的数据
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(X)
    print(f"PCA降维完成，原始维度: {X.shape[1]}, 降维后维度: {X_pca.shape[1]}")
    return X_pca

def create_patches(data, gt, patch_size=15, use_pca=False, pca_components=30):
    """
    为每个带标签的像素创建数据立方体（patch）
    参数:
        data (np.array): 高光谱数据 (高, 宽, 波段数)
        gt (np.array): 真实标签 (高, 宽)
        patch_size (int): patch的边长 (必须是奇数)
        use_pca (bool): 是否在创建patch前进行PCA降维
        pca_components (int): PCA主成分数量
    返回:
        patches (np.array): 数据patch数组，形状 (样本数, patch_size, patch_size, 波段数)
        labels (np.array): 对应的标签数组
    """
    h, w, c = data.shape
    
    if use_pca:
        # 重塑数据以应用PCA
        data_reshaped = data.reshape(-1, c)
        data_pca = apply_pca(data_reshaped, num_components=pca_components)
        data = data_pca.reshape(h, w, -1)
        print(f"数据已通过PCA降维至 {data.shape[2]} 个波段。")
    
    # 对数据进行padding，以处理边界像素
    margin = patch_size // 2
    padded_data = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), 'reflect')
    
    # 找到所有带标签的像素（忽略背景类别0）
    labeled_indices = np.argwhere(gt > 0)
    
    patches = []
    labels = []
    
    for y, x in tqdm(labeled_indices, desc="创建数据Patches"):
        patch = padded_data[y:y + patch_size, x:x + patch_size, :]
        patches.append(patch)
        labels.append(gt[y, x])
        
    # 将标签从1-9调整为0-8，以符合PyTorch的类别索引
    return np.array(patches), np.array(labels) - 1

def split_data(patches, labels, train_ratio=0.1, random_state=42):
    """
    按类别分层抽样划分训练集和测试集
    参数:
        patches: 数据patch数组
        labels: 标签数组
        train_ratio: 训练集比例
    返回:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        patches, labels, 
        train_size=train_ratio, 
        random_state=random_state,
        stratify=labels  # 关键：按类别分层抽样
    )
    print(f"数据划分完成。训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
    return X_train, X_test, y_train, y_test