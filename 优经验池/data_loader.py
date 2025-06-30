# =============================================================================
# 文件名: data_loader.py
# 描述: 负责加载和预处理高光谱数据。
# (此版本已修正函数返回值数量)
# =============================================================================
import numpy as np
from scipy.io import loadmat

def load_pavia_university():
    """
    加载Pavia University数据集。
    这是一个公开的高光谱数据集，需要用户自行下载。

    返回:
        X (np.ndarray): 图像数据，形状为 (高度, 宽度, 波段数)
        y (np.ndarray): 真实标签，形状为 (高度, 宽度)
    """
    try:
        # 从 .mat 文件加载数据
        data = loadmat('PaviaU.mat')['paviaU']
        gt = loadmat('PaviaU_gt.mat')['paviaU_gt']
        return data, gt
    except FileNotFoundError:
        print("错误：请将 'PaviaU.mat' 和 'PaviaU_gt.mat' 文件下载到项目根目录。")
        exit()

def normalize_data(X):
    """
    对数据进行归一化处理，将像素值缩放到 [0, 1] 区间。
    """
    orig_shape = X.shape
    # 将三维数据展平为二维进行归一化
    X_reshaped = X.reshape(-1, X.shape[2])
    # 最小-最大归一化
    X_norm = (X_reshaped - np.min(X_reshaped, axis=0)) / (np.max(X_reshaped, axis=0) - np.min(X_reshaped, axis=0))
    # 恢复原始形状
    return X_norm.reshape(orig_shape)

def create_train_test_split(X, y, train_percentage=0.05, random_seed=42):
    """
    根据专利描述，为每个类别随机抽取指定百分比的样本作为训练集。
    这是复现精度的关键步骤之一。
    """
    np.random.seed(random_seed)
    height, width, bands = X.shape
    # 将数据和标签展平，便于索引
    X_flat = X.reshape(-1, bands)
    y_flat = y.flatten()

    train_indices = []
    test_indices = []
    
    # Pavia University数据集的类别标签从1到9
    for class_id in range(1, 10): 
        # 找到当前类的所有像素索引
        class_indices = np.where(y_flat == class_id)[0]
        np.random.shuffle(class_indices)
        
        # 计算训练样本数量
        num_train_samples = int(len(class_indices) * train_percentage)
        
        # 分配索引到训练集和测试集
        train_indices.extend(class_indices[:num_train_samples])
        test_indices.extend(class_indices[num_train_samples:])

    # 背景像素（类别0）不参与训练，全部划入测试集
    background_indices = np.where(y_flat == 0)[0]
    test_indices.extend(background_indices)
    
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    X_train = X_flat[train_indices]
    y_train = y_flat[train_indices]
    X_test = X_flat[test_indices]
    y_test = y_flat[test_indices]

    # --- BUG修复: 只返回4个必需的值 ---
    return X_train, y_train, X_test, y_test
    # --- 修复结束 ---
