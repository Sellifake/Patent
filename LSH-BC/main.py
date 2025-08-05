# main.py
# 职责：项目主入口，配置和启动整个流程。

import torch
import torch.optim as optim
import numpy as np

from data_loader import load_hsi_data, stratified_split, HSIDataset
from torch.utils.data import DataLoader
from lsh_model import LSH_BC_Net
from loss_function import JointLoss
from trainer import Trainer

# --- 1. 配置参数 ---
# 数据集路径 (请根据您的实际路径修改)
# 专利中使用萨利纳斯(Salinas)数据集
DATA_PATH = r'C:\Project\GIthub_Project\HSI_data\Salinas\Salinas_corrected.mat'
GT_PATH = r'C:\Project\GIthub_Project\HSI_data\Salinas\Salinas_gt.mat'

# 模型和训练超参数
PATCH_SIZE = 15           # S101: 空间窗口边长
TEST_RATIO = 0.9          # S2: 测试集比例 (专利中训练集占90%, 此处设测试集比例为0.1)
                          # 注意：为了快速演示，我们用90%做测试，10%做训练。实际应为0.1
TRAIN_RATIO = 0.1

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LAMBDA_BAND = 0.01        # S703: 联合损失中波段正则项的权重
ENTROPY_WEIGHT = 0.5      # S702: 熵约束的权重
BETA_MOMENTUM = 0.9       # S504: 波段权重动量更新的系数

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"将使用设备: {DEVICE}")

def main():
    # --- 2. 数据加载与预处理 (S1, S2) ---
    print("--- 步骤1: 数据加载与预处理 ---")
    hsi_data, gt_data = load_hsi_data(DATA_PATH, GT_PATH)
    
    # 标准化数据
    hsi_data = (hsi_data - np.mean(hsi_data)) / np.std(hsi_data)
    hsi_data = hsi_data.astype(np.float32)

    num_bands = hsi_data.shape[-1]
    num_classes = np.max(gt_data)

    train_indices, test_indices = stratified_split(gt_data, test_ratio=1-TRAIN_RATIO)
    print(f"训练集样本数: {len(train_indices)}, 测试集样本数: {len(test_indices)}")

    train_dataset = HSIDataset(hsi_data, gt_data, train_indices, patch_size=PATCH_SIZE)
    test_dataset = HSIDataset(hsi_data, gt_data, test_indices, patch_size=PATCH_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. 模型、损失函数、优化器初始化 ---
    print("\n--- 步骤2: 初始化模型、损失函数和优化器 ---")
    model = LSH_BC_Net(num_bands=num_bands, num_classes=num_classes, patch_size=PATCH_SIZE).to(DEVICE)
    loss_fn = JointLoss(lambda_band=LAMBDA_BAND, entropy_weight=ENTROPY_WEIGHT)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- 4. 训练与评估 ---
    print("\n--- 步骤3: 开始训练与评估 ---")
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        beta_momentum=BETA_MOMENTUM
    )
    
    trainer.run(epochs=EPOCHS)

    # --- 5. 结果展示 ---
    print("\n--- 最终波段权重 ---")
    final_band_weights = model.band_weights.detach().cpu().numpy()
    # 排序并显示最重要的N个波段
    top_n = 20
    top_indices = np.argsort(final_band_weights)[-top_n:]
    print(f"模型选择的最重要的 {top_n} 个波段 (索引):")
    print(np.sort(top_indices))
    print("\n对应的权重值:")
    print(final_band_weights[np.sort(top_indices)])

if __name__ == '__main__':
    main()