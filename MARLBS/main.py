# =============================================================================
# 文件名: main.py
# 描述: 主程序入口，配置并运行整个流程。

# =============================================================================
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

from data_loader import load_pavia_university, normalize_data, create_train_test_split
from models import Autoencoder1, Autoencoder2, GCN
from trainer import Trainer

def main():
    # --- 1. 设置随机种子以保证实验的可复现性 ---
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- 2. 配置超参数 ---
    # 大部分参数严格遵循专利实施例五中的设置
    config = {
        'num_episodes': 150,           # 训练回合数
        'batch_size': 32,              # 批量大小
        'lr': 0.01,                    # 学习率
        'gamma': 0.99,                 # 折扣因子
        'memory_capacity': 200,        # 经验池大小
        'target_update_freq': 10,      # 目标网络更新频率
        'gmm_optimize_freq': 50,       # GMM优化频率
        'gmm_p1': 0.30,                # GMM高质量样本比例(PaviaU)
        'epsilon_start': 1.0,          # Epsilon初始探索率
        'epsilon_end': 0.01,           # Epsilon最终探索率
        'epsilon_decay': 0.99,         # Epsilon衰减率
        'action_dim': 2,               # 动作维度 (选择/不选择)
    }

    # --- 3. 加载和预处理数据 ---
    print("加载数据...")
    X, y = load_pavia_university()
    X = normalize_data(X)
    
    # 专利中使用5%的带标签样本作为训练集
    X_train, y_train, X_test, y_test = create_train_test_split(X, y, train_percentage=0.05, random_seed=seed)
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    num_bands = X.shape[2]

    # --- 4. 初始化模型 ---
    # 固定状态向量的维度，这是Q网络输入所必需的
    ae1_latent_dim = 16
    ae2_latent_dim = 8
    gcn_out_dim = 32
    # 使用三个状态向量长度的最大值作为最终的固定维度
    state_dim = max(49, ae2_latent_dim, gcn_out_dim) # ae2的输出是聚合后的，所以是ae2_latent_dim
    config['state_dim'] = state_dim

    models = {
        'ae1': Autoencoder1(input_dim=X_train.shape[0], latent_dim=ae1_latent_dim),
        'ae2': Autoencoder2(input_dim=ae1_latent_dim, latent_dim=ae2_latent_dim),
        'gcn': GCN(n_features=ae1_latent_dim, n_hidden2=gcn_out_dim), 
    }
    # --- 修复结束 ---

    # --- 5. 实例化并运行训练器 ---
    marl_trainer = Trainer(config, X_train, y_train, X_test, y_test, models)
    
    train_accuracies = marl_trainer.train()
    marl_trainer.evaluate()

    # --- 6. 绘制训练过程中的ACC变化图 ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies)
    plt.title("训练过程中选择波段子集的分类精度 (Train ACC)")
    plt.xlabel("训练回合 (Episode)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("training_accuracy_plot.png")
    plt.show()
    print("训练精度变化图已保存为 'training_accuracy_plot.png'")


if __name__ == '__main__':
    main()
