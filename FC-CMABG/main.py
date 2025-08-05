# main.py
# 职责：项目主入口，串联所有模块，执行完整的训练和生成流程。

import torch
import numpy as np
from sklearn.model_selection import train_test_split

# 从各个模块导入所需的类和函数
from data_loader import load_hsi_data, create_patches, HSIDataset
from feature_clustering import FusionClusterer
from state_encoder import StateEncoder, Autoencoder3D, train_autoencoder
from agent_and_env import BandGenerationEnv, A2CAgent

# --- 1. 配置参数 ---
DATA_PATH = r'C:\Project\GIthub_Project\HSI_data\Pavia_University\PaviaU.mat'
GT_PATH = r'C:\Project\GIthub_Project\HSI_data\Pavia_University\PaviaU_gt.mat'

PATCH_SIZE = 13
TEST_RATIO = 0.9
NUM_CLUSTERS = 20        
PCA_COMPONENTS = 10      
LATENT_DIM = 64          
GCN_EMBED_DIM = 32       
NUM_EPISODES = 500       
BATCH_SIZE = 64          
LEARNING_RATE = 1e-4     

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"将使用设备: {DEVICE}")


def main():
    # --- 2. 数据加载与预处理 (S1, S2) ---
    print("--- 步骤1: 数据加载与预处理 ---")
    hsi_data, gt_data = load_hsi_data(DATA_PATH, GT_PATH)
    patches, labels = create_patches(hsi_data, gt_data, patch_size=PATCH_SIZE)
    
    # 标准化数据
    patches = (patches - np.mean(patches)) / np.std(patches)
    # 显式转换为float32，防止numpy运算自动提升为float64
    patches = patches.astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(
        patches, labels, test_size=TEST_RATIO, random_state=42, stratify=labels
    )
    print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
    
    train_dataset = HSIDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 3. 光谱特征自适应聚类 (S3) ---
    print("\n--- 步骤2: 光谱特征自适应聚类 ---")
    fusion_clusterer = FusionClusterer(n_clusters=NUM_CLUSTERS)
    band_groups = fusion_clusterer.cluster_bands(X_train, y_train)
    print(f"已将 {hsi_data.shape[-1]} 个波段划分为 {NUM_CLUSTERS} 组。")

    # --- 4. 联合光谱-空域状态编码 (S4) ---
    print("\n--- 步骤3: 状态编码器设置 ---")
    autoencoder = Autoencoder3D(num_bands=hsi_data.shape[-1], latent_dim=LATENT_DIM)
    autoencoder = train_autoencoder(autoencoder, train_loader, epochs=5, device=DEVICE)

    state_encoder = StateEncoder(
        autoencoder_model=autoencoder,
        pca_components=PCA_COMPONENTS,
        gcn_embed_dim=GCN_EMBED_DIM
    )
    state_encoder.fit(X_train, band_groups)
    
    # --- 5. 级联多智能体特征生成 (S5, S6) ---
    print("\n--- 步骤4: 强化学习训练 ---")
    env = BandGenerationEnv(X_train, band_groups, state_encoder)
    
    state_dim = env.observation_space.shape[0]
    
    # 创建Agent时传入device
    agent0 = A2CAgent(state_dim, env.action_space_agent0.n, device=DEVICE, lr=LEARNING_RATE)
    agent1 = A2CAgent(state_dim, env.action_space_agent1.n, device=DEVICE, lr=LEARNING_RATE)
    agent2 = A2CAgent(state_dim, env.action_space_agent2.n, device=DEVICE, lr=LEARNING_RATE)

    for episode in range(NUM_EPISODES):
        # 适配gymnasium的reset返回值
        state, _ = env.reset()
        done = False
        
        # 选择动作时，将state张量移动到正确的设备
        state_gpu = state.to(DEVICE)
        
        action0, _, _ = agent0.policy.select_action(state_gpu)
        # 适配gymnasium的step返回值
        next_state, reward0, terminated, truncated, _ = env.step(action0, agent_id=0)
        done = terminated or truncated
        agent0.buffer.add(state.numpy(), action0, reward0, next_state.numpy(), done)
        state = next_state # 更新状态

        state_gpu = state.to(DEVICE)
        action1, _, _ = agent1.policy.select_action(state_gpu)
        next_state, reward1, terminated, truncated, _ = env.step(action1, agent_id=1)
        done = terminated or truncated
        agent1.buffer.add(state.numpy(), action1, reward1, next_state.numpy(), done)
        state = next_state # 更新状态

        state_gpu = state.to(DEVICE)
        action2, _, _ = agent2.policy.select_action(state_gpu)
        final_state, reward2, terminated, truncated, _ = env.step(action2, agent_id=2)
        done = terminated or truncated
        agent2.buffer.add(state.numpy(), action2, reward2, final_state.numpy(), done)

        if (episode + 1) % 10 == 0: 
            agent0.update_policy(batch_size=min(len(agent0.buffer), BATCH_SIZE//2))
            agent1.update_policy(batch_size=min(len(agent1.buffer), BATCH_SIZE//2))
            agent2.update_policy(batch_size=min(len(agent2.buffer), BATCH_SIZE//2))

        if (episode + 1) % 50 == 0:
            print(f"Episode [{episode+1}/{NUM_EPISODES}], "
                  f"最新生成特征: {env.selected_bands[-1] if env.selected_bands else 'N/A'}")

    print("\n--- 训练完成 ---")
    print(f"最终生成的优化波段组合 ({len(env.selected_bands)}个):")
    for i, band_combo in enumerate(env.selected_bands):
        print(f"特征 {i+1}: (组{band_combo[0]}, 运算{band_combo[1]}, 组{band_combo[2]})")


if __name__ == '__main__':
    main()