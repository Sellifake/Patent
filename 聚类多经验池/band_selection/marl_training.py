# band_selection/marl_training.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from tqdm import tqdm

# --- 强化学习环境 ---
class BandSelectionEnv:
    """
    多智能体波段选择环境
    这个环境模拟了专利中的核心流程：智能体选择波段组合，环境返回一个奖励值。
    """
    def __init__(self, hsi_data, gt_data, band_groups):
        self.hsi_data = hsi_data.reshape(-1, hsi_data.shape[2])
        self.gt_data = gt_data.flatten()
        
        # 只使用有标签的像素进行奖励计算
        labeled_mask = self.gt_data > 0
        self.hsi_data = self.hsi_data[labeled_mask]
        self.gt_data = self.gt_data[labeled_mask]

        self.band_groups = band_groups
        self.num_bands = self.hsi_data.shape[1]
        self.num_agents = len(band_groups)
        
        self.state = np.zeros(self.num_bands, dtype=np.float32)

    def reset(self):
        """重置环境状态"""
        self.state = np.zeros(self.num_bands, dtype=np.float32)
        return self.state

    def step(self, actions):
        """
        执行一步动作
        参数:
            actions (list): 包含每个智能体动作的列表。
                            每个动作是一个二进制向量，决定其组内哪些波段被选中。
        返回:
            next_state (np.array): 新的状态
            reward (float): 该动作获得的奖励
            done (bool): 是否结束 (在这个问题中恒为False)
        """
        # 根据所有智能体的动作，更新全局状态
        next_state = np.copy(self.state)
        for agent_idx, action_vec in enumerate(actions):
            group = self.band_groups[agent_idx]
            for i, selected in enumerate(action_vec):
                band_idx = group[i]
                next_state[band_idx] = selected

        self.state = next_state
        selected_band_indices = np.where(self.state > 0)[0]
        
        # --- 奖励计算 ---
        # 专利中使用随机森林分类精度作为奖励，这在每一步都计算会非常耗时。
        # 我们使用一个高效的代理指标：类间可分性（Fisher准则简化版）作为奖励。
        # 目标是最大化类间距离，同时最小化类内方差。
        if len(selected_band_indices) < 2:
            reward = -1.0 # 惩罚选择过少的波段
        else:
            reward = self._calculate_separability_reward(selected_band_indices)
            # 添加对波段数量的轻微惩罚，鼓励选择更少的波段
            reward -= 0.01 * len(selected_band_indices)

        return self.state, reward, False

    def _calculate_separability_reward(self, selected_bands):
        """计算类间可分性作为奖励"""
        data = self.hsi_data[:, selected_bands]
        labels = self.gt_data
        
        class_means = []
        class_scatters = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            class_data = data[labels == label]
            class_mean = np.mean(class_data, axis=0)
            class_means.append(class_mean)
            # 类内散度
            scatter = np.mean(np.sum((class_data - class_mean)**2, axis=1))
            class_scatters.append(scatter)

        # 类间散度
        total_mean = np.mean(data, axis=0)
        between_class_scatter = 0
        for i, class_mean in enumerate(class_means):
            n_class = np.sum(labels == unique_labels[i])
            between_class_scatter += n_class * np.sum((class_mean - total_mean)**2)

        within_class_scatter = np.mean(class_scatters)
        
        # 奖励是类间散度与类内散度的比值
        reward = between_class_scatter / (within_class_scatter + 1e-6)
        return float(reward)


# --- 智能体定义 ---
class DQNAgent:
    """
    DQN智能体，每个智能体负责一个波段组
    """
    def __init__(self, global_state_dim, group_size, device):
        self.device = device
        self.group_size = group_size
        # 动作空间大小为2^group_size，每个动作是一个二进制向量
        self.action_space_size = 2**group_size
        
        self.q_network = self._create_network(global_state_dim, self.action_space_size).to(device)
        self.target_network = self._create_network(global_state_dim, self.action_space_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

    def _create_network(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def select_action(self, state, epsilon):
        """使用Epsilon-Greedy策略选择动作"""
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
        else:
            action_idx = random.randrange(self.action_space_size)
        
        # 将动作索引解码为二进制向量
        action_vec = [int(x) for x in bin(action_idx)[2:].zfill(self.group_size)]
        return action_vec, action_idx
    
    def learn(self, batch, gamma):
        """从经验回放中学习"""
        states, actions, rewards, next_states = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        # 计算当前Q值
        q_values = self.q_network(states).gather(1, actions)
        
        # 计算目标Q值
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + gamma * next_q_values

        # 计算损失并更新网络
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# --- 经验回放池 ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def soft_update(target, source, tau):
    """软更新目标网络参数"""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


# --- 主训练函数 ---
def train_marl_for_band_selection(hsi_data, gt_data, band_groups, target_band_count):
    """
    训练多智能体进行波段选择
    """
    print("\n---------- 开始多智能体强化学习波段选择 ----------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = BandSelectionEnv(hsi_data, gt_data, band_groups)
    num_agents = env.num_agents
    
    # 警告：如果组内波段数过多，动作空间会爆炸。这里限制一下，否则无法运行。
    for i, group in enumerate(band_groups):
        if len(group) > 8:
            print(f"警告: 组 {i} 的大小为 {len(group)}，超过8。为保证可运行，将截断为前8个波段。")
            band_groups[i] = group[:8]

    agents = [DQNAgent(env.num_bands, len(group), device) for group in band_groups]
    
    # --- 多样经验池 ---
    teacher_buffer = ReplayBuffer(1000)
    agent_buffer = ReplayBuffer(5000)

    # --- 1. 生成教师经验 ---
    print("生成教师经验...")
    # 简单的教师策略：随机选择接近目标数量的波段
    for _ in range(200):
        state = env.reset()
        # 模拟一个动作：随机选择一些波段
        random_actions = []
        random_action_indices = []
        for agent in agents:
            action_vec, action_idx = agent.select_action(state, epsilon=1.0) # 完全随机
            random_actions.append(action_vec)
            random_action_indices.append(action_idx)
        
        next_state, reward, _ = env.step(random_actions)
        # 只保存高质量的经验
        if reward > 0: 
            for i in range(num_agents):
                teacher_buffer.add(state, random_action_indices[i], reward, next_state)
    print(f"教师经验池已填充，包含 {len(teacher_buffer)} 条高质量经验。")


    # --- 2. 智能体探索与学习 ---
    print("开始智能体探索与学习...")
    batch_size = 128
    gamma = 0.99
    tau = 0.005 # 软更新系数
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 500
    
    best_reward = -float('inf')
    best_band_selection = []
    
    num_episodes = 200 # 训练回合数
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)
        
        # 每个回合走20步
        for step in range(20):
            actions = []
            action_indices = []
            for agent in agents:
                action_vec, action_idx = agent.select_action(state, epsilon)
                actions.append(action_vec)
                action_indices.append(action_idx)
            
            next_state, reward, _ = env.step(actions)
            episode_reward += reward

            # 将经验存入各自的智能体经验池
            for i in range(num_agents):
                agent_buffer.add(state, action_indices[i], reward, next_state)
            
            state = next_state
            
            # --- 从多样经验池中采样学习 ---
            if len(agent_buffer) > batch_size:
                # 动态beta，初期多依赖教师
                beta = max(0.1, 1.0 - episode / (num_episodes * 0.7))
                
                num_teacher_samples = int(batch_size * beta)
                num_agent_samples = batch_size - num_teacher_samples
                
                # 分别从两个池中采样
                teacher_batch = teacher_buffer.sample(num_teacher_samples) if len(teacher_buffer) > num_teacher_samples else []
                agent_batch = agent_buffer.sample(num_agent_samples)
                
                # 合并batch并为每个智能体学习
                full_batch = teacher_batch + agent_batch
                for i in range(num_agents):
                    # 提取该智能体的动作
                    agent_specific_batch = [(s, a_all[i], r, ns) for s, a_all, r, ns in zip(*zip(*full_batch))]
                    agents[i].learn(agent_specific_batch, gamma)

        # 更新目标网络
        if episode % 5 == 0:
            for agent in agents:
                soft_update(agent.target_network, agent.q_network, tau)

        # 记录本回合找到的最佳波段组合
        current_selection = np.where(state > 0)[0]
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_band_selection = current_selection

        print(f"回合 {episode+1}/{num_episodes} | Epsilon: {epsilon:.2f} | 最终奖励: {episode_reward:.2f} | 最佳奖励: {best_reward:.2f} | 选中波段数: {len(best_band_selection)}")
    
    print("---------- 强化学习训练完成 ----------")
    print(f"找到的最佳波段组合 (共{len(best_band_selection)}个): {best_band_selection.tolist()}")
    return best_band_selection.tolist()