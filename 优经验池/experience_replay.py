# =============================================================================
# 文件名: experience_replay.py
# 描述: 实现基于GMM优化的经验池。
# (此版本已修正新生成样本的设备不一致问题)
# =============================================================================
import random
from collections import deque, namedtuple
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

# 定义一个经验元组，方便存储和访问
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state'))

class GMMExperienceReplay:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        """保存一个经验到经验池中。"""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """从经验池中随机采样一个批次的经验。"""
        return random.sample(self.memory, batch_size)

    # --- BUG修复: 添加device参数 ---
    def optimize_with_gmm(self, device, p1=0.3, p2=0.7):
        """
        创新点3的实现：使用高斯混合模型(GMM)优化经验池。
        p1: 高质量样本的比例
        p2: 从GMM生成并用于替换的样本比例
        """
        if len(self.memory) < self.capacity / 2: # 经验池未满时不执行优化
            return

        print("--- 开始GMM经验池优化 ---")
        
        # 1. 筛选高质量样本
        #    根据专利思想，这里我们简化为按奖励值排序来筛选
        all_samples = list(self.memory)
        all_samples.sort(key=lambda x: x.reward, reverse=True)
        
        num_high_quality = int(len(all_samples) * p1)
        high_quality_samples = all_samples[:num_high_quality]
        low_quality_samples = all_samples[num_high_quality:]

        if not high_quality_samples:
            print("--- 无高质量样本，跳过GMM优化 ---")
            return

        # 2. 准备数据并训练GMM
        #    我们将状态和奖励向量化，作为GMM的训练数据
        train_data = [np.concatenate([s.state.cpu().numpy(), np.array([s.reward])]) for s in high_quality_samples]
        train_data = np.array(train_data)
        
        # 根据专利，使用两个高斯分布来拟合数据
        gmm = GaussianMixture(n_components=2, random_state=42)
        try:
            gmm.fit(train_data)
        except ValueError as e:
            # 当所有高质量样本都相同时，GMM会拟合失败，这里捕获异常并跳过
            print(f"GMM拟合失败: {e}。跳过本轮优化。")
            return

        # 3. 生成模拟样本并替换低质量样本
        num_to_replace = int(len(low_quality_samples) * p2)
        if num_to_replace == 0:
            print("--- 无需替换，优化结束 ---")
            return
            
        generated_data, _ = gmm.sample(n_samples=num_to_replace)
        
        # 将生成的连续数据重新构建为经验元组
        new_samples = []
        state_dim = high_quality_samples[0].state.shape[0]
        for data_point in generated_data:
            # --- BUG修复: 将新创建的张量移动到正确的设备 ---
            state = torch.tensor(data_point[:state_dim], dtype=torch.float32).to(device)
            reward = data_point[state_dim]
            # 动作和下一状态是离散的，GMM无法直接生成。
            # 简化处理：从真实高质量样本中找到与之最相似的，并借用其动作和下一状态
            # 同样需要将借用的next_state确保在正确的设备上
            closest_sample = min(high_quality_samples, key=lambda s: np.linalg.norm(s.state.cpu().numpy() - state.cpu().numpy()))
            next_state_on_device = closest_sample.next_state.to(device) if closest_sample.next_state is not None else None
            new_samples.append(Experience(state, closest_sample.action, reward, next_state_on_device))

        # 构建新的经验池：高质量样本 + 生成的模拟样本
        new_memory = high_quality_samples + new_samples + low_quality_samples[num_to_replace:]
        
        self.memory = deque(new_memory, maxlen=self.capacity)
        print(f"--- GMM优化完成，当前经验池大小: {len(self.memory)} ---")

    def __len__(self):
        return len(self.memory)
