# 4_agent_and_env.py
# 职责：定义强化学习环境(Env)、智能体(Agent)及其定制化的奖励逻辑。

import gymnasium as gym  
from gymnasium import spaces
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

from a2c_models import ActorCriticNetwork

class ExperienceBuffer:
    """经验回放池，用于存储交互经验。"""
    def __init__(self, buffer_size=1000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

class A2CAgent:
    """A2C智能体，包含策略更新逻辑。"""
    def __init__(self, state_dim, action_dim, device='cpu', lr=1e-4, gamma=0.99, entropy_coef=0.01):
        # 接收device参数并移动网络
        self.device = device
        self.policy = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.buffer = ExperienceBuffer()

    def update_policy(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # 将Numpy数组转换为张量并移动到指定设备
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 评估动作
        log_probs, state_values, entropy = self.policy.evaluate_state(states, actions)
        
        # 计算目标价值
        with torch.no_grad():
            _, next_state_values = self.policy(next_states)
            target_values = rewards + self.gamma * next_state_values * (1 - dones)
            
        # 计算优势
        advantages = target_values - state_values

        # 计算损失
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(state_values, target_values.detach())
        entropy_loss = -entropy.mean()
        
        loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class BandGenerationEnv(gym.Env):
    """
    级联多智能体波段生成环境。
    """
    def __init__(self, train_data, band_groups, state_encoder):
        super(BandGenerationEnv, self).__init__()
        self.train_data = train_data
        self.band_groups = band_groups
        self.state_encoder = state_encoder
        self.num_groups = len(band_groups)
        
        self.action_space_agent0 = spaces.Discrete(self.num_groups)
        self.action_space_agent1 = spaces.Discrete(5) # +, -, *, /, concat
        self.action_space_agent2 = spaces.Discrete(self.num_groups)
        
        dummy_state = self.state_encoder.encode(self.train_data[:1], self.band_groups)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dummy_state.shape[0],), dtype=np.float32)

        self.reset()

    def reset(self, **kwargs): # gymnasium reset需要接受kwargs
        """重置环境状态。"""
        super().reset(seed=kwargs.get('seed')) # 处理可选的seed参数
        self.current_step = 0
        self.selected_bands = [] 
        self.first_group_idx = None
        self.operation_idx = None
        
        initial_state = self.state_encoder.encode(self.train_data, self.band_groups)
        return initial_state, {} # gymnasium reset返回 (obs, info)

    def step(self, action, agent_id):
        """
        环境执行一步。
        """
        reward = 0
        done = False
        
        if agent_id == 0:
            self.first_group_idx = action
            reward = 1.0
            
        elif agent_id == 1:
            self.operation_idx = action
            reward = 1.0 if action == 4 else 0.5
            
        elif agent_id == 2:
            second_group_idx = action
            new_feature_id = (self.first_group_idx, self.operation_idx, second_group_idx)
            self.selected_bands.append(new_feature_id)
            reward = np.random.rand() 
            done = True

        next_state = self.state_encoder.encode(self.train_data, self.band_groups)
        
        # gymnasium step返回 (obs, reward, terminated, truncated, info)
        terminated = done
        truncated = False 
        return next_state, reward, terminated, truncated, {}