# 5_a2c_models.py
# 职责：定义通用的A2C（优势演员-评论家）网络结构。

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCriticNetwork(nn.Module):
    """
    一个通用的演员-评论家网络模型。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        初始化网络。
        参数:
            state_dim (int): 输入状态的维度。
            action_dim (int): 输出动作空间的维度。
            hidden_dim (int): 隐藏层的神经元数量。
        """
        super(ActorCriticNetwork, self).__init__()
        
        # 共享网络层
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 演员网络头 (Actor Head)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1) # 输出动作概率
        )
        
        # 评论家网络头 (Critic Head)
        self.critic_head = nn.Linear(hidden_dim, 1) # 输出状态价值

    def forward(self, state):
        """
        前向传播。
        参数:
            state (torch.Tensor): 输入的状态。
        返回:
            action_probs (torch.Tensor): 动作的概率分布。
            state_value (torch.Tensor): 状态的价值估计。
        """
        shared_features = self.shared_layer(state)
        action_probs = self.actor_head(shared_features)
        state_value = self.critic_head(shared_features)
        return action_probs, state_value

    def select_action(self, state):
        """
        根据当前策略选择一个动作。
        参数:
            state (torch.Tensor): 输入的状态。
        返回:
            action (int): 选择的动作。
            log_prob (torch.Tensor): 该动作的对数概率。
            entropy (torch.Tensor): 策略的熵。
        """
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy

    def evaluate_state(self, state, action):
        """
        评估一个状态和动作。
        参数:
            state (torch.Tensor): 状态。
            action (torch.Tensor): 动作。
        返回:
            log_prob (torch.Tensor): 动作的对数概率。
            state_value (torch.Tensor): 状态的价值。
            entropy (torch.Tensor): 策略的熵。
        """
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, state_value, entropy