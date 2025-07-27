# agent.py
# --------
# 定义单个智能体及其行为

import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from models import QNetwork
from experience_replay import Experience


class Agent:
    def __init__(self, agent_id, state_dim, action_dim, lr, device):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.device = device
        
        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def select_action(self, state, epsilon):
        """
        使用Epsilon-greedy策略选择动作
        动作0: 取消选择, 动作1: 选择
        """
        if random.random() > epsilon:
            with torch.no_grad():
                # 返回Q值最高的动作
                return self.policy_net(state).argmax().item()
        else:
            return random.randrange(self.action_dim)

    def update_policy(self, memory, batch_size, gamma):
        """
        更新策略网络
        """
        if len(memory) < batch_size:
            return
        
        transitions = memory.sample(batch_size)
        batch = Experience(*zip(*transitions))

        # 过滤掉None的next_state
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device)

        # 计算Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 计算V(s_{t+1})
        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # 计算期望的Q值
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # 计算Huber损失
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_net(self):
        """更新目标网络权重"""
        self.target_net.load_state_dict(self.policy_net.state_dict())