# marl_environment.py
# -------------------
# 定义多智能体强化学习环境，核心是奖励计算

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

class BandSelectionEnv:
    def __init__(self, X_train, y_train):
        """
        初始化环境
        Args:
            X_train (np.ndarray): 训练数据
            y_train (np.ndarray): 训练标签
        """
        self.X_train = X_train
        self.y_train = y_train
        self.num_bands = X_train.shape[1]
        # 环境是随机森林分类器 
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.prev_acc = 0.0

    def calculate_reward(self, selected_band_indices, num_active_agents):
        """
        计算奖励，包含共享奖励和竞争奖励 
        """
        if not selected_band_indices:
            self.prev_acc = 0.0
            return {i: 0.0 for i in range(self.num_bands)}, 0.0

        X_subset = self.X_train[:, selected_band_indices]
        
        # 训练分类器并获取ACC
        self.classifier.fit(X_subset, self.y_train)
        y_pred = self.classifier.predict(X_subset)
        current_acc = accuracy_score(self.y_train, y_pred)

        # 1. 计算共享奖励 
        acc_improvement = current_acc - self.prev_acc
        shared_reward = acc_improvement / num_active_agents if num_active_agents > 0 else 0

        # 2. 计算竞争奖励 
        #    通过排列重要性计算每个波段的贡献度
        importances = {}
        original_score = current_acc
        for i, band_idx in enumerate(selected_band_indices):
            X_shuffled = X_subset.copy()
            shuffle(X_shuffled[:, i], random_state=42)
            shuffled_score = accuracy_score(self.y_train, self.classifier.predict(X_shuffled))
            importances[band_idx] = original_score - shuffled_score
        
        total_importance = sum(importances.values())
        
        # 计算每个智能体的最终奖励
        rewards = {}
        for band_idx in range(self.num_bands):
            if band_idx in selected_band_indices:
                # 归一化权重 
                weight = importances[band_idx] / total_importance if total_importance > 0 else 0
                competitive_reward = weight * current_acc
                # 最终奖励 = 共享 + 竞争 
                rewards[band_idx] = shared_reward + competitive_reward
            else:
                # 未被选中的智能体奖励为0
                rewards[band_idx] = 0.0
        
        self.prev_acc = current_acc
        return rewards, current_acc