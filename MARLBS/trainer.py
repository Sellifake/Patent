# =============================================================================
# 文件名: trainer.py
# 描述: 负责编排整个训练和评估流程。
# (此版本已修正GMM优化时的设备传递问题)
# =============================================================================
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from agent import Agent
from marl_environment import BandSelectionEnv
from experience_replay import GMMExperienceReplay
from state_representation import get_fused_state
from utils import calculate_metrics

class Trainer:
    def __init__(self, config, X_train, y_train, X_test, y_test, models):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"当前使用设备: {self.device}")

        self.X_train_np, self.y_train_np = X_train, y_train
        self.X_test_np, self.y_test_np = X_test, y_test
        self.X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)

        self.num_bands = X_train.shape[1]
        self.state_dim = config['state_dim']
        
        self.models = {name: model.to(self.device) for name, model in models.items()}
        self.env = BandSelectionEnv(X_train, y_train)
        self.memory = GMMExperienceReplay(config['memory_capacity'])
        
        # 创建所有智能体
        self.agents = [
            Agent(i, self.state_dim, config['action_dim'], config['lr'], self.device)
            for i in range(self.num_bands)
        ]
        
        self.selected_bands = list(range(self.num_bands)) # 初始时选择所有波段
        self.epsilon = config['epsilon_start']
        self.final_selected_bands = []

    def train(self):
        print("--- 开始训练 ---")
        overall_accuracies = []

        for episode in range(self.config['num_episodes']):
            
            # 1. 获取当前状态
            current_state_data = self.X_train[:, self.selected_bands] if self.selected_bands else torch.empty(self.X_train.shape[0], 0, device=self.device)
            state = get_fused_state(current_state_data, self.models, self.device, self.state_dim)
            
            # 2. 所有智能体并行决策
            actions = {}
            for agent in self.agents:
                actions[agent.agent_id] = agent.select_action(state, self.epsilon)
            
            # 3. 根据动作更新选择的波段
            next_selected_bands = [i for i, act in actions.items() if act == 1]
            
            # 4. 环境反馈奖励
            rewards, current_acc = self.env.calculate_reward(next_selected_bands, len(next_selected_bands))
            overall_accuracies.append(current_acc)
            
            # 5. 获取下一状态
            next_state_data = self.X_train[:, next_selected_bands] if next_selected_bands else torch.empty(self.X_train.shape[0], 0, device=self.device)
            next_state = get_fused_state(next_state_data, self.models, self.device, self.state_dim)
            
            # 6. 存储经验并更新策略
            for agent_id, action in actions.items():
                reward = rewards[agent_id]
                self.memory.push(state, action, reward, next_state)
            
            for agent in self.agents:
                agent.update_policy(self.memory, self.config['batch_size'], self.config['gamma'])

            # 7. 更新状态和探索率
            self.selected_bands = next_selected_bands
            self.epsilon = max(self.config['epsilon_end'], self.epsilon * self.config['epsilon_decay'])
            
            # 8. 定期更新目标网络
            if episode % self.config['target_update_freq'] == 0:
                for agent in self.agents:
                    agent.update_target_net()
            
            # 9. 定期执行GMM优化经验池
            if episode > 0 and episode % self.config['gmm_optimize_freq'] == 0:
                # --- BUG修复: 传递device参数到GMM优化函数 ---
                self.memory.optimize_with_gmm(self.device, p1=self.config['gmm_p1'], p2=1-self.config['gmm_p1'])
                
            if episode % 10 == 0:
                print(f"回合 {episode}/{self.config['num_episodes']}, "
                      f"选中波段数: {len(self.selected_bands)}, "
                      f"训练集ACC: {current_acc:.4f}, Epsilon: {self.epsilon:.4f}")

        print("--- 训练结束 ---")
        self.final_selected_bands = self.selected_bands
        return overall_accuracies

    def evaluate(self):
        print("\n--- 开始评估 ---")
        if not self.final_selected_bands:
            print("警告: 训练后没有选择任何波段，无法进行评估。")
            return
            
        print(f"最终选择的波段数量: {len(self.final_selected_bands)}")
        print(f"最终选择的波段索引: {sorted(self.final_selected_bands)}")
        
        # 使用选择的波段在测试集上进行最终分类
        X_train_selected = self.X_train_np[:, self.final_selected_bands]
        X_test_selected = self.X_test_np[:, self.final_selected_bands]
        
        final_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        final_classifier.fit(X_train_selected, self.y_train_np)
        
        y_pred = final_classifier.predict(X_test_selected)
        
        # 评估时需要过滤掉背景像素（标签为0）
        test_mask = self.y_test_np != 0
        
        oa, aa, kappa, per_class_acc = calculate_metrics(self.y_test_np[test_mask], y_pred[test_mask])
        
        print("\n--- 最终评估结果 ---")
        print(f"总体精度 (OA): {oa:.4f}  ")
        print(f"平均精度 (AA): {aa:.4f}")
        print(f"Kappa系数:    {kappa:.4f}")
        print("\n各类别精度:")
        for i, acc in enumerate(per_class_acc):
            print(f"  类别 {i+1}: {acc:.4f}")

        return oa, aa, kappa
