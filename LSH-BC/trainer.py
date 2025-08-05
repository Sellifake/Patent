# 4_trainer.py
# 职责：实现完整的训练、验证和权重动态更新逻辑。

import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score

class Trainer:
    """
    模型训练与评估器。
    """
    def __init__(self, model, loss_fn, optimizer, train_loader, test_loader, device, beta_momentum=0.9):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.beta_momentum = beta_momentum # S504: 动量更新系数
        self.best_accuracy = 0.0

    def train_epoch(self):
        """执行一个训练轮次。"""
        self.model.train()
        total_loss = 0
        for data, labels in tqdm(self.train_loader, desc="Training"):
            data, labels = data.to(self.device), labels.to(self.device)

            # S3-S6: 前向传播
            logits, band_importance = self.model(data)

            # S504: 动量更新可学习波段权重
            with torch.no_grad():
                current_weights = self.model.band_weights.data
                # 归一化重要性得分
                importance_norm = band_importance / (band_importance.sum() + 1e-9)
                # 执行动量更新
                new_weights = self.beta_momentum * current_weights + (1 - self.beta_momentum) * importance_norm
                # 重新归一化权重以保持和为1的性质 (可选，但推荐)
                self.model.band_weights.data = new_weights / new_weights.sum()

            # S7: 计算联合损失
            loss = self.loss_fn(logits, labels, self.model.band_weights)

            # S8: 反向传播与优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)

    def evaluate(self):
        """执行一次评估。"""
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data, labels in tqdm(self.test_loader, desc="Evaluating"):
                data, labels = data.to(self.device), labels.to(self.device)
                logits, _ = self.model(data)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        oa = accuracy_score(all_labels, all_preds)
        # 此处可以添加AA和Kappa的计算
        return oa

    def run(self, epochs):
        """启动完整的训练-评估流程。"""
        for epoch in range(epochs):
            avg_loss = self.train_epoch()
            accuracy = self.evaluate()
            
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Test OA: {accuracy:.4f}")
            
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                # 此处可以添加保存最佳模型权重的逻辑
                print(f"  -> New best accuracy! Saved model.")
        
        print(f"\nTraining finished. Best Test OA: {self.best_accuracy:.4f}")