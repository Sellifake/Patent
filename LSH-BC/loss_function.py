# 3_loss_function.py
# 职责：实现专利S7中的联合优化损失函数。

import torch
import torch.nn as nn
import torch.nn.functional as F

class JointLoss(nn.Module):
    """
    联合优化损失，包含分类损失和波段正则损失。
    """
    def __init__(self, lambda_band=0.01, entropy_weight=0.5):
        """
        初始化损失函数。
        参数:
            lambda_band (float): 波段正则损失的权重系数。
            entropy_weight (float): 熵约束项在波段正则损失中的权重。
        """
        super().__init__()
        self.lambda_band = lambda_band
        self.entropy_weight = entropy_weight
        self.classification_loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels, band_weights):
        """
        计算总损失。
        参数:
            logits (torch.Tensor): 模型的分类输出。
            labels (torch.Tensor): 真实标签。
            band_weights (torch.Tensor): 模型的可学习波段权重向量 (作为logits)。
        返回:
            total_loss (torch.Tensor): 计算得到的总损失。
        """
        # S701: 计算分类损失
        classification_loss = self.classification_loss_fn(logits, labels)

        # S702: 计算波段正则损失
        
        # 【修改1】将原始权重通过Softmax转换为合法的概率分布
        # 这一步是解决nan问题的关键
        band_probs = F.softmax(band_weights, dim=0)
        
        # 稀疏项 L1 norm, 作用于原始权重以鼓励其值接近0
        sparsity_loss = torch.norm(band_weights, p=1)

        # 【修改2】在合法的概率分布 band_probs 上计算熵
        # 为防止log(0), 添加一个极小值
        entropy_loss = -torch.sum(band_probs * torch.log(band_probs + 1e-9))

        band_reg_loss = sparsity_loss + self.entropy_weight * entropy_loss
        
        # S703: 组合联合损失
        total_loss = classification_loss + self.lambda_band * band_reg_loss
        
        return total_loss