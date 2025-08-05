# 2_lsh_model.py
# 职责：定义核心的LSH-BC网络模型，包括轻谱混合模块和交叉注意力更新器。

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightHybridBlock(nn.Module):
    """轻谱混合模块 (S4)"""
    def __init__(self, in_channels, out_channels_conv, patch_size=5, transformer_depth=1, num_heads=4):
        super().__init__()
        # S401: 提取局部卷积特征
        # 卷积核的深度(in_channels)将完全覆盖输入的光谱维，输出深度为1
        self.conv3d_local = nn.Conv3d(
            in_channels=1, 
            out_channels=out_channels_conv, 
            kernel_size=(in_channels, 3, 3), # 核深度等于输入深度
            padding=(0, 1, 1) # 谱维不填充，空间维填充
        )
        self.bn = nn.BatchNorm3d(out_channels_conv) # BN层在Conv后
        self.silu = nn.SiLU()

        # S402: 图像块划分参数
        self.patch_size = patch_size
        
        # S403: 全局关系建模 (Transformer)
        transformer_dim = out_channels_conv * patch_size * patch_size
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads, batch_first=True, dropout=0.1),
            num_layers=transformer_depth
        )

        # S404: 特征融合
        self.fusion_conv = nn.Conv2d(out_channels_conv * 2, out_channels_conv * 2, kernel_size=1)

    def forward(self, x):
        # x shape: (B, C, H, W)
        
        # 【修改2】调整前向传播的逻辑
        # 1. 准备5D输入，移除错误的permute
        x_3d = x.unsqueeze(1) # (B, 1, C, H, W) - 这是正确的输入格式

        # 2. 卷积 -> BN -> 激活
        local_feat_5d = self.conv3d_local(x_3d) # 输出: (B, Cout, 1, H, W)
        local_feat_5d = self.silu(self.bn(local_feat_5d))
        
        # 3. squeeze将深度维移除，得到4D张量用于后续处理
        local_feat = local_feat_5d.squeeze(2) # 输出: (B, Cout, H, W)

        # 全局关系建模 (后续逻辑不变)
        B, C_out, H, W = local_feat.shape
        tokens = F.unfold(local_feat, kernel_size=self.patch_size, stride=self.patch_size).permute(0, 2, 1)
        global_feat_tokens = self.transformer(tokens)
        
        global_feat_folded = F.fold(
            global_feat_tokens.permute(0, 2, 1),
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # 特征融合
        fused_feat = torch.cat([local_feat, global_feat_folded], dim=1)
        fused_feat = self.fusion_conv(fused_feat)

        return fused_feat, local_feat

class CrossAttentionUpdater(nn.Module):
    """交叉注意力模块 (S5)"""
    def __init__(self, high_level_channels, low_level_channels, num_heads=4):
        super().__init__()
        # 简化版实现，直接使用线性层学习从高层特征到低层通道重要性的映射
        self.importance_mapper = nn.Sequential(
            nn.Linear(high_level_channels, low_level_channels),
            nn.Sigmoid()
        )

    def forward(self, high_level_feat, low_level_feat):
        # high_level_feat (query): (B, C_high, H, W)
        # low_level_feat (key/value): (B, C_low, H, W)
        
        # 使用全局平均池化来获取每个特征图的全局表示
        global_high_level_info = F.adaptive_avg_pool2d(high_level_feat, 1).squeeze() # (B, C_high)

        # 映射到低维通道的重要性
        if global_high_level_info.dim() == 1: # 处理batch_size=1的情况
            global_high_level_info = global_high_level_info.unsqueeze(0)
            
        sample_importance = self.importance_mapper(global_high_level_info) # (B, C_low)

        # S503: 计算批次波段重要性
        batch_importance = torch.mean(sample_importance, dim=0) # (C_low,)
        
        return batch_importance


class LSH_BC_Net(nn.Module):
    """主网络模型"""
    def __init__(self, num_bands, num_classes, patch_size=15):
        super().__init__()
        # S3: 可学习波段权重向量
        self.band_weights = nn.Parameter(torch.ones(num_bands) / num_bands)
        
        # 主干网络
        # 注意：这里的out_channels_conv和low_level_channels必须等于num_bands
        # 才能使交叉注意力的输出维度与band_weights的维度匹配
        conv_out_channels = num_bands 
        
        self.hybrid_block = LightHybridBlock(in_channels=num_bands, out_channels_conv=conv_out_channels, patch_size=5)
        self.updater = CrossAttentionUpdater(high_level_channels=conv_out_channels*2, low_level_channels=conv_out_channels)
        
        # S6: 分类头
        self.cls_head = nn.Conv2d(conv_out_channels * 2, num_classes, kernel_size=1)
        self.center_idx = patch_size // 2

    def forward(self, x):
        # x shape: (B, C, H, W)
        # S302: 逐波段加权
        weighted_x = x * self.band_weights.view(1, -1, 1, 1)
        
        # S4: 轻谱混合特征提取
        fused_feat, local_feat = self.hybrid_block(weighted_x)
        
        # S5: 交叉注意波段筛选
        band_importance = self.updater(fused_feat, local_feat)
        
        # S6: 输出预测
        logits_map = self.cls_head(fused_feat) # (B, Num_Classes, H, W)
        # S602: 取中心像素的logits
        center_logits = logits_map[:, :, self.center_idx, self.center_idx]
        
        return center_logits, band_importance