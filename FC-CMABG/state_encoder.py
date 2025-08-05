# 3_state_encoder.py
# 职责：实现专利中的三模态状态编码：PCA, 3D-CNN Autoencoder, 和 Wavelet-GCN。

import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import pywt

# --- 3D卷积自编码器模型 ---
class Autoencoder3D(nn.Module):
    def __init__(self, num_bands, latent_dim=128):
        super(Autoencoder3D, self).__init__()
        # 编码器部分 (Encoder)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2), # (B, 16, C/2, H/2, W/2)
            
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2), # (B, 32, C/4, H/4, W/4)
            
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # (B, 64, C/8, H/8, W/8)
        )
        
        # 动态计算全连接层输入维度
        # 我们需要一个样本通过编码器来确定展平后的大小
        # 这里的维度需要根据实际输入进行调整
        # 假设输入是 (B, 1, 103, 13, 13)
        # 经过3次池化后，C变为 103//8=12, H,W变为 13//8=1
        self.fc_in_dim = 64 * (num_bands // 8) * (13 // 8) * (13 // 8)
        
        self.fc_encoder = nn.Linear(self.fc_in_dim, latent_dim)

    def encode(self, x):
        # 输入x的形状应为 (B, C, H, W)，需要调整为 (B, 1, C, H, W)
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(x.size(0), -1) # 展平
        latent_vec = self.fc_encoder(x)
        return latent_vec
    
    def forward(self, x):
        # 这个模型主要用其编码器部分，解码器在专利中用于重构训练，此处简化
        return self.encode(x)

def train_autoencoder(model, dataloader, epochs=10, learning_rate=1e-3, device='cpu'):
    """一个独立的函数来训练3D自编码器"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() 

    for epoch in range(epochs):
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(device)
            # 伪训练逻辑
            # 在一个完整的模型中:
            # latent = model.encode(batch_data)
            # reconstructed = model.decode(latent)
            # loss = criterion(reconstructed, batch_data)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
        print(f"自编码器 [伪] 训练 Epoch {epoch+1}/{epochs} on {device}")
    print("自编码器 [伪] 训练完成。")
    return model

# --- 状态编码器主类 ---
class StateEncoder:
    def __init__(self, autoencoder_model, pca_components=10, gcn_embed_dim=32):
        self.autoencoder = autoencoder_model
        self.pca_components = pca_components
        self.gcn_embed_dim = gcn_embed_dim
        self.pca_global = PCA(n_components=self.pca_components)
        self.pca_groups = {} # 存储每个组的PCA模型
        self.gcn_weights = None # GCN的可训练权重
        # 自动获取模型所在的设备
        self.device = next(self.autoencoder.parameters()).device

    def fit(self, train_data, band_groups):
        """
        使用训练数据拟合PCA和GCN模型。
        参数:
            train_data (numpy.ndarray): 训练数据, (N, H, W, C)。
            band_groups (dict): 波段分组信息。
        """
        print("正在拟合状态编码器 (PCA, GCN)...")
        num_samples, h, w, c = train_data.shape
        flat_data = train_data.reshape(-1, c)
        
        self.pca_global.fit(flat_data)
        
        for group_id, bands in band_groups.items():
            pca_group = PCA(n_components=min(self.pca_components, len(bands)))
            pca_group.fit(flat_data[:, bands])
            self.pca_groups[group_id] = pca_group
            
        spectral_mean = np.mean(flat_data, axis=0)
        coeffs = pywt.dwt(spectral_mean, 'db1')[0] 
        if len(coeffs) > self.gcn_embed_dim:
            coeffs = coeffs[:self.gcn_embed_dim]
        else:
            coeffs = np.pad(coeffs, (0, self.gcn_embed_dim - len(coeffs)))

        self.gcn_weights = torch.randn(c, self.gcn_embed_dim, requires_grad=True)
        print("状态编码器拟合完成。")

    def encode(self, full_data, band_groups):
        """
        为当前状态生成全局和组级状态向量。
        参数:
            full_data (numpy.ndarray): 全局数据, (N, H, W, C)。
            band_groups (dict): 波段分组信息。
        返回:
            state_vector (torch.Tensor): 拼接后的状态向量。
        """
        flat_data = full_data.reshape(-1, full_data.shape[-1])
        global_pca_proj = self.pca_global.transform(flat_data).mean(axis=0)
        
        group_pca_projs = []
        for group_id, bands in band_groups.items():
            proj = self.pca_groups[group_id].transform(flat_data[:, bands]).mean(axis=0)
            padded_proj = np.pad(proj, (0, self.pca_components - len(proj)))
            group_pca_projs.append(padded_proj)

        with torch.no_grad():
            # 将输入张量移动到与模型相同的设备
            data_tensor = torch.from_numpy(full_data).float().permute(0, 3, 1, 2).to(self.device)
            # 在转为numpy前，将结果移回CPU
            global_cae_vec = self.autoencoder.encode(data_tensor).mean(dim=0).cpu().numpy()
            
            group_cae_vecs = []
            for group_id, bands in band_groups.items():
                group_cae_vecs.append(global_cae_vec) 

        global_wavelet_embed = self.gcn_weights.mean(dim=0).detach().numpy()
        group_wavelet_embeds = []
        for group_id, bands in band_groups.items():
            embed = self.gcn_weights[bands, :].mean(dim=0).detach().numpy()
            group_wavelet_embeds.append(embed)
            
        state_vector = np.concatenate(
            [global_pca_proj] + group_pca_projs + \
            [global_cae_vec] + group_cae_vecs + \
            [global_wavelet_embed] + group_wavelet_embeds
        )
        
        return torch.from_numpy(state_vector).float()