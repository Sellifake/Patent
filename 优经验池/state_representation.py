# =============================================================================
# 文件名: state_representation.py
# 描述: 实现专利中核心的“融合状态”计算逻辑。
# (此版本已修正GCN输入特征的生成方式)
# =============================================================================
import torch
import numpy as np

def get_descriptive_stats(data_tensor):
    """辅助函数：计算单个张量的7个描述性统计量。"""
    if data_tensor.numel() == 0:
        return torch.zeros(7, device=data_tensor.device)
    
    std, mean = torch.std_mean(data_tensor)
    min_val = torch.min(data_tensor)
    max_val = torch.max(data_tensor)
    # 计算四分位数
    q = torch.quantile(data_tensor, torch.tensor([0.25, 0.5, 0.75], device=data_tensor.device))
    return torch.stack([mean, std, min_val, max_val, q[0], q[1], q[2]])

def get_meta_stats_state(selected_data, device):
    """
    创新点1的实现：特征子空间的元描述统计方法。
    """
    if selected_data.shape[1] == 0:
        return torch.zeros(49, device=device)
        
    # 步骤1: 计算每个特征（列）的描述性统计数据
    desc_stats_matrix = torch.stack([get_descriptive_stats(selected_data[:, i]) for i in range(selected_data.shape[1])])
    
    # 步骤2: 计算元描述性统计矩阵
    meta_desc_stats_matrix = torch.stack([get_descriptive_stats(desc_stats_matrix[:, i]) for i in range(desc_stats_matrix.shape[1])])
    
    # 步骤3: 连接成固定长度为49的状态向量
    return meta_desc_stats_matrix.T.flatten()

def get_autoencoder_state(selected_data, ae1, ae2, device):
    """
    创新点1的实现：基于自编码器的特征子空间深度表示方法。
    """
    if selected_data.shape[1] == 0:
        final_output_dim = ae2.encoder[-1].out_features
        return torch.zeros(final_output_dim, device=device)
        
    with torch.no_grad():
        # 步骤1: 使用第一个AE对每个特征列进行编码
        latent_vectors_c = [ae1.encoder(selected_data[:, i]) for i in range(selected_data.shape[1])]
        potential_matrix = torch.stack(latent_vectors_c)
        
        # 步骤2: 使用第二个AE对潜在矩阵的每一行进行编码
        latent_vectors_r = [ae2.encoder(potential_matrix[i, :]) for i in range(potential_matrix.shape[0])]
        static_encoding_matrix = torch.stack(latent_vectors_r)

        # 步骤3: 对结果进行池化以获得固定长度的向量
        final_vector = torch.mean(static_encoding_matrix, dim=0)
            
        return final_vector

def get_gcn_state(selected_data, models, device):
    """
    创新点1的实现：基于动态图的图卷积网络方法。
    """
    gcn = models['gcn']
    ae1 = models['ae1']

    if selected_data.shape[1] <= 1: 
        return torch.zeros(gcn.gc2.linear.out_features, device=device)

    # 步骤1: 计算特征相关性图的邻接矩阵
    corr_matrix = torch.corrcoef(selected_data.T)
    adj = corr_matrix.to_sparse()
    
    # --- BUG修复: 为GCN节点创建固定长度的特征向量 ---
    # 不再使用torch.eye，而是使用AE1为每个节点生成特征
    with torch.no_grad():
        features = torch.stack([ae1.encoder(selected_data[:, i]) for i in range(selected_data.shape[1])])
    # --- 修复结束 ---

    with torch.no_grad():
        latent_representation = gcn(features, adj)
    
    # 步骤3: 对每个特征的潜在表示取平均值，得到固定长度的向量
    return torch.mean(latent_representation, dim=0)

def get_fused_state(selected_data, models, device, fixed_state_dim):
    """
    将三种状态表示向量相加，形成最终的融合状态。
    """
    s1 = get_meta_stats_state(selected_data, device)
    s2 = get_autoencoder_state(selected_data, models['ae1'], models['ae2'], device)
    # --- BUG修复: 修正函数调用 ---
    s3 = get_gcn_state(selected_data, models, device)
    # --- 修复结束 ---
    
    # 关键步骤：由于三个状态向量长度可能不同，需要填充至固定长度再相加
    s1_padded = torch.nn.functional.pad(s1, (0, fixed_state_dim - len(s1)))
    s2_padded = torch.nn.functional.pad(s2, (0, fixed_state_dim - len(s2)))
    s3_padded = torch.nn.functional.pad(s3, (0, fixed_state_dim - len(s3)))
    
    return s1_padded + s2_padded + s3_padded
