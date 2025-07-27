# models.py
# ---------
# 定义所有PyTorch模型：Q网络, 自编码器, GCN

import torch
import torch.nn as nn
import torch.nn.functional as F

# 注：输出层改为2，代表选择/不选择两个动作的Q值
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim=2):
        super(QNetwork, self).__init__()
        # 中间层节点数设为32，比8更常用
        self.layer1 = nn.Linear(state_dim, 64)
        self.layer2 = nn.Linear(64, 32) 
        self.layer3 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# 第一个自编码器，用于对每个特征列进行编码 
class Autoencoder1(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(Autoencoder1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# 第二个自编码器，用于对潜在矩阵的行进行编码 
class Autoencoder2(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(Autoencoder2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# 图卷积网络层 
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.sparse.mm(adj, support)
        return output

# GCN模型，两层，节点数为128和32 
class GCN(nn.Module):
    def __init__(self, n_features, n_hidden1=128, n_hidden2=32):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_features, n_hidden1)
        self.gc2 = GraphConvolution(n_hidden1, n_hidden2)
    
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x