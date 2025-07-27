# classification_model/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFAW_Net(nn.Module):
    """
    基于自适应权重的多尺度特征融合分类网络 (FFAW-Net)
    结合3D CNN和2D CNN来提取空谱特征
    [修正版]：修复了通道数计算错误，并动态计算全连接层输入大小。
    """
    def __init__(self, num_bands, num_classes, patch_size=15):
        super(FFAW_Net, self).__init__()
        
        # --- 3D CNN部分 ---
        # 专利中描述了不同大小的卷积核，这里我们实现该结构
        # 输入形状: (N, 1, Bands, H, W)
        # 注意：这里的padding设计是为了在卷积后保持光谱和空间维度不变
        self.conv3d_layers = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # --- 2D CNN部分 ---
        # [修复]：这里的in_channels应该是 32 * num_bands，因为3D卷积的光谱维度未变
        self.conv2d_layers = nn.Sequential(
            nn.Conv2d(32 * num_bands, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # --- 全连接分类层 ---
        # [修复]：动态计算展平后的大小，避免硬编码错误
        self.flatten_size = self._get_flatten_size(num_bands, patch_size)
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _get_flatten_size(self, num_bands, patch_size):
        """
        通过一次虚拟的前向传播来自动计算卷积层输出的特征维度
        """
        # 创建一个与真实输入形状相同的虚拟张量
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, num_bands, patch_size, patch_size)
            x = self.conv3d_layers(dummy_input)
            # 展平光谱维度
            x = x.view(x.size(0), -1, x.size(3), x.size(4))
            x = self.conv2d_layers(x)
            # 计算展平后的大小
            flatten_size = x.numel() // x.shape[0]
            print(f"动态计算的全连接层输入大小为: {flatten_size}")
            return flatten_size

    def forward(self, x):
        # x 初始形状: (N, H, W, C), C是波段数
        # 调整为 (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # 增加一个通道维度以适应3D卷积: (N, 1, C, H, W)
        x = x.unsqueeze(1)
        
        # 3D卷积
        x = self.conv3d_layers(x)
        
        # 展平光谱维度，为2D卷积做准备
        # 形状从 (N, 32, D, H, W) -> (N, 32 * D, H, W)
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        
        # 2D卷积
        x = self.conv2d_layers(x)
        
        # 展平以输入全连接层
        x = x.view(x.size(0), -1)
        
        # 全连接层分类
        output = self.fc(x)
        
        return output