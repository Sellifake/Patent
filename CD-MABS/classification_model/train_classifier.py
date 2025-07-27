# classification_model/train_classifier.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from .network import FFAW_Net # 从同级目录下的network.py导入模型
from utils.metrics import calculate_metrics # 从utils模块导入评估函数

def train_and_evaluate(X_train, y_train, X_test, y_test, num_classes, epochs, batch_size, learning_rate):
    """
    训练并评估FFAW-Net模型
    """
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    # 获取数据维度信息
    num_samples_train, patch_size, _, num_bands = X_train.shape
    num_samples_test = X_test.shape[0]

    # 将数据转换为PyTorch Tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = FFAW_Net(num_bands=num_bands, num_classes=num_classes, patch_size=patch_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("---------- 开始训练分类模型 ----------")
    for epoch in range(epochs):
        model.train() # 设置为训练模式
        train_loss = 0.0
        train_correct = 0
        
        # 使用tqdm显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计损失和准确率
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

            # 更新进度条信息
            pbar.set_postfix({'loss': loss.item()})
        
        # epoch_loss = train_loss / num_samples_train
        # epoch_acc = train_correct.double() / num_samples_train
        # print(f"Epoch {epoch+1}/{epochs} -> 训练损失: {epoch_loss:.4f}, 训练准确率: {epoch_acc:.4f}")

    print("---------- 训练完成，开始评估模型 ----------")
    model.eval() # 设置为评估模式
    all_preds = []
    all_labels = []
    
    with torch.no_grad(): # 在评估阶段不计算梯度
        for inputs, labels in tqdm(test_loader, desc="评估中"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 计算并打印各项指标
    # target_names = [f'Class {i+1}' for i in range(num_classes)]
    # calculate_metrics(np.array(all_labels), np.array(all_preds), target_names)
    
    return model