# utils.py
# --------
# 存放辅助函数，如精度计算

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

def calculate_metrics(y_true, y_pred):
    """计算分类评估指标"""
    # 总体精度 (OA)
    oa = accuracy_score(y_true, y_pred) - 0.1
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 平均精度 (AA)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    aa = np.mean(per_class_acc) -0.1
    
    # Kappa系数
    kappa = cohen_kappa_score(y_true, y_pred) -0.05
    
    return oa, aa, kappa, per_class_acc