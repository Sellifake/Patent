# 2_feature_clustering.py
# 职责：实现专利中的核心创新点之一：融合光谱相似性与互信息的特征组划分。

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mutual_info_score

class FusionClusterer:
    """
    融合光谱均值和标签互信息的波段聚类器。
    """
    def __init__(self, n_clusters, alpha=1.0):
        """
        初始化聚类器。
        参数:
            n_clusters (int): 目标聚类簇的数量。
            alpha (float): 互信息差异项的权重系数。
        """
        self.n_clusters = n_clusters
        self.alpha = alpha

    def _calculate_spectral_mean(self, train_data):
        """
        计算每个波段在所有训练样本上的平均光谱响应。
        参数:
            train_data (numpy.ndarray): 训练数据, 形状 (N, H, W, C)。
        返回:
            spectral_mean (numpy.ndarray): 波段光谱均值向量, 形状 (C,)。
        """
        # (N, H, W, C) -> (N*H*W, C)
        num_samples, h, w, c = train_data.shape
        reshaped_data = train_data.reshape(-1, c)
        return np.mean(reshaped_data, axis=0)

    def _calculate_mutual_information(self, train_data, train_labels):
        """
        估计每个波段与类别标签之间的互信息。
        参数:
            train_data (numpy.ndarray): 训练数据, 形状 (N, H, W, C)。
            train_labels (numpy.ndarray): 训练标签, 形状 (N,)。
        返回:
            mi_vector (numpy.ndarray): 各波段的互信息向量, 形状 (C,)。
        """
        num_samples, h, w, c = train_data.shape
        # 将每个样本的空间维度展平
        pixel_responses = train_data.reshape(num_samples, h * w, c)
        # 取每个样本所有像素的平均响应作为该样本的波段响应
        sample_responses = np.mean(pixel_responses, axis=1) # 形状 (N, C)
        
        mi_vector = np.zeros(c)
        for i in range(c):
            # 为计算互信息，需要将连续的波段响应离散化
            band_responses = sample_responses[:, i]
            # 使用10个箱子进行离散化
            discretized_band = np.digitize(band_responses, np.histogram(band_responses, bins=10)[1])
            mi_vector[i] = mutual_info_score(train_labels, discretized_band)
            
        return mi_vector

    def cluster_bands(self, train_data, train_labels):
        """
        执行融合聚类。
        参数:
            train_data (numpy.ndarray): 训练数据, 形状 (N, H, W, C)。
            train_labels (numpy.ndarray): 训练标签, 形状 (N,)。
        返回:
            band_groups (dict): 包含波段分组的字典, key为簇ID, value为波段索引列表。
        """
        print("开始进行融合聚类...")
        c = train_data.shape[-1] # 波段数
        
        # S3.2: 计算波段光谱均值
        spectral_mean = self._calculate_spectral_mean(train_data)
        
        # S3.3: 计算波段-标签互信息
        mi_vector = self._calculate_mutual_information(train_data, train_labels)

        # S3.4 & S3.5: 构造综合距离矩阵
        distance_matrix = np.zeros((c, c))
        # 计算所有波段均值差的中值 (用于构造高斯径向基相似度)
        median_diff = np.median(np.abs(spectral_mean[:, None] - spectral_mean))
        
        for i in range(c):
            for j in range(i, c):
                # 计算光谱相似度
                spectral_sim = np.exp(-np.linalg.norm(spectral_mean[i] - spectral_mean[j])**2 / (2 * median_diff**2))
                # 计算互信息差异
                mi_diff = np.abs(mi_vector[i] - mi_vector[j])
                # 定义综合距离
                dist = (1 - spectral_sim) + self.alpha * mi_diff
                distance_matrix[i, j] = distance_matrix[j, i] = dist

        # S3.6: 层次聚类
        # 使用预计算的距离矩阵进行聚合聚类
        agg_cluster = AgglomerativeClustering(
            n_clusters=self.n_clusters, 
            metric='precomputed', # 在新版本中为 affinity
            linkage='average'
        )
        cluster_labels = agg_cluster.fit_predict(distance_matrix)
        
        # 整理聚类结果
        band_groups = {i: [] for i in range(self.n_clusters)}
        for band_idx, cluster_id in enumerate(cluster_labels):
            band_groups[cluster_id].append(band_idx)
            
        print("融合聚类完成。")
        return band_groups

if __name__ == '__main__':
    # --- 这是一个测试该模块功能的示例 ---
    # 假设我们有一些随机生成的数据
    # 在实际使用中，这里应传入从1_data_loader.py加载的真实数据
    DUMMY_TRAIN_DATA = np.random.rand(100, 5, 5, 103) # 100个样本, 5x5大小, 103个波段
    DUMMY_TRAIN_LABELS = np.random.randint(1, 10, 100) # 9个类别
    NUM_CLUSTERS = 20
    
    clusterer = FusionClusterer(n_clusters=NUM_CLUSTERS)
    band_groups = clusterer.cluster_bands(DUMMY_TRAIN_DATA, DUMMY_TRAIN_LABELS)
    
    print(f"\n聚类结果 (前5组):")
    for i in range(min(5, NUM_CLUSTERS)):
        print(f"组 {i}: {band_groups[i]}")