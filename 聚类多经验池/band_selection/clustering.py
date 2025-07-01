# band_selection/clustering.py
import numpy as np
from tqdm import tqdm

def correlation_entropy(b_i, b_j, sigma=1.0):
    """
    实现专利中的基于高斯核的相关熵计算公式 E[κ_σ(b_i, b_j)]
    这本质上是一种衡量两个波段向量相似度的核方法。
    参数:
        b_i, b_j: 两个波段的像素值向量, 形状 (num_pixels,)
        sigma: 高斯核的带宽参数
    返回:
        ce: 相关熵值
    """
    diff_sq = (b_i - b_j)**2
    kernel_vals = np.exp(-diff_sq / (2 * sigma**2))
    return np.mean(kernel_vals)

def greedy_clustering(data, num_groups, max_group_size=10):
    """
    实现专利中描述的基于相关熵和贪心算法的波段聚类方法
    参数:
        data: 高光谱数据，形状 (高, 宽, 波段数)
        num_groups: 期望的组数
        max_group_size: 每个组的最大波段数
    返回:
        groups: 一个包含各组波段索引的列表
    """
    h, w, num_bands = data.shape
    # 将数据重塑为 (像素数, 波段数)
    data_reshaped = data.reshape(-1, num_bands)
    
    print("开始计算波段间相关熵...")
    cor_entropies = []
    # 为了加速，我们可以只用部分像素来计算相关熵
    sample_indices = np.random.choice(h * w, size=min(h * w, 5000), replace=False)
    data_sample = data_reshaped[sample_indices, :]

    for i in tqdm(range(num_bands), desc="计算相关熵"):
        for j in range(i + 1, num_bands):
            ce = correlation_entropy(data_sample[:, i], data_sample[:, j])
            cor_entropies.append(((i, j), abs(ce)))
            
    # 按相关熵降序排序
    cor_entropies.sort(key=lambda x: x[1], reverse=True)
    print("相关熵计算和排序完成。")

    print("开始执行贪心聚类...")
    groups = [[] for _ in range(num_groups)]
    assigned_mask = np.zeros(num_bands, dtype=int) - 1 # -1表示未分配

    # 专利中的算法描述比较模糊，这里我们实现一个清晰的贪心逻辑：
    # 优先将最相似的波段对放入同一个组中
    for (b_i, b_j), _ in tqdm(cor_entropies, desc="贪心分组"):
        # 查找b_i和b_j所在的组
        group_i_idx, group_j_idx = assigned_mask[b_i], assigned_mask[b_j]
        
        if group_i_idx == -1 and group_j_idx == -1:
            # 如果两个都未分配，找到最小的组并尝试放入
            smallest_group_idx = min(range(num_groups), key=lambda i: len(groups[i]))
            if len(groups[smallest_group_idx]) + 2 <= max_group_size:
                groups[smallest_group_idx].extend([b_i, b_j])
                assigned_mask[b_i] = smallest_group_idx
                assigned_mask[b_j] = smallest_group_idx
        elif group_i_idx != -1 and group_j_idx == -1:
            # 如果i已分配，j未分配，尝试将j放入i所在的组
            if len(groups[group_i_idx]) < max_group_size:
                groups[group_i_idx].append(b_j)
                assigned_mask[b_j] = group_i_idx
        elif group_i_idx == -1 and group_j_idx != -1:
            # 如果j已分配，i未分配，尝试将i放入j所在的组
            if len(groups[group_j_idx]) < max_group_size:
                groups[group_j_idx].append(b_i)
                assigned_mask[b_i] = group_j_idx
    
    # 将剩余未分配的波段随机分配到未满的组中
    unassigned_bands = np.where(assigned_mask == -1)[0]
    for band_idx in unassigned_bands:
        # 找到一个未满的最小组
        available_groups = [i for i in range(num_groups) if len(groups[i]) < max_group_size]
        if not available_groups: break # 如果所有组都满了
        target_group_idx = min(available_groups, key=lambda i: len(groups[i]))
        groups[target_group_idx].append(band_idx)
        assigned_mask[band_idx] = target_group_idx

    print("波段聚类完成。")
    for i, group in enumerate(groups):
        print(f"  组 {i+1}: {len(group)}个波段 -> {group}")
        
    return groups