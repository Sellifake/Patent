# main.py
import numpy as np

# [修改] 导入所有需要的函数
from utils.data_loader import load_pavia_university, create_patches, split_data
from classification_model.train_classifier import train_and_evaluate
from band_selection.clustering import greedy_clustering
from band_selection.marl_training import train_marl_for_band_selection


# --- 参数配置 ---
# 数据集和模型参数
PATCH_SIZE = 15          # Patch大小，必须为奇数
TRAIN_RATIO = 0.1        # 训练集占总样本的比例

# 分类模型训练参数
EPOCHS = 50              # 训练轮数
BATCH_SIZE = 64          # 每批次样本数
LEARNING_RATE = 0.0005   # 学习率

# 波段选择参数
NUM_BAND_GROUPS = 12     # 期望将波段聚成的组数
MAX_GROUP_SIZE = 10      # 每个波段组的最大容量
TARGET_BAND_COUNT = 30   # 期望选择出的波段数量（此参数目前主要用于教师策略）


def main_classification_only():
    """
    主函数：仅运行分类流程（使用PCA降维）
    """
    print("---------- 步骤1: 加载和预处理数据 (PCA方案) ----------")
    hsi_data, gt_data = load_pavia_university()
    num_classes = np.max(gt_data)
    print(f"数据集加载成功，图像尺寸: {hsi_data.shape}, 标签类别数: {num_classes}")
    
    patches, labels = create_patches(
        hsi_data, gt_data, 
        patch_size=PATCH_SIZE, 
        use_pca=True, 
        pca_components=TARGET_BAND_COUNT
    )

    X_train, X_test, y_train, y_test = split_data(patches, labels, train_ratio=TRAIN_RATIO)

    print("\n---------- 步骤2: 训练和评估分类网络 ----------")
    train_and_evaluate(
        X_train, y_train, X_test, y_test,
        num_classes=num_classes,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    print("\n仅分类流程结束！")


def main_full_pipeline():
    """
    运行包含波段选择和分类的完整流程
    """
    print("---------- 步骤1: 加载数据 ----------")
    hsi_data, gt_data = load_pavia_university()
    num_classes = np.max(gt_data)

    print("\n---------- 步骤2: 基于相关熵的波段聚类 ----------")
    # 注意：波段聚类和MARL训练非常耗时，可以只运行一次，然后保存结果
    band_groups = greedy_clustering(hsi_data, num_groups=NUM_BAND_GROUPS, max_group_size=MAX_GROUP_SIZE)

    print("\n---------- 步骤3: 多智能体强化学习波段选择 ----------")
    selected_bands = train_marl_for_band_selection(
        hsi_data, gt_data, band_groups, target_band_count=TARGET_BAND_COUNT
    )
    
    if not selected_bands or len(selected_bands) < 5:
        print("\n强化学习波段选择未能选出足够的波段，切换到PCA方案进行分类。")
        main_classification_only()
        return

    print(f"\n---------- 步骤4: 使用选定波段({len(selected_bands)}个)进行最终分类 ----------")
    # 使用选定的波段创建数据
    data_selected = hsi_data[:, :, selected_bands]
    patches, labels = create_patches(data_selected, gt_data, patch_size=PATCH_SIZE, use_pca=False)
    X_train, X_test, y_train, y_test = split_data(patches, labels, train_ratio=TRAIN_RATIO)
    
    # 训练和评估
    train_and_evaluate(
        X_train, y_train, X_test, y_test,
        num_classes=num_classes,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    print("\n完整流程结束！")


if __name__ == "__main__":
    # 运行仅分类的版本进行快速验证
    # main_classification_only() 
    
    print("\n<<<<<<<<<< 开始运行完整流程 (强化学习+分类) >>>>>>>>>>")
    main_full_pipeline()