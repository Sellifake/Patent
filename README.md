MARLBS - Multi-Agent Reinforcement Learning Band Selection - 基于融合状态表示和经验池优化的多智能体高光谱波段选择方法
CD-MABS- A Multi-Agent Band Selection Method Based on Clustering and Diverse Experience Pools-基于聚类和多样经验池的多智能体波段选择方法

# 基于多智能体强化学习的高光谱波段选择方法

本项目包含了两种基于多智能体强化学习（MARL）的高光谱图像（HSI）波段选择方法，旨在解决高光谱数据的高维度和高冗余性问题。两种方法均将每个波段（或波段组）视为一个智能体，通过不同的策略学习选择最优的波段组合。

1.  **MARLBS (Multi-Agent Reinforcement Learning Band Selection)**: 一种基于 **融合状态表示** 和 **GMM优化经验池** 的多智能体方法。
2.  **CD-MABS (Clustering and Diverse-experience-pools-based Multi-Agent Band Selection)**: 一种基于 **波段聚类** 和 **多样化经验池** 的多智能体方法。

---

## 项目一: MARLBS (融合状态表示与GMM优化)

该方法的核心思想是为智能体提供一个信息极其丰富的“融合状态”，并利用GMM优化经验池来加速学习进程。

### 核心创新点

* **融合状态表示 (`state_representation.py`)**: 为了让智能体更好地感知环境，项目融合了三种不同的状态表示方法，共同构成一个全面的状态向量：
    * **元描述统计**: 通过对当前选中波段子空间数据的多层描述性统计（均值、方差、四分位数等），获得一个固定长度为49的特征向量。
    * **深度自编码器表示**: 使用两个级联的自编码器（Autoencoder）对波段子空间数据进行深度特征提取和编码，捕捉其非线性特征。
    * **动态图卷积网络表示**: 将波段间的相关性构建成一个动态图，并利用图卷积网络（GCN）学习其结构化特征。
* **混合式奖励机制 (`marl_environment.py`)**: 为了平衡整体目标和个体贡献，环境设计了一种混合奖励：
    * **共享奖励**: 基于整个波段子集分类精度的提升，所有被选中的智能体共享这份奖励，鼓励协作。
    * **竞争奖励**: 通过计算每个波段的排列重要性（Permutation Importance），评估其对分类任务的独立贡献，实现个体间的优胜劣汰。
* **GMM优化经验池 (`experience_replay.py`)**: 为了提升学习效率，项目使用高斯混合模型（GMM）对经验回放池进行优化。算法会筛选出奖励值高的“高质量”样本，用GMM拟合其分布，并生成新的、高质量的模拟样本，替换掉一部分低质量样本，从而加速收敛。

### 项目结构 (`MARLBS/`)

```
MARLBS/
├── agent.py                  # 定义单个智能体(Agent)的行为
├── data_loader.py            # 加载和预处理Pavia University数据集
├── experience_replay.py      # 实现基于GMM优化的经验回放池
├── main.py                   # 主程序入口，配置并运行MARLBS流程
├── marl_environment.py       # 定义MARLBS的环境与混合式奖励
├── models.py                 # 定义QNetwork, Autoencoder, GCN等PyTorch模型
├── state_representation.py   # 实现核心的“融合状态”计算逻辑
├── trainer.py                # 编排整个训练和评估流程
├── utils.py                  # 存放辅助函数，如分类指标计算
└── requirements.txt          # 项目依赖
```

### 环境依赖

所有依赖项都列在 `MARLBS/requirements.txt` 文件中。

```
torch
scikit-learn
numpy
scipy
matplotlib
```

### 如何运行

1.  **准备数据**: 运行前，请确保 `PaviaU.mat` 和 `PaviaU_gt.mat` 数据文件位于项目根目录下。`data_loader.py` 会自动加载它们。
2.  **启动训练**: 直接运行 `main.py` 文件即可开始训练和评估。

    ```bash
    python MARLBS/main.py
    ```
    程序将自动完成数据加载、归一化、训练、评估，并在结束后显示训练过程的准确率曲线图。

---

## 项目二: CD-MABS (聚类与多样化经验池)

该方法首先通过聚类将相似的波段分组，然后让代表每个“波段簇”的智能体进行选择。其核心是处理波段的高度相关性，并结合一个更复杂的分类网络进行评估。

### 核心创新点

* **基于相关熵的波段聚类 (`clustering.py`)**: 在智能体选择之前，项目首先使用一种基于 **相关熵** 的贪心算法对所有波段进行聚类。这大大减少了智能体的数量（从波段数减少到簇数），并降低了决策空间的复杂度。
* **分组式多智能体训练 (`marl_training.py`)**: 智能体的数量由聚类后的 `num_groups` 决定。每个智能体负责决定是否选择其代表的整个波段簇，而不是单个波段。
* **多样化经验池与教师策略 (`marl_training.py`)**: 代码中实现了两个经验池：`teacher_buffer` 和 `agent_buffer`。这暗示了一种教师-学生策略，其中“教师”的优质经验可以引导“学生”（即智能体）更快地学习，从而构成一个多样化的经验池。
* **强大的空谱分类网络 (`network.py`)**: 项目使用了一个名为 **FFAW-Net** 的复杂分类模型。该模型结合了3D卷积和2D卷积，用于同时提取高光谱图像的 **空-谱特征**，为波段选择的有效性提供更精确的评估。

### 项目结构 (`CD-MABS/`)

```
CD-MABS/
├── band_selection/
│   ├── clustering.py         # 实现基于相关熵的波段聚类
│   └── marl_training.py      # 实现分组式MARL训练
├── classification_model/
│   ├── network.py            # 定义FFAW-Net空谱分类网络
│   └── train_classifier.py   # 训练和评估分类器
├── utils/
│   ├── data_loader.py        # 加载、下载和预处理数据
│   └── metrics.py            # 分类评估指标计算函数
├── main.py                   # 主程序入口，串联聚类、选择和分类流程
└── requirements.txt          # 项目依赖
```

### 环境依赖

所有依赖项都列在 `CD-MABS/requirements.txt` 文件中。

```
numpy
scikit-learn
torch
scipy
requests
tqdm
```

### 如何运行

1.  **准备数据**: `utils/data_loader.py` 被设计为可以自动从网络下载 `PaviaU.mat` 数据集（如果本地不存在）。
2.  **启动完整流程**: 直接运行 `main.py` 文件。

    ```bash
    python CD-MABS/main.py
    ```
    程序会依次执行以下步骤：
    1.  加载数据。
    2.  进行基于相关熵的波段聚类。
    3.  运行多智能体强化学习进行波段选择。
    4.  使用选择出的波段，训练 `FFAW-Net` 分类器并报告最终精度。
    5.  如果强化学习未能选出足够波段，程序会切换到PCA降维方案作为备选。