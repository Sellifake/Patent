
## 项目一: MARLBS (融合状态表示与GMM优化)
MARLBS - Multi-Agent Reinforcement Learning Band Selection - 基于融合状态表示和经验池优化的多智能体高光谱波段选择方法
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

1. **准备数据**: 运行前，请确保 `PaviaU.mat` 和 `PaviaU_gt.mat` 数据文件位于项目根目录下。`data_loader.py` 会自动加载它们。
2. **启动训练**: 直接运行 `main.py` 文件即可开始训练和评估。

   ```bash
   python MARLBS/main.py
   ```

   程序将自动完成数据加载、归一化、训练、评估，并在结束后显示训练过程的准确率曲线图。

---

## 项目二: CD-MABS (聚类与多样化经验池)
CD-MABS- A Multi-Agent Band Selection Method Based on Clustering and Diverse Experience Pools-基于聚类和多样经验池的多智能体波段选择方法
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

1. **准备数据**: `utils/data_loader.py` 被设计为可以自动从网络下载 `PaviaU.mat` 数据集（如果本地不存在）。
2. **启动完整流程**: 直接运行 `main.py` 文件。

   ```bash
   python CD-MABS/main.py
   ```

   程序会依次执行以下步骤：

   1. 加载数据。
   2. 进行基于相关熵的波段聚类。
   3. 运行多智能体强化学习进行波段选择。
   4. 使用选择出的波段，训练 `FFAW-Net` 分类器并报告最终精度。
   5. 如果强化学习未能选出足够波段，程序会切换到PCA降维方案作为备选。

---

## 项目三: FC-CMABG (融合聚类与级联智能体)
FC-CMABG - Fusion-Clustering & Cascaded Multi-Agent Hyperspectral Band Generation - 基于融合聚类和级联多智能体的高光谱波段生成方法
该方法将重点从“波段选择”转向“波段生成”，旨在通过智能体的协作来创造出全新的、比原始波段更具判别力的特征波段。

### 核心创新点

* **融合信息的特征组划分 (`2_feature_clustering.py`)**: 在生成新特征前，项目首先通过一种创新的融合聚类方法对原始波段进行分组。该方法定义的波段距离不仅考虑了波段间的 **光谱相似性**，还融入了每个波段与最终分类任务的 **关联度**（通过互信息计算），使得分组结果同时具备物理意义和任务导向性。
* **跨域协同的状态编码 (`3_state_encoder.py`)**: 为强化学习智能体提供了极其丰富的决策依据。它并非使用原始数据，而是将三个不同维度的信息融合成一个全面的状态向量：
  * **主成分统计投影 (PCA)**: 从统计学角度捕捉全局和组级的主要判别方向。
  * **空谱级联自编码 (3D-CNN Autoencoder)**: 利用3D卷积网络同时提取空间纹理和光谱形态。
  * **小波图谱嵌入 (Wavelet-GCN)**: 通过离散小波变换和图卷积网络（GCN），捕捉波段之间的拓扑和频率域相关性。
* **级联多智能体决策框架 (`4_agent_and_env.py`)**: 这是该方法最大的创新。它将复杂的“波段生成”问题分解为一个有序的三步决策链，由三个专门的智能体序贯完成：
  * **智能体0 (组选择)**: 负责“从哪里开始”，选出第一个特征组。
  * **智能体1 (运算选择)**: 负责“如何组合”，从 `{加, 减, 乘, 除, 拼接}` 中选择一个运算。
  * **智能体2 (组配对与生成)**: 负责“和谁组合”，选择第二个特征组，完成新特征的生成，并立即进行预筛选。
    这种 “组 → 算 → 组” 的级联结构，极大地简化了每个智能体的决策难度。
* **独立的A2C更新与定制化奖励 (`4_agent_and_env.py` 和 `5_a2c_models.py`)**: 每个智能体都采用独立的优势演员-评论家（A2C）算法进行训练，并且拥有量身定制的奖励函数，以驱动它们完成各自的子目标，确保最终生成的波段组合既与原始特征相关，又富含新的信息增益。

### 项目结构 (`FC-CMABG/`)

```
FC-CMABG/
├── 1_data_loader.py            # 负责数据加载、预处理和样本生成
├── 2_feature_clustering.py     # 负责实现融合信息的特征组划分
├── 3_state_encoder.py          # 负责实现三模态状态编码
├── 4_agent_and_env.py          # 负责定义RL环境、三个智能体和其定制化的奖励
├── 5_a2c_models.py             # 负责定义通用的A2C演员和评论家网络结构
├── main.py                     # 主程序，负责串联所有模块，执行训练和生成流程
└── requirements.txt            # 项目依赖
```

### 环境依赖

```
所有依赖项都列在 `FC-CMABG/requirements.txt` 文件中。
torch
scikit-learn
numpy
scipy
gymnasium
pywavelets
```

### 如何运行

1. **准备数据**: 运行前，请确保 `PaviaU.mat` 和 `PaviaU_gt.mat` 数据文件位于项目根目录下，并相应修改 `main.py` 文件中的路径配置。
2. **启动完整流程**: 直接运行 `main.py` 文件。

   ```bash
   python FC-CMABG/main.py
   ```
   程序将依次执行以下步骤：

   1. 加载数据并创建数据立方体。
   2. 进行融合聚类，将原始波段划分为特征组。
   3. 设置并（伪）训练状态编码器。
   4. 运行级联多智能体强化学习，生成新特征“配方”。
   5. 结束后，打印出最终生成的优化波段组合。
