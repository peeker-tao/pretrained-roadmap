# SwAV (Swapping Assignments between Views)

## 基本信息

- **论文**: [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882)
- **作者**: Mathilde Caron et al. (INRIA / Meta)
- **发表**: NeurIPS 2020

## 创新点

1. **在线聚类 (Online Clustering)**: 使用 Sinkhorn-Knopp 算法在 batch 内平衡聚类分配
2. **交换预测**: 一个视图的编码经聚类后，预测另一个视图的聚类分配
3. **多分辨率裁剪 (Multi-Crop)**: 用小分辨率裁剪增加样本多样性，几乎零计算成本
4. **无需 Memory Bank 或大 batch**

## 核心原理

### SwAV 框架

1. 对同一样本生成两个增强视图
2. 编码器提取特征
3. 将特征投影到聚类代码 $q$（使用 Sinkhorn-Knopp 算法计算）
4. 交换预测：用视图 1 的编码预测视图 2 的代码，反之亦然

### 交换预测损失

$$\mathcal{L} = \mathcal{L}(\text{code}(\text{aug}_1), \text{feat}(\text{aug}_2)) + \mathcal{L}(\text{code}(\text{aug}_2), \text{feat}(\text{aug}_1))$$

### Sinkhorn-Knopp 算法

对每个 batch 的聚类分配进行正则化，确保：
- 每个 batch 中的聚类分配是均匀的（防止崩溃）
- 计算可微分，支持端到端反向传播

### Multi-Crop

使用多种分辨率裁剪：
- 2 个 224×224 的全局视图
- 多个 96×96 的局部视图

## 预训练方法

### 核心思想：不需要比较两个视图的嵌入——比较它们的"聚类标签"就行

SwAV（Swapping Assignments between Views）是自监督学习中的对比-聚类混合方法。不同于 SimCLR（直接比较两个视图的特征）或 MoCo（维护负样本队列），SwAV 的核心是：**将每个视图的特征映射到一组"聚类原型"上，然后让一个视图的聚类标签去预测另一个视图的特征**。

> SwAV = 在线聚类 + 交换预测 + Multi-Crop。它用 Sinkhorn-Knopp 聚类替代了传统的负例对比，既避免了 Memory Bank，也不需要大 batch——在 2020 年是最具创新性的自监督方法之一。

### 训练流水线（Step by Step）

#### Step 1 — 多视图生成（Multi-Crop）

SwAV 引入了 Multi-Crop 策略——使用不同分辨率的裁剪组合：

| 视图类型 | 数量 | 分辨率 | 计算成本 |
|---------|------|--------|---------|
| 全局视图 | 2 | 224×224 | 正常 |
| 局部视图 | $V$ (通常 4-8) | 96×96 | **极低**（面积仅 18%） |

**Multi-Crop 的妙处**：
- 局部视图的计算成本远低于全局视图（96² / 224² ≈ 18%）
- 但局部视图提供了宝贵的"不同视角"——迫使模型学习局部到全局的一致性
- 几乎零成本地增加了正例的多样性

> 类比：全局视图就像从远处看一栋房子（整体结构），局部视图就像看房子的窗户或门（细节）。一个好的表征应该同时捕捉整体和细节。

#### Step 2 — 特征提取和投影

每个视图通过编码器（ResNet-50）提取特征，然后通过两个投影头：

1. **投影头 1（用于对比）**：将特征投影到低维空间（128 维）
2. **投影头 2（用于聚类）**：将特征投影到原型空间（3000 维 = 原型数）

#### Step 3 — Sinkhorn-Knopp 在线聚类

这是 SwAV 的核心算法。对于每个 batch 的特征和原型向量：

**问题定义**：给定 $B$ 个样本的特征 $Z \in \mathbb{R}^{B \times d}$ 和 $K$ 个原型 $C \in \mathbb{R}^{K \times d}$，计算软聚类分配 $Q \in \mathbb{R}^{B \times K}$。

**直接 softmax 的问题**：如果直接使用 softmax($Z C^T / \tau$)，所有样本可能被分配到同一原型（聚类坍塌）。

**Sinkhorn-Knopp 的解决方案**：在保持软分配与原型相似度一致的前提下，强制约束分配矩阵的行和列和都均匀：

$$\max_{Q \in \mathcal{Q}} \text{Tr}(Q^T Z C^T) + \epsilon H(Q)$$

约束条件 $\mathcal{Q} = \{Q \in \mathbb{R}^{B \times K}_+ | Q\mathbf{1}_K = \mathbf{1}_B/B, Q^T\mathbf{1}_B = \mathbf{1}_K/K\}$。

- **行约束**：每个样本的总分配权重为 $1/B$（样本间均匀）
- **列约束**：每个原型的被分配权重为 $1/K$（原型间均匀）← 防止坍塌

**算法**：通过迭代行列归一化求解（与 Sinkhorn 算法等价，3 次迭代即可收敛）：

```text
1: Q ← exp(Z C^T / ε)     # 初始化：基于相似度的软分配
2: for iter in 1:3:
3:     Q ← Q / sum(Q, dim=1) * (1/B)   # 行归一化
4:     Q ← Q / sum(Q, dim=0) * (1/K)   # 列归一化
5: return Q
```

#### Step 4 — 交换预测

关键创新：**不直接比较特征，而是交换聚类标签来比较**。

$$\mathcal{L}_{\text{SwAV}} = \ell(Q_1, Z_2) + \ell(Q_2, Z_1)$$

其中 $\ell(Q_1, Z_2) = -\sum_k Q_1^{(k)} \log P_2^{(k)}$，$P_2 = \text{softmax}(Z_2 C^T / \tau)$。

**"交换"的含义**：
- 用视图 1 的聚类标签 $Q_1$ 来监督视图 2 的特征 $Z_2$
- 用视图 2 的聚类标签 $Q_2$ 来监督视图 1 的特征 $Z_1$

> 这本质上是一种"知识蒸馏"的形式——一个视图的"聚类观点"被用来教导另一个视图。它避免了直接比较嵌入向量，而是比较更高层的"语义标签"。

**Multi-Crop 扩展**：

$$\mathcal{L} = \sum_{i} \sum_{j \neq i} \ell(Q_i, Z_j)$$

所有局部视图预测全局视图的聚类标签。局部视图数量多但分辨率低，全局视图数量少但分辨率高——所有视图向全局视图"学习"。

### 为什么 SwAV 如此高效？

#### SwAV vs SimCLR vs MoCo

| 方法 | 负例来源 | 是否需要大 batch | 是否需要 Memory Bank | 是否需要聚类 |
|------|---------|----------------|-------------------|-----------|
| SimCLR | In-batch | **是**（≥4096） | 否 | 否 |
| MoCo v2 | Memory Queue | 否 | **是**（65K 样本） | 否 |
| **SwAV** | **不需要显式负例** | **否（256 即可）** | **否** | **是（在线聚类）** |

SwAV 用聚类替代了显式负例比较——聚类本质上是"隐式的对比"，因为聚类分配迫使不同样本被分配到不同原型。

### 详细训练配置

| 参数 | SwAV (ResNet-50) | 说明 |
|------|-----------------|------|
| 全局视图 | 2×224² | 标准分辨率 |
| 局部视图 | 8×96² | Multi-Crop |
| 原型数 $K$ | 3000 | 聚类中心数 |
| Sinkhorn 迭代 | 3 | 行列归一化轮数 |
| Sinkhorn 温度 $\epsilon$ | 0.05 | 控制分配软硬 |
| 预测温度 $\tau$ | 0.1 | softmax 锐度 |
| 投影头 | 2 层 MLP → 128 维 | 用于对比 |
| 聚类头 | 线性 → 3000 维 | 用于聚类分配 |
| 优化器 | SGD + Momentum (0.9) | — |
| 学习率 | 0.48（基础 lr × Batch/256） | 线性缩放 |
| 学习率调度 | 余弦衰减 + Warmup (10 ep) | — |
| 权重衰减 | 1e-6 | — |
| Batch Size | 4096 | SwAV 对 batch 不敏感 |
| Epoch | 400 | — |

### 预训练性能（ImageNet 线性探测）

| 方法 | ResNet-50 Top-1 | Epoch |
|------|----------------|-------|
| SimCLR | 70.4% | 1000 |
| MoCo v2 | 71.1% | 800 |
| SwAV | **75.3%** | 400 |
| SwAV (w/ multi-crop) | **75.3%** | **400（比 SimCLR 快 2.5×）** |

SwAV 用不到一半的训练 epoch 达到了更高的线性探测精度——Multi-Crop 和聚类对比的高效性功不可没。

### 预训练的实用价值

1. **在线聚类的对比范式**：不依赖显式负例，不需要 Memory Bank 或大 batch
2. **Multi-Crop 策略**：零成本提升正例多样性，被 DINO、iBOT 等后续工作广泛采用
3. **Sinkhorn-Knopp 的引入**：带约束的最优传输在自监督学习中大放异彩
4. **DINO 的技术前身**：DINO 的自蒸馏 + centering 机制直接继承自 SwAV 的在线聚类思想
5. **SOTA 的训练效率**：400 epoch 达到 75.3%，远快于同时期方法

- 2 个 224×224 全局视图
- 6 个 96×96 局部视图

### 预训练迁移价值

SwAV 在 ImageNet 线性探测上达到 75.3%（ResNet-50），超越了 BYOL 和 SimCLR，且无需大 batch。
