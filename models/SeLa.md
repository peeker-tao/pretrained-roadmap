# SeLa (Self-Labelling)

## 基本信息

- **论文**: [Self-labelling via Simultaneous Clustering and Representation Learning](https://arxiv.org/abs/1911.05371)
- **作者**: Yuki M. Asano, Christian Rupprecht, Andrea Vedaldi (Oxford)
- **发表**: ICLR 2020

## 创新点

1. **自标注 (Self-Labelling)**: 将聚类生成的伪标签作为监督信号
2. **等分约束**: 约束每个聚类分配的样本数相等（均匀分布）
3. **交替优化**: 使用 Sinkhorn-Knopp 算法求解最优运输问题

## 核心原理

### 最优运输视角

SeLa 将聚类标签分配问题形式化为最优运输 (Optimal Transport) 问题：
- 约束：每个类分配相等数量的样本
- 求解：使用 Sinkhorn-Knopp 算法

## 预训练方法

### 核心思想：无监督学习的终极问题——如果没有标签，模型怎么知道"应该学到什么"？SeLa 的回答大胆而优雅：**自己给自己贴标签！** 编码器给每张图片分配一个"伪标签"（聚类 ID），然后用这些伪标签像监督学习一样训练——但这个伪标签分配不是随便的，必须满足"每个类分到的人数要一样多"（防止所有人挤到一个类）

SeLa（Self-Labelling）将无监督聚类转化为最优运输（Optimal Transport）问题。它的核心约束是等分约束：N 张图片分配到 K 个伪类，每个类恰好分到 N/K 张——这是一个"公平分配"问题，可以用 Sinkhorn-Knopp 算法高效求解。

> SeLa = 编码器提取特征 + Sinkhorn-Knopp 求解最优标签分配（等分约束）+ 用伪标签做交叉熵训练。与 SwAV 共享核心思想，但 SeLa 离线做 Sinkhorn，SwAV 在线做。

### 训练流水线（Step by Step）

#### Step 1 — 自标注的目标

```text
给定: N 张无标签图片
编码器: 输出 K 维向量（每维对应一个伪类）

目标: 给每张图片分配一个伪标签 q_i（one-hot 或软分配）
约束:
  1. 每个伪类分配相同数量的图片（等分约束）
  2. 最大化"分配的确定性"（减少熵）
```

**为什么需要等分约束？**

| 无等分约束 | 有等分约束 |
|----------|---------|
| 所有图片分配到少数类 | **每个类分到 N/K 张** |
| 模型退化（trivial solution） | **模型必须学到有意义的区分** |
| 类似所有人挤到"常见类" | **强制探索数据中的所有模式** |

> 类比：如果老师让 30 个学生自选座位——没有约束的话所有人会挤到前排。等分约束 = "每排只能坐 6 人"——强制分散到所有座位。

#### Step 2 — 最优运输视角

SeLa 将标签分配形式化为最优运输问题：

$$\max_{Q} \langle Q, \log P \rangle + \epsilon H(Q)$$

$$\text{s.t.} \quad Q \cdot \mathbf{1}_N = \frac{1}{K} \cdot \mathbf{1}_K, \quad Q^T \cdot \mathbf{1}_K = \frac{1}{N} \cdot \mathbf{1}_N$$

| 符号 | 含义 |
|------|------|
| $P \in \mathbb{R}^{N \times K}$ | 编码器对 N 张图片在 K 个类的预测概率 |
| $Q \in \mathbb{R}^{N \times K}$ | 要优化的伪标签分配矩阵 |
| $\langle Q, \log P \rangle$ | 最大化分配与预测的一致性 |
| $\epsilon H(Q)$ | 熵正则化（$\epsilon$ 控制软硬程度） |
| $Q \cdot \mathbf{1} = \frac{1}{K}$ | **等分约束（每列和 = 1/K）** |

#### Step 3 — Sinkhorn-Knopp 算法

等分约束的最优运输问题的解可以用 Sinkhorn-Knopp 迭代高效求得：

```text
算法: Sinkhorn-Knopp（O(NK) 每迭代）

初始化: Q = P / sum(P)

重复直到收敛:
  1. 行归一化: Q_{i,:} = Q_{i,:} / sum(Q_{i,:})
  2. 列归一化: Q_{:,j} = Q_{:,j} / (N/K)  ← 强制等分!
  3. 如果变化 < 阈值 → 停止
  
约 5-20 次迭代即可收敛!
```

| 属性 | Sinkhorn-Knopp |
|------|---------------|
| 时间复杂度 | O(NK) 每次 |
| 收敛速度 | 指数级（线性收敛） |
| 计算量 | 极低（仅归一化操作） |
| 与 SwAV 的区别 | SeLa 离线做（全数据），SwAV 在线做（batch 内） |

#### Step 4 — 完整训练流程

```text
每个 epoch:
  
  步骤 1: 编码所有图片
    编码器 f_θ → 所有图片的特征
  
  步骤 2: 计算 P（预测概率）
    P = softmax(W · feature + b)
  
  步骤 3: Sinkhorn-Knopp 求解 Q（伪标签分配）
    → 满足: 每类 N/K 张图片
  
  步骤 4: 用伪标签训练编码器
    损失 = CrossEntropy(P, Q)
    即: 让编码器的预测 P 逼近 Sinkhorn 分配 Q
  
  步骤 5: 更新编码器参数
    θ ← θ - η · ∇Loss
```

| 参数 | SeLa | 说明 |
|------|------|------|
| 预训练数据 | ImageNet (无标签) | — |
| 编码器 | ResNet-50 | — |
| 伪类数 K | 3000（ImageNet） | 约 3× 真实类数 |
| Sinkhorn ε | 0.05 | 控制软分配程度 |
| Sinkhorn 迭代 | 10-20 | 极快收敛 |
| 优化器 | SGD | 动量 0.9 |
| 学习率 | 0.05 | — |
| 训练 Epoch | 200 | — |

#### Step 5 — SeLa vs SwAV vs DeepCluster

| 维度 | DeepCluster | SeLa | SwAV |
|------|------------|------|------|
| 聚类方法 | K-means | **Sinkhorn-Knopp（OT）** | **Sinkhorn-Knopp（OT）** |
| 等分约束 | 隐式（K-means 自然有） | **显式（OT 约束）** | **显式（OT 约束）** |
| 在线/离线 | 离线（每 epoch 聚类一次） | **离线** | **在线（batch 内）** |
| GPU 计算 | K-means 在 CPU | Sinkhorn 在 GPU | Sinkhorn 在 GPU |
| 损失 | 交叉熵（伪标签） | 交叉熵（伪标签） | **交换预测损失** |

### SeLa 的预训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 伪类数 K | 3000 | 3× ImageNet 类数 |
| Sinkhorn ε | 0.05 | 熵正则化系数 |
| Sinkhorn 迭代 | 10 | — |
| 批次大小 | 256 | — |
| 优化器 | SGD | — |
| 学习率 | 0.05（余弦衰减） | — |

### 预训练的实用价值

1. **最优运输在 SSL 中的首次成功应用**：Sinkhorn-Knopp 求解等分约束
2. **SwAV 的直接前身**：SwAV 就是在线的 SeLa
3. **K-means → OT 的范式转换**：K-means 强制硬分配，Sinkhorn 允许软分配
4. **等分约束的理论贡献**：防止模型坍塌（collapse）的有效机制
5. **自标注范式**：证明了"自己贴标签"可以学到媲美监督学习的表征
