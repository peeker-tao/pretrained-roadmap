# MoCo (Momentum Contrast)

## 基本信息

| 版本 | 论文 | 发表 |
|------|------|------|
| MoCo v1 | [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) | CVPR 2020 |
| MoCo v2 | [Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297) | arXiv 2020 |
| MoCo v3 | [An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.02057) | ICCV 2021 |

## 创新点

### MoCo v1
1. **动量编码器 (Momentum Encoder)**: 对编码器参数做指数滑动平均，保持字典的一致性
2. **队列机制 (Queue)**: 存储大量负例表征，复用计算结果，**解耦负例数量与 batch size**

### MoCo v2
1. **吸收 SimCLR 的设计**: 加入 MLP 投影头和更强的数据增强
2. **不增加计算量**: 简单改进即可显著提升性能

### MoCo v3
1. **扩展到 ViT**: 将 MoCo 应用于 Vision Transformer
2. **冻结 patch 投影层**: 解决 ViT 在自监督训练中的不稳定问题

## 核心原理

### 动量对比机制

MoCo 维护一个高容量、一致的字典：
- **队列**: 存储过去 batch 的表征（负例）
- **动量编码器**: 参数更新慢于查询编码器，保持字典的时空一致性

$$\theta_k \leftarrow m\theta_k + (1-m)\theta_q$$

其中 $m$ 是动量系数（通常 0.999），$\theta_k$ 是键编码器，$\theta_q$ 是查询编码器。

### InfoNCE Loss

$$\mathcal{L} = -\log \frac{\exp(q \cdot k^+ / \tau)}{\sum_{i=0}^K \exp(q \cdot k_i / \tau)}$$

### MoCo v3 的 ViT 训练稳定性

ViT 在自监督训练中会出现**振荡现象**——Loss 突然升高。MoCo v3 发现：
- 冻结 Patch Projection 层可以稳定训练
- 这与 Patch Projection 层的**训练不稳定**有关

## 预训练方法

### 核心思想：用"字典查询"的方式做对比学习

MoCo 的核心创新是将对比学习重新定义为**字典查询问题**：把一张图片的增强视图编码为"查询"（query），在一个大型"字典"中查找与之匹配的"键"（key）。这个字典通过**队列**和**动量编码器**来维护，既大又一致。

> 类比：你在图书馆里要找一本书的副本。你手上的残破封面照片是"查询"，图书馆书架上所有书的封面是"字典"（即"负例"）。MoCo 让你不需要把整个图书馆搬到桌上也能做这个搜索——只需要维护一个高效的索引卡片系统（队列）。

### 训练流水线（Step by Step）

#### Step 1 — 构建查询和键

对于输入图片 $x$，做两次不同的数据增强得到 $x^q$（查询视图）和 $x^k$（键视图）。

#### Step 2 — 编码查询和键

- **查询编码器 $f_q$**（正常梯度更新）：$q = f_q(x^q)$
- **键编码器 $f_k$**（动量更新）：$k_+ = f_k(x^k)$

两个编码器初始权重相同，但更新方式不同。

#### Step 3 — 队列中查找正例

从队列（包含过去 $K$ 个 batch 的键）中取出负例 $\{k_0, k_1, ..., k_{K-1}\}$。队列中**最新的 mini-batch 的键入队，最老的出队**——就像一个先进先出（FIFO）的循环缓冲。

```
队列：[k_oldest, ..., k_old, ..., k_recent, k_new_batch]  ← 新键入队
                                                          k_oldest 出队 →
```

#### Step 4 — 计算 InfoNCE 损失

$$\mathcal{L}_q = -\log \frac{\exp(q \cdot k_+ / \tau)}{\exp(q \cdot k_+ / \tau) + \sum_{i=1}^{K} \exp(q \cdot k_i / \tau)}$$

其中 $K=65536$（MoCo v1/v2 的默认队列大小），$\tau$ 是温度系数。

分子是查询与正例的相似度，分母是查询与正例 + 所有 $K$ 个负例的相似度之和。

#### Step 5 — 动量更新键编码器

$$\theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q$$

$m$ 通常设为 0.999（v1/v2）或 0.99（v3）。这意味着键编码器每次只"向查询编码器靠近 0.1%"。

### 为什么这样设计——深入理解每个组件

#### 1. 为什么要用队列？（MoCo 最关键的工程创新）

在 SimCLR 中，负例来自同一个 batch，因此负例数量被 batch size 限制（GPU 显存）。MoCo 用队列**解耦了负例数量和 batch size**：

| 方法 | 负例来源 | 负例数量 | 限制 |
|------|---------|---------|------|
| SimCLR | 当前 batch | = batch size × 2 | GPU 显存 |
| 端到端 Memory Bank | 存储所有样本的表征 | 整个数据集 | 内存+陈旧表征 |
| **MoCo 队列** | 过去 K 个 batch | K（可配置） | 几乎无限制 |

**队列 vs Memory Bank 的关键区别**：

- **Memory Bank（如 PIRL）**：存储整个数据集中每个样本的表征。问题是——当参数更新后，旧 batch 的表征立即"过时"（stale），不同样本的表征来自不同训练时刻的编码器，缺乏一致性。
- **MoCo 队列**：只存储最近 $K$ 个 batch 的键，旧键不断被新键替换。加之动量编码器更新极慢（$m=0.999$），队列中的键在短时间内几乎来自"同一个编码器"——一致性远好于 Memory Bank。

#### 2. 动量更新为什么重要？

想象一个极端情况：如果键编码器和查询编码器**完全相同且同时更新**——那么队列中所有键都来自不同时刻的编码器，有些键是 1000 步之前的参数产生的，有些是 1 步之前的——完全不一致。

动量更新（$m=0.999$）让键编码器**变化极慢**，队列中的键虽然来自不同时间点，但因为编码器变化微乎其微，它们在特征空间中的"含义"是基本一致的。

直观类比：查询编码器是一个快速运动的相机，键编码器是一个几乎静止的三脚架上的相机。队列存储的是三脚架相机拍的历史照片——虽然拍摄时间不同，但因为相机几乎没动，所有照片的"视角"是一致的。

#### 3. 为什么 MoCo v2 要加入投影头？

MoCo v1（2019）没有 MLP 投影头——直接从编码器输出的特征做对比。SimCLR（2020）证明了投影头的重要性。MoCo v2 吸收了这个发现，加入 2 层 MLP 投影头（2048→2048→128），性能从 60.6% 跃升至 71.1%——**几乎零成本获得 10+ 个百分点的提升**，展示了对比学习中各组件的"可组合性"（composability）。

#### 4. MoCo v3：ViT 的训练稳定性问题

MoCo v3 将 MoCo 框架扩展到 Vision Transformer（ViT）时，发现了一个重要问题：ViT 的自监督训练**不稳定**——训练中途 loss 会突然飙升（"振荡"）。

**原因**：Patch Projection 层（将像素映射为 patch embedding 的线性层）在自监督训练中容易学到退化映射，导致低层的梯度不稳定。

**解决方案**：随机初始化 patch projection 层后**冻结它**（不参与训练）。这几乎不影响性能（因为随机投影已经很好了），但彻底解决了训练不稳定的问题。

```python
# MoCo v3 的关键代码逻辑
for name, param in model.named_parameters():
    if 'patch_embed' in name:
        param.requires_grad = False  # 冻结 patch embedding
```

### 各版本详细配置

| 配置 | MoCo v1 | MoCo v2 | MoCo v3（ViT-B） |
|------|---------|---------|-----------------|
| 编码器 | ResNet-50 | ResNet-50 | ViT-B/16 |
| 投影头 | 无 | 2 层 MLP (128维) | 3 层 MLP (256维) |
| 预测头 | 无 | 无 | 3 层 MLP (256维) |
| 动量系数 $m$ | 0.999 | 0.999 | 0.99 |
| 队列大小 $K$ | 65536 | 65536 | 65536 |
| 温度 $\tau$ | 0.07 | 0.2 | 0.2 |
| 优化器 | SGD | SGD | AdamW |
| 学习率 | 0.03 | 0.03 | 1.5e-4 |
| 权重衰减 | 1e-4 | 1e-4 | 0.1 |
| Batch Size | 256 | 256 | 4096 |
| 训练 Epoch | 200 | 200 | 300/600 |
| 数据增强 | 基础 | SimCLR 风格 | SimCLR 风格 |
| 特殊配置 | Shuffling BN | Shuffling BN | 冻结 Patch Embed |

#### MoCo 的 Shuffling BN 技巧

MoCo 使用了一种巧妙的批归一化（BN）技巧：在多 GPU 训练时，BN 的统计信息可能"泄露"正例信息给负例（因为同一个 batch 的正负例可能在同一 GPU 上计算 BN）。解决方案是：在计算键编码器的 BN 时，**先打乱（shuffle）数据的 GPU 分配**，计算完后再恢复（unshuffle）。

### MoCo v3 训练配置详情

| 参数 | 值 | 说明 |
|------|-----|------|
| 编码器 | ViT-B/16 / ViT-L/16 / ViT-H/14 | Vision Transformer |
| 投影头 | 3 层 MLP：dim→4096→256 | BN + ReLU 在隐藏层 |
| 预测头 | 2 层 MLP：256→4096→256 | BN + ReLU（类似 BYOL） |
| 优化器 | AdamW | β₁=0.9, β₂=0.95 |
| 学习率 | 1.5e-4 × BatchSize/256 | 线性缩放 |
| 学习率调度 | 余弦衰减 | 40 epoch warmup |
| 动量 $m$ | 0.99 → 1.0（余弦调度） | 与 BYOL 类似的调度 |
| 权重衰减 | 0.1 | 比卷积网络更大 |
| Batch Size | 4096 | 64 GPU / TPU |
| Epoch | 300（ViT-B）/ 600（ViT-L） | — |

### 预训练性能

| 方法 | 编码器 | ImageNet 线性探测 |
|-----|--------|-----------------|
| MoCo v1 | ResNet-50 | 60.6% |
| MoCo v2 | ResNet-50 | 71.1% |
| MoCo v3 | ResNet-50 | 74.6% |
| MoCo v3 | ViT-B/16 | 76.7% |
| MoCo v3 | ViT-L/16 | 77.6% |

### 迁移学习的实用价值

MoCo 系列的核心贡献是**工程与算法的结合**：

1. **队列机制是低成本对比学习的基础**：在单 GPU 上也能用 65536 个负例做对比学习，不需要 SimCLR 那样的 4096 batch size
2. **动量编码器的思想影响深远**：BYOL、DINO、SimSiam 的工作都在不同程度上借鉴了动量的概念
3. **组件可组合性**：MoCo v2 证明了对比学习的各组件（动量、投影头、增强）可以"即插即用"地组合
4. **ViT 训练稳定性的发现**：MoCo v3 对 patch embedding 冻结的发现，影响了后续所有 ViT 自监督训练工作
