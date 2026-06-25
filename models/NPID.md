# NPID (Non-Parametric Instance-level Discrimination)

## 基本信息

- **论文**: [Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination](https://arxiv.org/abs/1805.01978)
- **作者**: Zhirong Wu et al. (CMU / Meta)
- **发表**: CVPR 2018

## 创新点

1. **实例判别 (Instance Discrimination)**: 首次明确提出将每个样本作为一个独立类别进行判别
2. **非参数 Softmax**: 使用内存缓存 (Memory Bank) 存储所有样本的表征，避免传统 Softmax 的参数化分类器
3. **对比学习的早期奠基**: 为后续 SimCLR、MoCo 等工作奠定了基础

## 核心原理

### 实例判别

将每个训练样本视为一个类别（ImageNet 有 128 万类），模型需要区分每个实例。

### Memory Bank

由于实例数太大（数百万），不可能直接计算所有实例的表征。NPID 维护一个 Memory Bank 存储每个样本的最新表征。

### 非参数 Softmax

$$P(i|v) = \frac{\exp(v \cdot f_i / \tau)}{\sum_{j=1}^n \exp(v \cdot f_j / \tau)}$$

其中 $v$ 是查询表征，$f_i$ 是第 $i$ 个样本的 Memory Bank 表征。

## 预训练方法

### 核心思想：每张图都是一个"类"——在百万量级的实例判别中，网络必须学会区分每个样本的细微特征

NPID（Non-Parametric Instance Discrimination）是自监督学习中**实例判别（Instance Discrimination）**路线的开创者。它的核心洞察：如果把每张训练图片都视为一个独立的类别，那么要正确判别 128 万张图片，网络就必须学习极其精细的视觉特征——这比区分 1000 个粗粒度类别更能迫使网络理解视觉内容。

> NPID = 128 万个实例类 + Memory Bank（存储所有样本特征）+ 非参数 Softmax（NCE 近似）。它直接启发了 MoCo、SimCLR 等对比学习方法——实例判别是对比学习的核心思想来源。

### 训练流水线（Step by Step）

#### Step 1 — 实例判别问题定义

传统 ImageNet 分类有 $K=1000$ 个类别。NPID 将 $N=128$ 万个训练样本视为 $N$ 个独立的类别：

$$P(i|v) = \frac{\exp(v^T f_i / \tau)}{\sum_{j=1}^N \exp(v^T f_j / \tau)}$$

其中：
- $v$：当前样本的特征向量
- $f_i$：第 $i$ 个样本的特征向量（存储在 Memory Bank 中）
- $\tau$：温度参数（控制分布的"锐度"）
- $N$：总样本数（128 万）

**为什么实例判别有效？**

| 任务 | 类别数 | 粒度 | 需要的特征 |
|------|--------|------|-----------|
| ImageNet 分类 | 1K | 粗（"狗" vs "猫"） | 类别间差异 |
| **NPID 实例判别** | **1.28M** | **极细（"这张狗照" vs "那张狗照"）** | **实例级差异** |

> 类比：1000 类分类就像学会区分"狗"和"猫"；实例判别就像学会区分"这只德国牧羊犬的照片 A"和"那只德国牧羊犬的照片 B"——后者要求网络注意每张图的独特细节（光照、姿态、背景），从而学到更丰富的表征。

#### Step 2 — Memory Bank：非参数分类的核心

直接计算 128 万类别的 softmax 在计算上不可行。NPID 使用 Memory Bank：

```text
Memory Bank M ∈ R^(N × d):
  M[i] = 第 i 个训练样本的最新特征向量
  
每次迭代：
  1. 正向传播：提取当前 batch 的特征 v
  2. 从 M 中读取所有样本的特征用于计算损失
  3. 更新 Memory Bank：M[i_batch] ← v（EMA 或直接替换）
```

**Memory Bank 的优缺点**：

| 优点 | 缺点 |
|------|------|
| 避免重复计算全数据集特征 | 存储 $N \times d$ 个浮点数（~640MB for 128M×128） |
| 特征一致性比实时计算好 | 特征滞后（旧样本的特征在 bank 中停留多 epoch） |

#### Step 3 — NCE（Noise Contrastive Estimation）近似

$N=128$ 万的 softmax 仍然太昂贵——需要对 128 万个分数的指数求和。NPID 使用 NCE 近似：

$$\mathcal{L}_{\text{NCE}} = -\log \frac{\exp(v^T f_+ / \tau)}{\exp(v^T f_+ / \tau) + \sum_{j=1}^m \exp(v^T f_j / \tau)}$$

其中：
- $f_+$：正例（当前样本的 Memory Bank 特征）
- $\{f_1, ..., f_m\}$：$m$ 个随机采样负例（$m \ll N$，通常 4096）

> NCE 的直觉：不需要考虑所有 128 万个类别，只需要在"正例 + 一小批随机负例"中判别。只要负例足够多且采样均匀，就能近似真实分布。

#### Step 4 — 负例采样的策略

| 采样策略 | 效果 |
|---------|------|
| 均匀随机 | 基准 |
| 难负例挖掘 | 更好（但计算成本高） |
| 类别感知 | 避免"同类假阴性" |

NPID 使用均匀随机采样——简单有效。但均匀采样有一个问题：**假阴性**——采样到的负例可能恰好与正例相似（但被标记为负例）。这是一致性问题，NPID 论文对此有详细讨论。

#### Step 5 — 训练流程

```text
初始化：
  1. 随机初始化 CNN
  2. 用随机 CNN 提取所有样本特征 → 填充 Memory Bank

每轮迭代：
  1. 读取一个 batch 的图像
  2. 正向传播 → 特征 v
  3. 从 Memory Bank 读取对应的正例特征 f_+
  4. 随机采样 m 个负例特征
  5. 计算 NCE 损失
  6. 反向传播更新 CNN 参数
  7. 更新 Memory Bank：M[i] ← v（当前 batch 对应的条目）
```

### 为什么 NPID 会出现"表征退化"？

NPID 存在一个重要问题：**特征空间的不均匀分布**。由于 Memory Bank 中的特征在不同时间更新，且缺乏强制的均匀约束，特征可能集中在特征空间的某些区域。

**表现**：某些样本的相似度极高，某些极低——导致 softmax 概率分布过度极端，训练不稳定。

MoCo 的改进正是针对这一点：
- **动量编码器**：缓慢更新，保证 Memory Bank 中特征的一致性
- **队列**：滑动窗口，及时淘汰旧特征

### 详细训练配置

| 参数 | NPID (ResNet-50) | 说明 |
|------|-----------------|------|
| 数据集 | ImageNet（无标签） | 128 万张图 |
| 特征维度 $d$ | 128 | $\ell_2$ 归一化后 |
| 温度 $\tau$ | 0.07 | 控制 softmax 锐度 |
| 负例数 $m$ | 4096 | NCE 采样数 |
| Memory Bank 大小 | 128 万 × 128 | ~640MB |
| 优化器 | SGD + Momentum (0.9) | — |
| 学习率 | 0.03 | — |
| Batch Size | 256 | — |
| Epoch | 200 | — |

### NPID 的局限与后续改进

| 问题 | NPID 的表现 | 改进方法 |
|------|-----------|---------|
| Memory Bank 过期 | 旧特征与新模型不一致 | MoCo：动量编码器 |
| 负例采样质量 | 随机采样，假阴性 | MoCo v2/SimCLR：更大 batch/队列 |
| 无数据增强一致性 | 没有考虑同一图像的变体 | SimCLR：两视图一致性 |
| 训练不稳定 | NCE 梯度方差高 | 温度参数调优 |

### 预训练的实用价值

1. **实例判别范式的开创者**：首次明确提出"每个样本是一个类"的自监督学习目标
2. **对比学习的直接前身**：MoCo、SimCLR、SwAV 都从实例判别 + NCE 的思想发展而来
3. **Memory Bank 的设计思想**：启发了 MoCo 的队列设计和后续的动量编码器
4. **非参数分类的理论贡献**：NCE/InfoNCE 在自监督学习中的应用由 NPID 开创
5. **证明了"没有标签也能学到好特征"**：在 ImageNet 线性探测上达到 ~54%（ResNet-50），远超之前方法
