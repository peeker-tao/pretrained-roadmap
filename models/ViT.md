# ViT (Vision Transformer)

## 基本信息

- **论文**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- **作者**: Alexey Dosovitskiy et al. (Google)
- **发表**: ICLR 2021

## 创新点

1. **图像块序列化**: 将图像分割为 16×16 的 patch 序列，直接应用 Transformer
2. **无 CNN 归纳偏置**: 证明纯 Transformer 无需卷积即可处理视觉任务
3. **大规模预训练必**: 在足够大数据上预训练，ViT 可超越 CNN

## 核心原理

### 图像分块 (Patch Embedding)

1. 将 $H×W×3$ 图像分为 $N$ 个 $P×P×3$ 的 patch
2. 每个 patch 展平后通过线性投影映射到 $D$ 维
3. 添加可学习的位置编码

### Transformer 编码器

标准 Transformer 编码器层：
- LayerNorm → Multi-Head Self-Attention → LayerNorm → MLP
- 使用 [CLS] token 作为分类表征

## 预训练方法

### 核心思想：把图片当作"句子"来处理

ViT 的预训练可以这样理解：把一张图片切成很多小方块（16×16 像素），每个小方块就像 NLP 中的一个"词"，然后把这一串"视觉词"送入标准的 Transformer 做处理。ViT 本身不预设 CNN 的归纳偏置（局部性、平移不变性），完全靠 Transformer 的注意力机制自己学习图像中的空间关系。

> 简单来说，ViT = "把图片打散成 patch + 用 Transformer 当作一个通用特征提取器"。

### 训练流水线（Step by Step）

#### Step 1 — 图像分块（Patchify）

输入一张 $224 \times 224 \times 3$ 的 RGB 图像，将其分割为 $14 \times 14 = 196$ 个不重叠的 $16 \times 16$ patch。每个 patch 的像素值被展平成一个 $16 \times 16 \times 3 = 768$ 维向量。

```text
[224×224×3] → 切割成 196 个 [16×16×3] → 展平成 196 个 [768维]
```

这就像把一篇文章拆成 196 个"词"。

#### Step 2 — Patch Embedding（线性投影）

每个 768 维的 patch 向量通过一个**可学习的线性投影矩阵** $E \in \mathbb{R}^{768 \times D}$ 映射到 Transformer 的隐藏维度 $D$（例如 ViT-Base 的 $D=768$）。

$$\text{Patch Embedding}_i = \text{Flatten}(\text{Patch}_i) \cdot E$$

> 这个线性投影相当于 CNN 中的 $16 \times 16$ 卷积，stride=16——把局部像素压缩成语义向量。

#### Step 3 — 添加 [CLS] Token 和位置编码

在 196 个 patch embedding 前面**拼接**一个特殊的 `[CLS]` token（可学习的向量，维度 $D$），用于汇总全局信息。然后给所有 197 个 token 加上**可学习的位置编码**：

$$z_0 = [x_{\text{class}}; x_p^1 E; x_p^2 E; ...; x_p^{196} E] + E_{\text{pos}}$$

- `[CLS]` token：借鉴 BERT 的设计，这个 token 的最终输出将用作整张图像的表示
- 位置编码 $E_{\text{pos}} \in \mathbb{R}^{197 \times D}$：**可学习的 1D 位置编码**（而非 2D），让模型自行从数据中发现 patch 之间的空间关系

#### Step 4 — Transformer 编码

像标准 Transformer 编码器一样，$L$ 层（ViT-Base 为 12 层），每层包含：

1. **多头自注意力（MSA）**：让每个 token（包括 [CLS]）关注所有其他 token
2. **前馈网络（MLP）**：两个全连接层 + GELU 激活
3. **LayerNorm + 残差连接**：Pre-LN 风格

$$z'_\ell = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}$$
$$z_\ell = \text{MLP}(\text{LN}(z'_\ell)) + z'_\ell$$

经过 $L$ 层后，取出 `[CLS]` 位置的输出 $z_L^0$，通过一个分类头做 ImageNet 1000 类的分类。

### 为什么这样设计——深入理解

#### 1. 为什么用 Patch 而不是像素？

直接用像素（$224 \times 224 = 50176$ 个 token）会让 Transformer 的注意力计算复杂度 $O(n^2)$ 爆炸。$16 \times 16$ 的 patch 大小是一个经验性的权衡：

| Patch 大小 | Patch 数量 | 计算量 | Top-1 (JFT 预训练) |
|-----------|-----------|--------|-------------------|
| $14 \times 14$ | 256 | 较高 | 略好 |
| $16 \times 16$ | 196 | 适中 | **标准选择** |
| $32 \times 32$ | 49 | 较低 | 明显下降 |

#### 2. 为什么需要大量预训练数据？

这是 ViT 论文最关键的发现：**CNN 有内置的归纳偏置（局部连接、平移不变性），所以中等数据量就能训练好；Transformer 没有这些偏置，需要从海量数据中自己"学到"这些规律。**

| 预训练数据 | 数据量 | ViT-B/16 Top-1 | ResNet-50 Top-1 |
|-----------|--------|---------------|-----------------|
| ImageNet-1K | 1.28M | 77.9% | **79.3%** |
| ImageNet-21K | 14M | 84.0% | 82.4% |
| JFT-300M | 300M | **88.6%** | 87.1% |

观察：在 ImageNet-1K 上 ViT 不如 ResNet，但在 JFT-300M 上 ViT 超越了 ResNet。**数据越多，Transformer 的优势越大**——这就是"大数据 + 大模型"范式的早期实证。

#### 3. 位置编码：1D vs 2D vs Relative

ViT 尝试了多种位置编码方式：

| 位置编码类型 | Top-1 | 说明 |
|-------------|-------|------|
| 无位置编码 | 极差 | 缺失空间信息 |
| 1D 学习位置编码 | **最好** | 论文默认选择 |
| 2D 学习位置编码 | 相似 | 未明显优于 1D |
| 相对位置编码 | 相似 | 只在微小模型上略好 |

1D 位置编码虽不包含显式的 2D 空间结构，但 Transformer 的注意力可以从数据中学习到 patch 之间的空间关系。可视化显示：低层注意力确实学会了关注相邻 patch，高层注意力学会了关注语义相关的远距离 patch。

### 详细训练配置

#### JFT-300M 监督预训练（ViT 原论文的核心）

| 参数 | ViT-B/16 | ViT-L/16 | ViT-H/14 |
|------|---------|---------|---------|
| 层数 $L$ | 12 | 24 | 32 |
| 隐藏维度 $D$ | 768 | 1024 | 1280 |
| MLP 维度 | 3072 | 4096 | 5120 |
| 注意力头数 | 12 | 16 | 16 |
| Patch 大小 | 16×16 | 16×16 | 14×14 |
| 参数量 | 86M | 307M | 632M |
| 优化器 | Adam | Adam | Adam |
| 基础学习率 | 0.03 | 0.03 | 0.03 |
| 权重衰减 | 0.3 | 0.3 | 0.3 |
| 学习率调度 | Cosine Decay | Cosine Decay | Cosine Decay |
| Warmup | 10K 步 | 10K 步 | 10K 步 |
| Batch Size | 4096 | 4096 | 4096 |
| 训练步数 | 300K | 300K | 300K |
| 微调分辨率 | 384×384 | 384×384 | 512×512 |

#### 微调策略（Fine-tuning）

ViT 从 JFT 预训练迁移到 ImageNet 时有两种微调方式：

1. **标准微调**：在 ImageNet 上以 $224 \times 224$ 分辨率训练
2. **高分辨率微调**：在 $384 \times 384$（或更高）分辨率上微调。此时 patch 大小不变（仍为 16×16），但 patch 数量增加，需要**插值位置编码**来适配新的 token 数量

$$\text{Position}_\text{new} = \text{Interpolate}(\text{Position}_\text{old}, \text{new\_shape})$$

> 高分辨率微调通常带来 1-2% 的额外提升，因为更精细的 patch 粒度让模型能看到更多细节。

#### 自监督预训练（ViT 的重要扩展）

ViT 原论文只用监督预训练。后续工作（MAE、DINO、MoCo v3）开发了专门针对 ViT 的自监督预训练方法：

| 方法 | 核心思路 | ViT-B Top-1 |
|------|---------|------------|
| **MoCo v3** | 动量对比学习，冻结 patch 投影层解决训练不稳定 | 76.7% |
| **DINO** | 自蒸馏（学生看局部，教师看全局） | 78.2% |
| **MAE** | 掩码 75% patch，重建缺失像素 | 79.1% |
| **BEiT** | 掩码 + 离散视觉 token 预测 | 79.7% |

### 预训练中的关键现象

#### Patch Projection 的冻结技巧（MoCo v3 发现）

ViT 在自监督训练中容易出现"训练振荡"——损失突然飙升。MoCo v3 的解决方案是**冻结 patch projection 层**（随机初始化后不再更新）。

直觉解释：patch projection 把像素映射到语义空间，这个映射对训练稳定至关重要。如果让它在自监督任务中也参与梯度更新，可能会学到"投机取巧"的映射，导致训练崩溃。冻结它相当于提供了一个稳定的输入表示基础。

#### 注意力距离分析

ViT 预训练后的注意力可视化揭示了一个有趣的现象：

- **低层**：注意力集中在相邻 patch（类似 CNN 的局部感受野）
- **中层**：注意力开始关注更长距离的语义关系
- **高层**：[CLS] token 的注意力聚焦在图像中最有辨识度的区域（如人脸、物体主体）

这说明 ViT **从数据中学到了类似 CNN 的分层视觉处理模式**，尽管它本身没有任何空间归纳偏置。

### 预训练的迁移价值

ViT 的预训练范式的核心启示：

1. **Transformer 是跨模态通用架构**——NLP 中的 Transformer 不做修改就能用于视觉
2. **规模是关键**——Transformer 需要大数据才能超越 CNN（归纳偏置 vs 数据驱动的权衡）
3. **预训练方式灵活**——监督/自监督/多模态预训练都可以用于 ViT
4. **Patch 化是视觉 Transformer 的基础**——后续所有 ViT 变体（Swin、DeiT、PVT 等）都沿用了这个设计
