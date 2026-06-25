# iBOT (Image BERT Pre-Training with Online Tokenizer)

## 基本信息

- **论文**: [iBOT: Image BERT Pre-Training with Online Tokenizer](https://arxiv.org/abs/2111.07832)
- **作者**: Jinghao Zhou et al. (字节跳动 / 约翰霍普金斯大学)
- **发表**: ICLR 2022

## 创新点

1. **在线 Tokenizer**: 与 BEiT 不同，iBOT 的 tokenizer 与编码器**同时训练**，无需离线训练的 dVAE
2. **MIM + 自蒸馏联合**: 同时进行掩码图像建模和自蒸馏
3. **ViT 的语义分割涌现**: 自注意力图自动学习语义分割

## 核心原理

### 在线 Tokenizer

- 教师网络对学生网络的输出做 softmax 聚类（在线 k-means）
- 学生网络预测被掩码 patch 的聚类分配
- 教师网络通过动量更新与学生同步

### 联合损失

$$\\mathcal{L} = \\mathcal{L}_{\\text{MIM}} + \\mathcal{L}_{\\text{Self-Distill}}$$

- **MIM 损失**: 预测被掩码 patch 的聚类标签
- **自蒸馏损失**: 教师网络和学生网络的输出一致（类似 DINO）

## 预训练方法

### 核心思想：DINO（全局语义）+ BEiT（局部细节）的统一

iBOT 的名字是 "Image BERT Pre-Training with Online Tokenizer"——它的目标是**把 BEiT 和 DINO 的优点融合到一个框架里**。BEiT 的 MIM 学习局部 patch 级别的语义，但需要离线训练的 dVAE（工程复杂）；DINO 的自蒸馏学习全局图像级别语义，但缺乏局部细节。iBOT 的解决方案是：**用教师网络的在线聚类替代离线 dVAE，同时在同一个框架里做 MIM 和自蒸馏**。

> iBOT 的一句话总结：用 DINO 的在线蒸馏方式来做 BEiT 的掩码图像建模——不需要离线训练 tokenizer，教师和学生一起成长。

### 训练流水线（Step by Step）

#### Step 1 — 在线 Tokenizer：教师网络聚类

与 BEiT 不同，iBOT 不需要预先训练的 dVAE 来提供离散 token。而是使用**在线 Tokenizer**：

1. 教师网络处理完整图像，输出每个 patch 的表征
2. 将教师输出的 patch 表征投影到一个 $K=8192$ 维空间
3. 用 Sinkhorn-Knopp 算法（在线聚类）将 patch 表征映射为伪标签（soft assignment）
4. 学生网络处理被掩码的图像，预测被掩码 patch 的伪标签

```
教师网络 (EMA更新): 完整图像 → patch表征 → Sinkhorn-Knopp聚类 → 伪标签
学生网络:            掩码图像 → patch表征 → 预测头 → 匹配伪标签
```

#### Step 2 — 自蒸馏：教师引导的全局学习

同时，iBOT 使用类似 DINO 的自蒸馏机制：

1. 教师网络看全局（大裁剪），学生网络看局部（多裁剪）
2. 学生输出与教师输出的 softmax 分布匹配
3. 教师通过动量 EMA 更新

$$\mathcal{L}_{\text{Self-Distill}} = H(P_t^{\text{[CLS]}}, P_s^{\text{[CLS]}})$$

#### Step 3 — 联合损失

$$\mathcal{L}_{\text{iBOT}} = \underbrace{\mathcal{L}_{\text{MIM}}}_{\text{掩码patch预测}} + \underbrace{\mathcal{L}_{\text{Self-Distill}}}_{\text{全局自蒸馏}}$$

**MIM 损失**：学生预测被掩码 patch 的聚类标签（交叉熵）
**自蒸馏损失**：[CLS] token 的教师-学生匹配

> 这两个损失在同一个网络中联合训练——视觉编码器既输出 [CLS] token（用于全局蒸馏），也输出 patch 级表征（用于 MIM 预测）。这种联合训练使得特征同时具备全局语义和局部细节。

#### Step 4 — 在线聚类的关键技术：Sinkhorn-Knopp

在线 tokenizer 的核心是 Sinkhorn-Knopp 算法，它与 DINO 中使用的 centering + sharpening 原理类似：

1. 计算教师网络中所有 patch 在 $K=8192$ 个聚类中心上的软分配矩阵
2. 通过交替行列归一化，确保**每个聚类中心被均等使用**（防止坍塌）
3. 温度参数控制软分配的锐度

> Sinkhorn-Knopp 的妙处：纯 k-means 硬分配会导致梯度不稳定，纯 softmax 会导致坍塌。Sinkhorn-Knopp 在最优点收敛性上有理论保证，使得在线聚类既稳定又有意义。

### 为什么 iBOT 能涌现语义分割？

iBOT 最令人惊叹的特性之一——类似于 DINO，它的自注意力图**自动涌现出语义分割**，而且比 DINO 更精细。

**原因**：MIM 损失让模型必须在 patch 级别上进行精细化预测，这迫使自注意力图具有空间局部性和语义一致性。结合自蒸馏损失提供的全局上下文，模型学会了**在注意力中自然形成"物体边界"**。

> 一个训练好的 iBOT ViT，居然可以用来做无监督的语义分割——这完全是预训练中"非刻意"产生的能力。

### iBOT vs BEiT vs MAE vs DINO

| 维度 | BEiT | MAE | DINO | iBOT |
|------|------|-----|------|------|
| 学习目标 | 离散 token（dVAE） | 像素值 | 全局语义（自蒸馏） | 在线聚类标签 + 自蒸馏 |
| Tokenizer | 离线 dVAE | 无 | 无 | **在线聚类** |
| 预训练任务 | MIM | MIM | 自蒸馏 | **MIM + 自蒸馏** |
| 线性探测 | 37.6% | 68.0% | 78.2% | **79.5%** |
| 微调 | 83.2% | 83.6% | 82.8% | **84.0%** |
| 语义分割涌现 | 弱 | 弱 | 强 | **最强** |
| 工程复杂度 | 高（需 dVAE） | 低 | 低 | 中 |

iBOT 是**唯一同时在线性探测和微调上都达到 SOTA 的方法**——这得益于 MIM（局部细节→好微调）和自蒸馏（全局语义→好线性探测）的互补。

### 详细训练配置

| 参数 | ViT-B/16 | ViT-L/16 |
|------|---------|---------|
| 输入尺寸 | 224×224 | 224×224 |
| Patch 大小 | 16×16 | 16×16 |
| 掩码率 | 10-50%（分块掩码） | 10-50% |
| 教师动量 τ | 0.996 → 1.0（余弦） | 0.996 → 1.0 |
| 聚类数 K | 8192 | 8192 |
| Sinkhorn 迭代 | 3 | 3 |
| 温度 | 0.1 | 0.1 |
| 优化器 | AdamW | AdamW |
| 学习率 | 5e-4 | 5e-4 |
| 学习率调度 | 余弦衰减 | 余弦衰减 |
| Warmup | 10 epoch | 10 epoch |
| 权重衰减 | 0.04 → 0.4 | 0.04 → 0.4 |
| Batch Size | 1024 | 1024 |
| Epoch | 300 | 300 |
| 全局裁剪 | 2×224² | 2×224² |
| 局部裁剪 | 8×96² | 8×96² |

### 预训练的迁移价值

1. **在线 tokenizer 的范式**：不需要离线训练 tokenizer，简化了 MIM 的工程流程
2. **MIM + 对比学习的统一**：证明了两种学习范式可以互补而非竞争
3. **DINOv2 的前身**：iBOT 的设计被直接整合到 DINOv2 的联合训练中
4. **涌现语义分割**：iBOT 的注意力图在无监督语义分割上的表现至今仍是一流水平
