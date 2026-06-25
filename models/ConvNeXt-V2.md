# ConvNeXt V2

## 基本信息

- **论文**: [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)
- **作者**: Sanghyun Woo et al. (Meta)
- **发表**: CVPR 2023

## 创新点

1. **全卷积掩码自编码器 (FCMAE)**: 为 ConvNeXt 设计的纯卷积 MIM 框架
2. **GRN (Global Response Normalization)**: 增强特征多样性，提升高层表征质量
3. **与 ViT 的自监督协同设计**: ConvNeXt + FCMAE 媲美 ViT + MAE

## 核心原理

### FCMAE

- ConvNeXt 作为编码器
- 使用**稀疏卷积**处理被掩码的图像
- 解码器重建像素

### GRN (Global Response Normalization)

$$\\text{GRN}(X) = \\gamma \\cdot \\frac{X}{\\|X\\|_2} + \\beta$$

- 全局归一化增强不同通道之间的竞争
- 防止特征坍塌，提升大模型训练稳定性

## 预训练方法

### 核心思想：MAE 已经证明了 MIM 在 ViT 上的力量——ConvNeXt V2 证明 CNN 同样可以，而且需要一个"CNN 特供版"的 MAE

ConvNeXt V1 仅使用了监督预训练（ImageNet-21K 分类）。ConvNeXt V2 的核心贡献是**为 ConvNeXt 设计了专用的自监督预训练框架 FCMAE（Fully Convolutional Masked Autoencoder）**，以及配套的 GRN（Global Response Normalization）技术来防止 CNN 在 MIM 中的特征坍塌。

> ConvNeXt V2 = FCMAE（卷积版 MAE）+ GRN（全局响应归一化）+ ConvNeXt 架构。它证明了 CNN + MIM 的组合可以匹敌甚至超越 ViT + MAE。

### 训练流水线（Step by Step）

#### Step 1 — FCMAE：为 CNN 定制的掩码自编码器

MAE 基于 ViT：将图像切分为 patch，随机丢弃一部分 patch，Encoder 只处理可见 patch。CNN 没有"patch"的概念——它的基本计算单元是滑动窗口。

**FCMAE 的解决方案：使用稀疏卷积处理掩码**。

```text
MAE (ViT):                     FCMAE (CNN):
  图像 → Patch化               图像 → 掩码
  → 丢弃被遮 patch             → 稀疏卷积 Encoder（跳过遮住区域）
  → Transformer Encoder        → 轻量卷积 Decoder
  → 拼接 mask tokens           → 预测被遮像素
  → Transformer Decoder
```

**稀疏卷积的关键**：只在有效像素（未被掩码的像素）上计算卷积。当 60% 的图像被遮住时，稀疏卷积只计算 40% 的区域——节省约 60% 的计算量。

#### Step 2 — GRN（Global Response Normalization）

GRN 是 ConvNeXt V2 的重要组件，旨在防止**特征坍塌**：

$$\text{GRN}(X) = \gamma \cdot \frac{X}{\|X\|_2} + \beta$$

其中 $\|X\|_2$ 是**全局**（跨空间和通道）的 L2 范数。

| 归一化方式 | 范围 | 效果 |
|-----------|------|------|
| BatchNorm | 每个通道独立 | 通道间无竞争 |
| LayerNorm | 跨通道 | 通道间隐式竞争 |
| **GRN** | **全局（所有像素+所有通道）** | **强通道竞争 → 防止坍塌** |

**为什么 GRN 防坍塌？**

在 MIM 训练中，CNN 容易陷入"懒惰解"——所有通道学同样的东西，输出的特征图高度冗余。GRN 的全局归一化引入了**零和竞争**：如果一个通道的响应太强，它会被归一化减弱——这迫使不同通道学习不同的特征。

> 类比：GRN 就像给每个通道设置了"预算限制"——如果你这个通道在这个位置花了很多激活预算，那别的通道就会被迫在这里"省着点"。结果就是每个通道必须"精打细算"，只在自己最擅长的位置激活。

#### Step 3 — FCMAE 的训练流程

```text
1. 输入图像 224×224
2. 随机生成 60% 掩码（大块掩码）
3. 稀疏卷积 Encoder 处理可见区域
   - 不对遮住区域做任何计算（节省计算）
4. 卷积 Decoder 处理：
   - Encoder 输出 + mask tokens → 预测完整图像
5. L2 损失：仅计算被遮区域的像素误差
6. 反向传播
```

**掩码策略**：

| 参数 | FCMAE | MAE (ViT) |
|------|-------|-----------|
| 掩码率 | 60% | 75% |
| 掩码块大小 | 32×32 | 随机 patch |
| Encoder 处理 | 稀疏卷积（跳过遮住区域） | 仅处理可见 patch |
| Decoder | 轻量卷积 | 轻量 Transformer |

#### Step 4 — 监督预训练（与 V1 相同）

ConvNeXt V2 同样支持监督预训练，配置与 V1 一致：

| 阶段 | 数据 | 分辨率 | Epoch |
|------|------|--------|-------|
| 预训练 | ImageNet-21K (14M) | 224² | 90 |
| 微调 | ImageNet-1K | 224² / 384² | 30 |

### FCMAE + 监督微调的两阶段范式

| 阶段 | 方法 | 数据 | 目标 |
|------|------|------|------|
| **Stage 1（预训练）** | FCMAE 自监督 | ImageNet-1K（无标签） | 像素重建 |
| **Stage 2（微调）** | 监督微调 | ImageNet-1K（有标签） | 分类 |

> FCMAE 预训练为 ConvNeXt 提供了比随机初始化好得多的起点——就像 BERT 的 MLM 预训练为 Transformer 提供了语言理解的基础。

### 详细训练配置（FCMAE）

| 参数 | ConvNeXt V2 FCMAE | 说明 |
|------|------------------|------|
| 数据集 | ImageNet-1K（无标签） | 128 万张 |
| 编码器 | ConvNeXt (V2 架构) | GRN + 稀疏卷积 |
| Decoder | 轻量卷积 Decoder | 逐点卷积 |
| 掩码率 | 60% | 大块掩码 |
| 掩码块大小 | 32×32 | — |
| 损失 | L2（MSE） | 仅掩码区域 |
| 优化器 | AdamW | — |
| 学习率 | 8e-4 | 余弦衰减 |
| Batch Size | 4096 | — |
| Epoch | 800（预训练）+ 100（微调） | — |

### ConvNeXt V2 vs ViT + MAE

| 维度 | ConvNeXt V2 + FCMAE | ViT + MAE |
|------|---------------------|-----------|
| 架构类型 | CNN | Transformer |
| MIM 方式 | 稀疏卷积 | 丢弃 patch |
| 计算效率 | 高（稀疏卷积） | 高（仅处理可见 patch） |
| 最高性能 (ImageNet-1K) | **88.9% (ConvNeXt V2-H)** | 87.8% (ViT-H MAE) |
| 迁移学习 | 好（CNN 有局部偏置） | 好 |

> ConvNeXt V2-H（660M 参数，FCMAE 预训练）在 ImageNet-1K 上达到 88.9% Top-1——这是当时纯 CNN 架构的新高度。

### 预训练的实用价值

1. **CNN + MIM 的成功示范**：打破了"MIM 只适用于 ViT"的认知
2. **FCMAE 的稀疏卷积框架**：为纯卷积架构提供了计算高效的 MIM 方案
3. **GRN 的防坍塌机制**：在 CNN 自监督训练中证明了通道竞争的重要性
4. **两阶段预训练范式**：自监督 + 监督微调，充分利用有标签和无标签数据
5. **CNN 的复兴信号**：证明 CNN 在大规模预训练中仍然可以与 ViT 竞争
