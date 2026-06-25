# EfficientNet

## 基本信息

- **论文**: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- **作者**: Mingxing Tan, Quoc V. Le (Google)
- **发表**: ICML 2019

## 创新点

1. **复合缩放 (Compound Scaling)**: 同时缩放深度 (depth)、宽度 (width) 和分辨率 (resolution) 三个维度
2. **NAS 基线网络**: 使用神经架构搜索 (MnasNet) 找到的 EfficientNet-B0 作为基线
3. **效率极高**: 在同等参数量下达到 SOTA 性能，EfficientNet-B7 用更少参数超越当时所有模型

## 核心原理

### 复合缩放

传统方法通常只在一个维度上缩放网络，EfficientNet 发现**联合缩放三个维度**效果更好：

$$\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi$$

其中 $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$（FLOPs 约束），$\phi$ 是用户指定的缩放系数。

### MBConv 模块

EfficientNet 的核心构建块是 MobileNetV2 的 MBConv（Mobile Inverted Bottleneck Conv）：
- 1×1 升维 → 3×3/5×5 Depthwise Conv → SE 模块 → 1×1 降维
- 使用 Swish 激活函数和 DropConnect

### 网络变体

| 模型 | 深度系数 | 宽度系数 | 分辨率 | Top-1 (ImageNet) |
|------|---------|---------|--------|-----------------|
| B0 | 1.0 | 1.0 | 224 | 77.1% |
| B3 | 1.8 | 1.2 | 300 | 81.1% |
| B7 | 3.1 | 2.0 | 600 | 84.3% |

## 预训练方法

### 核心思想：同时放大深度、宽度和分辨率

以前的模型缩放是很"粗暴"的——ResNet 从 50 层变成 101 层、152 层，只改变深度；Wide ResNet 只改变宽度。EfficientNet 的核心洞察是：**深度（层数）、宽度（通道数）、分辨率（输入尺寸）三者不是独立的**——更深需要更宽来捕获更多特征，更宽需要更高分辨率来提供更多细节。只有三者协调缩放，才能让模型在给定的计算预算下达到最优性能。

> EfficientNet 的复合缩放公式：$\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi$，其中 $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$。这组系数是通过网格搜索在小型基线网络 (B0) 上找到的，然后用同一个 $\phi$ 放大到更大的变体。

### 训练流水线（Step by Step）

#### Step 1 — 基线网络：EfficientNet-B0

EfficientNet 的基线网络 B0 不是手工设计的——它是通过**神经架构搜索（NAS）**使用 MnasNet 框架找到的。

B0 的核心组件是 **MBConv（Mobile Inverted Bottleneck Convolution）**：

```text
输入 (H×W×C_in)
  ↓
1×1 Conv (升维: C_in → C_in × expand_ratio, 通常 6×)
  ↓ BN + Swish
  ↓
k×k Depthwise Conv (k=3或5, 不改变通道数)
  ↓ BN + Swish
  ↓
SE (Squeeze-and-Excitation) 模块: 全局池化 → FC(降维) → FC(升维) → Sigmoid → 通道加权
  ↓
1×1 Conv (降维: C_in×expand_ratio → C_out)
  ↓ BN
  ↓
残差连接 (如果 C_in = C_out 且 stride = 1)
  ↓
输出
```

SE 模块是 MBConv 的一个重要组件——它为每个通道学习一个权重，让模型"关注重要的通道，忽略不重要的通道"。这像是在告诉模型："在理解这张图时，边缘检测通道比纹理检测通道更重要——听边缘的。"

#### Step 2 — 复合缩放的搜索

在 EfficientNet-B0（基线网络）上，用网格搜索找到最佳的缩放系数：

1. **固定 $\phi=1$**（即先不放大）
2. 搜索 $\alpha$（深度系数）、$\beta$（宽度系数）、$\gamma$（分辨率系数）
3. 约束：$\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$（总 FLOPs 约 2× B0）
4. 对 EfficientNet-B0 搜索得到：$\alpha=1.2, \beta=1.1, \gamma=1.15$

> 为什么用 $\beta^2$ 和 $\gamma^2$？因为在卷积网络中，宽度（通道数）翻倍和分辨率翻倍都会导致 FLOPs 翻两倍。这个约束确保所有变体在效率曲线上都是最优的。

#### Step 3 — 按 $\phi$ 放大网络

有了 $\alpha$, $\beta$, $\gamma$ 三个系数后，只需改变 $\phi$ 即可得到不同大小的网络：

| 模型 | $\phi$ | 深度 (×1.2^φ) | 宽度 (×1.1^φ) | 分辨率 (×1.15^φ) |
|------|------|-----------|-----------|----------------|
| B0 | 0 | 1.0× | 1.0× | 224 |
| B1 | 1 | 1.2× | 1.1× | 240 |
| B2 | 2 | 1.4× | 1.2× | 260 |
| B3 | 3 | 1.8× | 1.4× | 300 |
| B4 | 4 | 2.2× | 1.5× | 380 |
| B5 | 5 | 2.6× | 1.7× | 456 |
| B6 | 6 | 3.1× | 1.9× | 528 |
| B7 | 7 | 3.7× | 2.1× | 600 |

> B7 在 600×600 的分辨率上进行训练——这意味着训练速度极慢，但也是精度最高的 EfficientNet 变体。

### 详细训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据集 | ImageNet-1K | 128 万张图片，1000 类 |
| 优化器 | RMSProp | decay=0.9, momentum=0.9, ε=0.1 |
| 初始学习率 | 0.256 | 较高的起始学习率 |
| 学习率调度 | 指数衰减 0.97 / 2.4 epoch | 较缓慢的衰减 |
| Batch Size | 4096 | 大的 batch size |
| 权重衰减 | 1e-5 | 很小的权重衰减 |
| Epoch | 350 | 比标准的 90 epoch 长得多 |
| 数据增强 | AutoAugment + 随机裁剪 + 水平翻转 | — |
| Stochastic Depth | survival prob 0.8 | 随机丢弃层 |
| DropConnect | 是（替代 Dropout） | 随机丢弃权重连接 |
| 激活函数 | Swish | $x \cdot \sigma(x)$ |
| Label Smoothing | 0.1 | — |

#### 数据增强：AutoAugment

EfficientNet 使用 AutoAugment 自动数据增强策略。AutoAugment 用强化学习搜索最佳增强组合（如 "ShearX, 0.3 | Invert, 0.1 | ..."），找到的策略显著优于手工设计的增强。

#### 为什么使用 RMSProp 而不是 Adam/AdamW？

这是 EfficientNet 与众不同的一个训练选择。论文发现 RMSProp + 大 batch + 长训练在 EfficientNet 上比 Adam 表现更好。RMSProp 的自适应学习率归一化在深度可分离卷积的梯度分布中特别有效。

#### DropConnect

DropConnect 是 Dropout 的一个变种——不是随机置零神经元的输出，而是随机置零**权重矩阵的元素**。它比 Dropout 提供了更强的正则化效果，对于 MBConv 中稀疏的深度可分离卷积特别有效。

### 预训练性能

| 模型 | 参数量 | FLOPs | ImageNet Top-1 |
|------|--------|-------|---------------|
| ResNet-50 | 25.6M | 4.1B | 76.1% |
| EfficientNet-B0 | 5.3M | 0.39B | 77.1% |
| ResNet-152 | 60.2M | 11.6B | 78.3% |
| EfficientNet-B3 | 12.2M | 1.8B | 81.6% |
| EfficientNet-B5 | 30.0M | 9.9B | 83.6% |
| EfficientNet-B7 | 66.4M | 37.7B | **84.4%** |

EfficientNet-B0 用 ResNet-50 的 1/10 的 FLOPs 达到了更高的精度，B7 以相似的参数量大幅超越 ResNet-152。

### 预训练的迁移价值

1. **复合缩放的系统方法**：不再靠经验拍板，而是用数学约束来指导网络放大
2. **极高的参数效率**：B0 只有 5.3M 参数却达到 77.1%，极为适合移动端部署
3. **NAS + 手工优化的结合**：用 NAS 找基线（B0），用复合缩放放大——这是"AI+人"协作的典范
4. **标准视觉骨干**：EfficientNet 在分类、检测、分割上都有广泛应用
5. **EfficientNetV2** 进一步改进了训练速度和对小 batch 的适应性
