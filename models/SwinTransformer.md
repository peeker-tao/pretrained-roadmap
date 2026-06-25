# Swin Transformer

## 基本信息

- **论文**: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- **作者**: Ze Liu et al. (Microsoft)
- **发表**: ICCV 2021 (最佳论文奖)

## 创新点

1. **层次化特征图**: 类似 CNN 的多尺度特征金字塔结构
2. **移位窗口注意力 (Shifted Window)**: 在非重叠窗口间建立跨窗口连接
3. **线性计算复杂度**: 窗口注意力复杂度 $O(N)$ 而非全局注意力的 $O(N^2)$

## 核心原理

### 层次化结构

- 每经过一个 stage，特征图分辨率减半（类似 CNN 的下采样）
- 通道数倍增
- 适合检测、分割等稠密预测任务

### 移位窗口注意力

相邻两个 Swin Transformer Block 交替使用：
1. **W-MSA (Window MSA)**: 在规则窗口内做自注意力
2. **SW-MSA (Shifted Window MSA)**: 窗口移位后做自注意力，建立跨窗口连接

## 预训练方法

### 核心思想：让 ViT 像 CNN 一样分层

ViT 只有一个尺度的特征图（14×14），这对分类没问题，但对检测和分割（需要多尺度特征金字塔）很不友好。Swin Transformer 的核心创新是：**让 Transformer 像 CNN 一样逐层下采样**，产生多层次的特征图。这使 Swin 成为第一个真正适合做"通用骨干"（替代 ResNet）的 Transformer 架构。

### 训练流水线（Step by Step）

#### Step 1 — Patch Partition + Linear Embedding

输入图像 $H \times W \times 3$，通过 $4 \times 4$ 的卷积（stride=4）将图像分割为 $\frac{H}{4} \times \frac{W}{4}$ 个不重叠 patch。每个 patch 被映射到 $C$ 维（Swin-T 中 $C=96$）。

#### Step 2 — 四个阶段的层次化处理

| Stage | 输出尺寸 | 通道数 | Swin-T Block 数 | 功能 |
|-------|---------|--------|----------------|------|
| 1 | $\frac{H}{4} \times \frac{W}{4}$ | 96 | 2 | 浅层纹理 |
| 2 | $\frac{H}{8} \times \frac{W}{8}$ | 192 | 2 | 中层特征 |
| 3 | $\frac{H}{16} \times \frac{W}{16}$ | 384 | 6 | 语义特征 |
| 4 | $\frac{H}{32} \times \frac{W}{32}$ | 768 | 2 | 高层语义 |

这与 ResNet 的 4 个 stage 结构完全对等——所以 Swin 可以**直接替换**任何用 ResNet 作为骨干的检测/分割框架。

#### Step 3 — 窗口注意力（关键创新）

在每个 Swin Transformer Block 内，自注意力被限制在**局部窗口**（$M \times M$）内：

**W-MSA（Window Multi-head Self-Attention）**：
- 将特征图划分为 $M \times M$ 的规则窗口（默认 $M=7$）
- 每个 patch 只关注同窗口内的其他 patch
- 复杂度从 $O(N^2)$ 降至 $O(N \times M^2)$

**SW-MSA（Shifted Window MSA）**：
- 将窗口沿对角线方向偏移 $(\lfloor M/2\rfloor, \lfloor M/2\rfloor)$
- 在偏移后的新窗口中做自注意力
- 建立了原窗口之间的信息交换

**两个 Block 交替使用**：W-MSA → SW-MSA → W-MSA → ...

> 类比：W-MSA 是"在自己房间里和室友聊天"，SW-MSA 是"走到隔壁房间串门"。交替这两个操作确保了信息逐步在整个图像中流动。

#### Step 4 — Patch Merging（下采样）

每个 stage 之间通过 Patch Merging 实现 2× 下采样：

```text
[特征图: H×W×C] → 每2×2相邻patch拼接 → [H/2 × W/2 × 4C] → 线性投影 → [H/2 × W/2 × 2C]
```

这与 CNN 的 $2 \times 2$ pooling + stride 2 的功能完全对应。

### 预训练策略

#### 监督预训练（ImageNet）

Swin Transformer 可以使用标准的监督分类预训练，训练配置高度借鉴了现代 CNN 训练的最佳实践：

| 参数 | Swin-T | Swin-S | Swin-B | Swin-L |
|------|--------|--------|--------|--------|
| 窗口大小 M | 7 | 7 | 7 | 7 |
| 隐藏维度 C | 96 | 96 | 128 | 192 |
| 层数 (Stage) | {2,2,6,2} | {2,2,18,2} | {2,2,18,2} | {2,2,18,2} |
| 头数 | {3,6,12,24} | {3,6,12,24} | {4,8,16,32} | {6,12,24,48} |
| 参数量 | 28M | 50M | 88M | 197M |
| 优化器 | AdamW | AdamW | AdamW | AdamW |
| 学习率 | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| 权重衰减 | 0.05 | 0.05 | 0.05 | 0.05 |
| Batch Size | 1024 | 1024 | 1024 | 1024 |
| Epoch | 300 | 300 | 300 | 300 |
| Warmup | 20 epoch | 20 epoch | 20 epoch | 20 epoch |
| 学习率调度 | 余弦衰减 | 余弦衰减 | 余弦衰减 | 余弦衰减 |

#### 数据增强（现代训练配方）

Swin 使用了现代 CNN/ViT 训练的全套增强策略：

| 增强方法 | 参数 |
|---------|------|
| RandAugment | magnitude=9, magnitude-std=0.5 |
| Mixup | $\alpha = 0.8$ |
| CutMix | $\alpha = 1.0$ |
| Random Erasing | probability=0.25 |
| Label Smoothing | 0.1 |
| Stochastic Depth | 0.2 (Swin-T), 0.3 (Swin-S), 0.5 (Swin-B) |
| Repeated Augmentation | — |

#### ImageNet-22K 预训练 + ImageNet-1K 微调

Swin 的一个重要训练策略是：

**第一步**：在 ImageNet-22K（14M 图片，21841 类）上预训练 90 epoch
**第二步**：在 ImageNet-1K 上微调 30 epoch

两阶段训练带来了显著提升：

| 模型 | ImageNet-1K 直接训练 | 22K→1K | 提升 |
|------|-------------------|--------|------|
| Swin-B | 83.5% | **85.2%** | +1.7% |
| Swin-L | 83.5% | **86.3%** | +2.8% |

### Swin 预训练作为检测/分割骨干

Swin 作为预训练骨干的关键价值在于**多尺度特征输出**：

```text
Swin Stage 1 → C1 (H/4, W/4) → FPN → 小物体检测
Swin Stage 2 → C2 (H/8, W/8) → FPN → 中物体检测
Swin Stage 3 → C3 (H/16, W/16) → FPN → 大物体检测
Swin Stage 4 → C4 (H/32, W/32) → FPN → 全局语义
```

这使得 Swin 可以直接接入 Mask R-CNN、Cascade R-CNN、HTC 等检测/分割框架，**完全替代 ResNet 作为骨干网络**。

在 COCO 检测和 ADE20K 分割上的表现：

| 骨干 | 检测 (Mask R-CNN AP^box) | 分割 (UperNet mIoU) |
|------|------------------------|-------------------|
| ResNet-50 | 41.0 | 42.1 |
| Swin-T | **46.0** | **46.1** |
| ResNet-101 | 42.6 | 43.8 |
| Swin-S | **48.5** | **49.3** |
| Swin-B | **51.9** | **53.5** |

Swin 在检测和分割上大幅超越 ResNet（约 +5 AP），证明了层次化 Transformer 是优于 CNN 的通用视觉骨干。

### 预训练的迁移价值

1. **通用骨干的地位**：Swin 是第一个真正能"即插即用"替代 ResNet 的 Transformer 骨干
2. **层次化设计的典范**：后续的 ConvNeXt、PVT、CSWin 等都沿用了层次化思路
3. **窗口注意力的实用性**：线性复杂度让 Swin 可以处理高分辨率输入（800×1333 的 COCO 图像）
4. **多任务适配**：分类、检测、分割均有出色表现，不需要专门的架构调整
