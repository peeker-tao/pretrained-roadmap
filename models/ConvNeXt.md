# ConvNeXt

## 基本信息

- **论文**: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- **作者**: Zhuang Liu et al. (Meta)
- **发表**: CVPR 2022

## 创新点

1. **现代化 CNN**: 将 Swin Transformer 的设计理念反向应用于 CNN
2. **渐进式改进**: 系统地 Modernize 标准 ResNet

## 核心原理

### 改进点

1. **训练策略**: AdamW + 数据增强 (ImageNet 训练策略现代化)
2. **宏观设计**: Stage 比例调整为 3:3:9:3
3. **Patchify Stem**: 使用 4×4 卷积替代 7×7 卷积
4. **深度可分离卷积**: 使用 7×7 Depthwise Conv
5. **逆瓶颈 (Inverted Bottleneck)**: 同 Transformer
6. **大卷积核**: 7×7 替代 3×3
7. **各层微设计**: GELU, LayerNorm, 更少的激活

## 预训练方法

### 核心思想：用 Transformer 的训练方式训练 CNN

ConvNeXt 的核心洞察是：**CNN 本身没有过时，过时的是 CNN 的训练方式和设计细节**。如果给 CNN 配上 Transformer 的训练配方和现代化微设计，它依然可以匹敌甚至超越 ViT。这是一场"反方向"的实验——不是让 Transformer 更像 CNN（如 Swin），而是让 CNN 更像 Transformer。

### 训练流水线（Step by Step）

#### Step 1 — Patchify Stem：像 ViT 一样处理输入端

传统 ResNet 的 Stem 使用 7×7 卷积 + stride 2，然后是 3×3 max pooling + stride 2，总共做了 4× 下采样。ConvNeXt 将其改为：

```text
4×4 Conv, stride 4（通道数 96）
```

这与 Swin Transformer 的 patch partitioning 完全一致——把图片打成有重叠的 patch。这个改动减少了早期过多的不必要计算。

#### Step 2 — 现代化的宏观设计

| 设计元素 | ResNet-50 | Swin-T | ConvNeXt-T |
|---------|-----------|--------|------------|
| Stage 比例 | (3,4,6,3) | (2,2,6,2) | **(3,3,9,3)** |
| 通道数 | (256,512,1024,2048) | (96,192,384,768) | **(96,192,384,768)** |
| Stem | 7×7 Conv + Pool | 4×4 Conv stride 4 | 4×4 Conv stride 4 |
| 参数量 | 25.6M | 28.3M | 28.6M |

ConvNeXt 采用了与 Swin 对等的通道数和 Stage 比例，这使得公平比较成为可能——任何性能差异都来自架构差异，而非规模和计算量的差异。

#### Step 3 — 深度可分离卷积 + 逆瓶颈

ConvNeXt Block 的设计"刻意"模仿了 Transformer Block 的结构：

**Transformer Block 的处理逻辑**：
```
输入 → MSA (全局交互) → MLP (通道混合) → 输出
```

**ConvNeXt Block 的处理逻辑**：
```
输入 → 7×7 Depthwise Conv (空间交互) → LayerNorm
     → 1×1 Conv (升维 4×) → GELU → 1×1 Conv (降维) → 输出
```

关键设计对应：
| Transformer | ConvNeXt | 功能 |
|------------|----------|------|
| 多头自注意力（MSA） | 7×7 深度可分离卷积 | 空间信息交互 |
| MLP（升维 4× → 降维） | 1×1 Conv（升96→384→降96） | 通道混合 |
| LayerNorm | **LayerNorm**（非 BN） | 归一化 |
| GELU | **GELU**（非 ReLU） | 激活函数 |
| 残差连接（Scale<1） | 更少的归一化层 | 简化 |

#### Step 4 — 现代化训练配方（最关键的变化）

ConvNeXt 不再使用 ResNet 时代的"老式训练法"（SGD + Step Decay + 基础增强），而是全盘采用 ViT 的训练配方：

**旧配方（ResNet 时代，2015-2019）**：
- SGD + Momentum，阶梯式学习率衰减
- 基础增强：随机裁剪 + 翻转
- 90 epoch

**新配方（ViT/Swin 时代，2020-）**：
- AdamW，余弦衰减 + Warmup
- 全系列增强：RandAugment + Mixup + CutMix + Random Erasing
- 300 epoch

这个训练配方的现代化**单独贡献了约 2-3% 的 ImageNet Top-1 提升**——这说明很多"CNN 不如 Transformer"的结论，其实混淆了架构差异和训练配方差异。

### 监督预训练配置

| 参数 | ConvNeXt-T | ConvNeXt-S | ConvNeXt-B | ConvNeXt-L |
|------|-----------|-----------|-----------|-----------|
| 通道数 C | 96 | 96 | 128 | 192 |
| Blocks per Stage | (3,3,9,3) | (3,3,27,3) | (3,3,27,3) | (3,3,27,3) |
| 参数量 | 28.6M | 50.2M | 88.6M | 197.8M |
| FLOPs | 4.5G | 8.7G | 15.4G | 34.4G |
| 优化器 | AdamW | AdamW | AdamW | AdamW |
| 学习率 | 4e-3 | 4e-3 | 4e-3 | 4e-3 |
| 权重衰减 | 0.05 | 0.05 | 0.05 | 0.05 |
| Batch Size | 4096 | 4096 | 4096 | 4096 |
| Epoch | 300 | 300 | 300 | 300 |
| Warmup | 20 epoch | 20 epoch | 20 epoch | 20 epoch |
| 学习率调度 | 余弦衰减 | 余弦衰减 | 余弦衰减 | 余弦衰减 |
| Stochastic Depth | 0.1 | 0.4 | 0.5 | 0.5 |
| Label Smoothing | 0.1 | 0.1 | 0.1 | 0.1 |
| RandAugment | (9, 0.5) | (9, 0.5) | (9, 0.5) | (9, 0.5) |
| Mixup α | 0.8 | 0.8 | 0.8 | 0.8 |
| CutMix α | 1.0 | 1.0 | 1.0 | 1.0 |
| Random Erasing | 0.25 | 0.25 | 0.25 | 0.25 |

### 自监督预训练：ConvNeXt V2（FCMAE）

ConvNeXt V2 进一步发展了自监督预训练方案——**全卷积掩码自编码器（FCMAE）**：

**核心思路**：将 MAE 的"掩码+重建"范式应用于纯卷积网络。

| 设计选择 | ViT (MAE) | ConvNeXt (FCMAE) |
|---------|-----------|-----------------|
| 掩码方式 | 随机掩码 patch | 随机掩码像素（稀疏卷积） |
| 编码器 | ViT 编码器 | ConvNeXt 编码器 |
| 解码器 | 轻量 ViT 解码器 | 轻量 ConvNeXt 解码器 |
| 重建目标 | 掩码 patch 像素 | 掩码区域像素 |
| 关键技巧 | 非对称编码-解码 | 稀疏卷积避免无效计算 |

FC-MAE 使得 ConvNeXt 可以像 ViT 一样做高效的自监督预训练，且训练速度和下游性能都十分出色。

### 监督预训练性能

| 模型 | ImageNet-1K Top-1 | 参数量 | FLOPs |
|------|------------------|--------|-------|
| ResNet-50 | 76.1% | 25.6M | 4.1G |
| ConvNeXt-T | **82.1%** | 28.6M | 4.5G |
| Swin-T | 81.3% | 28.3M | 4.5G |
| ConvNeXt-S | **83.1%** | 50.2M | 8.7G |
| Swin-S | 83.0% | 49.6M | 8.7G |
| ConvNeXt-B | **83.8%** | 88.6M | 15.4G |
| Swin-B | 83.5% | 87.8M | 15.4G |
| ConvNeXt-L | **84.3%** | 197.8M | 34.4G |

ConvNeXt 在几乎相同的参数和 FLOPs 下，略优于对标的 Swin Transformer——证明了**现代化 CNN 完全可以匹敌 Transformer**。

### 预训练的迁移价值

1. **"CNN 没有死"**：ConvNeXt 是对"CNN 已死"叙事的有力回应——好的设计比架构类型更重要
2. **证明训练配方的重要性**：2-3% 的性能提升仅来自训练配方的现代化
3. **兼容现有的 CNN 生态**：所有为 ResNet 设计的推理优化框架、部署工具、量化方案都可以直接用于 ConvNeXt
4. **纯卷积的高效推理**：在移动端和边缘设备上，纯卷积比 Transformer 更易部署和加速
