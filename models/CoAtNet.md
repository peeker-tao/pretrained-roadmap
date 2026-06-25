# CoAtNet

## 基本信息

- **论文**: [CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://arxiv.org/abs/2106.04803)
- **作者**: Zihang Dai et al. (Google)
- **发表**: NeurIPS 2021

## 创新点

1. **卷积 + 注意力的系统融合**: 前几层用卷积（局部），后几层用注意力（全局）
2. **MBConv + Transformer 混合**: 类似 EfficientNet 的 MBConv 块 + Transformer 块
3. **所有数据规模的卓越性能**: 小数据优于 CNN，大数据优于 ViT

## 核心原理

### 设计原则

| 阶段 | 操作 | 原因 |
|------|------|------|
| S0-S1 (浅层) | MBConv (Depthwise Conv) | 强局部归纳偏置 |
| S2 (中层) | MBConv | 过渡 |
| S3-S4 (深层) | Transformer (Global MSA) | 全局关系建模 |

### 架构规格

| 版本 | 参数量 | ImageNet-21K Top-1 |
|------|--------|-------------------|
| CoAtNet-0 | 25M | 87.1% |
| CoAtNet-2 | 75M | 88.3% |
| CoAtNet-3 | 168M | 88.9% |
| CoAtNet-4 | 275M | 89.1% |

## 预训练方法

### 核心思想：卷积擅长局部（前几层），注意力擅长全局（后几层）——何不把两者按深度"拼"起来？

CoAtNet 是 CNN 和 ViT 的**深度维度融合**。其核心洞察：不同深度的层需要不同的操作类型。浅层（靠近输入）处理局部纹理、边缘——卷积是天然高效的选择。深层（靠近输出）做全局推理、物体间关系——注意力机制是最佳工具。在中间层，两者可以混合。

> CoAtNet = MBConv（浅层）+ Transformer（深层）= 在所有数据规模上都表现最好的视觉模型之一。小数据时卷积提供必要的归纳偏置，大数据时注意力释放全局建模能力。

### 训练流水线（Step by Step）

#### Step 1 — 架构设计原则：深度维度的操作分配

CoAtNet 的 5 个 Stage 使用不同操作：

| Stage | 操作 | 分辨率 | 通道数 | 设计理由 |
|-------|------|--------|--------|---------|
| S0 (Conv) | 2× 标准卷积 | H/2 × W/2 | 小 | 最浅层 → 强局部偏置 |
| S1 (MBConv) | MBConv 块 | H/4 × W/4 | 中 | 浅层 → 局部特征 |
| S2 (MBConv) | MBConv 块 | H/8 × W/8 | 中 | 过渡区 → 局部 + 中等范围 |
| S3 (TFMR) | Transformer 块 | H/16 × W/16 | 大 | 深层 → 全局建模 |
| S4 (TFMR) | Transformer 块 | H/32 × W/32 | 大 | 最深层 → 纯全局语义 |

**为什么 MBConv 在浅层？**
- MBConv（Mobile Inverted Bottleneck Conv）= 深度可分离卷积 + SE 注意力
- 局部感受野 + 通道注意力 → 高效提取多尺度局部特征
- 浅层特征图分辨率高（$H/4$），全局自注意力的 $O(N^2)$ 成本在这里是灾难性的

**为什么 Transformer 在深层？**
- 深层特征图分辨率低（$H/32$），自注意力的 $O(N^2)$ 可承受
- 深层需要跨空间的信息整合 → 全局注意力天然擅长

#### Step 2 — 相对位置编码的统一处理

CoAtNet 在 Transformer 阶段使用**相对位置编码**，但在卷积阶段使用**隐式位置编码**（卷积的局部感受野自带位置信息）。

这种设计的巧妙之处：
- 卷积阶段不需要显式位置编码（卷积核的位置是固定的）
- Transformer 阶段使用相对位置编码（更灵活，适应全局建模）

#### Step 3 — 预训练数据规模的关系

CoAtNet 的核心主张：**在所有数据规模上都最优**。

| 数据规模 | CNN 优？ | ViT 优？ | CoAtNet 优？ |
|---------|---------|---------|-------------|
| ImageNet-1K (1.2M) | ✓（归纳偏置帮助大） | ✗（欠拟合） | **✓（卷积浅层补偿）** |
| ImageNet-21K (14M) | ≈ | ≈ | **✓** |
| JFT-300M/3B | ✗（归纳偏置限制） | ✓ | **✓（深层注意力释放能力）** |

> 在 JFT-3B（30 亿张图）上，CoAtNet-4 达到 89.1% ImageNet Top-1——这是当时该 benchmark 的最高分之一。

#### Step 4 — 监督预训练配置

| 参数 | CoAtNet-0 | CoAtNet-3 | CoAtNet-4 |
|------|----------|----------|----------|
| 预训练数据 | ImageNet-1K | JFT-300M | JFT-3B |
| 分辨率 | 224² | 224² → 384² | 224² → 384² |
| 优化器 | AdamW | AdamW | AdamW |
| 学习率 | 按模型大小 | — | — |
| 权重衰减 | 0.05 | — | — |
| 数据增强 | RandAug + MixUp | RandAug + MixUp | RandAug + MixUp |
| Epoch | 300 | 90 + 30 (finetune) | 90 + 30 |

#### Step 5 — 从预训练到微调

```text
预训练阶段 (ImageNet-21K 或 JFT):
  → 大分辨率（224²）
  → 交叉熵 + 正则化
  → 90 epoch

微调阶段 (ImageNet-1K):
  → 更大分辨率（384² 或 512²）
  → 交叉熵
  → 30 epoch
  → 学习率降低 10×
```

### MBConv vs Transformer Block

| 组件 | MBConv | Transformer Block |
|------|--------|-------------------|
| 核心操作 | Depthwise Conv | Multi-Head Self-Attention |
| 感受野 | 局部（3×3 或 5×5） | 全局 |
| 计算复杂度 | $O(HW \cdot k^2)$ | $O((HW)^2)$ |
| 位置编码 | 隐式（卷积自带） | 显式（相对位置编码） |
| SE 注意力 | ✓（通道注意力） | — |
| 最佳使用深度 | 浅层（高分辨率） | 深层（低分辨率） |

### 详细训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据集 | ImageNet-1K / 21K / JFT | 视模型大小 |
| 优化器 | AdamW | — |
| 数据增强 | RandAug, MixUp, CutMix | 现代增强策略 |
| 学习率调度 | 余弦衰减 + Warmup | — |
| 正则化 | Dropout, Stochastic Depth | — |
| 微调分辨率 | 384² 或 512² | 预训练后用更高分辨率 |

### 预训练的实用价值

1. **深度维度融合的范式**：不同深度用不同操作——这是 CoAtNet 最核心的设计洞察
2. **小数据/大数据通吃**：ImageNet-1K 和 JFT-3B 上均达到 SOTA
3. **计算效率的优化**：浅层用高效卷积，深层用强大的自注意力——兼顾速度和精度
4. **CNN 与 ViT 的最优融合示范**：MBConv + Transformer 的组合被多个后续工作采纳
5. **高分辨率微调策略**：预训练 224² → 微调 384²+，充分利用大规模预训练
