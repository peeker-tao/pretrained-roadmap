# TimeSformer (Time-Space Transformer)

## 基本信息

- **论文**: [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095)
- **作者**: Gedas Bertasius, Heng Wang, Lorenzo Torresani (Meta)
- **发表**: ICML 2021

## 创新点

1. **分割的时空注意力**: 将时空注意力分解为空间注意力和时间注意力
2. **高效视频建模**: 相比 3D 卷积，计算量大幅降低

## 核心原理

### Divided Space-Time Attention

1. **空间注意力**: 在单帧内计算 patch 间的注意力
2. **时间注意力**: 跨帧的同一位置 patch 间计算注意力
3. 两者串行执行：先空间后时间

### 注意力变体

| 变体 | 描述 | 计算量 |
|------|------|--------|
| 单帧空间 | 只做空间注意力 | 低 |
| 分割时空 | 空间 + 时间 | 中 |
| 联合时空 | 时空同时 | 高 |

## 预训练方法

### 核心思想：视频是 3D 数据（时间×空间），但 3D 卷积和联合时空注意力计算量都太大了。TimeSformer 把时空注意力拆成两步——先看空间的物体（空间注意力），再看它们如何随时间变化（时间注意力）——计算量从 $O((THW)^2)$ 降到 $O(THW^2 + T^2HW)$

TimeSformer 是第一个证明"纯 Transformer 可以在视频理解上达到 SOTA"的工作。它的核心创新 **Divided Space-Time Attention** 既保持了 Transformer 的全局建模能力，又将计算复杂度控制在了可接受的范围内。

> TimeSformer = Divided Space-Time Attention（先空间后时间的分步注意力）+ ViT-style Backbone。在 Kinetics-400 上以 3× 更少的计算量超越了 I3D 和 SlowFast 等 3D CNN。

### 训练流水线（Step by Step）

#### Step 1 — 视频 Tokenization

```text
视频: T 帧 × H × W × 3
Patch 化: 每帧切分为 P×P patch
  → T × (H/P) × (W/P) 个 token
  如: 8 × 14 × 14 = 1568 个 token (16×16 patch)
```

每个 token 加上**空间位置编码**（同类 token 在不同帧共享）和**时间位置编码**（区分不同帧）。

#### Step 2 — Divided Space-Time Attention

这是 TimeSformer 的核心。标准联合时空注意力：

$$A_{\text{joint}} = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V, \quad Q,K,V \in \mathbb{R}^{THW \times d}$$

**问题**：Attention 矩阵大小是 $(THW) \times (THW)$ → 对 $T=8, H=14, W=14$ 就是 $1568 \times 1568$。

**TimeSformer 的解决方案**：拆成两步：

**Step 2a — 空间注意力**（逐帧计算）：

```text
对于第 t 帧的所有 HW 个 token:
  Q_t, K_t, V_t ∈ R^{HW × d}
  A_space(t) = softmax(Q_t K_t^T / √d) V_t
```

每帧独立计算空间自注意力——所有帧共享权重但处理不同的帧。

**Step 2b — 时间注意力**（逐空间位置计算）：

```text
对于空间位置 (i,j) 的所有 T 个 token:
  Q_ij, K_ij, V_ij ∈ R^{T × d}
  A_time(ij) = softmax(Q_ij K_ij^T / √d) V_ij
```

对每个空间位置，沿时间轴计算自注意力——不同位置独立但共享权重。

| 注意力类型 | 计算复杂度 | 关注什么 |
|-----------|----------|---------|
| 空间注意力 | $T \cdot O((HW)^2)$ | 同一帧内的物体关系 |
| 时间注意力 | $HW \cdot O(T^2)$ | 同一位置随时间的变化 |
| 联合时空 | $O((THW)^2)$ | 所有 token 之间的全局关系 |

> **复杂度对比**：Divided Space-Time 是 $O(THW^2 + T^2HW)$，联合是 $O(T^2H^2W^2)$。当 $T=8, HW=196$ 时：Divided ≈ 8×38416 + 196×64 = 319K，联合 ≈ 2.46M——**约 8 倍的差距**。

#### Step 3 — 五种时空注意力变体

TimeSformer 比较了五种排列方式：

| 变体 | 排列 | 效果 |
|------|------|------|
| **Space Only** | 仅空间注意力（忽略时间） | 最差（没有时序理解） |
| **Joint Space-Time** | 全局联合注意力 | 理论上最好，但计算爆炸 |
| **Divided S-T** | **先空间 → 后时间** | **最佳性价比** |
| **Divided T-S** | 先时间 → 后空间 | 略差（时间注意力缺少空间上下文） |
| **Sparse Local Global** | 局部注意力 + 全局注意力 | 中等 |

**Divided S-T 效果最佳的原因**：
- 先空间再时间 → 时间注意力已经收到了"第一帧有哪些物体"的信息 → 可以更好地跟踪物体的运动
- 先时间再空间 → 空间注意力收到的是"同一位置在不同帧的变化"信息 → 缺少物体的全局语义

#### Step 4 — 图像预训练（关键！）

TimeSformer 的一个关键发现：**用图像预训练初始化空间注意力，视频微调时再加入时间注意力**：

| 参数 | 图像预训练 | 说明 |
|------|----------|------|
| 数据集 | ImageNet-21K (14M 图像) | **纯图像数据——非视频！** |
| 输入 | 单帧 224×224 | 标准图像 |
| 架构 | 只使用空间注意力（无时间分支） | 类 ViT |
| 损失 | 交叉熵 | — |
| Epoch | 300 | — |

**为什么图像预训练有效？** TimeSformer 的空间注意力模块与标准 ViT 的自注意力完全相同——图像预训练的权重可以直接加载到空间注意力部分。时间注意力随机初始化，在视频数据上微调即可。

#### Step 5 — 视频微调

| 参数 | 视频微调 | 说明 |
|------|---------|------|
| 数据集 | Kinetics-400/600 | 视频分类 |
| 帧数 | 8-96 帧 × 224² | 空间分辨率保持 |
| 时间注意力 | 随机初始化 | 从头学习 |
| 空间注意力 | 加载图像预训练权重 | 迁移学习 |
| 损失 | 交叉熵 | — |
| 优化器 | SGD + Momentum | — |
| 学习率 | 0.005 | 小 lr（fine-tune 模式） |
| Epoch | 30 | — |

### TimeSformer vs 3D CNN

| 维度 | SlowFast (3D CNN) | TimeSformer |
|------|------------------|-------------|
| 核心操作 | 3D 卷积 | **Divided Space-Time Attention** |
| 时序建模 | 快路径（高帧率）+ 慢路径 | **逐位置时间注意力** |
| 计算量 | 高 | **中（~3× 低于 SlowFast）** |
| Kinetics-400 Top-1 | 79.8% | **80.7%** |
| 预训练数据 | Kinetics-400（视频） | **ImageNet-21K（图像！）** |
| 长视频支持 | 有限 | **96 帧（~3.2 秒@30fps）** |

### 详细训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 图像预训练数据 | ImageNet-21K (14M) | 与 ViT 共享 |
| 视频微调数据 | Kinetics-400/600 | — |
| 输入帧数 | 8 或 96 | 8 帧 = 标准，96 帧 = 长视频 |
| Patch 大小 | 16×16 | — |
| 编码器层数 | 12 (ViT-B) / 24 (ViT-L) | 每层含空间+时间注意力 |
| 优化器 | SGD + Momentum(0.9) | — |
| 学习率 | 0.005（微调） | — |
| Batch Size | 64 | — |

### 预训练的实用价值

1. **图像预训练 → 视频微调的范式**：证明了视觉 Transformer 的图像预训练可以直接迁移到视频任务
2. **计算效率的优化**：Divided Attention 将视频 Transformer 从"不可行"变为"可行"
3. **长视频理解**：96 帧输入能力 → 适合慢动作分析、异常检测等长时任务
4. **纯 Transformer 视频 baseline**：为 VideoMAE、VideoCLIP 等工作奠定了基础
5. **时空分解的方法论**：分离空间和时间处理 → 被 ViViT、Motionformer 等继承
