# Mamba (Selective State Space Model)

## 基本信息

- **论文**: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- **作者**: Albert Gu, Tri Dao
- **发表**: arXiv, 2023

## 创新点

1. **选择性状态空间模型 (Selective SSM)**: 根据输入内容选择性地传递或遗忘信息
2. **线性复杂度**: 序列长度 $L$ 的计算复杂度为 $O(L)$，优于 Transformer 的 $O(L^2)$
3. **去除注意力机制**: 完全基于状态空间模型的序列建模

## 核心原理

### 状态空间模型 (SSM)

SSM 使用隐状态 $h(t)$ 建模输入 $x(t)$ 到输出 $y(t)$ 的映射：

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

### 选择性机制

Mamba 的核心改进是让参数 $(A, B, C)$ **随输入变化**：
- 模型可以选择性地关注或忽略特定输入
- 解决了传统 SSM 无法进行内容感知推理的问题

### 硬件感知算法

使用扫描算法 (Parallel Scan) 实现高效的并行计算。

## 预训练方法

### 核心思想：注意力机制的 $O(L^2)$ 复杂度是诅咒——用状态空间模型替换，降到 $O(L)$

Transformer 的 Self-Attention 是一个 $O(L^2)$ 的操作——序列长度翻倍，计算量翻四倍。当处理长序列（如长文档、DNA、高分辨率图像）时，这成为了致命瓶颈。

Mamba 的根本思想：**用选择性状态空间模型（Selective SSM）替换注意力机制**，使得序列建模的复杂度从 $O(L^2)$ 降至 $O(L)$，同时通过"选择性"机制保留了注意力机制最核心的能力——**内容感知**。

> Mamba = 状态空间模型 (SSM) + 选择性机制（输入依赖）+ 硬件感知扫描算法。它证明了 SSM 可以在语言建模上达到 Transformer 级别的质量，且推理速度不受序列长度影响。

### 训练流水线（Step by Step）

#### Step 1 — 理解状态空间模型 (SSM)

SSM 是一个经典的连续时间系统：

$$h'(t) = A h(t) + B x(t)$$
$$y(t) = C h(t) + D x(t)$$

其中 $h(t)$ 是隐状态，$x(t)$ 是输入，$y(t)$ 是输出。

**离散化**后的 SSM（通过零阶保持 ZOH）：

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t$$
$$y_t = C h_t$$

其中 $\bar{A} = \exp(\Delta A)$，$\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \Delta B$。

> 如果你学过控制理论，SSM 就像一个 RNN——但它的 $A$ 矩阵被精心设计（HiPPO 初始化），使得 $h_t$ 能够高效地记住长距离依赖。

#### Step 2 — Mamba 的选择性机制（核心创新）

传统 SSM（如 S4、H3）的 $(A, B, C, \Delta)$ 是**时间不变**的——无论输入是什么，状态转移方式都一样。

Mamba 的关键突破：让 $(B, C, \Delta)$ **随输入 $x_t$ 变化**：

$$B_t = s_B(x_t), \quad C_t = s_C(x_t), \quad \Delta_t = \text{softplus}(s_\Delta(x_t))$$

其中 $s_B, s_C, s_\Delta$ 是小型线性投影。

**选择性意味着什么？**

| 机制 | 传统 SSM | Mamba SSM |
|------|---------|----------|
| 记忆策略 | 固定（对所有 token 相同） | **自适应**（可以"选择性遗忘"） |
| 内容感知 | ✗（无法区分重要/不重要） | **✓**（根据 token 调整） |
| 类比 | 固定窗口的积分 | **类似 Attention 的"选择性关注"** |

> 选择性机制是 Mamba 弥补 SSM 与 Attention 性能差距的关键。它让 SSM 拥有了类似注意力机制的能力：根据当前内容决定"记住什么、忽略什么"。

#### Step 3 — Mamba Block

Mamba Block 的结构（替代 Transformer Block）：

```text
输入 x
  ↓
  ├→ LayerNorm → SSM(B_t, C_t, Δ_t) → SiLU →
  │                                            ↓ 门控
  └→ LayerNorm → Linear → SiLU ────────────→ 逐元素乘法 → Linear → 输出
```

Mamba Block 的巧妙之处在于：
- **H3 风格的"门控"**：类似于 LSTM 的遗忘门/输入门，控制信息流动
- **SiLU 激活**：非线性变换
- **残差连接**：缓解深层网络的退化

#### Step 4 — 硬件感知扫描算法

选择性 SSM 的一个挑战是：由于 $(B_t, C_t, \Delta_t)$ 随时间变化，无法像传统 SSM 那样使用 FFT 加速（FFT 要求卷积核是时不变的）。

Mamba 的解决方案：**硬件感知的并行扫描算法（Parallel Scan）**：

1. 将 SSM 的递归计算表示为**前缀和（Prefix Sum）**问题
2. 使用 GPU 优化的并行扫描（Blelloch scan）实现
3. 通过 kernel fusion 减少内存读写

**效率**：

| 操作 | 复杂度 | GPU 实现 |
|------|--------|---------|
| Self-Attention | $O(L^2 d)$ | FlashAttention 优化 |
| Mamba SSM | $O(L d)$ | **并行扫描 + kernel fusion** |
| 推理（自回归） | Transformer: $O(L^2)$ | Mamba: **$O(L)$** |

> Mamba 的推理效率优势在长序列上尤其明显：序列长度 1M 时，Transformer 需要 $10^{12}$ 次操作，Mamba 只需 $10^6$ 级别的操作。

### 详细训练配置

| 参数 | Mamba-370M | Mamba-1.4B | Mamba-2.8B |
|------|-----------|-----------|-----------|
| 层数 | 48 | 48 | 64 |
| 隐藏维度 | 1024 | 2048 | 2560 |
| SSM 状态维度 | 16 | 16 | 16 |
| 卷积核大小 | 4 | 4 | 4 |
| 扩展比 | 2 | 2 | 2 |
| 优化器 | AdamW | AdamW | AdamW |
| 学习率 | 3e-3 | 1.5e-3 | 1.5e-3 |
| Batch Size | 0.5M tokens | 0.5M tokens | 0.5M tokens |
| 训练数据 | Pile (300B tokens) | Pile | Pile |
| 训练步数 | — | — | — |

### Mamba vs Transformer

| 维度 | Transformer | Mamba |
|------|-----------|-------|
| 核心机制 | Self-Attention | Selective SSM |
| 训练复杂度 | $O(L^2)$ | **$O(L)$** |
| 推理复杂度 | $O(L^2)$（KV-cache 优化后 $O(L)$） | **$O(1)$ 每 token** |
| 长序列速度 | FlashAttention 接近 $O(L)$ | **天然 $O(L)$** |
| 语言建模质量 (3B) | 相当 | **相当** |
| DNA 建模 | 有限 | **强**（长序列优势） |
| 音频建模 | 有限 | **强** |

### 预训练的实用价值

1. **线性复杂度的序列建模**：破解了 Transformer 的 $O(L^2)$ 瓶颈
2. **长序列处理的新可能**：DNA（百万级碱基）、长文档、高分辨率图像
3. **推理效率的革命**：每 token 的推理时间是常数，不随序列增长
4. **启发式新架构**：催生了 Vision Mamba、Jamba、Mamba-2 等衍生工作
5. **选择性机制的创新**：为 SSM 注入了内容感知能力，缩小了与 Attention 的差距
