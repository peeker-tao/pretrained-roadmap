# Vision Mamba (Vim)

## 基本信息

- **论文**: [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)
- **作者**: Lianghui Zhu et al.
- **发表**: ICML 2024

## 创新点

1. **Mamba 用于视觉**: 将状态空间模型（SSM）应用于视觉任务
2. **双向 SSM**: 类似 ViT 的双向上下文建模
3. **线性复杂度**: 相比 ViT 的 $O(N^2)$，Vision Mamba 为 $O(N)$

## 核心原理

### 架构

1. **Patch Embedding**: 同 ViT，将图像分割为 patch
2. **Vim Block**: Mamba SSM + MLP
3. **双向处理**: 前向和后向各一个 Mamba 模块

### 与 ViT 的对比

| 特性 | ViT | Vision Mamba |
|------|-----|-------------|
| 核心操作 | Self-Attention | SSM (Mamba) |
| 复杂度 | $O(N^2)$ | $O(N)$ |
| 长序列 | GPU 内存爆炸 | 高效 |
| 高分辨率 | 需要窗口/分片 | 天然友好 |

## 预训练方法

### 核心思想：Transformer 的自注意力需要每个 patch 和所有其他 patch 交互 → $O(N^2)$ 复杂度 → 高分辨率图像处理非常吃力。Vision Mamba 用状态空间模型（SSM）替换自注意力——SSM 像"海浪"一样，信息沿着 patch 序列方向传播，每次只和"邻居"交互——复杂度降到 $O(N)$

Vision Mamba（Vim）是 Mamba（Selective State Space Model）在视觉领域的应用。它的核心主张：**线性复杂度的 SSM 可以在保持 Transformer 质量的同时，大幅降低高分辨率图像的计算开销**。

> Vision Mamba = Patch Embedding（同 ViT）+ 双向 Mamba 块（正向 SSM + 反向 SSM）+ MLP。复杂度从 ViT 的 $O(N^2)$ 降为 $O(N)$，输入分辨率可以翻倍而计算量仅线性增长。

### 训练流水线（Step by Step）

#### Step 1 — 图像 Patch 化 + 序列化

与 ViT 相同的第一步，但多了"序列化"：

```text
图像: 224×224×3
  → Patch: 16×16 → 14×14 = 196 个 patch
  → Patch Embedding: 196 × d
```

**然后关键步骤**：将 2D patch 排列为 1D 序列（SSM 需要有序输入）：

| 序列化方式 | 描述 | 效果 |
|----------|------|------|
| 行扫描（Raster） | 逐行展开 | 简单但效果一般 |
| 希尔伯特曲线 | 保持空间 2D 连续性 | **最佳** |
| 蛇形扫描 | 相邻行反向拼接 | 较好 |
| 随机 | 打乱 patch 顺序 | 差（破坏空间结构） |

> 希尔伯特曲线扫描：将 2D 空间中的相邻 patch 在 1D 序列中也尽量相邻——减少 2D → 1D 的信息损失。

#### Step 2 — Mamba SSM：状态空间模型的"选择性"记忆

Mamba 是 Vision Mamba 的核心。它的工作方式是：

```text
输入序列: x_1, x_2, ..., x_L (L 个 patch token)

对于每个位置 t:
  1. 选择性扫描: 
     - 输入门: Δ_t = softplus(W_Δ · x_t)    (决定"关注多少")
     - 状态转换: A_t = exp(Δ_t · A)          (A 为固定 HiPPO 矩阵)
     - 输入: B_t = W_B · x_t                  (线性投影)
     
  2. SSM 递归:
     h_t = A_t · h_{t-1} + B_t · x_t         (状态更新，O(1) per step!)
     
  3. 输出:
     y_t = C_t · h_t                          (从状态到输出)
```

**Mamba 的"选择性"**：

与传统的 S4（状态空间模型）不同，Mamba 的 $B_t, C_t, \Delta_t$ 都**依赖于输入** $x_t$——这意味着模型可以"选择性地"记住或忘记信息。

| 操作 | 传统 S4 | Mamba |
|------|---------|-------|
| 参数 $\Delta, B, C$ | 固定（与输入无关） | **输入相关（选择性）** |
| 记忆 | 均匀衰减 | **灵活衰减（重要的不忘，不重要的快忘）** |
| 类比 | 固定学习率 | 每个 token 有自己的"记忆机制" |

#### Step 3 — 双向 Mamba 块

标准 Mamba 是单向的（从左到右）——但视觉理解需要双向上下文。解决方案：

```text
Patch 序列 (1D)

前向 Mamba (→): x_1 → x_2 → ... → x_L
  状态从前向后流动

后向 Mamba (←): x_L → x_{L-1} → ... → x_1
  状态从后向前流动

输出拼接: y = [y_forward ⊕ y_backward]
```

**双向 Mamba 模块**：

```text
  x (输入)
    ├ → LayerNorm → 前向 Mamba →
    ├ → LayerNorm → 后向 Mamba →
    └ → 残差连接
  y = x + Concat(y_forward, y_backward)
```

#### Step 4 — 监督预训练配置

| 参数 | Vim-Ti | Vim-S | 说明 |
|------|--------|-------|------|
| 数据集 | ImageNet-1K | ImageNet-1K | 监督预训练 |
| 图像分辨率 | 224² | 224² | — |
| Patch 大小 | 16×16 | 16×16 | — |
| Vim 块数 | 16 | 24 | Mamba 层数 |
| 隐藏维度 | 384 | 768 | — |
| 序列化方式 | 希尔伯特曲线 | 希尔伯特曲线 | — |
| 损失 | 交叉熵 | 交叉熵 | — |
| 优化器 | AdamW | AdamW | — |
| 学习率 | 5e-4 | 5e-4 | — |
| Epoch | 300 | 300 | — |

#### Step 5 — 自监督预训练（可选）

Mamba 也支持 MAE 风格的预训练：

```text
类似 MAE: 随机 mask 75% 的 patch → 仅可见 patch 通过 Mamba → 轻量 Decoder 重建
```

### Vision Mamba vs ViT

| 维度 | ViT | Vision Mamba |
|------|-----|-------------|
| 核心操作 | Self-Attention | **选择性 SSM (Mamba)** |
| 复杂度 | $O(N^2 d)$ | **$O(N d^2)$** |
| 64×64 高分辨率推理 | 需要窗口/分片 | **天然高效** |
| 长序列 | GPU 内存爆炸 | **极低内存** |
| ImageNet Top-1 | 82.3 (ViT-B) | 81.8 (Vim-S) |
| 参数利用效率 | 高 | **更高（同等参数下更快）** |

### 详细训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 预训练数据 | ImageNet-1K / 21K | — |
| 序列化 | 希尔伯特曲线 | 保持 2D 结构 |
| SSM 状态维度 | 16 | — |
| SSM 扩展因子 | 2 | — |
| 双向 | 是 | 正反向 Mamba 各一 |
| 位置编码 | 可学习 | — |
| 损失 | 交叉熵 | — |

### 预训练的实用价值

1. **线性复杂度的视觉模型**：$O(N)$ 替代 $O(N^2)$ → 高分辨率图像处理的新可能
2. **SSM 在视觉的首次大规模验证**：证明了 Mamba 不仅在 NLP 有效，在视觉也有效
3. **长序列视觉任务**：医学影像（4K×4K）、遥感 → 以前不可行，现在可行
4. **与 ViT 互补**：ViT 适合中等分辨率（224²），Vim 适合高分辨率（≥512²）
5. **后 Transformer 时代的探索**：Attention 不是唯一答案 → Mamba/SSM 是强有力的替代方案
