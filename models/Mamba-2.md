# Mamba-2

## 基本信息

- **论文**: [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)
- **作者**: Tri Dao, Albert Gu (Princeton / CMU)
- **发表**: ICML 2024

## 创新点

1. **SSM-Attention 对偶性**: 证明 SSM 和线性注意力在数学上等价
2. **结构化矩阵视角**: 将 SSM 重新解释为半可分矩阵乘法
3. **2-8× 速度提升**: 相比 Mamba-1 的显著加速

## 核心原理

### SSM-Attention 对偶性

Mamba-2 的核心发现：**SSM 可以表示为结构化矩阵乘法，而线性注意力也是矩阵乘法**。两者在结构化半可分矩阵的框架下统一。

### 关键改进

| 特性 | Mamba-1 | Mamba-2 |
|------|---------|---------|
| SSM 参数化 | $A \in \mathbb{R}$ | $A$ 为标量 |
| 计算方式 | 并行扫描 | 矩阵乘法 |
| 速度 | 基线 | 2-8× 更快 |
| 理论框架 | SSM | SSM + Attention 统一 |

## 预训练方法

### 核心思想：SSM 和 Attention 其实是"一家人"

Mamba-1 证明了 SSM 可以匹敌 Transformer，但它和 Transformer 看起来仍然像是两个完全不同的世界：一个是状态空间递归，一个是注意力矩阵乘法。Mamba-2 的贡献在于**从数学上证明了这两个世界其实是等价的**——SSM 和线性注意力都可以被统一为"结构化半可分矩阵乘法"。

> Mamba-2 = SSM-Attention 统一理论 + 结构化矩阵计算 + 2-8× 比 Mamba-1 更快的训练和推理。它不仅是 Mamba 的工程优化，更是一次理论重构。

### 训练流水线（Step by Step）

#### Step 1 — SSM-Attention 对偶性（理论突破）

Mamba-2 的核心发现：**状态空间模型在数学上等价于结构化半可分矩阵乘法**。

**SSM 的矩阵形式**：

给定输入序列 $X = [x_1, ..., x_L]^T$，SSM 的输出可以表示为：

$$Y = M \cdot X$$

其中 $M$ 是一个**半可分矩阵（Semi-Separable Matrix）**：

$$M_{ij} = C_i \left(\prod_{k=j+1}^i \bar{A}_k\right) \bar{B}_j$$

对于因果（下三角）系统，$M$ 是一个下三角矩阵。

**Attention 的矩阵形式**：

$$Y = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

这也是一个矩阵乘法——但使用的是 softmax 归一化的注意力矩阵。

**统一视角**：两者都是 $Y = M \cdot V$ 的形式，区别仅在于 $M$ 的结构：
- SSM：$M$ 是半可分矩阵（满足特定的低秩结构）
- Attention：$M$ 是 softmax 归一化的相似度矩阵
- 线性 Attention：$M = QK^T$ 乘 $V$（去掉 softmax）

#### Step 2 — Mamba-2 的 SSM 参数化简化

Mamba-1 的 $A$ 是一个对角矩阵。Mamba-2 将 $A$ 简化为**标量**：

| 参数 | Mamba-1 | Mamba-2 |
|------|---------|---------|
| $A$ | 对角矩阵 $\in \mathbb{R}^{N}$ | **标量** $\in \mathbb{R}$ |
| $B, C$ | 输入依赖的向量 | 输入依赖的向量 |
| $\Delta$ | 输入依赖的标量 | 输入依赖的标量 |
| 状态维度 $N$ | 16 | 64-128 |

将 $A$ 简化为标量使得 Mamba-2 可以表示为**结构化掩码注意力（Structured Masked Attention, SMA）**——一种可以高效实现的矩阵乘法。

#### Step 3 — 结构化掩码注意力（SMA）计算

Mamba-2 的 SSM 被重新表述为：

$$Y = (L \circ CB^T) \cdot X$$

其中 $L$ 是一个下三角矩阵（由 $\Delta$ 和 $A$ 决定），$C$ 和 $B$ 是 $B_t, C_t$ 组成的矩阵，$\circ$ 是逐元素乘法。

**关键**：这个形式允许使用标准的矩阵乘法（GEMM）来计算 SSM——而不需要并行扫描。

**计算效率**：

| 计算方式 | Mamba-1 | Mamba-2 |
|---------|---------|---------|
| SSM 核心 | 并行扫描 | **矩阵乘法（GEMM）** |
| GPU 利用率 | 中 | **高（GEMM 是 GPU 最强项）** |
| 训练速度 | 基线 | **2-8× 更快** |

> GPU 的矩阵乘法（Tensor Core GEMM）经过了数十年的极致优化；将 SSM 表示为矩阵乘法就能充分利用这些优化。

#### Step 4 — 与 Attention 的混合

Mamba-2 支持**与 Attention 层混合使用**：

```text
Layer 1-4:  Mamba-2 SSM Block
Layer 5-6:  Self-Attention Block
Layer 7-12: Mamba-2 SSM Block
```

这种混合架构在保留 Attention 的全局建模能力的同时，利用 SSM 的线性复杂度处理大部分层。Jamba 模型就是这种混合架构的成功案例。

### Mamba-1 vs Mamba-2

| 维度 | Mamba-1 | Mamba-2 |
|------|---------|---------|
| SSM 参数化 | 对角 $A$ 矩阵 | **标量 $A$** |
| 计算方式 | 并行扫描 | **矩阵乘法（GEMM）** |
| 训练速度 | 基线 | **2-8× 更快** |
| 推理吞吐 | 高 | **更高** |
| 理论框架 | SSM | **SSM + Attention 统一** |
| 状态维度 | 16 | **64-128** |
| 注意力混合 | 不直接支持 | **天然支持** |
| 实现复杂度 | 高（需要自定义 kernel） | **低（标准 GEMM）** |

### 详细训练配置

| 参数 | Mamba-2 配置 | 说明 |
|------|------------|------|
| SSM 层 | 与 Mamba-1 类似 | 或与 Attention 交替 |
| 状态维度 $N$ | 64-128 | 更大的状态空间 |
| $A$ 参数化 | 标量 | 简化的 SSM 核心 |
| 训练数据 | 类似 Mamba-1 | Pile, SlimPajama 等 |
| 优化器 | AdamW | — |
| 损失 | 下一个 token 预测 | 标准 LM 损失 |

### SSM-Attention 统一理论的含义

| 特性 | 传统 SSM 视角 | Mamba-2 统一视角 |
|------|------------|----------------|
| SSM 实现 | 需要自定义硬件 kernel | **标准 GEMM 即可** |
| Attention 和 SSM | 两个不同世界 | **同一频谱的两端** |
| 混合架构 | 复杂设计 | **自然选择** |
| 计算优化 | 需要领域专家 | **利用现有 AI 编译器** |

### 预训练的实用价值

1. **SSM-Attention 统一理论**：揭示了两者本质上的等价性，是架构设计的基础性成果
2. **训练效率飞跃**：2-8× 比 Mamba-1 更快，使 SSM 模型的大规模训练更可行
3. **标准 GEMM 实现**：不需要定制硬件 kernel，降低了工程门槛
4. **混合架构的自然支持**：Jamba、Zamba 等混合模型受益于这一理论
5. **为架构搜索（NAS）提供了新的设计空间**：可以在 SSM 和 Attention 之间自由组合
