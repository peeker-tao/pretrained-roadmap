# Jamba

## 基本信息

- **论文**: [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)
- **作者**: AI21 Labs
- **发表**: arXiv, 2024

## 创新点

1. **Transformer + Mamba 混合架构**: 交替使用 Mamba 层和 Transformer 层
2. **MoE (Mixture of Experts)**: 部分 FFN 层使用 MoE 提升容量
3. **256K 长上下文**: 结合 SSM 和 Attention 的优势

## 核心原理

### 混合架构

每个 Jamba Block 包含：
1. **Mamba SSM 层**: 高效的长程建模
2. **Transformer 注意力层**: 精确的上下文感知
3. **MoE FFN 层**: 大容量但计算高效

### 设计原则

- Mamba 层处理长程依赖（$O(N)$）
- Attention 层处理关键 token 的精确交互（部分层）
- MoE 层提供大规模模型的容量

## 预训练方法

### 核心思想：Transformer 的注意力擅长精确的"点对点"交互，但 $O(N^2)$ 太贵；Mamba 擅长高效的长程"流式"处理，但缺少精确交互。Jamba 的答案是——两种都要！注意力层提供"强交互"，Mamba 层提供"长记忆"，MoE 层提供"大知识库"

Jamba 是第一个商业级别的 **Transformer-Mamba 混合架构**。它的设计哲学：不要让任何单一机制承担所有负载——注意力层处理关键长程交互、Mamba 层高效维护全局状态、MoE 层在不增加计算的情况下扩展知识容量。

> Jamba = Mamba SSM 层（长距离效率）+ Transformer 注意力层（精确交互）+ MoE FFN 层（大容量）+ 256K 上下文。52B 总参数，但每次前向传播仅有 12B 活跃（MoE 稀疏激活）。

### 训练流水线（Step by Step）

#### Step 1 — Jamba 模块的设计

每个 Jamba 模块是 **Mamba + Attention + MoE** 的交替组合：

```text
Jamba Block 的布局 (周期性 8 层一个循环):

Layer 1: Mamba SSM
Layer 2: Mamba SSM
Layer 3: Mamba SSM
Layer 4: Mamba SSM
Layer 5: Mamba SSM
Layer 6: Mamba SSM
Layer 7: Mamba SSM
Layer 8: Transformer Attention + MoE FFN  ← "强交互" + "大容量"

然后重复...
```

**各层的角色分工**：

| 层类型 | 频率 | 角色 | 复杂度 |
|--------|------|------|--------|
| **Mamba SSM** | 7/8 层 | 高效维护全局上下文状态，$O(N)$ | 底 |
| **Transformer Attention** | 1/8 层 | 精确的长程交互（全局注意力），$O(N^2)$ | 高 |
| **MoE FFN** | 1/8 层（+Attention 层后） | 大容量知识存储，稀疏激活 | 低（每次仅激活 2-3 个专家） |

> 1/8 的注意力层意味着总计算量比纯 Transformer 低约 85%——但保留了关键的"全局精确交互"能力。

#### Step 2 — Mamba 层的处理

Mamba SSM 在 Jamba 中的角色是**维护全局上下文状态**：

```text
输入 token 序列: x_1, x_2, ..., x_L

每个位置:
  选择性 SSM 更新:
    h_t = A(x_t) · h_{t-1} + B(x_t) · x_t
    y_t = C(x_t) · h_t

信息流动: x_1 → h_1 → h_2 → ... → h_L
  每个状态 h_t 包含"从 x_1 到 x_t 的压缩摘要"
```

**Mamba 层的优势**：
- $O(L)$ 复杂度 → 即使 256K 长度的输入也无压力
- 选择性记忆 → 重要的信息跨长距离传播，噪声信息快速遗忘
- 流式处理 → 信息自然向前流动

**Mamba 层的局限**：
- 信息是"单向流动"的 → 后方的 token 无法影响前方 token 的表示
- 缺少精确的 token-token 交互 → 不适合需要精确比较的推理

#### Step 3 — Attention 层的精确补充

每 8 层插入一个注意力层来处理精确交互：

```text
查询: 当前序列需要哪些"精确的 token 对 token 关系"？

案例 1: 代词消解
  "Alice called Bob. She was happy."
  "She" 需要精确关注 "Alice" → Attention 层处理

案例 2: 多步推理
  "如果 A > B 且 B > C，那么 A ? C"
  需要精确比较多个 token → Attention 层处理

案例 3: 代码生成
  "def foo(x): return x + " (预测函数体)
  需要精确匹配变量名、括号 → Attention 层处理
```

**Attention 层的安排**：
- 每 8 层才做一次全局注意力 → 大大减少计算
- 但注意力是"标准全局注意力"（无窗口/稀疏） → 保证交互质量
- 配合 RoPE 旋转位置编码 → 精确的相对位置感知

#### Step 4 — MoE（Mixture of Experts）扩展容量

FFN 的 MoE 设计让 Jamba 在不增加计算的前提下扩展参数：

```text
标准 FFN: 
  y = W₂ · GELU(W₁ · x)

MoE FFN:
  路由器: g = softmax(W_r · x)  → 选择 top-k 专家
  y = Σ_i g_i · FFN_i(x)        → 仅激活选中的专家
```

| 参数 | 值 | 说明 |
|------|-----|------|
| 总专家数 | 16 | 每层 |
| 每次激活 | 2 | Top-2 routing |
| 总参数 | 52B | — |
| 活跃参数 | 12B | 仅 ~23% 参数参与每次前向传播 |
| Router 类型 | 软路由（加权组合） | — |

**MoE 在 Jamba 中的特殊角色**：MoE 层只在 Attention 层之后出现（同样 1/8 频率），因为注意力层的输出是"精确交互后的关键信息"，最适合路由到"最了解"该类型信息的专家。

#### Step 5 — 256K 长上下文训练

Jamba 支持 256K tokens 的上下文窗口：

| 策略 | 说明 |
|------|------|
| Mamba 连续状态 | SSM 自然支持任意长度，无额外成本 |
| 注意力仅 1/8 层 | 长上下文的注意力成本可控 |
| RoPE 位置编码 | 支持任意长度的相对位置 |
| 渐进式扩展 | 预训练从 4K → 16K → 64K → 256K |

### Jamba vs Pure Mamba vs Pure Transformer

| 维度 | Pure Transformer | Pure Mamba | **Jamba** |
|------|-----------------|-----------|----------|
| 核心层 | 100% Attention | 100% SSM | **87.5% SSM + 12.5% Attention** |
| 复杂度 | $O(N^2 d)$ | $O(N d^2)$ | **介于两者之间** |
| 精确交互 | ✓✓ | ✗ | **✓（1/8 层）** |
| 长程效率 | ✗ | ✓✓ | **✓✓** |
| 知识容量 | 标准 FFN | 标准 FFN | **MoE（16 专家）** |
| 256K 上下文 | 困难 | 自然 | **自然** |

### 详细训练配置

| 参数 | Jamba | 说明 |
|------|-------|------|
| 总参数 | 52B | 含 16 个 MoE 专家 |
| 活跃参数 | 12B | 每次前向传播 |
| 层数 | 48 | Jamba Block × 6 |
| Mamba 层数 | 42 | 7/8 × 48 |
| Attention 层数 | 6 | 1/8 × 48 |
| MoE 专家 | 16 | 每个 Attention 层 |
| 上下文长度 | 256K | 训练后支持 |
| 词汇量 | 256K | — |
| 优化器 | AdamW | — |
| 学习率 | 1e-4 | — |

### 预训练的实用价值

1. **混合架构的设计原则**：不是"Attention vs Mamba"，而是"两者如何搭配最好"
2. **长上下文的商业化**：256K 上下文 + $O(N)$ 接近的效率 → 长文档处理
3. **MoE 的高效容量扩展**：52B 模型仅需 12B 活跃参数的计算量
4. **后 Transformer 时代的里程碑**：与 GPT-4o、Gemma 并列的混合架构方案
5. **架构融合的方向**：未来的 LLM 可能不是纯 Transformer——混合是趋势
