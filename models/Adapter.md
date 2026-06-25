# Adapter

## 基本信息

- **论文**: [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)
- **作者**: Neil Houlsby et al. (Google)
- **发表**: ICML 2019

## 创新点

1. **轻量级模块插入**: 在 Transformer 层中插入小型瓶颈网络（Adapter）
2. **接近全量微调性能**: 仅训练 ~3% 参数即达到全量微调的 97%+ 性能
3. **多任务友好**: 不同任务存储不同的 Adapter，切换成本极低

## 核心原理

### Adapter 结构

在每个 Transformer 层的两个位置插入 Adapter：
1. **Attention 之后**: Multi-Head Attention → Adapter → LayerNorm → FFN
2. **FFN 之后**: FFN → Adapter → LayerNorm

### 瓶颈设计

$$\\text{Adapter}(x) = x + W_{\\text{up}}\\cdot\\text{ReLU}(W_{\\text{down}}\\cdot x)$$

- 下投影: $d \\to b$（压缩，$b \\ll d$）
- 上投影: $b \\to d$（恢复）
- 跳跃连接: 保证初始化时不影响原模型

### 与其他 PEFT 方法的对比

| 方法 | 可训练参数 | 推理延迟 | 多任务 |
|------|-----------|---------|--------|
| Adapter | ~3% | 有增加 | ✅ 极好 |
| LoRA | ~1% | 可合并 | ✅ 好 |
| Prefix Tuning | ~0.1% | 序列延长 | ⚠️ |
| 全量微调 | 100% | 无 | ❌ 差 |

## 适配方法

### 核心思想：不需要微调整个 175B 模型——在每层插入一个小型"适配器"，只训练这些适配器

Adapter 是参数高效微调的**开山之作**（2019 年，早于 LoRA 和 Prompt Tuning）。它的核心洞察：预训练模型已经包含了丰富的通用知识——下游任务适配只需要在每层添加小型的"翻译模块"，将通用特征"翻译"为任务特定特征。

> Adapter = 每层插入 2 个瓶颈网络（下投影 → 非线性 → 上投影）+ 残差连接。仅训练 ~3% 参数即可达到全量微调的 97%+ 性能。这是第一次证明"微调不需要动预训练权重"。

### 训练流水线（Step by Step）

#### Step 1 — Adapter 插入位置

在每个 Transformer 层中插入两个 Adapter：

```text
Transformer 层:
  Multi-Head Attention
    ↓
  [Adapter 1] ← 插入点 1（Attention 之后）
    ↓
  LayerNorm
    ↓
  Feed-Forward Network (FFN)
    ↓
  [Adapter 2] ← 插入点 2（FFN 之后）
    ↓
  LayerNorm
```

**为什么选择这两个位置？**
- **Adapter 1（Attention 之后）**：调整"刚注意到的信息"如何被后续层理解
- **Adapter 2（FFN 之后）**：调整"刚处理过的信息"如何被下一层使用

> 两个 Adapter 覆盖了 Transformer 的两个核心子层——注意力和前馈——确保对信息的"收集阶段"和"处理阶段"都能进行任务适配。

#### Step 2 — 瓶颈架构（Bottleneck）

Adapter 的核心是瓶颈设计：

$$\text{Adapter}(x) = x + W_{\text{up}} \cdot \text{ReLU}(W_{\text{down}} \cdot x)$$

其中：
- $W_{\text{down}} \in \mathbb{R}^{d \times b}$：下投影（$d$ 是隐藏维度，$b \ll d$）
- $W_{\text{up}} \in \mathbb{R}^{b \times d}$：上投影
- $b$ 是瓶颈维度，通常 4-64

| $d$ | $b$ | 压缩比 | 每 Adapter 参数 |
|-----|------|--------|----------------|
| 768 (BERT-base) | 64 | 12× | $2 \times 768 \times 64 = 98K$ |
| 768 | 8 | 96× | $2 \times 768 \times 8 = 12K$ |
| 4096 (LLaMA-7B) | 64 | 64× | $2 \times 4096 \times 64 = 524K$ |

**瓶颈的妙处**：
- 下投影将信息压缩到低维空间——类似"总结关键变化"
- 非线性 ReLU 在压缩空间中学习"如何调整"
- 上投影将调整"解压"回原始维度

> 类比：Adapter 就像一个"翻译官"——先把信息浓缩（下投影），理解后（ReLU），再用目标语言表达（上投影）。瓶颈维度 $b$ 就是翻译官的"工作记忆容量"。

#### Step 3 — 残差连接的关键作用

$$y = x + \text{Adapter}(x)$$

Adapter 带残差连接，初始时 $W_{\text{up}}$ 用零初始化，$W_{\text{down}}$ 用正态分布初始化。

**初始状态**：Adapter(x) ≈ 0 → y ≈ x → Adapter 不影响原模型输出

> 这保证了 Adapter "从零开始学"——不会在训练初期破坏预训练模型的良好行为。类似于 LoRA 中 A 零初始化 B 随机初始化的设计。

#### Step 4 — 训练策略

| 策略 | 详情 |
|------|------|
| 冻结参数 | 所有预训练权重 + LayerNorm（可选冻结 LN） |
| 可训练 | 仅 Adapter 的 $W_{\text{down}}$ 和 $W_{\text{up}}$ |
| 训练参数占比 | 约 3%（$b=64$ 时） |
| 多任务切换 | 每个任务保存独立的 Adapter 权重 |

**LayerNorm 要不要训？** 实验表明，同时训练 LayerNorm 的参数（极少量参数）可以进一步提升性能——LN 提供了一种"全局适配"的补充能力。

#### Step 5 — 推理时的处理

Adapter 模块在推理时是**原位计算**的——不能像 LoRA 那样合并到权重中。这意味着：

- 推理延迟略有增加（额外的前向计算）
- 但多任务部署极其方便：只需替换 Adapter 权重文件

### Adapter vs LoRA vs 全量微调

| 维度 | 全量微调 | Adapter | LoRA |
|------|---------|---------|------|
| 可训练参数 | 100% | ~3% | ~1% |
| 推理延迟 | 无增加 | **略有增加** | **可合并 = 零增加** |
| 性能 vs 全量 | 100%（基准） | 97-99% | 98-99.5% |
| 多任务存储 | × 整个模型 | ✓ 仅 Adapter | ✓ 仅 LoRA 权重 |
| 实现复杂度 | 简单 | 中（需修改模型结构） | 中 |

### 详细配置

| 参数 | 典型值 | 说明 |
|------|--------|------|
| 瓶颈维度 $b$ | 8-64 | 在性能和参数间权衡 |
| 每层 Adapter 数 | 2 | Attention 后 + FFN 后 |
| 可训练参数占比 | 1-5% | 视 $b$ 而定 |
| 初始化 | $W_{\text{up}}=0$, $W_{\text{down}} \sim N(0, \sigma^2)$ | 保证初始不影响原模型 |
| 优化器 | Adam/AdamW | — |
| 学习率 | 1e-4 ~ 1e-3 | 通常高于全量微调 |

### 预训练的实用价值

1. **PEFT 领域的开创者**：首次证明"微调可以不动预训练权重"
2. **多任务 NLP 的轻量方案**：BERT + Adapter 可同时支持数十个任务
3. **瓶颈设计的通用范式**：下投影→非线性→上投影的结构被 LoRA、Compacter 等继承
4. **视觉领域的 Adapter**：ViT-Adapter、Conv-Adapter 将这一思想引入视觉
5. **连续学习的利器**：每学一个新任务添加新 Adapter，旧 Adapter 保持不变——天然无灾难遗忘
