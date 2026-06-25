# LLaMA 系列 (Large Language Model Meta AI)

## 基本信息

| 版本 | 论文 | 发表 | 最大参数量 |
|------|------|------|-----------|
| LLaMA-1 | [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | arXiv 2023 | 65B |
| LLaMA-2 | [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) | arXiv 2023 | 70B |
| LLaMA-3 | [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | arXiv 2024 | 405B |

## 创新点

### LLaMA-1
1. **高质量数据 > 超大模型**: 用更多数据训练较小模型（7B→65B）即可匹敌 GPT-3 (175B)
2. **完全开源**: 开源了模型权重，推动了开源大模型的爆发

### LLaMA-2
1. **商业可用**: 首次推出可商用版本
2. **RLHF 对齐训练**: 引入人类反馈强化学习

### LLaMA-3
1. **405B 参数**: 最大稠密模型之一
2. **改进分词器**: 125K 词汇量的 BPE 分词器
3. **大规模高质量数据**: 15T+ tokens 的训练数据

## 核心原理

### 架构特点

- **RMSNorm**: 使用 RMS Layer Norm，计算更高效
- **SwiGLU 激活**: 在 FFN 中使用 SwiGLU 替代 ReLU
- **旋转位置编码 (RoPE)**: 相对位置编码
- **分组查询注意力 (GQA)**: LLaMA-2 开始引入

### 训练效率优化

LLaMA-1 证明了在同等计算预算下：
- **更多数据 + 更小模型** → 更好性能
- 7B 模型在 1T tokens 上训练可匹敌 GPT-3

## 预训练方法

### 核心思想：用更多数据训练更小的模型

LLaMA 的核心洞察是一个反直觉的发现：**在相同的计算预算下，用更多高质量数据训练一个较小的模型，比用较少数据训练一个超大模型效果更好**。这与 GPT-3 的"超大模型优先"策略形成鲜明对比。

> LLaMA-7B（70 亿参数）用 1T tokens 训练，性能接近 GPT-3-175B（1750 亿参数，用 300B tokens 训练）——Chinchilla 法则在开源世界的实证。

### 训练流水线（Step by Step）

#### Step 1 — 数据混合与预处理

LLaMA-1 的预训练数据来自多个公开数据源的混合：

| 数据源 | 占比 | Token 数 | 特点 |
|--------|------|---------|------|
| CommonCrawl | 67.0% | ~0.94T | 互联网文本（经严格过滤） |
| C4 | 15.0% | ~0.21T | T5 的清洗数据集 |
| GitHub | 4.5% | ~0.06T | 代码数据（提升推理能力） |
| Wikipedia | 4.5% | ~0.06T | 20 种语言 |
| Books | 4.5% | ~0.06T | Gutenberg + Books3 |
| ArXiv | 2.5% | ~0.04T | 科学论文 |
| StackExchange | 2.0% | ~0.03T | 问答数据 |

**数据过滤流程（LLaMA-1）**：

1. 对 CommonCrawl 原始文本做**行级去重**（每个文档内）
2. 用 fastText 语言分类器**移除非英文文本**
3. 用 n-gram 语言模型**过滤低质量页面**（困惑度过滤）
4. 全量**文档级去重**

> LLaMA 的数据处理哲学：数据的**质量**和**多样性**比数据的**绝对数量**更重要。

LLaMA-2 进一步改进了数据处理流程，训练数据增加到 2T tokens。LLaMA-3 则使用了 15T+ tokens，但具体来源未详细公开。

#### Step 2 — 分词

LLaMA 使用 **Byte-Pair Encoding（BPE）** 分词，但有一个重要改进：

- **LLaMA-1/2**：使用 SentencePiece 实现的 BPE，32K 词汇量
- **LLaMA-3**：升级到 128K 词汇量

**改进的意义**：128K 词汇量意味着更多的常见词/词组被保留为单个 token。例如：

| 文本 | 32K 词汇量 | 128K 词汇量 |
|------|----------|-----------|
| "internationalization" | 可能拆为 3-4 个 token | 可能 1 个 token |
| "machine learning" | 2 个 token | 可能 1 个 token（如果是高频词组） |

更大的词汇量 = 更短的序列长度 = 更高效的训练和推理。对于同样的 1T tokens 训练量，128K 词汇量处理的文本量远远大于 32K 词汇量。

#### Step 3 — 前向传播：改进的 Transformer 架构

LLaMA 采用了多项架构改进：

**RMS Norm（Root Mean Square Layer Normalization）**：

相比标准 LayerNorm，RMS Norm 只做缩放不做平移，计算量减少约 20%：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \gamma$$

**SwiGLU 激活函数**：

在 FFN 层使用 SwiGLU（Swish-Gated Linear Unit）替代 ReLU/GELU：

$$\text{SwiGLU}(x) = (xW_1 \odot \text{SiLU}(xW_2)) W_3$$

SwiGLU 在 LLM 中被证明比标准 ReLU/GELU 具有更好的训练动力学。

**旋转位置编码（RoPE）**：

LLaMA 使用 RoPE 替代绝对位置编码。RoPE 通过旋转矩阵将位置信息编码到注意力计算中：

$$\langle q_m, k_n \rangle = \langle R_m q, R_n k \rangle = \langle q, R_{n-m} k \rangle$$

RoPE 的优势：天然支持**任意长度外推**（训练 2048 长度，推理时扩展到 4096 或更长）。

**分组查询注意力（GQA，LLaMA-2/3）**：

标准多头注意力（MHA）中，每个注意力头有独立的 Q、K、V。GQA 让多个 Q 头**共享一组 K、V 头**：

```text
MHA: Q₁(K₁,V₁)  Q₂(K₂,V₂)  Q₃(K₃,V₃)  Q₄(K₄,V₄) ...  8 Q, 8 K, 8 V
GQA: Q₁(K₁,V₁)  Q₂(K₁,V₁)  Q₃(K₂,V₂)  Q₄(K₂,V₂) ...  8 Q, 2 K, 2 V
```

GQA 大幅减少了 KV-cache 的显存占用（推理时可节省 4-8× 显存），对长序列推理至关重要。

#### Step 4 — 自回归损失计算

标准的因果语言模型（Causal LM）损失：

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_{<t})$$

所有 token 都参与损失计算——这与 BERT 的 MLM（仅被掩码的 15% 参与）不同。

### 为什么 LLaMA 的训练效率更高？

#### Chinchilla 最优法则

DeepMind 的 Chinchilla 论文（2022）通过大量实验发现了一个最佳关系：

**最优训练 token 数 = 20 × 模型参数量**

| 模型 | 参数量 | 训练 tokens | 是否最优？ |
|------|--------|------------|-----------|
| GPT-3 | 175B | 300B | ❌ 欠训练（应训练 3.5T） |
| Chinchilla | 70B | 1.4T | ✅ 接近最优 |
| LLaMA-7B | 7B | 1T | ✅ 非常充裕 |
| LLaMA-13B | 13B | 1T | ✅ 充裕（~77× 参数） |
| LLaMA-65B | 65B | 1.4T | ✅ 接近最优（~22× 参数） |
| LLaMA-3-405B | 405B | 15T+ | ✅ 充裕（~37× 参数） |

LLaMA 系列严格遵循 Chinchilla 法则——用"恰到好处"的参数和"充足"的数据进行平衡训练，而非 GPT-3 的"超大模型 + 不足数据"。

### 详细训练配置

| 参数 | LLaMA-7B | LLaMA-13B | LLaMA-65B |
|------|---------|----------|----------|
| 层数 | 32 | 40 | 80 |
| 隐藏维度 | 4096 | 5120 | 8192 |
| 注意力头 | 32 | 40 | 64 |
| FFN 维度 | 11008 | 13824 | 22016 |
| 序列长度 | 2048 | 2048 | 2048 |
| 词汇量 | 32000 | 32000 | 32000 |
| 优化器 | AdamW | AdamW | AdamW |
| β₁, β₂ | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 |
| 学习率 | 3e-4 | 3e-4 | 1.5e-4 |
| 学习率调度 | 余弦衰减 | 余弦衰减 | 余弦衰减 |
| Warmup | 2000 步 | 2000 步 | 2000 步 |
| 权重衰减 | 0.1 | 0.1 | 0.1 |
| 梯度裁剪 | 1.0 | 1.0 | 1.0 |
| Batch Size | 4M tokens | 4M tokens | 4M tokens |
| 训练硬件 | 2048 A100-80G GPU | 2048 A100-80G | 2048 A100-80G |
| 训练时间 | ~21 天（7B） | ~45 天 | ~63 天 |
| 训练成本 | 约 $230K | 约 $380K | 约 $620K |

#### 训练细节

- **减少内存占用**：使用 xFormers 库的**高效注意力实现**（Flash Attention 等价方案）
- **手动实现的 backward**：对 SwiGLU、RoPE 等操作做了定制的反向传播，减少中间激活的内存占用
- **模型并行**：65B 模型使用张量并行（tensor parallelism）跨 8 GPU
- **数据并行**：所有变体使用 FSDP（Fully Sharded Data Parallel）

### LLaMA-2/3 的强化训练

**LLaMA-2**：除了更大规模的预训练（2T tokens），还增加了**RLHF 对齐训练**：

1. **SFT（Supervised Fine-Tuning）**：在人类编写的指令-回复对上微调
2. **RLHF（Reward Modeling + PPO）**：训练奖励模型，用 PPO 优化模型输出
3. **拒绝采样 + PPO 迭代**：多轮交替优化

**LLaMA-3**：进一步将训练规模推至 **15T+ tokens**，并引入了以下改进：

1. **128K 词汇量**的改进分词器
2. **分组查询注意力（GQA）**在 8B 和 70B 模型中均使用
3. **4 阶段训练**：预训练 → 长上下文继续预训练 → 退火 → 后训练（SFT+RLHF）
4. **128K 上下文**支持
5. **多模态能力**（LLaMA-3 的某些版本）

### 预训练的实用价值

LLaMA 系列的重要性不仅仅在于技术本身，更在于其对 AI 生态的影响：

1. **开源大模型浪潮**：LLaMA 的权重公开引发了全球范围内的开源大模型运动
2. **Chinchilla 法则的实证**：证明了"小模型 + 多数据"策略的可行性
3. **衍生模型生态**：Alpaca、Vicuna、LLaVA、Mistral、Yi 等大量模型基于 LLaMA 或受其启发
4. **RLHF 的对齐实践**：LLaMA-2 的 RLHF 流程成为开源社区对齐训练的标准参考
5. **降低了 LLM 门槛**：7B/13B 模型可以在消费级硬件上运行，推动了本地 LLM 的普及
