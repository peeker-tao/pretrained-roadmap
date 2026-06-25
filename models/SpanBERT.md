# SpanBERT

## 基本信息

- **论文**: [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529)
- **作者**: Mandar Joshi et al. (Allen Institute for AI / UW)
- **发表**: TACL 2020

## 创新点

1. **Span 级别的掩码**: 掩码连续的 span 而非随机 token
2. **Span Boundary Objective (SBO)**: 利用 span 边界 token 的表示来预测被掩码的 span
3. **去除 NSP**: 仅使用单句训练，提升效率

## 核心原理

### Span Masking

随机选择文本中连续的 span（而非独立 token），掩码整个 span。几何分布采样 span 长度，使模型学习更长的依赖关系。

### Span Boundary Objective

$$\\mathcal{L}_{\\text{SBO}} = -\\log P(x_i | \\mathbf{h}_{s-1}, \\mathbf{h}_{e+1}, p_i)$$

其中 $\\mathbf{h}_{s-1}$ 和 $\\mathbf{h}_{e+1}$ 是 span 前后边界 token 的表示，$p_i$ 是 token 在 span 中的位置编码。

## 预训练方法

### 核心思想：BERT 随机 mask 单个 token——这就像做填空题时每个空都是一个字，太简单了。SpanBERT 改为 mask 连续的一段文本（短语），同时让模型的"边界 token"来推理被遮住的内容

SpanBERT 的目标很明确：**让预训练更适合需要 span 级别理解的下游任务**（如抽取式问答、指代消解、关系抽取）。BERT 的 token 级别 MLM 对于"遮住的短语是什么"的监督信号太弱——SpanBERT 的连续 span 掩码 + 边界目标（SBO）弥补了这一不足。

> SpanBERT = 连续 Span Masking（取代 BERT 的随机 token masking）+ Span Boundary Objective（用边界 token 预测 span 内容）+ 去除 NSP 任务。在 SQuAD 2.0 上比 BERT 高出 6.6%。

### 训练流水线（Step by Step）

#### Step 1 — Span Masking（连续片段掩码）

BERT 的掩码：随机选 15% 的 token，每个独立地 mask。

SpanBERT 的掩码：随机选连续的 span，mask 整个 span。

```text
BERT Masking:
  [CLS] The cat sat on the [MASK] and [MASK] [MASK] sleep [SEP]
  
SpanBERT Masking:
  [CLS] The cat sat on [MASK] [MASK] [MASK] [MASK] [MASK] and went to sleep [SEP]
  （"the mat" 被整体 mask 为连续的 [MASK] [MASK]）
```

**Span 长度采样**：$\ell \sim \text{Geometric}(p=0.2)$，平均长度 3.8 个 token。

| 分布 | 短 span (1-2) | 中 span (3-6) | 长 span (7+) |
|------|-------------|-------------|------------|
| Geometric(0.2) | 常见 | 中等 | 不常见 |
| 均匀分布 | 太平均 | 太平均 | 太多长 span |

> 几何分布的选择是因为自然语言中短短语比长短语更常见——"深度学习"（2 词）比"深度神经网络的反向传播算法"（7 词）更频繁出现。

#### Step 2 — Span Boundary Objective (SBO)

这是 SpanBERT 最精妙的设计。传统 MLM 用被 mask token 自身的隐藏状态来预测它——但这在连续 mask 场景中效果不佳（因为相邻被 mask token 之间缺乏有用信息）。

**SBO 的做法**：用 span **边界外**的 token 来预测 span 内的 token：

$$\mathcal{L}_{\text{SBO}} = -\log P(x_i | \mathbf{h}_{s-1}, \mathbf{h}_{e+1}, p_i)$$

其中：
- $\mathbf{h}_{s-1}$：span 左边界的隐藏状态
- $\mathbf{h}_{e+1}$：span 右边界的隐藏状态
- $p_i$：token $x_i$ 在 span 中的位置编码（相对位置）
- 预测网络：$\mathbf{h}_{s-1} \oplus \mathbf{h}_{e+1} \oplus \mathbf{p}_i \to \text{MLP} \to \text{Softmax}$

**为什么边界更有用？**

```text
文本: "The cat sat on the [furry mat] and went to sleep"

MLM 预测 "mat":
  使用的信息: [MASK] 位置自身的隐藏状态（空洞，相邻 [MASK] 也是空洞）
  
SBO 预测 "mat":
  使用的信息: "the" (s-1) + "and" (e+1) + 位置 2 (span 内的第 2 个 token)
  → "边界说: 前面是 the，后面是 and，你是 span 的第 2 个词"
  → 更容易推理出 "mat"
```

> 类比：MLM 是让一个人闭着眼睛猜自己手里拿的是什么；SBO 是让两个人站在一旁（边界 token），通过观察上下文来推断中间被遮住的内容。

#### Step 3 — 联合训练：MLM + SBO

总损失：

$$\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{SBO}}$$

两个损失**同时计算**，共用同一批被 mask 的 span。

- MLM 损失：用被 mask 位置的隐藏状态预测其 token
- SBO 损失：用边界 token 的隐藏状态预测 span 内的每个 token

两者互补——MLM 从"内部视角"学习，SBO 从"外部视角"学习。

#### Step 4 — 移除 NSP（Next Sentence Prediction）

SpanBERT 不训练 NSP 任务（BERT 的次要预训练目标），仅使用**单句训练**。

| 决策 | 理由 |
|------|------|
| 去除 NSP | NSP 对下游任务帮助有限（RoBERTa 已经证明） |
| 单句训练 | 每个训练样本是一段连续文本（而非句对） |
| 段落级采样 | 从文档中采样连续的 512 token 片段 |

> 去除 NSP 带来的效率提升：内存利用率更高（不需要同时存储两个句子），batch size 可以更大。

#### Step 5 — 训练流程

```text
1. 从语料库中采样连续文本片段（512 tokens）
2. 随机选择 15% 的文本内容形成 span：
   a. 从 Geometric(0.2) 采样 span 长度
   b. 随机选择 span 起始位置
   c. 将 span 内所有 token 替换为 [MASK]（80%）/ random（10%）/ 不变（10%）
3. 前向传播 → 获取所有位置的隐藏状态
4. 计算 MLM 损失 + SBO 损失
5. 反向传播
```

### SpanBERT vs BERT

| 维度 | BERT | SpanBERT |
|------|------|---------|
| 掩码方式 | 随机 token | **连续 span** |
| 预测方式 | 仅 MLM | **MLM + SBO** |
| NSP | ✓ | **✗（移除）** |
| 训练片段 | 句对（512 tokens） | **单段连续文本（512 tokens）** |
| 平均 mask 长度 | 1 | **3.8** |
| SQuAD 2.0 F1 | 76.8% | **83.4%** |
| Coreference (Ontonotes) | 79.6% | **84.9%** |

### 详细训练配置

| 参数 | SpanBERT-base | SpanBERT-large |
|------|-------------|---------------|
| 数据集 | BooksCorpus + English Wikipedia | 同 |
| 层数 | 12 | 24 |
| 隐藏维度 | 768 | 1024 |
| Span 掩码率 | 15% | 15% |
| Span 长度分布 | Geometric(0.2) | Geometric(0.2) |
| MLM: SBO 权重 | 1:1 | 1:1 |
| 优化器 | AdamW | AdamW |
| 学习率 | 1e-4 | 1e-4 |
| Batch Size | 256 | 256 |
| 训练步数 | ~1M | ~1M |

### 预训练的实用价值

1. **Span 级理解的特化**：抽取式问答、指代消解的 SOTA 提升
2. **SBO 的创新设计**：边界预测 → 在其他领域也被采用（如蛋白质序列建模）
3. **NSP 不必要的进一步验证**：与 RoBERTa 互相印证
4. **连续掩码的通用性**：连续 span mask 催生了 MASS、ERNIE-GEN 等生成式预训练方法
5. **几何分布的跨领域启示**：按自然分布采样 mask 而非均匀分布
