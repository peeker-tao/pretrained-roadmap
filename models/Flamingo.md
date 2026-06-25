# Flamingo

## 基本信息

- **论文**: [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
- **作者**: Jean-Baptiste Alayrac et al. (DeepMind)
- **发表**: NeurIPS 2022

## 创新点

1. **冻结的预训练模型**: 视觉编码器和 LLM 均冻结，不参与训练
2. **Gated Cross-Attention**: 插入到 LLM 中的可训练门控交叉注意力层
3. **少样本上下文学习**: 将 In-Context Learning 扩展到多模态领域

## 核心原理

### Flamingo 架构

1. **视觉编码器**: 预训练的 NormalizerFree ResNet + Perceiver Resampler
2. **语言模型**: 预训练的 Chinchilla
3. **门控交叉注意力 (Gated Cross-Attention)**: 在 LLM 的每层中插入可训练的交叉注意力层

### Perceiver Resampler

将可变数量的视觉特征压缩为固定数量的 token（64 个），供交叉注意力使用。

### 门控机制

$$\text{GatedCA}(x, v) = x + \alpha \cdot \text{CA}(\text{LN}(x), v)$$

其中 $\alpha$ 是可学习的门控参数，控制视觉信息的注入强度。

## 预训练方法

### 核心思想：冻结两个巨兽，只训练"胶水层"

Flamingo 的设计哲学与几乎同时期所有多模态模型都不同：**不对视觉编码器和语言模型做任何微调，只在两者之间插入少量可训练的"桥接层"**。这样做的好处是：保留了视觉编码器（NFNet）和语言模型（Chinchilla）的全部预训练知识，同时通过门控交叉注意力机制让它们"学会沟通"。

> Flamingo 的设计理念：视觉编码器和 LLM 是两个"专家"，已经各自在自己的领域（视觉/语言）训练得非常好。不需要让它们重新学——只需要教它们一个"语言"（交叉注意力+门控），让它们能对话。

### 训练流水线（Step by Step）

#### Step 1 — 数据准备：M3W（多模态 MassiveWeb）

Flamingo 的预训练使用了约 **1.8B（18 亿）图文对**和大量视频数据：

| 数据源 | 类型 | 特点 |
|--------|------|------|
| 网页图文对（ALIGN 风格） | 图文 | 弱对齐（alt-text） |
| 高质量图文对 | 图文 | 更强的图文匹配 |
| 视频数据 | 视频帧 + 字幕 | 时序信息 |

数据的关键设计是**图文交错（interleaved）**：不是"一张图+一段文字"，而是"图-文字-图-文字"交错排列——这让 Flamingo 学会了"看图说话、再看图再说话"的多轮视觉对话能力。

```
训练样本示例:
<image> This is a cat sitting on a sofa. <image> And this is the same cat playing with a toy.
```

#### Step 2 — 视觉编码：冻结的 NFNet + Perceiver Resampler

**NFNet（Normalizer-Free Network）**：DeepMind 的无归一化卷积网络，在 ImageNet 上预训练，完全不参与后续训练。

**Perceiver Resampler**：将 NFNet 输出的可变数量（取决于图像分辨率）的视觉特征压缩为**固定数量的 64 个 token**。

```text
图像 → NFNet(冻结) → [N个视觉特征] → Perceiver Resampler(可训练) → [64个压缩token]
```

Perceiver Resampler 通过交叉注意力机制工作：
- 维护 64 个可学习的"查询向量"
- 这些查询向量通过交叉注意力"读取"NFNet 的视觉特征
- 输出 64 个固定大小的视觉 token

> 64 是经过精心选择的——太多会增加 LLM 的计算负担，太少会丢失视觉信息。对于视频，Flamingo 每帧取 64 个 token，每秒 1 帧。

#### Step 3 — 门控交叉注意力（Gated XATTN）

这是 Flamingo **最核心的创新**——在冻结的 LLM 每一层之间**插入**交叉注意力层：

```text
原始 LLM 层:    x → [Self-Attn] → [FFN] → 输出
Flamingo 插入后: x → [Self-Attn] → [Gated XATTN] → [FFN] → 输出
                                 ↑
                          视觉token (64个)
```

**门控机制**：

$$\text{GatedXATTN}(x, v) = x + \tanh(\alpha) \cdot \text{CrossAttn}(\text{LayerNorm}(x), v)$$

其中 $\alpha$ 是**可学习的门控参数**（初始化为 0）：

- $\tanh(\alpha)$ 的范围是 $[-1, 1]$，控制视觉信息注入的强度
- 初始化为 0 意味着初始时视觉信息完全不影响输出——模型先学会用语言，再逐步打开"视觉之门"
- 这个初始化为 0 的策略确保了训练初期的稳定性

**门控参数逐层独立**：每一层有自己的 $\alpha$，不同层可以学到不同程度的视觉依赖。

> 类比：门控交叉注意力就像一个"滑动门"——门关着时（α≈0），视觉信息进不来，模型全靠语言；门开大时（α>>0），视觉信息大量涌入。初始化为 0 给了模型一个"安全"的起点——先确保语言能力不受影响，再慢慢学如何利用视觉。

#### Step 4 — 以视觉为条件的语言建模损失

Flamingo 的预训练损失是标准的**下一个 token 预测**，但以图像为条件：

$$\mathcal{L} = -\sum_{t} \log P(y_t | y_{<t}, v_1, v_2, ..., v_n)$$

其中 $v_i$ 是 Perceiver Resampler 输出的 64 个视觉 token。

**关键细节**：
- 仅训练 Perceiver Resampler + Gated XATTN 层（约 5B 参数）
- NFNet 和 Chinchilla LLM 完全冻结（约 75B 参数）
- 总参数量 80B，可训练仅 5B

### 为什么这样设计——深入理解

#### 1. 为什么冻结预训练模型？

| 方案 | 优点 | 缺点 |
|------|------|------|
| 全量微调 | 灵活性高 | 灾难性遗忘、训练成本极高 |
| 冻结+轻量桥接（Flamingo） | 保留原始能力、训练快、少样本泛化好 | 灵活性有限 |
| 投影层桥接（LLaVA 风格） | 简单 | 视觉-语言融合深度受限 |

Flamingo 选择冻结策略的核心原因：**少样本上下文学习需要 LLM 保持其原始的 in-context learning 能力**。如果微调了 LLM，它在纯文本上的少样本能力可能会退化。

#### 2. 门控机制的初始化为什么是 0？

$\alpha$ 初始化为 0 → $\tanh(0) = 0$ → 交叉注意力的贡献为 0 → 初始行为与纯文本 LLM 完全相同。

这有两个好处：
1. **训练稳定性**：不会因为视觉信息的突然注入导致 LLM 输出崩溃
2. **渐进式学习**：模型先输出流畅的文本，然后逐步学习"参考"视觉信息

#### 3. 为什么要图文交错训练？

标准的图文对训练（一张图 + 一段文字）只能学到"描述单张图"的能力。图文交错训练（图-文-图-文）让 Flamingo 学到了**多模态对话和推理**的能力——例如：

- "这是图 A（猫）。这是图 B（狗）。图 A 和图 B 有什么共同点？"
- Flamingo 能比较两幅图并生成分析

这在少样本多模态对话中至关重要。

### 详细训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 视觉编码器 | NFNet-F6（冻结） | 438M 参数 |
| Perceiver Resampler | 可训练 | 输出 64 个视觉 token |
| LLM | Chinchilla 70B（冻结） | 80 层，8192 维 |
| Gated XATTN | 每个 LLM 层 1 个（共 80 个） | 可训练 |
| 总参数量 | 80B | — |
| 可训练参数量 | ~5B | Perceiver + Gated XATTN |
| 训练数据 | M3W（1.8B 图文对 + 视频） | 图文交错格式 |
| 优化器 | AdamW | — |
| 学习率 | 1e-4 | 余弦衰减 |
| Warmup | 5000 步 | — |
| Batch Size | 约 16K tokens | — |
| 训练步数 | 约 10B 图文样本 | — |
| 训练硬件 | TPU v4 | — |
| 训练时间 | 15 天（约 1500 TPUv4） | — |

### 少样本能力

Flamingo 最令人印象深刻的是其**少样本多模态对话**能力：

| 任务 | 零样本 | 4-shot | 32-shot |
|------|--------|--------|---------|
| VQA | 51.8 | 56.3 | 59.4 |
| COCO 描述 | 79.4 | 96.9 | 113.8 |
| OK-VQA | 44.7 | 49.4 | 52.7 |

Flamingo 在少样本设置下可以达到或超越许多**完全微调**的专用模型——这种"不需要微调、只需给几个例子"的能力是冻结设计哲学的直接结果。

### 预训练的实用价值

1. **冻结+桥接设计范式的开创者**：影响了 BLIP-2（Q-Former）、LLaVA（投影层）等大量后续工作
2. **图文交错训练的先驱**：证明了交错数据对多模态对话的重要性
3. **少样本多模态能力的标杆**：至今仍是少样本 VLM 的重要参考
4. **Google Gemini 的技术前身**：Flamingo 的多模态架构直接影响了 Gemini 的设计
