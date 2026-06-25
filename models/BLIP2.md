# BLIP-2

## 基本信息

- **论文**: [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)
- **作者**: Junnan Li et al. (Salesforce)
- **发表**: ICML 2023

## 创新点

1. **Q-Former (Querying Transformer)**: 轻量级可训练桥接模块
2. **冻结视觉编码器和 LLM**: 仅训练 Q-Former，训练成本极低
3. **三阶段训练**: 逐步桥接视觉和语言模态

## 核心原理

### Q-Former

Q-Former 是一个轻量级 Transformer，通过一组可学习的 query token 从冻结的视觉编码器中提取与文本相关的视觉信息：

1. **Image Transformer**: 接受冻结视觉编码器的输出
2. **Text Transformer**: 处理文本输入
3. **Cross-Attention**: query token 与视觉特征做交叉注意力

### 三阶段训练

**阶段 1**: 图文对比学习 + 图文匹配（冻结视觉编码器）
- 使 Q-Former 学会从图像中提取与文本相关的信息

**阶段 2**: 基于图像的文本生成（冻结视觉编码器）
- 使 Q-Former 的 query 能生成 LLM 可理解的文本

**阶段 3**: 解冻 LLM，指令微调
- 使模型遵循自然语言指令

## 预训练方法

### 核心思想：Q-Former——一座连接"眼睛"和"嘴巴"的轻量桥梁

BLIP-2 的核心问题是：**冻结的视觉编码器（如 CLIP ViT）的输出和冻结的 LLM（如 OPT/LLaMA）的期望输入之间存在巨大的"模态鸿沟"**。视觉编码器输出的是 257 个连续向量（1 个 [CLS] + 256 个 patch），LLM 期望的是离散 token 嵌入——二者根本不是同一"语言"。

BLIP-2 的答案：**Q-Former（Querying Transformer）**——一个小型的可训练的 Transformer（188M 参数），通过一组可学习的查询 token 从冻结的视觉编码器中选择性地提取信息，输出 LLM 可以理解的"软视觉提示"。

> BLIP-2 = 冻结的视觉编码器 + Q-Former（唯一可训练） + 冻结的 LLM。Q-Former 是整个系统的"翻译官"。

### Q-Former 详解

Q-Former 包含两个共享自注意力的子 Transformer：

```text
Q-Former 架构:
                    ┌──────────────┐
  32个可学习Query → │  Image       │ ← 视觉特征 (257个token)
                    │  Transformer │
                    └──────┬───────┘
                           │ 共享自注意力权重
                    ┌──────┴───────┐
  文本token →        │  Text        │ → 文本表征 / 生成token
                    │  Transformer │
                    └──────────────┘
```

**关键设计**：
- **32 个可学习的 Query**：作为"软锚点"从视觉特征中抽取语言相关信息
- **交叉注意力**：Query 通过交叉注意力与冻结的视觉特征交互，提取最有用的信息
- **共享自注意力**：Image Transformer 和 Text Transformer 共享自注意力层——减少参数量
- **输出**：32 个 Query 的输出向量作为"视觉前缀"输入 LLM

> Q-Former 的 32 个 Query 可以被理解为 32 个"提问"——"这个图里有什么？""什么颜色？""什么场景？"——它们通过训练学会了提出 LLM 最需要知道的问题。

### 三阶段训练流水线

#### 第一阶段：多任务视觉-语言表征学习

在冻结视觉编码器的情况下，Q-Former 通过三个任务学习多模态表征：

**任务 1 — 图文对比学习（ITC）**：
- 对齐 Query 输出（取最大值）与文本 [CLS] 嵌入
- 标准 InfoNCE 损失，同 CLIP
- 负样本：batch 内其他图文对

**任务 2 — 图文匹配（ITM）**：
- 二分类：图像和文本是否匹配
- Query 输出 + 文本 [CLS] → 二分类头

**任务 3 — 基于图像的文本生成（LM）**：
- 以 Query 输出为条件，自回归生成文本
- 交叉熵损失

$$\mathcal{L}_{\text{Stage1}} = \mathcal{L}_{\text{ITC}} + \mathcal{L}_{\text{ITM}} + \mathcal{L}_{\text{LM}}$$

> 这三个任务互相补充：ITC 学习全局对齐（"整张图和整段文字配不配"），ITM 学习匹配判断，LM 学习从视觉特征中生成文字。

#### 第二阶段：视觉-语言生成引导

**目标**：让 Q-Former 的输出可以被 LLM 理解。

Q-Former 输出的 32 个 Query 向量 $\{q_1, ..., q_{32}\}$ 通过一个**线性投影层**映射到 LLM 的嵌入空间，作为**软视觉前缀**输入 LLM：

$$\text{LLM_Input} = [\text{Proj}(q_1), ..., \text{Proj}(q_{32}), \text{Text_Tokens}]$$

LLM 被冻结，仅训练投影层和 Q-Former（微调）：

$$\mathcal{L}_{\text{Stage2}} = -\sum_{t} \log P_{\text{LLM}}(y_t | y_{<t}, q_1, ..., q_{32})$$

**为什么不直接微调 Q-Former+LLM？**
- 如果解冻 LLM，灾难性遗忘会破坏 LLM 的语言能力
- BLIP-2 的设计目标是"最低成本的桥接"，而非最佳性能

#### 第三阶段（可选）：LLM 微调 + 指令微调

对于一些设置，可以解冻 LLM 进行全量微调，实现更好的性能。

### 详细训练配置

| 参数 | 第一阶段 | 第二阶段 |
|------|---------|---------|
| 视觉编码器 | CLIP ViT-L/14（冻结） | 同左（冻结） |
| Q-Former | 可训练 | 可训练 |
| LLM | 不涉及 | OPT-2.7B/6.7B 或 LLaMA-7B（冻结） |
| 数据 | COCO+VG+CC3M+CC12M+SBU+LAION-400M 子集 | COCO |
| 优化器 | AdamW | AdamW |
| 学习率 | 1e-4 | 1e-5 |
| 学习率调度 | 余弦衰减 | 余弦衰减 |
| Batch Size | 512 | 256 |
| Epoch | 约 100K 步 | 约 10K 步 |
| 训练硬件 | 16×A100 | 8×A100 |

### 为什么 BLIP-2 是一个里程碑？

| 特性 | Flamingo | BLIP-2 | LLaVA |
|------|---------|--------|-------|
| 可训练参数 | ~5B | **188M** | ~50M |
| 跨模态桥接 | Gated XATTN | **Q-Former** | 线性/MLP 投影 |
| LLM 冻结 | 是 | 是（阶段2） | 否（阶段2微调） |
| 训练数据 | 1.8B | ~200M | 3M+158K |
| 训练效率 | 中 | **高** | 最高 |

BLIP-2 的 Q-Former 设计开创了"可训练的压缩桥接"范式——188M 参数就完成了视觉编码器和 LLM 之间的模态转换。

### 预训练的实用价值

1. **Q-Former 的创新**：将模态桥接问题转化为"查询-检索"问题
2. **冻结+轻量桥接的标准范式**：被 MiniGPT-4、InstructBLIP 等采用
3. **三任务联合训练的有效性**：ITC+ITM+LM 的组合在多模态表征学习中证明有效
4. **为 BLIP-3（xGen-MM）等后续工作奠定基础**
