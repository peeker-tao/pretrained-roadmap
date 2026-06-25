# Vicuna

## 基本信息

- **论文**: [Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality](https://lmsys.org/blog/2023-03-30-vicuna/)
- **作者**: LMSYS (UC Berkeley, CMU, Stanford, UC San Diego)
- **发表**: 技术报告, 2023

## 创新点

1. **LLaMA 指令微调**: 在 LLaMA 基础上使用 ShareGPT 对话数据微调
2. **高质量对话数据**: 7 万条 ShareGPT 真实用户对话
3. **开源聊天模型标杆**: 成为多模态模型（LLaVA 等）的标准语言骨干

## 核心原理

### 训练方法

1. 从 ShareGPT 收集约 7 万条真实对话
2. 将对话格式化为多轮对话模板
3. 在 LLaMA 上做全量监督微调（SFT）

### 与 LLaMA 的关系

| 模型 | 基础 | 训练数据 |
|------|------|---------|
| LLaMA | 预训练 | 公开文本 |
| Vicuna | LLaMA → SFT | ShareGPT 对话 |

## 预训练关系（指令微调方法）

### 核心思想：用高质量对话数据"激活"预训练模型

Vicuna 本身不涉及预训练——它是基于 Meta 的 LLaMA 预训练模型进行**监督微调（SFT）**得到的一个对话模型。但它证明了一个重要的观点：**你不需要自己预训练一个 LLM，只需要用高质量对话数据在开源的预训练模型上微调，就能获得接近 ChatGPT 的对话体验**。

> Vicuna 的"秘诀"就一条：ShareGPT 上 7 万条真实用户与 ChatGPT 的对话记录。这些数据质量极高——因为是真实用户在真实场景中与最强闭源模型的互动。用这些数据微调 LLaMA，效果惊人。

### 训练流水线

#### Step 1 — 数据收集：ShareGPT

ShareGPT 是一个用户可以分享 ChatGPT 对话的平台。Vicuna 团队从 ShareGPT 爬取了约 7 万条真实对话（2023 年 3 月数据）。

**数据特点**：
- 包含丰富的多轮对话（不只是"一问一答"）
- 覆盖各种主题：编程、写作、角色扮演、头脑风暴、问答……
- 数据由 ChatGPT 生成，质量有保证
- 但包含一些被拒绝的回答（表示 ChatGPT 的某些"规则/限制"）

**数据清洗**：
- 过滤掉过短/过长的对话
- 过滤掉被拒绝的回复（ChatGPT 说"I cannot..."的）
- 格式化为多轮对话模板

#### Step 2 — 对话格式化

将 ShareGPT 对话格式化为以下训练格式：

```text
USER: [用户问题]
ASSISTANT: [ChatGPT回复]
USER: [用户追问]
ASSISTANT: [ChatGPT回复]
...
```

训练时，只计算 ASSISTANT 回复部分的损失（USER 部分不参与损失计算）：

$$\mathcal{L} = -\sum_{t \in \text{ASSISTANT tokens}} \log P(x_t | x_{<t})$$

> 这个设计很重要——我们只想让模型学会"如何回答"，不想让它学会"如何提问"。

#### Step 3 — 监督微调

在 LLaMA-7B/13B 上做全量微调（不是 LoRA 等参数高效方法）：

| 参数 | Vicuna-7B | Vicuna-13B |
|------|----------|-----------|
| 基础模型 | LLaMA-7B | LLaMA-13B |
| 训练数据 | 70K ShareGPT 对话 | 70K ShareGPT 对话 |
| 优化器 | AdamW | AdamW |
| 学习率 | 2e-5 | 2e-5 |
| Epoch | 3 | 3 |
| Batch Size | 128 | 128 |
| 序列长度 | 2048 | 2048 |
| 训练时间 | ~1 天（8×A100） | ~1 天 |
| 训练成本 | ~$140 | ~$300 |

> 微调成本极低——$140 就能做出接近 ChatGPT 90% 质量的对话模型！这直接引爆了开源 LLM 微调的热潮。

#### Step 4 — 评估

Vicuna 使用 GPT-4 作为评估器（而非人工评估），这是 LMSYS 的 FastChat 评估框架的核心：

1. 对同一个问题，让 Vicuna 和其他模型（LLaMA、Alpaca、ChatGPT 等）各生成一个回答
2. 让 GPT-4 判断哪个回答更好（输出评分和理由）
3. 统计胜率

**评估结果**：
- Vicuna-13B 在约 90% 的情况下被认为质量接近或达到 ChatGPT
- Vicuna 全面超越了 Alpaca（Stanford 的 LLaMA+Self-Instruct 微调模型）

### 为什么 Vicuna 如此成功？

#### ShareGPT 数据的独特优势

| 数据来源 | 数据量 | 质量 | 多样性 |
|---------|--------|------|--------|
| Self-Instruct（Alpaca） | 52K | 中（GPT-3 生成的模板） | 中 |
| Dolly（Databricks） | 15K | 中（人工撰写） | 低 |
| **ShareGPT（Vicuna）** | **70K** | **极高（ChatGPT 真实对话）** | **极高** |
| OpenAssistant | 12K+ | 高（人工） | 高 |

ShareGPT 的数据质量是其他来源无法比拟的——因为它们是来自 ChatGPT 的真实对话，经过了实际用户的"压力测试"。

#### LLaMA 的强大预训练基础

Vicuna 的成功也归功于 LLaMA 的预训练质量。LLaMA-13B 在 1T tokens 上预训练——这为微调提供了一个非常强大的起点。如果底模不好，再好的微调数据也效果有限。

### Vicuna 作为多模态模型的骨干

Vicuna 最大的遗产之一是成为 **LLaVA（Large Language and Vision Assistant）** 的标准语言骨干。LLaVA 将视觉编码器（CLIP ViT）的输出通过一个线性投影层连接到 Vicuna，实现了视觉-语言理解。

```text
[图像] → CLIP ViT → 线性投影 → Vicuna → 文本回答
```

Vicuna 在这一角色中的成功，部分原因是它的对话训练数据让模型特别适合"看图回答问题"的场景。

### 预训练的实用价值

1. **开源对话模型的引爆点**：$140 微调成本实现了接近 ChatGPT 的体验，引发了开源 LLM 的爆发
2. **高质量微调数据 > 复杂训练方法**：Vicuna 的成功证明了数据质量的决定性作用
3. **LLaVA 等下游模型的基石**：作为视觉-语言模型的标准语言骨干
4. **LMSYS/FastChat 评估框架**：GPT-4-as-judge 的评估方法成为开源 LLM 社区的标准
5. **催生了大批 Vicuna 衍生模型**：如 WizardVicuna、Vicuna+RLHF 等
