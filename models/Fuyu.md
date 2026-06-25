# Fuyu

## 基本信息

- **论文**: [Fuyu: A Multimodal Architecture for AI Agents](https://www.adept.ai/blog/fuyu-8b)
- **作者**: Adept AI
- **发表**: 博客文章, 2024

## 创新点

1. **极简多模态架构**: 直接将图像嵌入拼接在文本序列中
2. **无专门视觉编码器**: 所有处理由同一 Transformer 完成
3. **支持任意分辨率**: 无需固定输入尺寸

## 核心原理

### Fuyu 架构

1. 将图像按 patch 展开为线性序列
2. 在文本序列前直接拼接图像 patch 序列
3. 所有操作由统一的 Transformer 处理
4. 无独立的视觉编码器 (如 ViT)

### 与传统方法的对比

| 方法 | 视觉编码器 | 桥接模块 | 架构复杂度 |
|------|-----------|---------|-----------|
| LLaVA | 有 (CLIP ViT) | 投影层 | 中 |
| BLIP-2 | 有 | Q-Former | 高 |
| Fuyu | **无** | **无** | **极低** |

## 预训练方法

### 核心思想：去掉所有专门的视觉组件——只有一个"看图和读文字都一样"的 Transformer

Fuyu 是多模态架构中的"异类"——它没有 CLIP ViT、没有 Q-Former、没有投影层、没有交叉注意力。它直接把图像分割成 patch，像 token 一样拼接到文本序列前面，送入一个**纯解码器 Transformer**。这是"大道至简"在多模态模型中的极致体现。

> Fuyu = 图像 patch 看作"视觉 token" + 标准的下一个 token 预测。不需要任何视觉专用模块——图像的 patch 嵌入和文本的 token 嵌入通过相同的嵌入矩阵和相同的 Transformer 处理。

### 训练流水线（Step by Step）

#### Step 1 — 图像转"视觉 token"

给定一张任意分辨率的图像：

1. 将图像分割为固定大小的 patch（如 30×30 像素）
2. 每个 patch 通过一个**线性投影层**变为一个向量
3. 添加可学习的位置编码
4. 所有 patch 向量排列成一个一维序列

```text
图像 (H×W) → 分割为 N 个 patch → 线性投影 → [patch_1, patch_2, ..., patch_N]
```

**关键**：Fuyu 使用**线性投影**而非卷积或 ViT——这是因为转换的"视觉 token"将与文本 token 共用同一个 Transformer 进行处理，只需要一个简单的维度映射。

#### Step 2 — 图文序列拼接

将视觉 token 序列直接拼接到文本 token 序列前面：

```text
[patch_1] [patch_2] ... [patch_N] [BOS] [token_1] [token_2] ... [token_M] [EOS]
```

- 没有 [IMG] 这样的特殊分隔 token
- 没有 [CLS] token
- 视觉和文本 token 在序列中"平权"——Self-Attention 可以自由地让任何 token 与任何其他 token 交互

#### Step 3 — 标准自回归训练

$$\mathcal{L} = -\sum_{t} \log P(x_t | x_{<t}, \text{patches})$$

其中 $x_1, ..., x_M$ 是文本 token，$\text{patches}$ 是图像 patch 序列作为条件。

**损失计算**：仅在文本 token 上计算（不在 patch 上计算），标准的因果语言模型损失。

#### Step 4 — 支持任意分辨率

Fuyu 的一个巧妙设计：不需要固定的输入尺寸。因为图像被分割为 patch，而 patch 大小固定，所以不同分辨率的图像会产生不同数量的视觉 token。

| 分辨率 | Patch 数（30px patch） |
|--------|---------------------|
| 224×224 | ~56 patch |
| 448×448 | ~224 patch |
| 任意尺寸 | 动态 |

> 这种灵活性对于 AI Agent 场景至关重要——Agent 看到的可能是任意尺寸的屏幕截图。

### 为什么 Fuyu 的极简架构有效？

#### Fuyu vs LLaVA vs BLIP-2

| 组件 | LLaVA | BLIP-2 | Fuyu |
|------|-------|--------|------|
| 视觉编码器 | CLIP ViT (300M) | CLIP ViT (300M) | **无** |
| 桥接模块 | MLP 投影 | Q-Former (188M) | **无** |
| LLM | Vicuna (7B) | LLaMA/OPT (7B) | 8B 纯解码器 |
| 视觉专用参数 | ~300M (冻结) | ~488M | **0** |
| 总参数量 | ~7.3B | ~7.5B | **8B（全是语言参数）** |
| 输入分辨率 | 固定 336²（LLaVA-1.5） | 固定 224² | **任意** |

Fuyu 的权衡是：**放弃专门的视觉编码器，换取极致的架构简洁性**。代价是在纯视觉任务（如 ImageNet 分类）上不如 CLIP ViT，但好处是：
- 不需要处理视觉编码器的输出格式
- 不需要协调冻结/微调
- 任意分辨率输入天然支持

#### 为什么直拼 patch 能工作？

原因在于 Transformer 的 Self-Attention 本质上是**位置无关的**——它对图像 patch 和文本 token 一视同仁：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}}\right)V$$

对于 Self-Attention 来说，图像 patch 1 和文本 token "cat" 只是两个向量——它们之间的注意力计算和"cat"与"dog"之间的注意力计算没有区别。

> Fuyu 相信：只要你给 Transformer 足够的数据和足够的计算，它就能自己学会"看懂"——不需要专门的视觉编码器来"翻译"。

### 详细训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | 8B 纯解码器 Transformer | 类似 Persimmon-8B |
| 视觉输入 | 原始像素 → patch → 线性投影 | 无视觉编码器 |
| Patch 大小 | 30×30 像素 | — |
| 文本分词 | SentencePiece | — |
| 训练数据 | 图文交错数据 + 文本数据 | 具体规模未公开 |
| 损失 | 标准因果 LM 损失 | 仅计算文本 token |
| 优化器 | AdamW | — |
| 最大序列长度 | 支持 16K+ tokens | 可容纳大量 patch |

### 预训练的实用价值

1. **架构极简主义的极致**：去掉了所有视觉专用模块，统一为纯 Transformer
2. **任意分辨率输入**：对 AI Agent（屏幕截图、文档扫描）等场景至关重要
3. **训练和推理的简化**：不需要处理多编码器的协调问题
4. **启发式设计**：影响了 Chameleon、Gemini 等原生多模态模型的设计思路
5. **Adept AI Agent 的基础视觉能力**：Fuyu 是 Adept 的 ACT-1 Agent 的视觉理解组件
