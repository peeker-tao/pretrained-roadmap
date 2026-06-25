# UNITER (UNiversal Image-TExt Representation)

## 基本信息

- **论文**: [UNITER: UNiversal Image-TExt Representation Learning](https://arxiv.org/abs/1909.11740)
- **作者**: Yen-Chun Chen et al. (Microsoft)
- **发表**: ECCV 2020

## 创新点

1. **统一多任务预训练**: 同时使用 4 种预训练任务
2. **条件掩码**: 在一种模态被掩码时，利用另一种模态提供条件信息
3. **大规模图文对预训练**: 使用多个数据集联合训练

## 核心原理

### 四种预训练任务

1. **掩码语言建模 (MLM)**: 掩码文本 token，以图像为条件预测
2. **掩码区域建模 (MRM)**: 掩码图像区域，以文本为条件预测
3. **图文匹配 (ITM)**: 判断图文是否匹配
4. **词-区域对齐 (WRA)**: 使用最优运输将词与图像区域对齐

### 条件掩码

与 BERT 不同，UNITER 的掩码预测可以利用另一模态的信息：
- MLM 可以参考图像特征
- MRM 可以参考文本特征

## 预训练方法

### 核心思想：图像和文字是"彼此的老师"

UNITER（UNiversal Image-TExt Representation）是 2019-2020 时代的 VLP（Vision-Language Pre-training）代表。它的核心创新是：**让图像和文本互相提供条件信息**——被掩码的文本 token 可以参考图像来预测，被掩码的图像区域可以参考文本。这与 BERT 的单模态 MLM 不同：BERT 只能依赖周围的文本，UNITER 可以跨模态求助。

> UNITER = 单塔 Transformer + 四种预训练任务。它的"条件掩码"设计使得在一种模态信息缺失时，另一种模态成为"上下文"。

### 训练流水线（Step by Step）

#### Step 1 — 输入构建：图文拼接

UNITER 使用**单塔（Single-Stream）架构**——图像区域和文本 token 拼接成一条序列，送入一个统一的 Transformer：

```text
[IMG] [Region1] [Region2] ... [RegionN] [CLS] [Token1] [Token2] ... [TokenM] [SEP]
```

- **图像区域**：使用 Faster R-CNN 检测物体的边界框，每个区域提取 RoI 特征
- **文本 token**：标准 WordPiece 分词
- **[IMG]** 和 **[CLS]**：特殊 token 标记模态边界

> UNITER 的单塔设计使得图像区域和文本 token 可以在 Self-Attention 中自由交互——"猫"这个 token 可以直接 attend 到猫的图像区域。

#### Step 2 — 四种预训练任务

**任务 1 — 掩码语言建模（MLM）**

掩码 15% 的文本 token，以图像区域和未掩码文本为条件预测：

$$\mathcal{L}_{\text{MLM}} = -\sum_{t \in \text{masked}} \log P(w_t | w_{\backslash t}, v)$$

其中 $v$ 是图像区域特征。

> 关键：当"猫"被掩码时，模型可以从图像中看到一个猫的边界框——这让预测变得更容易、更准确。

**任务 2 — 掩码区域建模（MRM）**

掩码 15% 的图像区域（替换为零向量），以文本和未掩码区域为条件预测：

$$\mathcal{L}_{\text{MRM}} = -\sum_{r \in \text{masked}} \log P(c_r | v_{\backslash r}, w)$$

其中 $c_r$ 是区域 $r$ 的语义类别（通过 Faster R-CNN 的分类器得到）。

> MRM 是"反向的 MLM"——当猫的图像区域被遮住，模型从文本"猫在沙发上"推测被遮住区域应该是猫。

**任务 3 — 图文匹配（ITM）**

给定 [CLS] token 的输出，做一个二分类：图像和文本是否匹配。训练时 50% 的图文对替换为不匹配的对（随机采样）。

$$\mathcal{L}_{\text{ITM}} = -\mathbb{E}_{(I,T)}[y \log p_{\text{[CLS]}}(I,T) + (1-y) \log(1-p_{\text{[CLS]}}(I,T))]$$

**任务 4 — 词-区域对齐（WRA）**

使用**最优运输（Optimal Transport）**将文本中的词与图像中的区域做"软"对齐：

$$\mathcal{L}_{\text{WRA}} = \min_{T \in \Pi(w,v)} \langle T, C \rangle$$

其中 $C_{ij}$ 是词 $i$ 和区域 $j$ 之间的距离（基于注意力权重计算），$T$ 是运输矩阵。

> WRA 教模型学会精细的词-区域对应——"cat"对应猫的区域，"sofa"对应沙发的区域。

#### Step 3 — 联合训练

$$\mathcal{L}_{\text{UNITER}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{MRM}} + \mathcal{L}_{\text{ITM}} + \lambda \mathcal{L}_{\text{WRA}}$$

四种任务互补：MLM 和 MRM 学习跨模态语义，ITM 学习全局对齐，WRA 学习细粒度对应。

### 为什么需要"条件掩码"？

BERT 的 MLM 只能利用单模态的上下文——当 BERT 预测被掩码的"猫"时，只能看周围的词。UNITER 的 MLM 可以跨模态求助——从图像中"看"到猫。

| 场景 | BERT MLM | UNITER MLM |
|------|---------|-----------|
| "The [MASK] is sleeping on the sofa" | 从上下文猜 | 从上下文**+图像中的猫**猜 |
| 被掩码的图像区域 [MASK] | 不存在（单模态） | 从文本 **"cat on sofa"** 推测 |

### 详细训练配置

| 参数 | UNITER-Base | UNITER-Large |
|------|-----------|-------------|
| 图像区域数 | 36 (Faster R-CNN) | 36 |
| 文本长度 | 动态 | 动态 |
| Transformer 层 | 12 | 24 |
| 隐藏维度 | 768 | 1024 |
| MLM 掩码率 | 15% | 15% |
| MRM 掩码率 | 15% | 15% |
| 优化器 | Adam | Adam |
| 学习率 | 5e-5 | 3e-5 |
| Warmup | 10% 步数 | 10% 步数 |
| Batch Size | 512 | 512 |
| 训练数据 | COCO + VG + CC + SBU (共 5.5M 图文对) | 同 Base |

### 预训练性能

| 任务 | UNITER-Base | UNITER-Large |
|------|-----------|-------------|
| VQA | 72.7 | 73.8 |
| NLVR² | 77.2 | 79.1 |
| 图文检索 (Flickr30K R@1) | 80.7 | 83.6 |
| SNLI-VE | 78.6 | 79.4 |

### 预训练的实用价值

1. **条件掩码的验证**：证明了跨模态条件信息对掩码预测的增强效果
2. **单塔架构的有效性**：最简单直接的跨模态融合方式
3. **多任务预训练的模板**：MLM+MRM+ITM 成为 VLP 领域的标准配置
4. **为后续 VLP 工作奠定基础**：影响了 VinVL、Oscar、VILLA 等后续模型
