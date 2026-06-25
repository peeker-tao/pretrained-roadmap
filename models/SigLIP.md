# SigLIP (Sigmoid Loss for Language Image Pre-training)

## 基本信息

- **论文**: [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)
- **作者**: Xiaohua Zhai et al. (Google)
- **发表**: ICCV 2023

## 创新点

1. **Sigmoid Loss 替代 Softmax**: 每对图文独立计算损失，不需要全局归一化
2. **小 batch 友好**: 不像 CLIP 需要大 batch 提供充分负例
3. **更高效的训练**: 单对损失计算，减少了跨样本依赖

## 核心原理

### Sigmoid Loss

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^N \left[ y_{ij} \log \sigma(s \cdot \text{sim}_{ij} + b) + (1-y_{ij}) \log(1 - \sigma(s \cdot \text{sim}_{ij} + b)) \right]$$

其中 $y_{ij} = 1$ 当 $i=j$（匹配对），否则为 0。$s$ 是可学习的温度参数，$b$ 是偏置。

### 与 InfoNCE 的对比

| 方面 | InfoNCE (CLIP) | Sigmoid Loss (SigLIP) |
|------|---------------|----------------------|
| 归一化方式 | 全局 Softmax | 逐对 Sigmoid |
| 负例利用 | 所有负例参与 | 每对独立 |
| 小 batch | 性能下降明显 | 性能稳定 |
| 训练效率 | 需要大 batch | batch 灵活 |

## 预训练方法

### 核心思想：谁说对比学习一定要 global softmax？改成 pairwise sigmoid 更香！

CLIP 使用 InfoNCE 损失——在一个大 batch 内，所有图文对互相竞争，通过 softmax 归一化来计算匹配概率。这个设计有一个问题：**InfoNCE 的性能严重依赖大 batch 提供充分的负例**，小 batch 下性能急剧下降。

SigLIP 提出了一个简单但有效的替代方案：**把全局 softmax 换成逐对的 sigmoid——每一对图文独立计算是否匹配**，不再需要跨样本的全局归一化。

> SigLIP = CLIP 的架构 + Sigmoid 损失替换 InfoNCE。它证明了：对比学习不需要全局归一化，逐对分类同样有效——而且更灵活、更高效。

### 训练流水线（Step by Step）

#### Step 1 — 相同的模型架构

SigLIP 使用与 CLIP 完全相同的双编码器架构：
- 图像编码器：ViT
- 文本编码器：Transformer
- 输出：图像嵌入 $z_I$、文本嵌入 $z_T$

**相似度计算**：

$$\text{sim}_{ij} = \frac{z_I^i \cdot z_T^j}{\|z_I^i\| \|z_T^j\|} \cdot e^t + b$$

其中 $t$ 是可学习的 log-温度参数（控制相似度的"锐度"），$b$ 是可学习的偏置（控制匹配判定的"阈值"）。

#### Step 2 — Sigmoid 损失（替代 InfoNCE）

**CLIP 的 InfoNCE 损失**：

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B}\sum_{i=1}^B \log \frac{\exp(\text{sim}_{ii}/\tau)}{\sum_{j=1}^B \exp(\text{sim}_{ij}/\tau)} - \frac{1}{B}\sum_{i=1}^B \log \frac{\exp(\text{sim}_{ii}/\tau)}{\sum_{j=1}^B \exp(\text{sim}_{ji}/\tau)}$$

问题：分母需要遍历 batch 中所有负例——batch 小时负例不够，性能下降。

**SigLIP 的 Sigmoid 损失**：

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{|B|} \sum_{i=1}^B \sum_{j=1}^B \left[ \mathbf{1}[i=j] \log \sigma(\text{sim}_{ij}) + \mathbf{1}[i \neq j] \log(1 - \sigma(\text{sim}_{ij})) \right]$$

其中 $\mathbf{1}[i=j]$ 是匹配指示器（对角线上为 1，其他为 0），$\sigma$ 是 sigmoid 函数。

> Sigmoid 损失的每个 $(i,j)$ 对独立计算——$i$ 和 $j$ 是否匹配是一个独立的二分类问题。不需要全局 softmax 归一化，$i$ 和 $j$ 之间的比较不依赖于 batch 中的其他样本。

#### Step 3 — Sigmoid vs Softmax 的深层含义

| 维度 | InfoNCE (Softmax) | Sigmoid Loss |
|------|-------------------|-------------|
| 数学本质 | 多分类（$B$ 选 1） | **$B^2$ 个二分类** |
| 归一化 | 全局 softmax | **逐对 sigmoid** |
| 负例数量敏感度 | 高（batch 依赖） | **低** |
| Batch Size | 至少 16K（CLIP） | **128 也可** |
| 小 batch 性能 | 差 | **好** |
| 大 batch 性能 | 好 | **也很好** |
| 计算量 | $O(B^2)$（softmax 归一化） | $O(B^2)$（逐对 sigmoid） |
| 理论性质 | softmax 的概率解释 | sigmoid 的二分类解释 |

#### Step 4 — 为什么 Sigmoid 更好？

**数学直觉**：

InfoNCE 可以被理解为："在 $B$ 个候选中，选出正确的那个"——这是一个 $B$ 分类问题。当 $B$ 小时，$B$ 分类太容易，模型学不到精细的区分能力。

Sigmoid 损失可以被理解为："对每一对 $(i,j)$，判断它们是否是匹配的"——这是 $B^2$ 个独立的二分类问题。即使 $B$ 小，二分类问题仍然有意义。

**关键优势**：
- **小 batch 友好**：每个正例对独立地与每个负例对比较，不依赖全局分布
- **灵活的负例利用**：可以自由选择使用哪些负例（in-batch / 全局 / 难例挖掘）
- **更好的梯度质量**：每个 $(i,j)$ 对独立计算梯度，不会因为 softmax 中某个强负例"支配"整个分母

### 详细训练配置

| 参数 | SigLIP Base (ViT-B/16) | SigLIP Large (ViT-L/16) | SigLIP Giant (ViT-g/14) |
|------|----------------------|------------------------|------------------------|
| 图像编码器 | ViT-B/16 | ViT-L/16 | ViT-g/14 |
| 文本编码器 | Transformer Base | Transformer Large | Transformer Large |
| 损失 | Sigmoid | Sigmoid | Sigmoid |
| 可学习温度 t | ✓ | ✓ | ✓ |
| 可学习偏置 b | ✓ | ✓ | ✓ |
| 训练数据 | WebLI（大规模图文对） | WebLI | WebLI |
| 优化器 | AdamW | AdamW | AdamW |
| 学习率 | 5e-4 | 3e-4 | 2e-4 |
| 学习率调度 | 余弦衰减 | 余弦衰减 | 余弦衰减 |
| Batch Size | 32768 | 32768 | 32768 |
| 训练步数 | — | — | — |

### 预训练性能

| 模型 | ImageNet 零样本 | 说明 |
|------|----------------|------|
| CLIP ViT-L/14 | 76.2% | InfoNCE 损失 |
| SigLIP ViT-L/16 | 78.5% | Sigmoid 损失 |
| SigLIP ViT-g/14 | **79.2%** | 更大的视觉编码器 |

SigLIP 用更简单的损失函数获得了更好的零样本分类性能。

### SigLIP 的影响

SigLIP 之后，Sigmoid 损失逐渐成为对比学习的可选方案：

| 工作 | 损失选择 |
|------|---------|
| CLIP (2021) | InfoNCE |
| SigLIP (2023) | **Sigmoid** |
| ImageBind (2023) | InfoNCE |
| EVA-CLIP (2023) | InfoNCE |
| BLIP-3/xGen-MM (2024) | **Sigmoid** |

### 预训练的实用价值

1. **对比损失的新范式**：证明逐对 sigmoid 可以替代全局 softmax
2. **降低了对大 batch 的依赖**：小 batch 也能有效对比训练
3. **可学习的温度和偏置**：模型自动调整匹配阈值
4. **更好的理论可解释性**：二分类比 softmax 多分类更容易分析和调优
5. **被 Google 多模态团队广泛采用**：作为 WebLI 数据训练的基础损失
