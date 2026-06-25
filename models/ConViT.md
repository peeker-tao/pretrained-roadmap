# ConViT (Convolution-augmented Vision Transformer)

## 基本信息

- **论文**: [ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases](https://arxiv.org/abs/2103.10697)
- **作者**: Stéphane d'Ascoli et al. (Meta / École Normale Supérieure)
- **发表**: ICML 2021

## 创新点

1. **软卷积归纳偏置**: 在自注意力中引入可学习的卷积偏置，初始化为卷积模式
2. **门控自注意力 (GPSA)**: 自注意力头可以"切换"为类卷积模式
3. **渐进式去偏**: 训练过程中自动学习是否需要卷积偏置

## 核心原理

### Gated Positional Self-Attention (GPSA)

$$\\text{GPSA}(X) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d}} + \\gamma \\cdot B\\right) V$$

- $B$ 是卷积偏置（局部先验）
- $\\gamma$ 是可学习门控系数，初始为 1（强卷积偏置）
- 训练过程中 $\\gamma$ 可以趋近 0（放弃卷积偏置）

### 与纯 ViT 的对比

- **ViT**: 无位置先验，需要大量数据学习空间关系
- **ConViT**: 初始有卷积偏置，小数据时表现更好

## 预训练方法

### 核心思想：ViT 在少数据时表现差是因为缺乏"局部优先"的归纳偏置——给它一个可学习的卷积偏置，让它自己决定什么时候"放弃偏置、拥抱全局"

ConViT 的动机来自一个关键观察：ViT 在小数据（ImageNet-1K）上不如 CNN，但在大数据（JFT-300M）上超越 CNN。区别在于 CNN 内置了**局部性**和**平移不变性**的归纳偏置——这些偏置在数据少时帮助模型，在数据多时可能成为限制。ConViT 的方案：**在自注意力中注入可学习的卷积偏置，让模型自己决定需要多少"局部性"。**

> ConViT = GPSA（门控位置自注意力）+ 可学习卷积偏置 $B$ + 可学习门控 $\gamma$。$\gamma$ 从 1（强卷积）开始，训练过程中可以衰减到 0（纯注意力）。

### 训练流水线（Step by Step）

#### Step 1 — GPSA（Gated Positional Self-Attention）

ConViT 的核心组件——带有门控卷积偏置的自注意力：

$$\text{GPSA}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + \gamma \cdot B\right) V$$

其中：
- $QK^T / \sqrt{d}$：标准自注意力的相似度矩阵（内容相关）
- $B \in \mathbb{R}^{N \times N}$：卷积偏置矩阵（位置相关）
- $\gamma$：可学习门控标量，控制卷积偏置的强度

**卷积偏置 $B$ 的设计**：

$B$ 是一个可学习的矩阵，初始化为局部卷积模式：

$$B_{ij} = \begin{cases} \text{可学习参数} & \text{如果 } |i-j| \leq \text{kernel\_size} \\ -\infty & \text{否则} \end{cases}$$

这意味着：每个 query token 只与 kernel_size 范围内的 key token 有额外的偏置——通过 softmax 后变成类似于卷积的局部加权。

#### Step 2 — 门控机制 $\gamma$

$\gamma$ 是一个小的可学习标量（每层、每个 head 独立）：

| $\gamma$ 的行为 | 含义 |
|---------------|------|
| $\gamma \approx 1$ | 强卷积偏置——类似 CNN 的行为 |
| $\gamma \approx 0$ | 无卷积偏置——退化为标准 Self-Attention（纯 ViT） |
| $\gamma$ 在训练中衰减 | **自动"放弃"卷积偏置，过渡到全局注意力** |

**初始化**：$\gamma_{\text{init}} = 1$（每个 head 都从強卷积模式开始）

**训练过程**：
- 浅层：$\gamma$ 倾向于保持较大（局部处理更高效）
- 深层：$\gamma$ 倾向于衰减至 0（全局语义推理更适合全局注意力）

> 类比：$\gamma$ 就像给自注意力装了一个"训练辅助轮"。刚开始骑车（训练初期，数据少），辅助轮（卷积偏置）提供稳定性和方向感；骑熟练了（训练后期），辅助轮收起来，完全靠自己（纯自注意力）。

#### Step 3 — GPSA 层的排列

ConViT 使用混合排列策略：

| 层 | 类型 | 说明 |
|----|------|------|
| 前几层 | GPSA（门控卷积注意力） | 局部特征提取，注入强卷积偏置 |
| 后几层 | SA（标准自注意力） | 全局语义建模，无偏置 |

**为什么不是所有层都用 GPSA？**
- 浅层：需要提取局部特征（边缘、纹理），卷积偏置加速这一过程
- 深层：需要整合全局信息（物体关系），卷积偏置反而限制全局视野

#### Step 4 — 监督预训练

| 参数 | ConViT 配置 | 说明 |
|------|------------|------|
| 数据集 | ImageNet-1K | 128 万张（有标签） |
| 架构 | GPSA + ViT Block | 混合排列 |
| 卷积核大小 | 3×3（空间） | 局部范围 |
| $\gamma$ 初始化 | 1 | 强卷积偏置起点 |
| 损失 | 交叉熵 | 标准分类 |
| 优化器 | AdamW | — |
| 学习率 | 按模型大小调整 | — |
| 数据增强 | RandAugment, Mixup, CutMix | 现代增强策略 |

#### Step 5 — 训练中 $\gamma$ 的动态变化

ConViT 的训练过程隐式实现了**从 CNN 到 ViT 的渐进过渡**：

```text
Epoch 0:   γ ≈ 1.0  → 每个 head 都是"类卷积"模式
Epoch 50:  γ ≈ 0.7  → 部分 head 开始放松卷积偏置
Epoch 150: γ ≈ 0.3  → 大多 head 偏向全局注意力
Epoch 300: γ ≈ 0.1  → 几乎纯自注意力（少数据时可能保持较高 γ）
```

**关键**：$\gamma$ 的演变是自动的——由梯度下降根据任务需求决定，无需人工干预。

### ConViT vs ViT vs CNN（小数据场景）

| 模型 | ImageNet-1K Top-1 | 数据需求 | 归纳偏置 |
|------|------------------|---------|---------|
| ViT-B/16 | 77.9% | 高 | 无 |
| **ConViT-B/16** | **81.3%** | **中** | **可学习** |
| ResNet-50 | 76.2% | 低 | 强（硬编码） |
| DeiT-B | 81.8% | 中（蒸馏） | 无 |

> ConViT 在小数据（ImageNet-1K）上远优于纯 ViT，接近需要知识蒸馏的 DeiT。可学习的卷积偏置弥合了 CNN 和 ViT 在小数据场景的差距。

### 详细训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据集 | ImageNet-1K | 128 万张 |
| Patch Size | 16×16 | — |
| GPSA 层数 | 前 10 层 | 其余为标准 SA |
| Heads 数 | 16 | $\gamma$ 每 head 独立 |
| 卷积核 (空间) | 3×3 | 局部偏置范围 |
| 损失 | 交叉熵 | — |
| 数据增强 | RandAug + Mixup + CutMix | — |
| 优化器 | AdamW | — |
| Epoch | 300 | — |

### 预训练的实用价值

1. **可学习归纳偏置**：证明了归纳偏置可以参数化——不需要在 CNN 和 ViT 之间"二选一"
2. **渐进式训练的自然实现**：从 CNN 到 ViT 的平滑过渡，训练过程中自动完成
3. **小数据 ViT 的解决方案**：在医学图像、遥感等小数据场景中表现优越
4. **GPSA 的通用性**：门控卷积偏置可以插入任何 ViT 架构
5. **理论可视化**：可以观察 $\gamma$ 的演变来理解模型学到了什么层次的特征
