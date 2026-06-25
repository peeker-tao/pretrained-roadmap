# DenseNet (Densely Connected Convolutional Networks)

## 基本信息

- **论文**: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- **作者**: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
- **发表**: CVPR 2017（最佳论文奖）

## 创新点

1. **密集连接**: 每一层都与之前所有层直接连接（L 层网络有 L(L+1)/2 个连接）
2. **特征复用**: 每一层接收所有先前层的特征图作为输入，最大化信息流动
3. **参数效率高**: 通过特征复用，用更少的参数达到超越 ResNet 的性能
4. **缓解梯度消失**: 短路径连接使梯度可以直接流向前层

## 核心原理

### Dense Block

每个 Dense Block 内部，第 $l$ 层的输入是所有前面层输出的拼接：

$$x_l = H_l([x_0, x_1, ..., x_{l-1}])$$

其中 $[·]$ 表示通道维度的拼接操作，$H_l$ 是组合操作（BN → ReLU → 3×3 Conv）。

### 增长率 (Growth Rate)

每个 Dense Block 中，每层输出的通道数 $k$ 称为增长率。通常取 $k = 12$ 或 $k = 32$，远小于其他网络的通道数。尽管每层输出很少，但所有前层输出的拼接使得输入仍然丰富。

### 过渡层 (Transition Layer)

Dense Block 之间的过渡层：BN → 1×1 Conv (降通道) → 2×2 Avg Pooling (降分辨率)

### 网络结构

DenseNet 有 DenseNet-121, DenseNet-169, DenseNet-201, DenseNet-264 等变体。

## 预训练方法

### 核心思想：ResNet 用"加法"融合特征（$x + f(x)$），DenseNet 用"拼接"融合特征（Concat）——让每一层都能"看到"之前所有层的原始输出

ResNet 的跳过连接是"把这一层的输出加到前面去"，信息流还是受限于残差学习。DenseNet 的密集连接更激进：**直接将之前所有层的输出在通道维度拼接起来**——这意味着第 100 层可以直接"看到"第 1 层的原始特征，没有经过任何变换或衰减。

> DenseNet 的密集连接是一种极端的特征复用策略——每一层都是前面所有层的"消费者"，同时也是后面所有层的"生产者"。这种设计使梯度流动极其顺畅，参数效率极高。

### 训练流水线（Step by Step）

#### Step 1 — Dense Block 内的数据流动

在一个 Dense Block 内部，第 $\ell$ 层接收的输入是：

$$x_\ell = H_\ell([x_0, x_1, x_2, ..., x_{\ell-1}])$$

其中 $[\cdot]$ 是通道维度的拼接（Concat），$H_\ell$ 是复合函数：BN → ReLU → 3×3 Conv。

**增长速度**：设增长率为 $k$（每层输出 $k$ 个特征图），则第 $\ell$ 层的输入通道数为：

$$\text{Input Channels} = C_0 + k \times (\ell - 1)$$

其中 $C_0$ 是该 Dense Block 的初始输入通道数。

> 如果 $k=32$，第 32 层的输入就有 $32 \times 31 = 992$ 个通道。这看起来会爆炸——但 DenseNet 的每层输出极少（$k$ 远小于普通网络），所以总参数量反而很低。

#### Step 2 — 增长率 $k$ 的选择

| 增长率 $k$ | 特点 |
|-----------|------|
| $k=12$ | 极致参数效率（DenseNet-121） |
| $k=32$ | 性能最强（DenseNet-169/201/264） |
| $k=48$ | 更大模型变体 |

$k=12$ 时，每层只输出 12 个特征图——但因为有密集连接，输入包含前面所有 12 通道的输出拼接，所以信息不会因输出通道少而丢失。

> 类比：增长率就像"每层对这个研究领域贡献的新知识量"——每层只贡献很少的新知识（$k=12$），但它可以访问所有前人的知识（拼接），所以不需要重复发明已有的知识。

#### Step 3 — 过渡层（Transition Layer）的压缩

Dense Block 之间的过渡层：

```text
Dense Block 输出 (大量通道)
  ↓ BN → ReLU → 1×1 Conv (压缩通道为 θ×输入通道数)
  ↓ 2×2 Average Pooling (stride 2, 降分辨率)
  ↓
下一个 Dense Block
```

**压缩因子** $\theta$：通常设 $\theta = 0.5$（如果输入有 600 通道，则压缩到 300 通道）。

过渡层的作用：
1. **通道压缩**：防止通道数爆炸式增长
2. **空间压缩**：通过 AvgPool 降低分辨率，扩大感受野

> 如果把 DenseNet 比作高速公路：Dense Block 是车辆汇入的区域，车道越来越多（通道增长）；过渡层是收费站，压缩车道、通过瓶颈后再恢复。

#### Step 4 — 损失函数与优化

**损失**：标准交叉熵

$$\mathcal{L} = -\sum_{c=1}^{1000} y_c \log\left(\frac{\exp(z_c)}{\sum_j \exp(z_j)}\right)$$

**优化器**：SGD + Nesterov Momentum

$$v_{t+1} = \mu v_t - \eta \nabla L(\theta_t + \mu v_t)$$
$$\theta_{t+1} = \theta_t + v_{t+1}$$

Nesterov 动量的关键在于 **先看一步**：在 $\theta_t + \mu v_t$（而非 $\theta_t$）处计算梯度。这就像"先探头看看前面路况，再决定这一步步长"——收敛更快。

**学习率调度 — 余弦退火**：

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

| 调度方式 | 特点 |
|---------|------|
| 阶梯衰减（AlexNet/VGG） | 硬性分阶段，手动设置衰减点 |
| **余弦退火（DenseNet）** | **平滑衰减，自动到最小值** |

#### Step 5 — 密集连接对梯度流的优化

DenseNet 的梯度流动比 ResNet 更优：

**ResNet 梯度**：

$$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_L} \cdot \prod_{i=0}^{L-1} \left(1 + \frac{\partial f_i(x_i)}{\partial x_i}\right)$$

**DenseNet 梯度**：由于 $x_L$ 直接拼接了 $x_0, x_1, ..., x_{L-1}$，梯度有 $L$ 条独立路径直接回传到 $x_0$：

$$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_0} \text{(直接路径)} + \sum_{i=1}^L \frac{\partial L}{\partial x_i} \cdot \frac{\partial x_i}{\partial x_0}$$

> 这就像信息从顶层到第 0 层有 $L$ 条独立的高速公路（而非一条需要经过多个收费站的公路）——梯度不会沿途衰减。

### 为什么 DenseNet 参数效率高？

| 网络 | 参数量 | ImageNet Top-1 | 参数效率 |
|------|--------|---------------|---------|
| ResNet-50 | 25.6M | 76.0% | 基准 |
| ResNet-101 | 44.6M | 77.4% | +74% 参数 = +1.4% |
| DenseNet-121 | **8.0M** | 75.0% | **仅 31% 参数** |
| DenseNet-201 | **20.0M** | 77.4% | **仅 45% 参数** |

> DenseNet-201 用 ResNet-101 一半不到的参数达到相同性能——密集连接对特征的复用率极高，每一层学到的特征被后续所有层直接利用，不浪费。

### 详细训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据集 | ImageNet | 120 万图 / 1000 类 |
| 架构变体 | DenseNet-121/169/201/264 | 数字 = 总层数 |
| 增长率 $k$ | 32 | 每层输出通道数 |
| 压缩因子 $\theta$ | 0.5 | 过渡层通道压缩 |
| 优化器 | SGD + Nesterov (0.9) | — |
| 学习率 | 0.1（初始），余弦退火 | 平滑衰减 |
| Batch Size | 256 | — |
| 权重衰减 | 1e-4 | — |
| Epoch | 90（标准）/ 300（更优） | — |
| 数据增强 | 随机裁剪 224×224 + 翻转 | — |

### 预训练性能

| 模型 | Top-1 Acc | Top-5 Error | 参数量 | FLOPs |
|------|----------|------------|--------|-------|
| DenseNet-121 | 74.98% | 7.71% | 8.0M | 2.9B |
| DenseNet-169 | 76.17% | 6.89% | 14.1M | 3.4B |
| DenseNet-201 | 77.42% | 6.14% | 20.0M | 4.4B |
| DenseNet-264 | 77.85% | 5.82% | 33.3M | 5.9B |

### 预训练的实用价值

1. **特征复用的极致示范**：DenseNet 的预训练特征具有天然的多尺度特性，每一层的特征都可以被后续所有层利用
2. **紧凑表征**：DenseNet 的预训练特征比同性能的 ResNet 更紧凑（通道数少），在迁移学习中存储和传输成本更低
3. **密集连接启发了后续架构设计**：MixNet（MixConv）、CSPNet 等都借鉴了密集连接的思想
4. **医学图像中的主流选择**：DenseNet 的紧凑特征在医学图像（数据少、标注贵）中表现突出
5. **与注意力机制的互补**：DenseNet 的密集连接 + SE 通道注意力的组合在多个任务中达到 SOTA
