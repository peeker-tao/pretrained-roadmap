# ResNet (Residual Network)

## 基本信息

- **论文**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **作者**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft)
- **发表**: CVPR 2016（最佳论文奖）

## 创新点

1. **残差连接 (Skip Connection)**: 引入跨层恒等映射 $y = F(x) + x$，解决深层网络退化问题
2. **超深网络训练**: 成功训练 152 层网络（VGG 的 8 倍深），ImageNet 错误率降至 3.57%
3. **瓶颈设计 (Bottleneck)**: 使用 1×1→3×3→1×1 的瓶颈结构大幅降低计算量
4. **批量归一化 (Batch Normalization)**: 加速训练、稳定梯度

## 核心原理

### 退化问题

实验发现，简单地堆叠更多层会导致**训练误差**反而增大（非过拟合），这说明优化器难以拟合恒等映射。ResNet 通过残差学习解决这一问题：

$$F(x) = H(x) - x$$

其中 $H(x)$ 是期望的底层映射，网络学习的是残差 $F(x)$。当恒等映射是最优时，网络只需将残差推至 0 即可，这比直接学习恒等映射容易得多。

### 残差块设计

**Basic Block**（用于 ResNet-18/34）：
- 3×3 conv → BN → ReLU → 3×3 conv → BN → 残差相加 → ReLU

**Bottleneck Block**（用于 ResNet-50/101/152）：
- 1×1 conv (降维) → 3×3 conv → 1×1 conv (升维) → 残差相加 → ReLU

### 网络变体

| 模型 | 层数 | 结构特点 |
|------|------|---------|
| ResNet-18 | 18 | Basic Block，轻量 |
| ResNet-34 | 34 | Basic Block |
| ResNet-50 | 50 | Bottleneck，最常用 |
| ResNet-101 | 101 | 更深，更高精度 |
| ResNet-152 | 152 | 最深版本 |

## 预训练方法

### 核心思想：在 ImageNet 上学习通用视觉特征

ResNet 的预训练是**监督学习**——直接用 ImageNet 的 1000 类标签来训练。虽然这看起来"朴实无华"，但 ResNet 预训练权重是计算机视觉领域**最广泛使用**的通用骨干，几乎所有下游任务（检测、分割、检索、跟踪……）的默认做法都是："先加载 ImageNet 预训练的 ResNet，然后微调"。

### 训练流水线（Step by Step）

#### Step 1 — 数据准备与增强

ImageNet ILSVRC 2012 数据集：
- **训练集**：约 128 万张图片，1000 个类别
- **验证集**：5 万张图片

每张图片经过以下数据增强：

1. **随机裁剪（RandomResizedCrop）**：将图片缩放到短边 256，然后随机裁剪为 $224 \times 224$
2. **随机水平翻转**：50% 概率
3. **PCA 颜色增强（Color Jittering）**：对 RGB 通道做 PCA 后施加随机噪声——这是一种轻微的颜色扰动，增加模型对光照变化的鲁棒性

测试时使用 **10-crop 评估**：从测试图片的四个角 + 中心 + 翻转，共 10 个裁剪结果，取平均预测。

#### Step 2 — 前向传播：残差学习

输入图片 $224 \times 224 \times 3$ 经过：

```
Conv1 (7×7, stride 2, 64通道, BN, ReLU)
  → MaxPool (3×3, stride 2)
  → [ResBlock] × N₁（Stage 1, 通道数=64或256）
  → [ResBlock] × N₂（Stage 2, 通道数=128或512, stride 2下采样）
  → [ResBlock] × N₃（Stage 3, 通道数=256或1024, stride 2）
  → [ResBlock] × N₄（Stage 4, 通道数=512或2048, stride 2）
  → Global Average Pooling → 1000-d FC → Softmax
```

其中最重要的结构是**残差块**：

**Bottleneck Block（ResNet-50/101/152）**：

```
输入: 256维 → 1×1 Conv (降维到64) → BN → ReLU
           → 3×3 Conv (64) → BN → ReLU
           → 1×1 Conv (升维到256) → BN
           → 与原始输入相加 → ReLU
```

瓶颈设计的巧妙之处：$1 \times 1 \to 3 \times 3 \to 1 \times 1$ 的结构把计算量降到了直接使用 $3 \times 3$ 卷积的约 $1/3$，同时因为 $1 \times 1$ 的"先降后升"增加了一层非线性变换。

#### Step 3 — 损失计算与反向传播

对 1000 类分类任务使用**多类交叉熵损失**：

$$\mathcal{L} = -\sum_{c=1}^{1000} y_c \log(\hat{y}_c)$$

其中 $\hat{y}_c$ 是 Softmax 输出的类别概率。

#### Step 4 — 阶梯式学习率下降

ResNet 使用"阶梯式"（step-wise）学习率衰减，这是早期深度学习训练的经典策略：

| Epoch | 学习率 | 说明 |
|-------|--------|------|
| 0-30 | 0.1 | 高学习率快速收敛 |
| 31-60 | 0.01 | 第一次衰减（÷10） |
| 61-90 | 0.001 | 第二次衰减（÷10） |

> 与现代的余弦衰减不同，阶梯式衰减更"粗暴"——但它是 2015 年前后的标准做法，而且效果很好。每个 loss plateau 阶段被学习率下降"解放"，继续优化到更低的 loss。

### 为什么 ResNet 预训练如此通用？

#### 1. 残差连接的"特征空间保持"

关键洞察：残差连接 $y = F(x) + x$ 不仅解决了梯度问题，还有一个重要效果——**特征空间在逐层传递时保持相对稳定**。

- 没有残差连接：每层都在从零开始"重新编码"，层间特征空间变化剧烈
- 有残差连接：每层只需要"微调"前一层的输出，特征空间逐层平滑演化

这意味着 ResNet 的各层特征都是"有意义的中间表示"——浅层学边缘、纹理，深层学语义——这使得它的中间层特征非常适于迁移到其他任务（如 FPN 特征金字塔用于检测、分割）。

#### 2. 批量归一化（BN）的关键作用

ResNet 在每个卷积层后、ReLU 之前都插入 BN：

$$ \hat{x} \leftarrow \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}, \quad y \leftarrow \gamma \hat{x} + \beta $$

BN 在 ResNet 中扮演了三个角色：
1. **加速训练**：减少 internal covariate shift，允许更大的学习率
2. **正则化**：mini-batch 的统计噪声起到了类似 Dropout 的正则化效果（因此 ResNet 不需要 Dropout）
3. **稳定梯度**：防止深层网络的梯度消失/爆炸

### 详细训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据集 | ImageNet-1K | 128 万训练图片，1000 类 |
| 输入尺寸 | 224×224 | 随机裁剪自 256 短边 |
| 优化器 | SGD + Momentum | momentum = 0.9 |
| 初始学习率 | 0.1 | 较大的初始学习率 |
| 学习率衰减 | ÷10 at 30/60 epoch | 阶梯式（共 90 epoch） |
| 权重衰减 | 0.0001 | L2 正则化 |
| 批量大小 | 256 | 分 8 GPU |
| 总迭代次数 | 60 万 | 约 90 epoch |
| 权重初始化 | He 初始化（MSRA） | 适用于 ReLU 的初始化 |
| BN epsilon | 1e-5 | — |
| 评估方式 | 中心裁剪 224×224 | 或用 10-crop |

#### He 初始化

ResNet 使用专门为 ReLU 激活设计的权重初始化方法：

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

其中 $n_{\text{in}}$ 是输入维度。这个初始化确保 ReLU 激活后的方差在各层之间保持稳定——这对于没有 BN 时的深层网络尤为重要。

### ResNet 预训练权重作为"标准基线"

ResNet-50 的 ImageNet 预训练权重（Top-1 ~76.1%）成为了 CV 领域的"标准度量衡"：

| 下游任务 | ResNet-50 迁移效果 |
|---------|-------------------|
| COCO 目标检测（Faster R-CNN） | ~37-39 AP |
| COCO 实例分割（Mask R-CNN） | ~34 AP |
| Pascal VOC 检测 | ~81 mAP |
| 图像检索（ReID） | rank-1 ~88-95% |
| 工业缺陷检测 | 高精度（fine-tune 后） |

**注意**：ResNet 的 "预训练" 是纯监督学习，与 BERT、GPT、SimCLR 等自监督预训练有本质区别。但正是因为它作为监督预训练的极致打磨，使其成为了后续所有自监督方法的**对标基线**——"我们的自监督方法在 ImageNet 线性探测上达到了 xx%，超越了监督预训练的 ResNet-50"——这句话几乎是所有自监督论文的标配。

### 预训练的迁移价值

1. **最通用的视觉骨干**：近 10 年来，ResNet 是计算机视觉中使用最广泛的预训练模型
2. **稳定可靠的基线**：新方法在 ResNet 上的表现可作为与现有工作的直接比较
3. **丰富的生态**：torchvision 直接提供预训练权重，一行代码即可加载
4. **适配性极强**：检测（Faster R-CNN/RetinaNet）、分割（FCN/DeepLab）、检索、生成……几乎所有视觉任务都有适配 ResNet 的方案
