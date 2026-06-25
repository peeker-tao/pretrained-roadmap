# CosFace

## 基本信息

- **论文**: [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/abs/1801.09414)
- **作者**: Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou, Zhifeng Li, Wei Liu
- **发表**: CVPR 2018

## 创新点

1. **余弦间隔 (Cosine Margin)**: 在余弦空间中引入特征间隔，替代传统的欧几里得间隔
2. **归一化特征**: 同时归一化特征向量和权重向量，使学习聚焦于角度/余弦差异
3. **大间隔分类**: 通过余弦余量 $m$ 实现更严格的分类边界

## 核心原理

### CosFace Loss (LMCL)

CosFace 损失函数（Large Margin Cosine Loss）在归一化空间中对 Softmax 进行修改：

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \log \frac{e^{s(\cos(\theta_{y_i})-m)}}{e^{s(\cos(\theta_{y_i})-m)} + \sum_{j \neq y_i} e^{s\cos\theta_j}}$$

其中：
- $s$: 特征缩放因子（固定为 64）
- $m$: 余弦间隔（通常取 0.35）
- $\cos\theta_j$: 特征与第 $j$ 类权重的余弦相似度

### 与 ArcFace 的对比

| 方法 | 间隔位置 | 几何解释 | 间隔形式 |
|------|---------|---------|---------|
| CosFace | 余弦空间 | $\cos(\theta_1)-m > \cos(\theta_2)$ | 减法间隔 |
| ArcFace | 角度空间 | $\cos(\theta_1+m) > \cos(\theta_2)$ | 加法间隔 |

## 预训练方法

### 核心思想：ArcFace 在角度上加间隔 $m$——CosFace 则在余弦值上减间隔 $m$。两者等价吗？不——它们在决策边界的几何形状上有微妙差别

CosFace（又名 LMCL：Large Margin Cosine Loss）与 ArcFace 同属 margin-based 人脸识别家族。区别在于**间隔的位置**：CosFace 在余弦空间中减 $m$，ArcFace 在角度空间中加 $m$。这一看似微小的差异导致了不同的梯度行为。

> CosFace = L2 归一化特征 + L2 归一化权重 + 余弦空间减法间隔 $m$ + 缩放 $s$。$\cos\theta_i - m > \cos\theta_j$ 意味着正类的余弦值至少要比负类高 $m$。

### 训练流水线（Step by Step）

#### Step 1 — 双重归一化

CosFace 首先对特征和权重进行 L2 归一化：

$$\|x\| = 1, \quad \|W_j\| = 1$$

这意味着 logit 完全由余弦相似度 $\cos\theta_j = W_j^T x$ 决定。

**为什么归一化和 ArcFace 一样重要？** 人脸图像的质量差异巨大（光照、姿态、表情）。如果不归一化，高质量图像的嵌入模长会远大于低质量图像——模型按"图像质量"而非"身份"来分类。归一化消除了这个捷径。

#### Step 2 — 余弦空间的减法间隔

$$\mathcal{L}_{\text{CosFace}} = -\frac{1}{N}\sum_i \log \frac{e^{s(\cos\theta_{y_i} - m)}}{e^{s(\cos\theta_{y_i} - m)} + \sum_{j \neq y_i} e^{s\cos\theta_j}}$$

**间隔在余弦空间的含义**：

| 情况 | $\cos\theta_{\text{正}}$ | $\cos\theta_{\text{负}}$ | 无间隔判决 | CosFace 判决 |
|------|------------------------|------------------------|----------|------------|
| 边界 | 0.6 | 0.55 | 0.6 > 0.55 ✓ | 0.6-0.35=0.25 < 0.55 ✗ |
| 合格 | 0.9 | 0.55 | ✓ | 0.9-0.35=0.55 = 0.55 (刚好) |
| 安全 | 0.95 | 0.3 | ✓ | 0.95-0.35=0.6 > 0.3 ✓ |

> 减 $m$ 本质上提升了"正类的门槛"——你的正类相似度必须比负类高出至少 $m$，才能被正确分类。

#### Step 3 — CosFace vs ArcFace：梯度行为的关键差异

| 维度 | CosFace | ArcFace |
|------|---------|---------|
| 间隔位置 | $\cos\theta - m$ | $\cos(\theta + m)$ |
| 几何空间 | 余弦空间 | 角度空间 |
| 间隔单位 | 无单位（余弦值） | 弧度 |
| 对小角度梯度 | 相同 | 相同 |
| **对大角度梯度** | **恒定（$\partial L / \partial \theta \approx s \cdot \sin\theta$）** | **更陡（$\partial L / \partial \theta \approx s \cdot \sin(\theta+m)$）** |

**大角度时 $\theta$ 大（正类特征偏离中心远）**：
- ArcFace 的梯度中有 $\sin(\theta+m)$ 的放大效应 → 对大角度样本（困难样本）惩罚更重
- CosFace 的梯度中是 $\sin\theta$ → 恒定惩罚

> 这意味着 ArcFace 比 CosFace 更关注"最难的正例"——训练后期，ArcFace 会更多地推动那些偏离中心的困难样本回到类中心。

#### Step 4 — $m$ 和 $s$ 的选择

| 参数 | CosFace 推荐值 | 作用 |
|------|-------------|------|
| $s$ | 64 | 与 ArcFace 相同 |
| $m$ | 0.35 | 比 ArcFace 的 0.5 小（因为操作在余弦空间，范围 [0,1] 而非角度空间 [0,π]） |

**为什么 CosFace 的 $m$ 比 ArcFace 小？**
- ArcFace 的 $m=0.5$ 弧度 ≈ 28.6°
- CosFace 的 $m=0.35$ 余弦值 → 相当于 $\cos^{-1}(0.35) \approx 69.5°$？不——间隔是减法，不是映射
- 实际等效：$\cos\theta - 0.35 > \cos\theta'$ 的几何约束 ≈ 在角度上有 ~20-30° 的等效间隔

#### Step 5 — 训练流程

```text
1. 人脸图像 → 检测 + 对齐 → 112×96 RGB
2. 数据增强: 水平翻转（训练时）
3. ResNet-50 → 512 维嵌入
4. L2 归一化: ‖x‖ = 1
5. 线性层: 权重 L2 归一化 ‖W_j‖ = 1
6. 计算余弦相似度: cosθ = W^T x
7. 正类余弦减 m: cosθ_y - 0.35
8. 所有 logit 乘 s=64
9. Softmax → 交叉熵 → 反向传播
```

### CosFace vs ArcFace vs SphereFace

| 方法 | 间隔公式 | 间隔位置 | 对大角度梯度 | 训练稳定性 |
|------|---------|---------|------------|----------|
| SphereFace | $\cos(m\theta)$ | 角度空间（乘法） | 非线性 | 不稳定（需要退火 $m$） |
| **CosFace** | $\cos\theta - m$ | **余弦空间（减法）** | **恒定** | **稳定** |
| ArcFace | $\cos(\theta + m)$ | 角度空间（加法） | 更强 | 稳定 |

### 详细训练配置

| 参数 | CosFace 配置 | 说明 |
|------|------------|------|
| 数据集 | CASIA-WebFace (0.5M) / MS-Celeb-1M (10M) | — |
| 架构 | ResNet-50 / ResNet-64 | — |
| 嵌入维度 | 512 | — |
| 间隔 $m$ | 0.35 | 余弦空间减法间隔 |
| 缩放 $s$ | 64 | — |
| 优化器 | SGD + Momentum(0.9) | — |
| 学习率 | 0.1 → 分段衰减 | — |
| 权重衰减 | 5e-4 | — |
| Batch Size | 64/GPU × 8 GPU = 512 | 多 GPU 训练 |

### 预训练的实用价值

1. **人脸识别的双雄之一**：与 ArcFace 并列为最常用的两种 margin-based 损失
2. **余弦空间的直观性**：$\cos\theta - m$ 的公式比 $\cos(\theta+m)$ 更易于实现和理解
3. **归一化嵌入的推广**：双重归一化（特征 + 权重）被广泛采用
4. **类别不平衡的鲁棒性**：相比于 Triplet Loss，margin-based 损失对类别分布不敏感
5. **细粒度分类的迁移**：CosFace/ArcFace 的 margin 策略被迁移到车辆识别、鸟类分类等任务
