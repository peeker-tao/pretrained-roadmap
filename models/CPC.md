# CPC (Contrastive Predictive Coding)

## 基本信息

- **论文**: [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
- **作者**: Aaron van den Oord, Yazhe Li, Oriol Vinyals (DeepMind)
- **发表**: arXiv, 2018

## 创新点

1. **InfoNCE Loss**: 提出 InfoNCE（Noise Contrastive Estimation）损失函数，成为对比学习的标准损失
2. **预测未来潜表征**: 在潜空间中预测未来时间步的表征，而非直接在原始信号空间中预测
3. **跨模态通用框架**: 适用于图像、语音、文本和强化学习

## 核心原理

### 对比预测编码

CPC 的核心思想是：**学习一个能够编码过去信息并预测未来潜表征的模型**。

1. 编码器将原始序列映射到潜空间
2. 自回归模型（GRU）总结历史潜表征
3. 使用 InfoNCE 区分真正的未来表征 vs. 负样本

### InfoNCE Loss

$$\mathcal{L} = -\mathbb{E}_X \left[ \log \frac{f_k(x_{t+k}, c_t)}{\sum_{x_j \in X} f_k(x_j, c_t)} \right]$$

其中 $f_k(x, c) = \exp(z_x^T W_k z_c)$，$c_t$ 是上下文表征，$x_{t+k}$ 是未来潜表征。

### InfoNCE 与互信息

InfoNCE 损失的下界是互信息：
$$I(x_{t+k}; c_t) \geq \log(N) - \mathcal{L}_{\text{InfoNCE}}$$

## 预训练方法

### 核心思想：给你一部电影的前 10 分钟——你能否预测接下来会发生什么？CPC 做的是同样的事：编码器把过去的信息压缩成"上下文"，然后用对比损失（InfoNCE）区分"真正的未来"和"假的未来"。这种"预测未来"的能力迫使模型学到真正有意义的表征——它必须理解世界的规律

CPC（Contrastive Predictive Coding）是 InfoNCE 损失函数的诞生论文，也是对比学习思想的最早系统性阐述。它的框架极其通用：适用于图像、语音、文本、强化学习——任何有序列结构的数据。

> CPC = 编码器（CNN/ResNet）+ 自回归模型（GRU）+ InfoNCE 损失（区分"真正的未来表征" vs "随机抽取的负样本"）。学到的表征 = 互信息最大化的压缩表示。

### 训练流水线（Step by Step）

#### Step 1 — 通用 CPC 框架

```text
原始序列: x_1, x_2, ..., x_T

步骤 1: 编码
  对于每个时间步 t:
    z_t = g_enc(x_t)    ← 编码器将原始信号压缩到潜空间

步骤 2: 上下文总结
  c_t = g_ar(z_1, ..., z_t)   ← 自回归模型总结 1 到 t 的历史

步骤 3: 预测未来
  对于未来步 k (k=1,2,3,...):
    预测: W_k · c_t           ← 线性投影从上下文"跳到"未来
    真实: z_{t+k}             ← 未来真实表征
    负样本: z_{random}        ← 随机抽取的其他表征
```

**为什么在潜空间中预测？**

| 原始信号空间 | 潜空间（CPC） |
|-----------|------------|
| 预测像素值 | 预测特征向量 |
| 需要建模所有细节 | **只需建模语义信息** |
| 损失: MSE（产生模糊预测） | **损失: InfoNCE（学到语义）** |
| 低效 | **高效** |

> 类比：预测下 5 分钟的电影不需要逐像素预测每个画面——只需要预测"接下来会发生什么"（语义）。潜空间预测就是这个意思。

#### Step 2 — InfoNCE 损失（核心贡献）

CPC 的最大贡献是 InfoNCE（Noise Contrastive Estimation）损失：

$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}_X \left[ \log \frac{\exp(z_{t+k}^T \cdot W_k \cdot c_t)}{\sum_{j=1}^N \exp(z_j^T \cdot W_k \cdot c_t)} \right]$$

| 符号 | 含义 |
|------|------|
| $z_{t+k}$ | 真正的未来表征（正样本） |
| $c_t$ | 上下文表征（历史摘要） |
| $z_j$ (j≠t+k) | 负样本（从 mini-batch / memory bank 中抽取） |
| $W_k$ | 第 k 步的线性预测矩阵 |
| $N$ | 负样本数量 |

**InfoNCE 的直觉**：

```text
分子: 正样本的相似度（上下文和真正未来的匹配度）
分母: 所有候选（正+负）的相似度之和
损失: -log(正样本概率)

→ 模型被迫使"正样本的匹配度 > 所有负样本的匹配度"
→ 这就是"对比"——在"对比"中找到真正的未来
```

#### Step 3 — InfoNCE 与互信息的关系

CPC 的一个关键理论贡献：InfoNCE 是互信息的变分下界：

$$I(x_{t+k}; c_t) \geq \log(N) - \mathcal{L}_{\text{InfoNCE}}$$

| 不等式 | 含义 |
|--------|------|
| $I(x_{t+k}; c_t)$ | 上下文 $c_t$ 和未来 $x_{t+k}$ 的互信息 |
| 越大越好 | $c_t$ 知道$x_{t+k}$越多越好 |
| $\log(N)$ | 负样本越多 → 互信息估计越紧 |
| $-\mathcal{L}_{\text{InfoNCE}}$ | 损失越小 → 互信息下界越高 |

> 训练 InfoNCE 本质上是**最大化上下文与未来的互信息**——这就是"学到的表征为什么好"的信息论解释。

#### Step 4 — 不同模态的 CPC 预训练

**语音 CPC（标准配置）**：

```text
数据: LibriSpeech (960 小时英语语音)
输入: 16kHz 原始波形
编码器: 5 层 CNN (kernel=10,5,8,4,4 步长=5,4,4,4,4)
  将 16kHz 波形 → 100Hz 潜表征序列

自回归: 1 层 GRU (256 维)
上下文: 来自当前 c_t，预测未来 1-12 步
负样本: 每次 10 个
```

| 参数 | 语音 CPC | 说明 |
|------|---------|------|
| 编码器 | 5 层 CNN | — |
| 自回归 | 1 层 GRU | 256 维 |
| 预测步数 | 1-12 | 预测 ~120ms 的未来 |
| 负样本数 | 10 | — |
| 训练数据 | LibriSpeech 960h | 英语有声书 |
| 下采样率 | 160× | 16kHz→100Hz |
| 损失 | InfoNCE | — |

**图像 CPC**：

```text
图像 CPC 的"序列":
  将图像分割为重叠的 patch (7×7 网格)
  从上到下逐行扫描 = 49 步的"序列"
  
上下文: 上方的 patch (c_above)
预测: 下方相邻 patch 的潜表征
负样本: 随机位置的其他 patch
```

| 参数 | 图像 CPC | 说明 |
|------|---------|------|
| 编码器 | ResNet-101 (去最后层) | — |
| Patch 网格 | 7×7（重叠） | 49 个 patch |
| 自回归 | 1 层 GRU | — |
| 预测目标 | 下方 0-5 行的 patch | — |
| 负样本 | 随机位置的 patch | — |

#### Step 5 — InfoNCE 的影响力

InfoNCE 成为**几乎所有现代对比学习的标准损失**：

| 方法 | 损失 | 与 CPC 的关系 |
|------|------|------------|
| CPC | InfoNCE | **原版** |
| SimCLR | NT-Xent (归一化的 InfoNCE) | 继承 |
| MoCo | InfoNCE + Memory Bank | 继承 |
| CLIP | 双向 InfoNCE | 继承 |
| wav2vec2 | Contrastive Loss = InfoNCE | 继承 |

### 预训练的实用价值

1. **InfoNCE 的奠基论文**：InfoNCE 成为对比学习的"标准损失" >10 万引用
2. **互信息最大化视角**：为无监督学习提供了信息论的理论基础
3. **跨模态通用框架**：同一架构（编码器 + 自回归 + InfoNCE）适用于语音/图像/文本/RL
4. **潜空间预测范式**："在特征空间中预测""在原始空间中预测"——影响了后续所有 SSL 方法
5. **思想遗产**：wav2vec2、BYOL、SimSiam 都可以追溯到 CPC 的框架

CPC 的 InfoNCE Loss 成为后续所有对比学习方法的基础，是自监督学习领域最具影响力的贡献之一。
