# FourCastNet

## 基本信息

- **论文**: [FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators](https://arxiv.org/abs/2202.11214)
- **作者**: Jaideep Pathak et al. (NVIDIA / Caltech / LBNL)
- **发表**: arXiv, 2022

## 创新点

1. **傅里叶神经算子 (AFNO)**: 在频域中建模球面天气动力学
2. **高分辨率全球预报**: 25km 分辨率，比传统 NWP 快 45000 倍
3. **Transformer 替代方案**: AFNO 比标准 Transformer 更高效

## 核心原理

### AFNO (Adaptive Fourier Neural Operator)

1. 空间域 → 傅里叶变换 → 频域
2. 在频域进行自适应线性变换（类似注意力）
3. 频域 → 逆傅里叶变换 → 空间域

### 与 Transformer 的对比

| 操作 | Transformer | AFNO |
|------|------------|------|
| 混合机制 | 自注意力 | 频域卷积 |
| 复杂度 | $O(N^2)$ | $O(N \\log N)$ |
| 归纳偏置 | 弱 | 强（谱域） |

## 预训练方法

### 核心思想：天气预报本质上是"球面图像到球面图像的转换"——输入当前时刻的全球气象场，输出未来时刻的全球气象场。传统数值预报用物理方程在超算上"算"出未来，FourCastNet 改用傅里叶神经算子在 GPU 上"学"出未来

FourCastNet 是 AI 天气预报的先驱之一。它的核心创新是 **AFNO（Adaptive Fourier Neural Operator）**——一种在频域中操作的自适应机制，结合了傅里叶变换的全局建模能力和神经网络的自适应学习能力。

> FourCastNet = Vision Transformer backbone + AFNO（替换标准 Self-Attention，在频域中进行信息混合）+ 全球气象监督预训练（ERA5）。25km 分辨率、预报 7 天、推理时间 <2 秒（vs 传统方法数小时）。

### 训练流水线（Step by Step）

#### Step 1 — 全球气象场的 Patch 化

将全球气象场转化为 Vision Transformer 可处理的 patch：

```text
输入: 20 个气象场 × 721 × 1440 (纬度×经度)
  - 包括: 温度、气压、风速(U/V)、比湿、位势高度...

Patch Tokenization:
  Patch 大小: 16×16
  → 45 × 90 = 4050 个 patch / 场
  → 20 个场 → 每个 patch 是 20 维

最终 Token 序列: 4050 个 token, 每个 d 维
```

| 维度 | 说明 |
|------|------|
| 纬度（721） | 0.25° 分辨率, 90°N – 90°S |
| 经度（1440） | 0.25° 分辨率, 0° – 360° |
| 场数（20） | 温度、风速、气压、湿度等 |
| 表面/层 | 多个大气层次（地表、850hPa、500hPa 等） |

#### Step 2 — AFNO：频域中的"自适应注意力"

这是 FourCastNet 最核心的技术创新。**AFNO（Adaptive Fourier Neural Operator）** 替换了标准 Transformer 的自注意力层：

**标准 Self-Attention 的瓶颈**：

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V, \quad O(N^2) \text{ 复杂度}$$

全球气象场的 token 数 $N=4050$，复杂度尚可接受。但问题是：Self-Attention 在空间上是"各向同性"的——它不对球面几何有任何偏好。

**AFNO 的做法**：在频域中进行信息混合：

```text
Step 1: 空间域 → FFT → 频域
  X_token (4050 × d) → FFT → X_freq (4050 × d)

Step 2: 在频域中进行自适应混合
  对每个频率成分 k:
    X_freq[k] = W_k · X_freq[k]  (逐频率的可学习线性变换)
    + 非线性（GELU）
    + 稀疏化（丢弃弱频成分）

Step 3: 频域 → IFFT → 空间域
  X_freq → IFFT → X_spatial（增强的 token）
```

**AFNO 的优势**：

| 维度 | Self-Attention | AFNO |
|------|---------------|------|
| 复杂度 | $O(N^2)$ | **$O(N \log N)$** |
| 全局感受野 | ✓ | **✓（傅里叶全局）** |
| 归纳偏置 | 弱 | **强（频域稀疏性）** |
| 球面几何 | 无偏好 | **自然（频谱反映大气波）** |
| 可解释性 | 低 | **高（频率成分 = 大气尺度的波动）** |

> AFNO 特别适合天气预报，因为大气动力学天然地用频谱描述——罗斯贝波（大尺度）和重力波（小尺度）在频域中自然分离。

#### Step 3 — 监督预训练（ERA5）

FourCastNet 在 ERA5 再分析数据上进行监督学习：

```text
输入: X(t) — 当前时刻的 20 个全球气象场
目标: X(t+Δt) — Δt 后的气象场
  Δt = 6h, 12h, ..., 168h (7 天)

训练: 最小化 MSE(X_pred, X_ERA5)
```

**ERA5 数据集**：

| 属性 | 值 |
|------|-----|
| 机构 | ECMWF（欧洲中期天气预报中心） |
| 时间范围 | 1979–现在 |
| 空间分辨率 | 0.25° (~25km) |
| 时间分辨率 | 1 小时 |
| 变量 | 67 个气象变量（使用 20 个） |
| 训练集 | 1979-2015 |
| 验证集 | 2016-2017 |
| 测试集 | 2018 |

#### Step 4 — 自回归预报

训练后，FourCastNet 通过自回归方式进行长期预报：

```text
X(0) → AFNO → X(6h)
X(6h) → AFNO → X(12h)
X(12h) → AFNO → X(18h)
...
X(t) → AFNO → X(t+6h)

预报 7 天 = 28 步自回归
预报 14 天 = 56 步自回归
```

**自回归的误差累积问题**：

每步都有小误差，28 步后可能累积。FourCastNet 的处理：
- 训练时使用多步损失（不仅优化 6h 误差，也优化 12h、24h...误差）
- 随机打乱预报长度 → 模型学会"不看历史，只根据输入预报任意时长"

#### Step 5 — 完整训练配置

| 参数 | FourCastNet | 说明 |
|------|-----------|------|
| 训练数据 | ERA5（1979-2015） | ~40 年全球数据 |
| 空间分辨率 | 0.25° (721×1440) | 25km |
| 输入变量 | 20 | 表面 + 多层大气 |
| Patch 大小 | 16×16 | — |
| AFNO 层数 | 12 | 每层含 FFT + 自适应混合 |
| 隐藏维度 | 768 | — |
| 混合块数 | 8 | 频域混合的通道数 |
| 稀疏阈值 | 由权重幅值决定 | 自适应丢弃弱频率 |
| 损失函数 | MSE | — |
| 优化器 | AdamW | — |
| 学习率 | 5e-4 | 余弦衰减 |
| Batch Size | 64 | — |
| 训练时间 | ~1 周（16×A100） | — |

### FourCastNet vs 传统 NWP

| 维度 | ECMWF IFS（NWP） | FourCastNet |
|------|-----------------|-------------|
| 方法 | 物理方程（Navier-Stokes + 热力学） | **AFNO（频域学习）** |
| 硬件 | 超算（数万 CPU 核） | **单 GPU** |
| 推理时间 | ~1-3 小时 | **<2 秒** |
| 精度（5 天 Z500） | SOTA | 接近 SOTA |
| 物理可解释性 | 高（方程已知） | 中（频域可解释） |
| 集合预报 | 物理扰动 | 需要额外处理 |

### 预训练的实用价值

1. **AI 天气预报的加速器**：从"小时级"到"秒级" → 实时预警成为可能
2. **频域方法的验证**：AFNO 证明了傅里叶变换是处理全球气象数据的优选方案
3. **低成本的集合预报**：单 GPU 推理 → 可以运行 100 个扰动版本 → 概率预报
4. **与 GraphCast/Pangu 的并列**：共同开启了 AI 天气的革命
5. **与 PINN 的互补**：FourCastNet = 数据驱动，PINN = 物理驱动 → 融合 = 最佳方案
