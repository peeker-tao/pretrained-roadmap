# Point-BERT

## 基本信息

- **论文**: [Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling](https://arxiv.org/abs/2111.14819)
- **作者**: Xumin Yu et al.
- **发表**: CVPR 2022

## 创新点

1. **3D BERT**: 将 BERT/MIM 范式引入 3D 点云
2. **掩码点云块 + 重建离散代码**: 类似 BEiT 的两阶段方法

## 核心原理

### 掩码点云建模

1. 将点云分割为块 (patch)
2. 随机掩码一部分块
3. 使用 Transformer 编码器预测被掩码块的离散代码
4. 离散代码通过预训练的点云 VQ-VAE 获得

## 预训练方法

### 核心思想：BERT 靠 MLM 学会了语言的语法——Point-BERT 把同样的思想搬到了 3D 点云：遮住一些点的"块"，让模型预测被遮住区域是什么形状

Point-BERT 是 MIM（Masked Image Modeling）和 BERT 范式在 3D 点云领域的迁移。核心挑战：**1）如何将不规则的点云分割成有意义的"块"？2）用什么作为重建目标（像素值 vs 离散token）？** Point-BERT 的方案：借鉴 BEiT 的两阶段方法——先用 dVAE 将点云块转换为离散 token，再训练 Transformer 预测被掩码块的 token。

> Point-BERT = 点云分块（Patchify）+ 随机掩码（Mask）+ dVAE tokenizer（离散化）+ Transformer 编码器 + 交叉熵重建。这是 3D 视觉从 NLP 范式受益的典型例子。

### 训练流水线（Step by Step）

#### Step 1 — 点云分块（Point Patchification）

点云是一个 $N \times 3$ 的矩阵（$N$ 个点的 XYZ 坐标）——没有规则的网格结构。Point-BERT 的分块策略：

```text
原始点云 (N=1024 points):
  点1: (x1, y1, z1)
  点2: (x2, y2, z2)
  ...
  点1024: (x1024, y1024, z1024)

分块（FPS + KNN）:
  1. 使用 FPS 采样 M 个中心点（如 64 个）
  2. 每个中心点取 K 个最近邻（如 32 个）
  3. → 64 个 patch，每个 patch 包含 32 个点（需用 DGCNN 编码为固定维度向量）
```

| 步骤 | 方法 | 输出 |
|------|------|------|
| 中心点采样 | FPS（最远点采样） | 64 个中心点 |
| 邻域分组 | KNN（K=32） | 64 × 32 个点 |
| Patch 编码 | 小型 PointNet / DGCNN | 64 × d 维向量 |

#### Step 2 — dVAE Tokenizer 预训练

在训练 Point-BERT 之前，需要先训练一个 dVAE tokenizer：

```text
Stage 0（预训练 dVAE）:
  点云 → 分块 → DGCNN Encoder → 离散 codebook → Decoder → 重建点云
  
  目标: 学习一个有意义的离散 token 空间
  Codebook 大小: 8192（类似 BERT 的词汇表）
```

这个 dVAE 将每个点云 patch 映射到离散的 token ID（1 到 8192）。训练好的 tokenizer 用**交叉熵损失**重建点云。

#### Step 3 — 掩码点云块（Masked Point Modeling）

有了 tokenizer 后，Point-BERT 的训练类似于 BEiT：

```text
预训练（Point-BERT）:
  输入点云 → 分块 → dVAE 生成所有 patch 的 token ID（标签）
  → 随机掩码 60% 的 patch
  → 可见 patch → 编码器（Point Transformer）
  → 被遮 patch 位置用 [MASK] token
  → 预测被遮 patch 的 token ID
  → 与真实 token ID 计算交叉熵
```

**掩码策略**：

| 参数 | Point-BERT | BEiT（图像） |
|------|----------|-----------|
| Patch 数 | 64 | 196 (14×14) |
| 掩码率 | 60% | 40-60% |
| 掩码方式 | 随机 patch | 随机 patch |
| 重建目标 | dVAE 离散 token | dVAE 离散 token |

#### Step 4 — 为什么用离散 token 而非连续坐标？

| 重建目标 | 优势 | 劣势 |
|---------|------|------|
| 原始 XYZ 坐标 | 直观 | 点云坐标是连续的——MSE 回归可能导致模糊的"平均点云" |
| **dVAE 离散 token** | **语义化训练目标，类似 BERT** | **需要额外预训练 tokenizer** |

> 离散 token 的预测是分类任务（8192 选 1）——比回归连续坐标的 MSE 更稳定、语义更丰富。这与 BEiT 选择 dVAE token 而非像素回归的理由一致。

#### Step 5 — 完整两阶段训练流程

```text
Stage 1: dVAE Tokenizer 预训练
  数据: ShapeNet（~50K 3D 模型）
  输入: 1024 点 × 3 (XYZ)
  分块: 64 patches × 32 points
  编码: DGCNN → 离散 latent
  Codebook: 8192
  解码: DGCNN → 重建 1024 点
  损失: Chamfer Distance（点云重建专用）

Stage 2: Point-BERT 预训练
  数据: ShapeNet（无标签）
  输入: 1024 点 × 3 (XYZ)
  分块 → dVAE tokenizer → 获得 64 个 token ID（标签）
  掩码: 随机 mask 60% patches
  Encoder: Point Transformer（12 层）
  预测: 被遮 patch 的 token ID
  损失: 交叉熵
```

### 详细训练配置

| 参数 | Stage 1 (dVAE) | Stage 2 (Point-BERT) |
|------|---------------|---------------------|
| 数据集 | ShapeNet | ShapeNet |
| 点云点数 | 1024 | 1024 |
| Patch 数 | 64 | 64 |
| 每个 Patch 点数 | 32 | 32 |
| Codebook 大小 | 8192 | 8192（复用） |
| 掩码率 | — | 60% |
| 优化器 | AdamW | AdamW |
| 学习率 | 1e-3 | 1e-3 |
| Batch Size | 64 | 64 |
| Epoch | 300 | 300 |

### Point-BERT vs 其他 3D 自监督方法

| 方法 | 范式 | 重建目标 | 需要 tokenizer？ |
|------|------|---------|----------------|
| PointContrast | 对比学习 | — | ✗ |
| **Point-BERT** | **MIM（掩码点云建模）** | **离散 token** | **✓（dVAE）** |
| Point-MAE | MIM | 连续坐标 | ✗ |

### 预训练的实用价值

1. **3D ⇔ NLP 范式的桥接**：将 BERT 的 MLM 范式引入不规则数据（点云）
2. **dVAE tokenizer for 3D**：展示了如何为点云学习有意义的离散"词汇"
3. **下游任务的双重受益**：在分类（ModelNet40）和分割（ShapeNetPart）上均有显著提升
4. **小数据 3D 任务的 starter**：在 ScanObjectNN 等小数据集上，Point-BERT 预训练 + 微调带来巨大提升
5. **推动了 3D MIM 线的后续工作**：Point-MAE、Point-M2AE、ACT 等
