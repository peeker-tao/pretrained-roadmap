# 3DETR

## 基本信息

- **论文**: [An End-to-End Transformer Model for 3D Object Detection](https://arxiv.org/abs/2109.08141)
- **作者**: Ishan Misra et al. (Meta)
- **发表**: ICCV 2021

## 创新点

1. **端到端 3D 检测 Transformer**: 无需 3D 特定的归纳偏置
2. **点云编码 → 集合预测**: 类似 DETR 的 3D 泛化
3. **预训练友好**: Transformer 架构天然适合自监督预训练

## 核心原理

### 架构

1. **点云编码器**: 将点云转换为 token 序列
2. **Transformer 编码器-解码器**: 标准 Transformer
3. **集合预测头**: 直接输出 3D 边界框集合

### 与 DETR 的对比

| 特征 | DETR (2D) | 3DETR |
|------|----------|-------|
| 输入 | 图像 patch | 点云 patch |
| 输出 | 2D 边界框 | 3D 边界框 |
| 查询 | Object Queries | 3D Object Queries |

## 预训练方法

### 核心思想：DETR 证明了 Transformer 可以直接输出 2D 检测框——3DETR 证明同样的思路在 3D 场景也成立，而且 Transformer 架构让自监督预训练变得天然可接入

3DETR 的核心贡献是将 DETR（DEtection TRansformer）的端到端检测范式从 2D 图像迁移到 3D 点云。传统 3D 检测器（如 VoteNet）充满 3D 特定的归纳偏置（投票机制、球查询等），而 3DETR 用标准 Transformer 替代了所有这些手工设计——这让预训练变得更容易接入。

> 3DETR = 点云分块编码 + Transformer Encoder-Decoder + 3D 集合预测头。它不依赖任何 3D 特定的操作（如球查询、FPN），整个检测流程是一个纯净的 Transformer 管道。

### 训练流水线（Step by Step）

#### Step 1 — 点云 Tokenization

将不规则的点云转换为 Transformer 可以处理的 token 序列：

```text
原始点云 (N points, XYZ + RGB):
  → 最远点采样 (FPS): 选取 M 个中心点
  → KNN 分组: 每个中心点取 K 个邻居
  → 小型 MLP: 将每组点编码为 d 维向量
  → Token 序列: M × d
```

| 步骤 | 操作 | 输出 |
|------|------|------|
| 1. 采样中心点 | FPS | M 个点（如 1024） |
| 2. K 近邻分组 | KNN, K=32 | M 组，每组 32 点 |
| 3. 局部编码 | MLP（PointNet 风格） | M × d（d=256） |
| 4. 位置编码 | 中心点 XYZ → 位置嵌入 | M × d |

#### Step 2 — Transformer Encoder

标准 Transformer 编码器处理点云 token（与 NLP 完全相同）：

```text
Token 序列 [M × d] + 位置嵌入
  → Multi-Head Self-Attention
  → FFN
  → × N_enc 层
  → 增强的点云 Token
```

**与 CNN 3D 检测器的关键区别**：
- CNN（如 VoteNet）：层级式下采样 + 球查询 → 需要 3D 特定的操作
- 3DETR：所有 token 通过自注意力全局交互 → 标准 Transformer，无需 3D 专用操作

#### Step 3 — Transformer Decoder with Object Queries

类似于 DETR，3DETR 使用一组可学习的 **Object Queries**：

```text
Object Queries [Q × d] (如 128 个，随机初始化)
  → Cross-Attention with Encoder 输出
  → Self-Attention between queries（防止重复检测）
  → FFN
  → × N_dec 层
  → 每个 Query 输出一个 3D 检测结果
```

**Query 的语义**：每个 Object Query 学会关注点云中的不同空间位置——训练后，某些 Query 倾向于检测椅子，另一些倾向于检测桌子。

#### Step 4 — 3D 集合预测头

每个 Object Query 输出一个 3D 边界框：

```text
每个 Query 的输出:
  → 分类头: 类别概率 (C 类 + ∅ 背景)
  → 框头: (x, y, z, w, h, d, θ) — 3D 中心 + 尺寸 + 朝向
```

**匈牙利匹配**：预测框与真实框之间的最优二分图匹配（与 DETR 相同），确保每个真实框只被一个预测负责。

#### Step 5 — 监督预训练 + 自监督预训练

**监督预训练**（标准方式）：

| 参数 | 3DETR | 说明 |
|------|-------|------|
| 数据集 | ScanNet / SUN RGB-D | 室内 3D 场景 + 3D 标注框 |
| 损失 | 分类损失 + 框回归（L1 + GIoU） | 同 DETR |
| 优化器 | AdamW | — |
| 学习率 | 1e-4 | — |
| Batch Size | 8-16 | 全场景点云内存受限 |
| Epoch | 600-1000 | 3D 检测训练慢 |

**自监督预训练**（结合 PointContrast）：

| 阶段 | 方法 | 数据 | 目标 |
|------|------|------|------|
| Stage 1 | PointContrast 对比学习 | 无标签 3D 场景 | 点云表征 |
| Stage 2 | 监督微调 | 有标签 3D 检测数据 | 3D 检测 |

> Transformer 架构使 3DETR 自然兼容自监督预训练——预训练学到的 token 表征直接可用于下游的 3D 检测微调。

### 3DETR vs VoteNet（Traditional 3D Detector）

| 维度 | VoteNet | 3DETR |
|------|---------|-------|
| 核心操作 | 球查询 + 投票 + 聚合 | **标准 Self-Attention** |
| 3D 专用设计 | 多（投票、聚类、球查询） | **无（纯 Transformer）** |
| 预训练兼容性 | 低（架构不标准） | **高（标准 Transformer）** |
| 检测精度 (ScanNet) | 58.6 mAP | 62.1 mAP |
| 架构复杂度 | 高 | 低（模块更少） |

### 详细训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 点云采样 | 20K-40K 点/场景 | 全场景 |
| Token 数 | 2048 | FPS 采样 |
| Encoder 层数 | 6 | — |
| Decoder 层数 | 8 | — |
| Object Queries | 128 | 每场景最多检测数 |
| 损失 | CE + L1 + GIoU | — |
| 优化器 | AdamW | lr=1e-4 |
| Epoch | 1080 | ScanNet 上 |

### 预训练的实用价值

1. **Transformer 检测范式在 3D 的验证**：证明了 DETR 类端到端检测器在 3D 领域同样有效
2. **架构简化**：消除了 3D 特定的手工设计（投票、球查询）→ 代码更简单、更易维护
3. **预训练友好**：标准 Transformer 使自监督预训练可以直接接入
4. **为 3D 基础模型铺路**：证明了 Transformer 可以处理多模态 3D 任务（检测、分割、分类）
5. **跨模态一致性**：2D DETR 和 3D 3DETR 共享相似的 Transformer 设计 → 技术可迁移
