# Florence

## 基本信息

- **论文**: [Florence: A New Foundation Model for Computer Vision](https://arxiv.org/abs/2111.11432)
- **作者**: Lu Yuan et al. (Microsoft)
- **发表**: arXiv, 2021

## 创新点

1. **统一视觉基础模型**: 一个模型覆盖分类、检测、分割、检索、描述等任务
2. **图文对比 + 适配器**: 预训练后通过轻量级适配器迁移到不同任务
3. **9 亿图文对训练**: 大规模弱监督预训练

## 核心原理

### 双塔对比预训练

- 类似 CLIP 的图文对比学习
- 使用 Swin Transformer 作为视觉编码器

### 任务适配

- 预训练 → 冻结骨干 → 添加轻量任务适配头
- 支持 8+ 种视觉任务

## 预训练方法

### 核心思想：CLIP 证明了"图文对比学习"在零样本分类上很强，但 Florence 问了一个关键问题——预训练后，能不能不微调整个模型，而是"冻结骨干 + 添加小适配器"来支持分类、检测、分割、检索等所有视觉任务？答案是：可以，而且效果超过专用模型

Florence（FLOrence-Revival — 佛罗伦萨复兴）是微软提出的大规模视觉基础模型。它的核心设计是**预训练-适配**分离——在 9 亿图文对上用对比学习预训练一个 Swin Transformer，然后针对不同下游任务添加轻量适配器（不更新骨干）。

> Florence = Swin-H 视觉编码器 + CLIP 风格图文对比预训练（FLOD-900M, 9 亿图）+ 冻结骨干 + 轻量任务适配头。覆盖分类、检测、分割、检索、描述等 8+ 任务。

### 训练流水线（Step by Step）

#### Step 1 — 大规模图文对比预训练

```text
阶段 1: 图文对比预训练

数据: FLOD-900M (Florence Large-scale Object Detection)
  规模: 9 亿图文对
  来源: 互联网爬取（图文配对）
  质量: 中等（ALT text 质量不一）

视觉编码器: Swin-H / CoSwin-H
  参数量: ~658M
  类型: 层级化 Transformer

文本编码器: 12 层 Transformer
  参数量: ~200M

训练:
  损失: InfoNCE 对比损失（双向）
    L = L_I→T + L_T→I
  温度 τ: 可学习
  Batch Size: 32,768
  优化器: AdamW
  学习率: 5e-4（cosine decay）
  GPU: 512 × A100 (80G)
  训练时间: ~10 天
```

| 参数 | Florence | CLIP | 对比 |
|------|---------|------|------|
| 训练数据 | FLOD-900M (9 亿) | WIT-400M (4 亿) | 更大 |
| 视觉编码器 | Swin-H (层级) | ViT-L (各向同性) | 更适合密集预测 |
| 文本编码器 | 12 层 Transformer | 12 层 Transformer | 相同 |
| 损失 | 双向 InfoNCE | 双向 InfoNCE | 相同 |
| 训练规模 | 512 A100 | 592 V100 | 更大 |

#### Step 2 — Swin Transformer 层级设计（关键选择）

为什么 Florence 用 Swin 而非 ViT？

| 维度 | ViT | Swin Transformer | 影响 |
|------|-----|-----------------|------|
| Patch 粒度 | 固定 16×16 | **逐步合并（4×→8×→16×→32×）** | 多尺度特征 |
| Attention | 全局 | **窗口内局部** | 适合密集预测 |
| 特征金字塔 | 无 | **天然多尺度** | 检测/分割友好 |

```text
Swin-H 的层级特征:

Stage 1: H/4 × W/4 × 192   (高分辨率 → 适合检测)
Stage 2: H/8 × W/8 × 384
Stage 3: H/16 × W/16 × 768
Stage 4: H/32 × W/32 × 1536  (低分辨率 → 适合分类)
```

> 这个多尺度设计让 Florence 对检测和分割任务天然友好——不需要 FPN（Feature Pyramid Network）就能获得多尺度特征。

#### Step 3 — 任务适配器（冻结骨干 + 轻量适配）

预训练后，针对不同任务添加不同的适配头：

```text
预训练骨干 (Swin-H)
  ↓ [冻结/微调? 任务决定]
适配器:

分类:
  FC 层 → 类别数

检测 (DETR 风格):
  查询向量 → Cross-Attention → 类别/框

分割 (Mask2Former 风格):
  像素级查询 → 逐像素分类

检索:
  跨模态投影 → 余弦距离

VQA:
  Cross-Modal Adapter → 答案分类

图像描述:
  Cross-Attention Decoder → 文本生成
```

| 任务 | 适配器类型 | 骨干 |
|------|---------|------|
| 分类 | Linear Probe (FC) | 冻结 |
| 检测 | DETR Adapter | 冻结/微调 |
| 分割 | Mask2Former Adapter | 冻结/微调 |
| 检索 | 余弦距离 | 冻结 |
| VQA | Cross-Modal Adapter | 微调 |
| 描述 | Transformer Decoder | 微调 |

#### Step 4 — Florence 的"文艺复兴"命名

Florence 这个名字暗示了"CV 的文艺复兴"——一个模型统治所有视觉任务：

| "复兴前" | "Florence 复兴后" |
|---------|---------------|
| 每个任务一个专用模型 | **一个骨干 + 不同适配器** |
| 分类用 ResNet, 检测用 FPN, 分割用 U-Net | **统一用 Swin + 适配器** |
| 每个任务从头训练 | **预训练 → 冻结 → 适配** |
| 数据壁垒 | **大规模预训练 + 共享** |

#### Step 5 — 完整训练配置

| 参数 | Florence | 说明 |
|------|---------|------|
| 预训练数据 | FLOD-900M | 9 亿图文对 |
| 视觉编码器 | Swin-H (658M) | — |
| 文本编码器 | 12 层 Transformer (~200M) | — |
| 总参数 | ~893M | — |
| 损失 | 双向 InfoNCE | — |
| Batch Size | 32,768 | — |
| 优化器 | AdamW | lr=5e-4 |
| 训练 GPU | 512 × A100 (80G) | ~10 天 |
| 适配器训练 | 任务特定 | 轻量 |

### Florence 预训练的实用价值

1. **视觉基础模型的"操作系统"范式**：一个骨干 = 所有视觉任务的"内核"
2. **冻骨+适配 = 低门槛**：下游任务只需训练适配器（<1% 参数）
3. **Swin 对密集预测的优势**：层级设计 → 检测/分割无痛迁移
4. **9 亿图文对的验证**：数据规模扩展到 9 亿后对比学习仍有正向收益
5. **从业界的验证**：微软内部多个产品使用 Florence 作为视觉骨干
