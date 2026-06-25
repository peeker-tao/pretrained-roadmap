# BiT (Big Transfer)

## 基本信息

- **论文**: [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370)
- **作者**: Alexander Kolesnikov et al. (Google)
- **发表**: ECCV 2020

## 创新点

1. **大规模预训练 + 简单迁移**: 在 JFT-300M 上预训练，下游仅需简单微调
2. **Group Normalization + Weight Standardization**: 替代 BatchNorm，适合大 batch 训练
3. **BiT-HyperRule**: 系统化的超参数迁移策略

## 核心原理

### 预训练-微调策略

1. **大规模预训练**: JFT-300M（3 亿标注图像）上 ResNet 预训练
2. **BiT-L 微调**: 在下游任务上做简单微调，使用固定的超参数调度

### BiT-HyperRule

| 超参数 | 值 |
|-------|-----|
| 学习率 | 0.003 × (image_size/224)² |
| 训练步数 | 500 / 10000 / 20000 (按数据量) |
| 调度 | Cosine decay |

### 架构改进

- GN (Group Normalization) 替代 BN（适合大 batch、跨分辨率迁移）
- WS (Weight Standardization) 稳定训练

## 预训练方法

### 核心思想：如果你在足够大的数据集上预训练了一个足够好的模型，下游任务的微调应该"只需改最后几层"——不需要复杂的适应策略。BiT 证明了：大规模预训练 + 简单的 GroupNorm 架构 + 标准微调 = 超越所有复杂的迁移方法

BiT（Big Transfer）的核心哲学极其简单："预训练要尽可能大，迁移要尽可能简单"。它在 3 亿张标注图像（JFT-300M）上预训练 ResNet，然后用一套系统化的超参数规则（BiT-HyperRule）迁移到 20+ 个下游数据集——没有复杂的适应模块，没有多阶段策略，就是简单的微调。

> BiT = 大规模监督预训练（JFT-300M, 3 亿标注图）+ GN+WS（GroupNorm + Weight Standardization 替代 BN）+ BiT-HyperRule（标准化微调超参数）。在 20+ 下游任务上超越当时的所有复杂迁移方法。

### 训练流水线（Step by Step）

#### Step 1 — GN + WS 架构（替代 BatchNorm）

BiT 的首要创新是架构层面的——用 GN+WS 替换 BN：

**为什么替换 BN？**

批量归一化（BatchNorm）有两个致命缺陷：
1. **Batch 敏感**：小 batch 时 BN 统计不稳定 → 大模型训练不能用小 batch
2. **分辨率迁移困难**：BN 在训练分辨率（224²）上学习统计 → 迁移到不同分辨率（如 384²）时统计不匹配

**BiT 的解决方案：GN + WS**：

| 组件 | 作用 | 与 BN 的对比 |
|------|------|-----------|
| **GroupNorm (GN)** | 每组通道内归一化 | 不受 batch size 影响 |
| **Weight Standardization (WS)** | 标准化卷积权重 | 稳定训练，与 GN 互补 |

```text
BatchNorm:
  y = γ · (x - μ_batch) / σ_batch + β
  问题: μ_batch, σ_batch 依赖 batch

GroupNorm + Weight Standardization:
  GN: y = γ · (x - μ_group) / σ_group + β
      μ_group, σ_group 在同一图片的通道组内计算
      → 不依赖 batch!
  
  WS: W_norm = (W - μ_W) / σ_W
      → 权重标准化 → 训练更稳定
```

**GN+WS 的优势**：
- 可以大 batch（4096）训练，也可以小 batch（4）微调
- 从 224² 预训练 → 直接 384² 微调无问题
- 训练更稳定，收敛更快

#### Step 2 — 大规模监督预训练

```text
阶段 1: 大规模监督预训练

数据: JFT-300M
  规模: 3 亿标注图像
  类别: 18,291 个类（细粒度标注）
  举例: "美洲短毛猫, 室内, 沙发" 

模型: ResNet-152×4 (GN+WS 替换所有 BN)
  参数量: ~940M

训练:
  损失: Sigmoid 交叉熵（多标签，因为 JFT 有层级标签）
  分辨率: 224²
  优化器: SGD + Momentum(0.9)
  学习率: 0.1（cosine decay）
  Batch Size: 4096
  Epoch: 70（JFT 约 14 epoch）

预训练时长: ~1 周（128 TPUv3）
```

**JFT-300M 的标签特点**：

| 特性 | ImageNet | JFT-300M |
|------|---------|----------|
| 图片数 | 1.28M | 300M |
| 类别数 | 1000 | 18,291 |
| 标签方式 | 单标签 | **层级多标签** |
| 标注方法 | 众包 | **半自动 + 人工验证** |
| 损失函数 | Softmax CE | **Sigmoid CE（多标签）** |

#### Step 3 — BiT-HyperRule（标准化微调）

大规模预训练后，BiT 用一套固定的超参数规则进行下游迁移——消除下游任务上的超参数调优：

| 超参数 | BiT-HyperRule 公式 | 示例（224²） |
|--------|-------------------|------------|
| 学习率 | $0.003 \times (S/224)^2$ | 224² → 0.003, 384² → 0.009 |
| 训练步数 | 500 / 10K / 20K | 按数据集大小选择 |
| 学习率调度 | Cosine decay | — |
| MixUp | $\alpha = 0.1$ | 固定 |
| 权重衰减 | 0.0001 | 固定 |

**为什么 HyperRule 有效？**
- 预训练质量足够高 → 微调不需要复杂的搜索
- 基于分辨率调整学习率 → 不同任务无需从头调参
- 标准化的迁移流程 → 任何新任务都能快速部署

#### Step 4 — BiT-L 微调

微调阶段极其简单——没有 Adapter、没有 Prompt Tuning、没有多阶段：

```text
阶段 2: 简单微调

数据: 下游任务数据
模型: 预训练的 ResNet-152×4
修改: 替换分类头（18,291 → 下游类别数）

微调:
  所有层可训练（全微调）
  仅使用 HyperRule 默认值
  无额外技巧
```

#### Step 5 — 完整训练配置

| 参数 | BiT-S | BiT-M | BiT-L |
|------|-------|-------|-------|
| 预训练数据 | ImageNet-1K | ImageNet-21K | **JFT-300M** |
| 预训练类别 | 1000 | 21K | **18,291** |
| 模型 | R50×1 | R50×4 | **R152×4** |
| GN+WS | ✓ | ✓ | ✓ |
| 微调学习率 | HyperRule | HyperRule | HyperRule |
| 微调步数 | 500/10K/20K | 10K/20K | 10K/20K |

### BiT vs 标准预训练+微调

| 维度 | 标准预训练 | BiT |
|------|----------|-----|
| 归一化 | BN | **GN + WS** |
| 分辨率迁移 | 需调整 | **无痛切换** |
| 微调策略 | 手动调参 | **HyperRule 自动化** |
| 预训练数据 | ImageNet | **JFT-300M** |
| 下游性能 | 基线 | **20+ 数据集全面超越** |

### 预训练的实用价值

1. **大规模预训练 + 简单微调**：证明了"预训练越大，迁移越简单"的原则
2. **GN+WS 的推广**：解决了大模型训练中的 BN 瓶颈
3. **HyperRule 的方法论**：标准化的迁移超参数 → 工业级部署
4. **多标签预训练的优势**：JFT 的层级标签 → 下游细粒度分类更好
5. **对 ViT/KiT 的影响**：BiT 的简单迁移思想被 ViT 继承
