# ChemBERTa

## 基本信息

- **论文**: [ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction](https://arxiv.org/abs/2010.09885)
- **作者**: Seyone Chithrananda, Gabriel Grand, Bharath Ramsundar
- **发表**: NeurIPS 2020 (Machine Learning for Molecules Workshop)

## 创新点

1. **RoBERTa 用于化学**: 将 RoBERTa 的预训练方法应用于 SMILES 分子序列
2. **规模效应**: 验证了分子预训练的 Scaling Law（更多数据 → 更好的性质预测）
3. **开源 ChemBERTa**: 推动分子 ML 的民主化

## 核心原理

### SMILES Tokenization

- 基于 BPE（Byte-Pair Encoding）的分词器
- 自动学习化学子结构 token

### 预训练任务

- **MLM**: 掩码 SMILES token 预测
- 类似 BERT/RoBERTa 的训练流程

## 预训练方法

### 核心思想：SMILES 字符串是分子的"语言"——`CCO` 是乙醇，`CC(=O)OC1=CC=CC=C1C(=O)O` 是阿司匹林。用 RoBERTa 的 MLM 预训练处理 SMILES，模型自己就会学会"羧基总是和苯环在某个位置配对"之类的化学规则

ChemBERTa 是将现代 NLP 预训练（RoBERTa 级别）系统性地应用于分子表征学习的先驱工作。它的核心贡献是验证了：**大规模 MLM 预训练在化学领域同样有效——更多的分子数据 + 更大的模型 = 更好的分子性质预测**。

> ChemBERTa = SMILES tokenization（BPE 子词）+ RoBERTa 架构 + MLM 预训练 + 大规模分子数据集。它证明了化学信息学可以从 NLP 的"大数据 + 大模型"范式中受益。

### 训练流水线（Step by Step）

#### Step 1 — SMILES Tokenization

SMILES（Simplified Molecular Input Line Entry System）是分子的线性字符串表示：

```text
乙醇: CCO
苯: c1ccccc1
阿司匹林: CC(=O)OC1=CC=CC=C1C(=O)O
咖啡因: CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

**BPE（Byte-Pair Encoding）分词**：

```text
原始: CC(=O)OC1=CC=CC=C1C(=O)O

逐字符: [C] [C] [(] [=] [O] [)] [O] [C] [1] [=] [C] [C] [=] [C] [C] [=] [C] [1] [C] [(] [=] [O] [)] [O]

BPE 合并后:
  [CC] [=O] [OC] [1] [=] [CC] [=] [CC] [=] [C1] [C] [=O] [O]
  
词汇表大小: 52K (包含原子字符 + 高频子结构)
```

**BPE 的优势**：自动学习化学上合理的子结构 token——如 `C=O`（羰基）、`c1ccccc1`（苯环）——这些 token 在化学上是有意义的官能团。

#### Step 2 — RoBERTa 架构的 MLM 预训练

ChemBERTa 完全复用 RoBERTa 的训练配方：

```text
输入: SMILES 字符串 → BPE tokenizer → token 序列

MLM 预训练:
  1. 随机 mask 15% 的 SMILES token
  2. Transformer Encoder (12 层, 768 维)
  3. 预测被 mask 的 token
  4. 交叉熵损失

关键改进（相比 BERT）:
  - 动态掩码: 每个 epoch 重新生成 mask（而非固定）
  - 去除 NSP: 仅使用 MLM 任务
  - 更大 batch: 8K 序列
```

**分子 MLM 的特殊性**：

| 维度 | 自然语言 MLM | 分子 SMILES MLM |
|------|------------|---------------|
| 被 mask 的 token | `The [MASK] sat on the mat` | `C[MASK]OC1=[MASK]C=CC=C1` |
| 预测难度 | 中等（语法 + 语义） | **高（化学规则 + 3D 结构）** |
| 信息冗余 | 高 | **低（分子结构精确）** |
| 错误代价 | 低 | **高（一个原子错了 = 不同分子）** |

> 分子的 MLM 比语言更难——语言的同义词多，分子的"同义词"（异构体）化学性质完全不同。

#### Step 3 — 预训练数据与规模效应

ChemBERTa 的一个核心贡献是验证了**分子领域的 scaling law**：

| 版本 | 训练数据 | 参数量 | 预训练 PPL | 下游性能 |
|------|---------|--------|----------|---------|
| ChemBERTa-5M | 1M | 5M | 高 | 基线 |
| ChemBERTa-5M | 10M | 5M | 中 | +5% |
| ChemBERTa-2-46M | 77M | 46M | 低 | +15% |

**关键发现**：更多数据 > 更大模型（在 5M-46M 参数范围内）。77M 分子的 ChemBERTa-2 性能优于 10M 分子的版本。

#### Step 4 — 完整训练配置

| 参数 | ChemBERTa | ChemBERTa-2 |
|------|----------|------------|
| SMILES 数据 | PubChem (10M) | PubChem (77M) |
| 模型大小 | 5M (6 层, 512 维) | 46M (12 层, 768 维) |
| Tokenizer | BPE (52K vocab) | BPE (52K vocab) |
| MLM 掩码率 | 15% | 15% |
| 序列长度 | 512 | 512 |
| 优化器 | AdamW | AdamW |
| 学习率 | 1e-4 | 1e-4 |
| Batch Size | 8K | 8K |
| 总训练步数 | 1M | 2M |

#### Step 5 — 下游任务的微调

ChemBERTa 的预训练权重在下游分子性质预测任务上微调：

| 任务 | 数据 | 输入 | 预测目标 |
|------|------|------|---------|
| 水溶性 (ESOL) | ~1.1K | SMILES | logS 值（回归） |
| 脂溶性 (Lipophilicity) | ~4.2K | SMILES | logD 值（回归） |
| 毒性 (Tox21) | ~10K | SMILES | 毒性分类 |
| BBBP（血脑屏障） | ~2K | SMILES | 渗透性分类 |
| HIV 活性 | ~41K | SMILES | 活性分类 |

### ChemBERTa vs 传统分子指纹

| 维度 | 传统 ECFP/MACCS 指纹 | ChemBERTa |
|------|---------------------|----------|
| 表征方式 | 手工规则（子图哈希） | **从数据中学习** |
| 维度 | 固定（1024/2048 位） | 768 维（可压缩） |
| 领域知识 | 需要（化学专家设计） | **无需（自动学习）** |
| 上下文敏感 | ✗（固定编码） | **✓（上下文感知）** |
| 预训练收益 | 无 | **数据越多越好** |

### 预训练的实用价值

1. **分子 MLM 可行性的验证**：SMILES 可以被视作语言——MLM 预训练学到的表征优于手工分子指纹
2. **分子 scaling law 的证明**：更多 SMILES 数据 → 更好的下游性质预测
3. **BPE tokenizer for chemistry**：自动学习化学子结构 → 比固定字符 tokenizer 更好
4. **药物发现的降门槛**：开源预训练权重降低了 AI 分子性质预测的入门成本
5. **与 MolFormer 共同奠定了化学语言模型的方向**
