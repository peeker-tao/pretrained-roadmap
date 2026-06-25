# DNABERT

## 基本信息

- **论文**: [DNABERT: Pre-trained Bidirectional Encoder Representations from Transformers for DNA](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680)
- **作者**: Yanrong Ji et al.
- **发表**: Bioinformatics, 2021

## 创新点

1. **DNA 序列的 BERT**: 将 DNA 序列视为"基因组语言"进行 MLM 预训练
2. **k-mer Tokenization**: 使用重叠的 k-mer 对 DNA 序列进行分词
3. **基因组功能预测**: 启动子、剪接位点、转录因子结合位点预测

## 核心原理

### k-mer Tokenization

DNA 序列 → k-mer 重叠分词 → token 序列

例如 k=6: `ATCGATCG` → `[ATCGA, TCGAT, CGATC, GATCG]`

### 预训练任务

- **MLM**: 随机掩码 k-mer token，根据上下文预测
- 捕获 DNA 序列的上下文依赖性

## 预训练方法

### 核心思想：基因组序列是一本用 4 个字母（A、T、C、G）写的"天书"——每个基因、启动子、增强子都是"功能词"。DNABERT 用 BERT 的 MLM 预训练来"阅读"这本天书，目标是理解非编码区的"语法规则"

DNABERT 开创了"基因组语言模型"的范式。它将人类基因组视为一种"语言"：A/T/C/G 是字母表，k-mer 是"单词"，基因调控元件（启动子、增强子、剪接位点）是"功能关键词"。MLM 预训练让模型学习到这些元件的语法上下文。

> DNABERT = k-mer 重叠分词（DNA 专用）+ BERT-base 架构 + MLM 预训练（遮住 k-mer，用上下文推测）。在启动子预测、剪接位点识别等任务上显著超越了传统方法。

### 训练流水线（Step by Step）

#### Step 1 — k-mer Tokenization：DNA 的"分词"

DNA 只有 4 个字母（A/T/C/G），但 BERT 不能以单个核苷酸为 token——那样太细粒度、语义太弱。DNABERT 使用**重叠的 k-mer** 作为 token：

```text
DNA 序列: "ATCGATCGTAGCT"

k=3 (3-mer):
  [ATC] [TCG] [CGA] [GAT] [ATC] [TCG] [CGT] [GTA] [TAG] [AGC] [GCT]

k=4 (4-mer):
  [ATCG] [TCGA] [CGAT] [GATC] [ATCG] [TCGT] [CGTA] [GTAG] [TAGC] [AGCT]

k=5 (5-mer):
  [ATCGATCG] [TCGATCGT] [CGATCGTA] ... 
```

| k 值 | 总可能 token 数 | 滑动窗口重叠 |
|------|--------------|-----------|
| k=1 | 4 | — |
| **k=3** | **64** | 2 bp 重叠 |
| k=4 | 256 | 3 bp 重叠 |
| k=6 | 4096 | 5 bp 重叠 |
| k=8 | 65536 | 7 bp 重叠 |

**k=3-6 是最常用的范围**——太小则 token 信息量不足，太大则词汇表膨胀且丢失位置精度。

**为什么用重叠 k-mer？**
- 非重叠：`ATCGATCG` → `[ATCG] [ATCG]`（失去"CGAT"和"GATC"的信息）
- 重叠：`ATCGATCG` → `[ATCG] [TCGA] [CGAT] [GATC] [ATCG]`（保留所有子序列）

> 重叠 k-mer 就像中文分词时保留所有可能的切分方式——不丢失任何连续子序列的信息。

#### Step 2 — MLM 预训练

```text
输入: k-mer 序列 [ATC] [TCG] [CGA] [GAT] [ATC] [TCG] [CGT] ...

1. 随机选择 15% 的 k-mer 进行 mask:
   - 80% → [MASK]
   - 10% → 随机替换为其他 k-mer
   - 10% → 保持不变

2. BERT-base Encoder (12 层, 768 维)
   → 每个 k-mer 位置的隐藏状态

3. 预测被 [MASK] 位置的原始 k-mer
   → 交叉熵损失（分类：256 选 1 for k=4）
```

**DNA MLM 与 NLP MLM 的区别**：

| 维度 | NLP BERT | DNABERT |
|------|---------|---------|
| Token 数（词汇量） | ~30K | **256（k=4）或 4096（k=6）** |
| 序列长度 | ~512 | 512（k-mer 序列） |
| 语义层次 | 词→短语→句子 | **k-mer→基序→调控元件** |
| 训练数据 | 维基+图书 | **人类基因组（GRCh38）** |
| 预训练 epoch | ~40 | **~100（基因组更小）** |

#### Step 3 — 基因组数据的预处理

人类参考基因组（GRCh38）包含 ~3.2G 碱基对（bp）——但其中大部分是重复序列和非编码区。

```text
GRCh38 基因组:
  ├ 编码区（~1.5%）：实际决定蛋白质的基因
  ├ 非编码调控区（~5-10%）：启动子、增强子、沉默子
  └ 重复序列/转座子（~50%+）：进化残留
```

DNABERT 的训练数据：
- 从全基因组中随机采样 512-mer 长度的片段
- 排除高度重复区域（如着丝粒/端粒）
- 正负链都采样（DNA 是双链的）

#### Step 4 — 完整训练配置

| 参数 | DNABERT | 说明 |
|------|---------|------|
| 基因组数据 | 人类参考基因组 (GRCh38) | ~3.2G bp |
| Tokenization | 重叠 k-mer (k=3-6) | 滑动窗口 |
| 模型架构 | BERT-base | 12 层, 768 维, 110M 参数 |
| MLM 掩码率 | 15% | BERT 标准 |
| 序列长度 | 512 k-mer tokens | 对应 ~520 bp DNA |
| 优化器 | AdamW | — |
| 学习率 | 5e-5 | — |
| Batch Size | 256 | — |
| Epoch | 100 | 基因组较小，多轮 epoch |

#### Step 5 — 下游基因组任务的微调

| 任务 | 数据 | 预测目标 | DNABERT 提升 |
|------|------|---------|------------|
| 启动子预测 | ~2K 序列 | 是否包含启动子 | +5-8% vs 传统方法 |
| 剪接位点识别 | ~10K 序列 | 供体/受体剪接位点 | +3-5% |
| 转录因子结合位点 | ChIP-seq (ENCODE) | TFBS 预测 | +8-12% |
| 染色质状态预测 | 多个细胞系 | 开放/关闭 | +5-7% |
| 增强子-启动子交互 | Hi-C 数据 | 是否交互 | +10-15% |

### DNABERT vs 传统基因组方法

| 维度 | 传统方法 (PWM/MEME) | DNABERT |
|------|-------------------|---------|
| 方法 | 位置权重矩阵（统计频率） | **上下文感知 Transformer** |
| 上下文 | 无（独立建模每个位置） | **双向上下文（~512 bp）** |
| 特征 | 手工设计（Gibbs sampling 等） | **自动学习** |
| 非线性 | 无（线性相加） | **深度非线性** |
| 精度 | 基线 | **+5-15%** |

### 预训练的实用价值

1. **基因组语言模型的开山之作**：证明了"DNA 序列 → MLM → 调控元件理解"的可行性
2. **非编码区的理解**：机器学习首次大规模地解读人类基因组的"暗物质"（非编码区）
3. **调控元件的功能预测**：突变 → 改变 k-mer 语境 → 预测致病性 → 精准医学
4. **跨物种迁移**：人类基因组预训练 → 微调到小鼠/斑马鱼 → 跨物种的调控元件识别
5. **基因组 NLP 的方向**：启发了 DNABERT-2（更长的上下文、更高效的 tokenizer）、Nucleotide Transformer 等
