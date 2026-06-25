# DCLM (DataComp for Language Models)

## 基本信息

- **论文**: [DCLM: Data Curation for Language Models](https://arxiv.org/abs/2406.11794)
- **作者**: Jeffrey Li et al. (UW / AI2 / 多机构)
- **发表**: NeurIPS 2024

## 创新点

1. **语言模型的数据筛选基准**: 类似 DataComp，固定模型和训练，竞赛数据策略
2. **DCLM-baseline**: 高质量 3.8T token 数据集，7B 模型 MMLU 达 64%
3. **系统性的数据消融**: 分析不同数据策略的边际贡献

## 核心原理

### 数据筛选流程

1. **原始数据**: Common Crawl 快照
2. **去重**: URL + 内容两级去重
3. **质量过滤**: 使用 fastText 分类器 + 启发式规则
4. **重排序**: 基于模型困惑度等指标

### 关键发现

- 数据筛选策略对模型质量的影响 > 模型架构
- 简单的 fastText 分类器就非常有效
- "质量 > 数量"在 LLM 预训练中同样成立

## 预训练方法

### 核心思想：LLM 的瓶颈不再是模型架构（GPT 架构已经很成熟），而是训练数据的质量。DCLM 把 DataComp 的"以数据为中心"范式引入 LLM——固定 7B Transformer，让研究者竞争数据筛选策略。结论：**仅通过更好的数据筛选，7B 模型的 MMLU 就能从 28% 飙升到 64%**

DCLM（DataComp for Language Models）把 DataComp 的"数据竞赛"思想应用到语言模型。核心发现：LLM 的预训练数据中有大量"垃圾"——去除这些垃圾比增加模型参数更有效。

> DCLM = 固定 7B Transformer + 固定训练超参数 + Common Crawl 原始语料 + 数据筛选竞赛。baseline 数据集 DCLM-baseline（3.8T token）仅用 fastText 分类器就筛选出高质量语料，7B 模型达到 64% MMLU。

### 训练流水线（Step by Step）

#### Step 1 — DCLM 的核心设计（同 DataComp 范式）

```text
DCLM = "语言模型版本的 DataComp"

固定不变的:
  ✓ 模型架构: 7B Transformer (GPT-style)
  ✓ 训练流程: 138B token 或更多
  ✓ 训练超参数: 学习率、batch size 等
  ✓ 评估协议: 53 个下游任务

唯一可变的:
  ✎ 数据筛选策略: 从 Common Crawl 中选取训练数据
```

#### Step 2 — 原始数据源：Common Crawl

| 属性 | 值 |
|------|-----|
| 来源 | Common Crawl（网络爬虫） |
| 总文本量 | 数百 TB |
| 质量 | 极不均匀（新闻、论坛、垃圾、SEO 内容） |
| 语言 | 以英文为主，含多语言 |

**Common Crawl 中的数据问题**：

| 问题类型 | 示例 | 占比（估计） |
|---------|------|---------|
| SEO 垃圾 | "最好的手机 买手机 便宜手机 手机推荐..." | ~10-20% |
| 模板文本 | "本网站使用 Cookie，点击同意..." | ~5-10% |
| 代码片段 | 无格式的 minified JS/CSS | ~10% |
| 低质量翻译 | 机器翻译无校对 | ~5% |
| 重复 | 同一内容被多网站复制 | ~10-20% |
| **高质量** | 维基百科、论文、书籍 | **~5-10%** |

> 未经筛选的 Common Crawl 中，**只有 5-10% 的内容是高质量的**——其余都是噪音。DCLM 的目标就是找到这 5-10% 的黄金。

#### Step 3 — DCLM 的数据筛选流水线

```text
Common Crawl (原始)
  ↓
1. 文本提取 (Trafilatura / resiliparse)
  去除 HTML 标记、导航栏、广告
  ↓
2. URL 去重 + 文档级去重 (MinHash)
  去除完全相同和近重复文档
  ↓
3. 语言检测 (fastText)
  仅保留英文
  ↓
4. 质量过滤 (fastText 分类器)
  训练一个 fastText 模型区分"高质量"和"低质量"文本
  训练信号: 维基百科 vs 随机 Common Crawl
  ↓
5. 启发式规则
  去除太短(<100 词)、太长(>100K 词)、无意义重复
  ↓
6. 模型困惑度过滤
  使用轻量 LM 计算困惑度 → 去除高困惑度文本
  ↓
DCLM-baseline (~3.8T token)
```

**步骤 4 的 fastText 质量分类器**（核心筛选器）：

```text
正例（高质量）: 维基百科 + OpenWebText2 + 书籍
负例（低质量）: 随机 Common Crawl 样本

训练: fastText 二分类器
  → 对每个文档预测 P(高质量|文本)
  → 保留 P > 阈值的文档

fastText 的优势:
  - 极快（可处理数百 TB 文本）
  - 线性复杂度（vs Transformer 的 O(N²)）
  - 效果惊人地好
```

#### Step 4 — 固定 Transformer 预训练

| 参数 | DCLM-7B | 说明 |
|------|----------|------|
| 模型架构 | GPT-style Decoder-only | 标准 Transformer |
| 参数量 | 7B | — |
| 层数 | 32 | — |
| 隐藏维度 | 4096 | — |
| 上下文长度 | 2048 | — |
| 训练 Token | 138B (baseline) / 2.5T (扩展) | — |
| 优化器 | AdamW | — |
| 学习率 | 3e-4 | 余弦衰减 |
| Batch Size | 1024 sequences | — |
| 评估 | 53 个下游任务 | 包括 MMLU, HellaSwag 等 |

#### Step 5 — 关键发现

| 发现 | 细节 | 数值影响 |
|------|------|---------|
| **fastText 过滤极有效** | 简单的线性分类器 → 巨大提升 | MMLU: 28% → 58% |
| **去重至关重要** | 不去重 → 训练不稳定 → 性能下降 | +5-8% MMLU |
| **模型困惑度过滤有效** | 高困惑度文本 → 难以学习 → 去除 | +3-5% |
| **合成数据有潜力** | 用大模型重写低质量文本 | +2-4% |
| **"质量 > 数量"** | 1T 高质量 token > 5T 低质量 token | — |

### DCLM-baseline 数据集

| 属性 | 值 |
|------|-----|
| 原始语料 | Common Crawl (2023-2024) |
| 筛选方法 | fastText 质量分类 + 去重 + 规则 |
| 最终规模 | 3.8T token |
| 训练模型 | 7B Transformer |
| MMLU（7B 模型） | 64% |
| 对比：LLaMA-7B MMLU | 35% |
| 对比：Mistral-7B MMLU | 62% |

> DCLM-baseline 的 7B 模型在 MMLU 上达到 64%，接近 Mistral-7B 的水平——仅用了简单数据筛选，没有复杂的训练技巧。

### DCLM 的预训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 原始数据 | Common Crawl | 网络文本 |
| 筛选方法 | fastText + 去重 + 规则 | — |
| 固定模型 | 7B Transformer | — |
| 训练 token | 138B (baseline) | — |
| 评估任务 | 53 | 综合评测 |

### 预训练的实用价值

1. **LLM 数据质量的标准化**：提供了筛选高质量文本的完整方法
2. **"以数据为中心"的 LLM 范式**：固定模型 → 改进数据 → 比改进模型更有效
3. **fastText 的工业实用性**：简单的线性分类器就能大幅提升数据质量
4. **开源高质量数据集**：DCLM-baseline (3.8T token) 公开可用
5. **数据筛选的 ROI**：在数据筛选上投入算力，比增加模型参数量更划算
