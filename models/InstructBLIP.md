# InstructBLIP

## 基本信息

- **论文**: [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://arxiv.org/abs/2305.06500)
- **作者**: Wenliang Dai et al. (Salesforce / 多伦多大学)
- **发表**: NeurIPS 2023

## 创新点

1. **指令感知 Q-Former**: 不同的指令对应不同的 query，实现指令感知的视觉特征提取
2. **视觉指令微调**: 在 BLIP-2 基础上引入指令微调
3. **13 个数据集 SOTA**: 广泛的指令跟随能力

## 核心原理

### 指令感知 Q-Former

在 BLIP-2 的 Q-Former 基础上，将指令文本与视觉 query 进行交互，使模型根据指令关注图像的不同部分。

## 预训练方法

### 核心思想：不同的指令，应该从图像中提取不同的信息

BLIP-2 的 Q-Former 从图像中提取了 32 个固定的查询向量——无论你问什么（描述图像？数物体？读 OCR？），这 32 个向量都是一样的。InstructBLIP 的洞察是：**指令本身应该决定 Q-Former 从图像中提取什么信息**。

> InstructBLIP = BLIP-2 + 指令感知 Q-Former + 大规模多任务指令微调。它的核心贡献是让视觉信息的提取变得"指令敏感"——当你问"图中有几只猫？"，Q-Former 应该聚焦于物体数量；当你问"猫是什么颜色？"，Q-Former 应该聚焦于颜色属性。

### 训练流水线（Step by Step）

#### Step 1 — 指令感知 Q-Former

InstructBLIP 对 BLIP-2 Q-Former 的关键修改：

**原始 BLIP-2 Q-Former**：
- 32 个可学习的 query token 与视觉特征做交叉注意力
- query token 对所有任务相同

**InstructBLIP 指令感知 Q-Former**：
- 指令文本通过 Q-Former 的文本 Transformer 编码
- 指令表征注入 query token，使其"指令感知"

```text
指令: "How many cats are in the image?"
        ↓
指令编码: [inst_1, inst_2, ..., inst_K]
        ↓ 与 query token 交互
指令感知 Query: [q_1 ⊕ inst, q_2 ⊕ inst, ..., q_32 ⊕ inst]
        ↓ 交叉注意力
视觉特征提取（聚焦于物体数量相关区域）
```

> 指令感知 query 的工作原理：Q-Former 的自注意力让 query token 与指令 token 交互——query 学会根据指令调整自己的"关注点"。一个 query 可能在计数任务中关注物体检测，在描述任务中关注场景语义。

#### Step 2 — 多任务指令数据构建

InstructBLIP 将 26 个多模态数据集转化为统一的指令格式：

| 原始任务 | 数据集 | 指令模板 |
|---------|--------|---------|
| 图像描述 | COCO Captions | "Describe this image in detail." |
| 视觉问答 | VQAv2 | "Question: {question} Answer:" |
| 图像推理 | ScienceQA | "Answer the following science question based on the image:" |
| 图像分类 | ImageNet | "What is in this image?" |
| OCR | TextVQA | "What text is written in the image?" |
| 对话 | LLaVA-Instruct | 多轮对话格式 |

**关键**：对每个数据集，InstructBLIP 设计了 10-15 个不同的指令模板——这确保模型学会理解不同的指令表达方式，而非记忆特定模板。

#### Step 3 — 训练策略

**两阶段训练**：

**阶段 1 — 预训练（同 BLIP-2）**：
- Q-Former + 视觉编码器 + LLM 的初始对齐
- 使用图文描述数据
- 学习率较高

**阶段 2 — 指令微调**：
- 在所有 26 个数据集上做指令微调
- 仅微调 Q-Former 和投影层
- 视觉编码器和 LLM 冻结
- 学习率较低

$$\mathcal{L} = -\sum_{t} \log P(y_t | y_{<t}, \text{InstrQFormer}(I, \text{instruction}))$$

#### Step 4 — 指令感知的详细设计

**指令如何与 Q-Former 交互**？两种方式：

1. **Q-Former 自注意力**：query token 与指令 token 共同参与自注意力——query 可以看到指令内容
2. **交叉注意力条件**：query token 与视觉特征的交叉注意力以指令为条件

这使得 Q-Former 的输出同时编码了"指令问了什么"和"图像显示了什么"——LLM 收到的视觉信息已经是"针对问题的视觉信息"。

### 详细训练配置

| 参数 | 阶段 1（预训练） | 阶段 2（指令微调） |
|------|---------------|----------------|
| 视觉编码器 | ViT-g/14（冻结） | ViT-g/14（冻结） |
| Q-Former | 可训练 | 可训练 |
| LLM | Vicuna-7B/13B（冻结） | Vicuna-7B/13B（冻结） |
| 训练数据 | COCO + VG + CC3M + ... | 26 个数据集指令化 |
| 每个数据集样本 | — | 10-100K |
| 优化器 | AdamW | AdamW |
| 学习率 | 1e-4 | 1e-5 |
| 学习率调度 | 余弦衰减 | 余弦衰减 |
| Batch Size | 512 | 256 |
| 训练硬件 | 16×A100 | 16×A100 |

### InstructBLIP vs BLIP-2 vs LLaVA

| 特性 | BLIP-2 | LLaVA | InstructBLIP |
|------|--------|-------|-------------|
| 视觉-语言桥接 | Q-Former | MLP | **指令感知 Q-Former** |
| 指令敏感性 | ✗ | 部分（通过 LLM） | **✓（从视觉提取开始）** |
| 多任务能力 | 需要分别微调 | 指令微调 | **多任务指令微调（26 数据集）** |
| 零样本泛化 | 一般 | 好 | **更好** |

### 预训练的实用价值

1. **指令感知视觉提取**：证明"看什么取决于问什么"——视觉特征应该被指令条件化
2. **多数据集指令微调模板**：26 数据集统一微调的范式被后续工作广泛借鉴
3. **BLIP-2 生态扩展**：将 BLIP-2 从"视觉-语言模型"升级为"多模态指令跟随模型"
4. **Salesforce BLIP 系列的完整闭环**：BLIP（图文理解）→ BLIP-2（冻结桥接）→ InstructBLIP（指令化）
