# PINN (Physics-Informed Neural Networks)

## 基本信息

- **代表性论文**: [Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- **作者**: Maziar Raissi et al. (Brown University)
- **发表**: Journal of Computational Physics, 2019

## 创新点

1. **物理约束融入神经网络**: 将 PDE（偏微分方程）作为正则化项加入损失函数
2. **正问题和反问题统一**: 同一框架求解 PDE 正问题和参数反问题
3. **小数据高效**: 在极少观测数据下准确推断物理场

## 核心原理

### 损失函数

$$\\mathcal{L} = \\mathcal{L}_{\\text{data}} + \\lambda \\mathcal{L}_{\\text{physics}}$$

- $\\mathcal{L}_{\\text{data}}$: 数据匹配损失（边界条件、初始条件、观测值）
- $\\mathcal{L}_{\\text{physics}}$: 物理方程残差（使用自动微分计算 PDE 残差）

### 与预训练的关系

PINN 不是传统意义上的预训练方法，而是一种**物理先验注入**的训练策略。但可被视为"使用物理定律作为预训练监督信号"。

## 训练方法

### 核心思想：传统的神经网络是"数据说了算"——有多少数据就学多少。PINN 的革命性在于让"物理定律也说了算"——损失函数不仅惩罚预测值与数据不一致，还惩罚预测值违背物理方程。这就像一个学生在答题时，不仅对答案，还被检查推导过程是否遵循物理规律

PINN 不是传统意义上的预训练，而是一种**先验注入**的训练策略。它的核心理念：**物理定律（PDE）是天然的无监督信号**——无穷多的时空点上都有"方程应该成立"这个约束。

> PINN = 神经网络（MLP）+ 物理正则化（PDE 残差损失）+ 自动微分（用 autograd 计算导数）。不需要标签数据，仅靠物理方程就能训练神经网络。

### 训练流水线（Step by Step）

#### Step 1 — 物理问题 → 神经网络

将物理系统建模为一个神经网络的输入-输出映射：

```text
输入: (x, y, z, t) — 时空坐标
输出: (u, v, w, p, T) — 速度、压力、温度等物理场

神经网络: MLP (8 层 × 40 神经元, tanh 激活)
参数 θ: 权重 + 偏置
```

**为什么用 MLP 而不是 CNN/Transformer？**
- PINN 的目标是在连续域上求解 PDE → 输入是连续的时空坐标，不是离散的网格数据
- MLP 是连续函数逼近器 → 自然适合连续输入

#### Step 2 — 双重损失函数

$$\mathcal{L}_{\text{PINN}} = \underbrace{\mathcal{L}_{\text{data}}}_{\text{数据匹配}} + \underbrace{\lambda \mathcal{L}_{\text{physics}}}_{\text{物理约束}}$$

**数据损失**（$\mathcal{L}_{\text{data}}$）：

$$\mathcal{L}_{\text{data}} = \frac{1}{N_d} \sum_{i=1}^{N_d} \|u_{\theta}(x_i, t_i) - u_i^{\text{obs}}\|^2$$

匹配边界条件、初始条件、和稀疏观测值。

**物理损失**（$\mathcal{L}_{\text{physics}}$）——PINN 的核心创新：

以 Navier-Stokes 方程（流体力学）为例：

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{1}{\rho} \frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)$$

将神经网络输出代入 PDE：

$$r(x, y, t) = \frac{\partial u_{\theta}}{\partial t} + u_{\theta} \frac{\partial u_{\theta}}{\partial x} + v_{\theta} \frac{\partial u_{\theta}}{\partial y} + \frac{1}{\rho} \frac{\partial p_{\theta}}{\partial x} - \nu \left( \frac{\partial^2 u_{\theta}}{\partial x^2} + \frac{\partial^2 u_{\theta}}{\partial y^2} \right)$$

$$\mathcal{L}_{\text{physics}} = \frac{1}{N_p} \sum_{j=1}^{N_p} \|r(x_j, y_j, t_j)\|^2$$

> 如果 $r \approx 0$，说明网络的输出在物理上是"自洽"的——满足 Navier-Stokes 方程。

#### Step 3 — 自动微分计算导数

这是实现 $\mathcal{L}_{\text{physics}}$ 的关键技术——在不需要解析求导的情况下计算神经网络输出的各阶偏导数：

```text
1. 输入: (x, y, t) → 前向传播 → u_θ(x, y, t)
2. autograd(u_θ, x) → ∂u/∂x
3. autograd(∂u/∂x, x) → ∂²u/∂x²
4. 同样计算 ∂u/∂t, ∂u/∂y, ∂²u/∂y² ...
5. 代入 PDE 公式计算残差 r
6. 残差 r → 损失 → 反向传播更新 θ
```

**自动微分的角色**：
- 传统数值方法（有限差分/有限元）需要手动离散化 PDE
- PINN 使用 autograd 自动计算导数 → 无需网格、无需离散化 → 任意时空点都可以计算

| 方法 | 导数计算 | 网格 | 维数诅咒 |
|------|--------|------|---------|
| 有限差分 | 手工离散化 | 需要 | **严重**（指数增长） |
| 有限元 | 弱形式推导 | 需要 | 严重 |
| **PINN** | **Autograd** | **不需要** | **缓解（随机采样）** |

#### Step 4 — 配点（Collocation Points）采样

**残差点**（用于计算物理损失）不需要是观测点——它们可以是域内的任意随机点：

```text
在时空域 (x, y, t) 中随机采样 N_p 个"配点"
  → 这些点不需要有观测数据
  → 只需要"在此处 PDE 应该成立"
  → 配点越多，物理约束越强

典型数量:
  - N_p: 10,000-100,000（物理配点）
  - N_d: 100-1,000（观测点/边界点）
```

#### Step 5 — 完整训练流程

```text
1. 初始化 MLP
2. 每轮迭代:
   a. 采样观测/边界点 (x_d, u_d)
   b. 采样物理配点 (x_p)
   c. 前向传播 → u_θ(x_d) 和 u_θ(x_p)
   d. 自动微分计算各阶导数
   e. 计算 L_data + λ L_physics
   f. 反向传播更新 θ
3. 收敛后 → u_θ 是整个域上的连续解
```

### PINN 的训练特殊挑战

| 挑战 | 原因 | 解决方案 |
|------|------|---------|
| **双目标竞争** | L_data 和 L_physics 可能梯度冲突 | 自适应 λ（attention to loss） |
| **高频分量难学** | PDE 解可能包含激波/高频 | 傅里叶特征、多尺度 PINN |
| **优化困难** | 损失景观不平坦 | L-BFGS（二阶优化器） |
| **长时间模拟** | 误差累积 | 顺序 PINN（分段训练） |

### 适用场景

| 问题类型 | 描述 | PINN 优势 |
|---------|------|---------|
| **正问题** | 已知 PDE + 边界条件 → 求解 | 替代数值求解器，对复杂几何更灵活 |
| **反问题** | 已知观测 + PDE → 反推参数 | 自然融合数据和物理 |
| **小数据推断** | 极少数观测 | 物理约束补偿数据不足 |
| **高维 PDE** | 维数诅咒问题 | 随机配点改变复杂度 |

### 训练的实用价值

1. **无需标注的物理学习**：物理方程是免费的"无限监督信号"
2. **数据融合**：稀疏观测 + 物理约束 → 完整物理场 → 医疗影像、天气重建
3. **反问题的统一求解**：同一框架求解 PDE 参数估计 → 材料参数识别、污染源定位
4. **与预训练的融合趋势**：PINN 中的物理损失可以作为 Foundation Model 的训练约束
5. **无网格求解**：无需网格生成 → 复杂几何（如血管网络）上的 PDE 求解
