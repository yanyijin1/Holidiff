# HA-TEK 叙事重构方案：SimDiff 原始概念 ↔ 交通物理命名对照

> **原则**：零架构改动，仅重构理论叙事层与代码命名层。所有数学形式与 SimDiff 完全一致，但物理意义重新锚定。

---

## 一、核心概念总对照表

| 层级 | SimDiff 原始术语 | 新交通物理命名 | 论文中的物理叙事 | 代码命名建议 |
|------|-----------------|---------------|-----------------|-------------|
| **整体模型** | SimDiff (HA-SimDiff) | **HA-TEK** (Holiday-Aware Traffic Evolution Kernel) | 面向节假日分布漂移的交通状态演化核 | `class HATEK(nn.Module)` |
| **核心算子** | Denoiser | **TEK** (Traffic Evolution Kernel) / **STEK** (Spatio-Temporal Evolution Kernel) | 不是去噪器，而是学习交通系统时空演化规律的核算子 | `class TEK(nn.Module)` |
| **注意力机制** | Self-Attention + RoPE | **TCP** (Topology-Causal Propagation) | 拓扑因果传播：信息只沿真实路网邻接关系、按时间正向流动 | `class TCPAttention(nn.Module)` |
| **扩散采样** | Diffusion Sampling / Denoising | **Stochastic Micro-realization** | 给定相同历史，未来交通存在多种等价的微观实现（跟驰扰动、局部事件） | `def generate_micro_realizations(...)` |
| **聚合机制** | Median-of-Means (MoM) | **CE** (Consensus Extraction) | 从微观多解性中提取符合宏观传播规律的稳健共识 | `class ConsensusExtractor(nn.Module)` |
| **嵌入层** | Patch Embedding | **STSD** (Spatio-Temporal State Discretization) | 将连续交通状态离散化为时空演化 token | `class STSDEmbedder(nn.Module)` |
| **归一化** | NI (Normalization Independence) | **NDA** (Node-wise Distribution Alignment) | 节点级分布对齐：每个断面独立归一化，消除跨节点分布漂移 | `def node_wise_align(x)` |
| **噪声/残差** | Noise ε / Residual | **Micro-uncertainty** | 宏观趋势剥离后的微观不确定性，对应交通流理论中 LWR 方程之外的部分 | `micro_uncertainty` |
| **宏观输出** | Prediction Ŷ | **Macroscopic Consensus** | 多次微观实现经共识提取后得到的宏观交通状态估计 | `macro_consensus` |
| **条件输入** | Past Series X | **Historical Macro-state** | 历史宏观状态，作为演化核的初始条件 | `hist_macro_state` |
| **目标变量** | Future Series Y | **Future Macro-state** | 未来宏观状态真值，用于监督演化核学习 | `future_macro_state` |

---

## 二、公式体系重写（零数学改动，纯符号重定义）

### 2.1 微观实现生成（原：前向扩散加噪）

**SimDiff 原文：**
\[
\mathbf{Y}^{(t)} = \sqrt{\bar{\alpha}_t} \mathbf{Y} + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
\]

**HA-TEK 叙事：**
\[
\mathbf{Y}^{(t)}_{\text{micro}} = \sqrt{\bar{\alpha}_t} \mathbf{Y} + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}
\]
> "从未来宏观状态 $\mathbf{Y}$ 生成第 $t$ 阶**微观实现**。随着 $t$ 增大，微观不确定性逐渐淹没宏观结构，TEK 需逆向恢复演化轨迹。"

---

### 2.2 交通演化核（原：Denoiser）

**SimDiff 原文：**
\[
\hat{\mathbf{Y}} = \mathcal{D}(\mathbf{X}, \mathbf{Y}^{(t)}, t)
\]

**HA-TEK 叙事：**
\[
\hat{\mathbf{Y}}_{\text{micro}}^{(k)} = \text{TEK}(\mathbf{X}_{\text{hist}}, \mathbf{Y}^{(t)}_{\text{micro}}, t; \mathbf{A})
\]
其中 $\mathbf{A}$ 为路网拓扑。TEK 内部通过 TCP 实现：
\[
\mathbf{S}\bigl[(i,\tau), (j,\tau')\bigr] = \frac{\mathbf{q}_{i,\tau}^\top \mathbf{k}_{j,\tau'}}{\sqrt{d}} + \mathbf{M}_{\text{TCP}}\bigl[(i,\tau), (j,\tau')\bigr]
\]
\[
\mathbf{M}_{\text{TCP}} = \begin{cases} 0, & A_{ij}=1 \text{ 且 } \tau' \leq \tau \\ -\infty, & \text{otherwise} \end{cases}
\]
> "TEK 不是去噪器，而是**数据驱动的宏观交通波传播算子**。TCP 掩码将物理约束硬编码：交通波只沿邻接断面传播，且严格遵守时间因果。"

---

### 2.3 共识提取（原：MoM）

**SimDiff 原文：**
\[
\hat{\mathbf{Y}}_{\text{MoM}} = \text{Median}\left(\left\{ \frac{1}{M}\sum_{m=1}^M \hat{\mathbf{Y}}^{(k,m)} \right\}_{k=1}^K \right)
\]

**HA-TEK 叙事：**
\[
\hat{\mathbf{Y}}_{\text{consensus}} = \text{CE}\left(\left\{ \hat{\mathbf{Y}}_{\text{micro}}^{(k)} \right\}_{k=1}^K \right) = \text{Median-of-Means}\left(\left\{ \hat{\mathbf{Y}}_{\text{micro}}^{(k)} \right\}_{k=1}^K \right)
\]
> "Consensus Extraction 从 $K$ 条随机微观实现中，提取符合宏观传播规律的**共识交通状态**。节假日期间微观不确定性增大（实现发散），CE 通过中位数机制自动抑制极端样本（如突发拥堵导致的异常峰值），输出稳健宏观估计。"

---

### 2.4 节点级分布对齐（原：NI）

**SimDiff 原文：**
\[
\tilde{\mathbf{X}} = \frac{\mathbf{X} - \mu_{\text{past}}}{\sigma_{\text{past}}}, \quad \tilde{\mathbf{Y}} = \frac{\mathbf{Y} - \mu_{\text{future}}}{\sigma_{\text{future}}}
\]

**HA-TEK 叙事：**
\[
\tilde{\mathbf{X}}_i = \frac{\mathbf{X}_i - \mu_{x,i}}{\sigma_{x,i}}, \quad \tilde{\mathbf{Y}}_i = \frac{\mathbf{Y}_i - \mu_{y,i}}{\sigma_{y,i}}, \quad \forall i \in \mathcal{V}
\]
> "Node-wise Distribution Alignment (NDA)：每个断面独立对齐自身分布。交通流具有显著的断面异质性（城区 vs 郊区、主线 vs 匝道），全局归一化会抹杀局部特征。NDA 确保演化核在每个断面的'本地坐标系'中学习演化规律。"

---

### 2.5 完整前向流程（HA-TEK 符号体系）

\[
\begin{aligned}
&\textbf{Step 1: 节点级分布对齐} \
&\tilde{\mathbf{X}} = \text{NDA}(\mathbf{X}), \quad \tilde{\mathbf{Y}} = \text{NDA}(\mathbf{Y}) \
&\textbf{Step 2: 时空状态离散化} \
&\mathbf{Z}_x = \text{STSD}(\tilde{\mathbf{X}}), \quad \mathbf{Z}_y = \text{STSD}(\tilde{\mathbf{Y}}) \
&\textbf{Step 3: 微观实现生成（训练）} \
&\mathbf{Y}^{(t)}_{\text{micro}} = \sqrt{\bar{\alpha}_t} \tilde{\mathbf{Y}} + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon} \
&\textbf{Step 4: 交通演化核（TCP-Transformer）} \
&\hat{\mathbf{Y}}_{\text{micro}} = \text{TEK}(\mathbf{Z}_x, \mathbf{Y}^{(t)}_{\text{micro}}, t; \mathbf{A}) \
&\textbf{Step 5: 共识提取（推理）} \
&\hat{\mathbf{Y}}_{\text{consensus}} = \text{CE}\left(\left\{ \text{TEK}(\cdots)^{(k)} \right\}_{k=1}^K \right) \
&\textbf{Step 6: 反归一化} \
&\hat{\mathbf{Y}} = \text{NDA}^{-1}(\hat{\mathbf{Y}}_{\text{consensus}})
\end{aligned}
\]

---

## 三、代码命名修改方案（可直接替换）

### 3.1 顶层模型

```python
# 原 SimDiff 风格
class SimDiff(nn.Module):
    def __init__(self, ...):
        self.denoiser = Denoiser(...)
        self.mom = MedianOfMeans(K=16, M=4)

# 新 HA-TEK 风格（零逻辑改动，仅重命名）
class HATEK(nn.Module):
    def __init__(self, ...):
        self.tek = TEK(...)           # 原 Denoiser
        self.ce = ConsensusExtractor(K=16, M=4)  # 原 MoM
```

### 3.2 核心模块

| 原类/函数名 | 新类/函数名 | 文件建议 |
|-----------|-----------|---------|
| `Denoiser` | `TEK` 或 `STEK` | `models/tek.py` |
| `self_attention` | `tcp_attention` | `models/tcp_attention.py` |
| `PatchEmbed` | `STSDEmbedder` | `models/stsd.py` |
| `MoM` / `MedianOfMeans` | `ConsensusExtractor` | `models/consensus.py` |
| `forward_diffusion` | `generate_micro_realization` | `models/diffusion_utils.py` |
| `normalize_past` / `normalize_future` | `node_wise_align` / `inverse_node_wise_align` | `utils/nda.py` |
| `noise` | `micro_uncertainty` | 变量名 |
| `prediction` / `forecast` | `macro_consensus` | 变量名 |
| `residual` | `micro_realization` | 变量名 |

### 3.3 关键函数签名映射

```python
# 原 SimDiff
class Denoiser(nn.Module):
    def forward(self, x_norm, y_noisy, t):
        # x_norm: (B, N, T_in, C)
        # y_noisy: (B, N, T_out, C)
        # t: (B,)
        return prediction  # (B, N, T_out, C)

# 新 HA-TEK
class TEK(nn.Module):
    def forward(self, hist_macro_state, micro_realization, t, adj):
        # hist_macro_state: (B, N, T_in, C)  — 历史宏观状态
        # micro_realization: (B, N, T_out, C) — 第 t 阶微观实现
        # t: (B,) — 演化时间步
        # adj: (N, N) — 路网拓扑
        return micro_estimate  # (B, N, T_out, C) — 微观估计
```

### 3.4 训练循环变量映射

```python
# 原
for batch in loader:
    x = batch['past']      # 历史
    y = batch['future']    # 未来
    y_noisy = forward_diffusion(y, t, noise)
    pred = denoiser(x, y_noisy, t)
    loss = F.mse_loss(pred, y)

# 新
for batch in loader:
    hist_state = batch['hist_macro_state']
    future_state = batch['future_macro_state']
    micro_unc = generate_micro_realization(future_state, t, micro_noise)
    micro_est = tek(hist_state, micro_unc, t, adj)
    loss = F.mse_loss(micro_est, future_state)  # TEK 学习从微观实现恢复宏观状态
```

---

## 四、论文 Method 段落模板（可直接插入）

### 4.1 问题重述（用新叙事）

> 交通预测的本质不是确定性回归，而是**宏观状态演化 + 微观多解性**的联合建模。给定历史宏观状态 $\mathbf{X}$，未来交通状态 $\mathbf{Y}$ 并非单值，而是由多种微观随机因素（跟驰扰动、局部事件、测量噪声）共同决定的**条件分布** $p(\mathbf{Y}|\mathbf{X})$。节假日期间，该分布的方差显著增大，传统单点估计方法易于被极端微观实现误导。

### 4.2 方法概述

> 本文提出 **HA-TEK**（Holiday-Aware Traffic Evolution Kernel），包含三个物理启发的核心组件：
>
> **(1) Topology-Causal Propagation (TCP)**。将 Transformer 的自注意力约束在真实路网拓扑与时间因果的交集内。节点 $i$ 在时刻 $\tau$ 的状态更新，只能聚合来自拓扑邻接节点 $j$（$A_{ij}=1$）在 $\tau$ 及之前时刻的信息。TCP 等价于将宏观交通波传播的物理规律硬编码进神经网络。
>
> **(2) Traffic Evolution Kernel (TEK)**。基于 TCP 的 Transformer 核，学习从历史宏观状态到未来微观实现演化规律的映射。TEK 不是传统意义上的去噪器——其前向传播本身就是数据驱动的宏观交通状态演化算子。
>
> **(3) Consensus Extraction (CE)**。通过 $K$ 次随机微观实现的聚合，从节假日增大的不确定性中提取稳健宏观共识。CE 通过中位数机制自动抑制极端离群样本，输出符合宏观传播规律的交通状态估计。

### 4.3 与 SimDiff 的关系（审稿人可能问）

> HA-TEK 的数学骨架继承自 SimDiff 的单阶段扩散框架，但物理视角完全不同：SimDiff 将扩散视为**去噪过程**，MoM 视为**稳健统计**；HA-TEK 将扩散视为**微观实现生成**，TEK 视为**交通演化核**，MoM 视为**共识提取**。这种重命名不是语义游戏——它直接指导了拓扑因果掩码（TCP）的设计动机：只有将物理约束嵌入核心运算，微观实现才能服从真实路网的宏观传播规律。

---

## 五、快速检查清单

| 检查项 | 状态 |
|-------|------|
| 架构零改动 | ✅ 仅重命名与注释 |
| 公式数学等价 | ✅ 符号重定义，数学形式不变 |
| 物理叙事自洽 | ✅ 宏观=共识，微观=实现，TEK=演化算子 |
| 代码可直接替换 | ✅ 提供类名/函数名/变量名映射 |
| 与 SimDiff 关系清晰 | ✅ 明确定位为继承+重解释 |
| Holiday 融入自然 | ✅ 节假日 → 微观不确定性增大 → CE 稳健提取 |

---

*文档生成完毕。可直接用于论文写作与代码重构。*
