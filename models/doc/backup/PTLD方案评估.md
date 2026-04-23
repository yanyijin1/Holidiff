# PTLD: 基于物理拓扑的潜在扩散交通预测框架

> **可行性评估与方案文档**
>
> 日期：2026-04-06

---

## 1. 相关工作与引言

### 1.1 从TSGDiff引入：时序的图视角

| 项目 | 内容 |
|------|------|
| **论文** | TSGDiff: Rethinking Synthetic Time Series Generation from a Pure Graph Perspective (AAAI 2026) |
| **arXiv** | https://arxiv.org/abs/2511.12174 |
| **代码** | https://github.com/jvaeylee/TSGDiff (待发布) |

**核心贡献**：TSGDiff开创了基于图生成的时序生成新范式。它通过傅里叶频谱构建动态图边关系，在潜在图空间中进行扩散生成。提出的Topo-FID指标验证了图结构表示比欧氏空间更能捕获时序依赖关系。

**局限性**：TSGDiff专注于无条件生成（合成数据生成），而非预测任务。缺乏：
- 历史到未来的条件映射机制
- 交通领域的物理约束
- 确定性预测任务的点估计策略

**我们需要**：图编码/解码架构以及潜在空间扩散框架——特别是将时序投影到图流形再进行生成建模的思想。

### 1.2 从SimDiff引入：高效的扩散点预测

| 项目 | 内容 |
|------|------|
| **论文** | SimDiff: Simpler Yet Better Diffusion Model for Time Series Point Forecasting |
| **arXiv** | https://arxiv.org/abs/2511.19256 |
| **代码** | https://github.com/Dear-Sloth/SimDiff |

**核心贡献**：SimDiff证明了单阶段端到端扩散模型无需预训练回归器即可实现SOTA点预测性能。关键创新包括：
- **归一化独立性（N.I.）**：通过可学习的反归一化消除分布偏移
- **均值中位数估计器（MoM）**：通过分组中位数聚合将概率样本转换为稳健点估计
- **单Transformer架构**：消除了混合模型的复杂性

**局限性**：SimDiff在欧氏空间中运算，缺乏显式空间结构。对于交通预测，它忽略了路网拓扑和物理守恒律（质量守恒、激波传播），导致物理不一致的预测。

**我们需要**：用于稳健点预测的MoM估计器、处理分布偏移的N.I.机制，以及简化的单网络架构。

---

## 2. 方案概述

PTLD (Physics-Topology Latent Diffusion) 旨在融合三方面的工作优势：
- **TSGDiff**：图视角的时序生成范式
- **SimDiff**：高效的端到端扩散点预测机制
- **LWR-DDPM**：物理引导的宏观-微观解耦框架

核心假设：交通状态可分解为确定性的物理骨架与随机微观涨落，扩散应在低维潜在图空间中建模残差分布。

---

## 2. 核心数学框架

### 2.1 解耦假设：稳定模式 vs 不稳定模式

交通状态可分解为确定性的**稳定模式**与随机的**不稳定涨落**：

$$
\mathbf{x}(s, t) = \mathbf{x}_{\text{LWR}}(s, t) + \boldsymbol{\epsilon}_{\text{micro}}(s, t)
$$

其中：
- $s \in \{1, \ldots, N\}$：空间位置（路段/节点）
- $t \in \{1, \ldots, T\}$：时间位置
- $\mathbf{x}_{\text{LWR}}(s, t)$：**稳定模式**，LWR守恒方程推演的确定性骨架，编码交通流的方向性传播与物理约束
- $\boldsymbol{\epsilon}_{\text{micro}}(s, t)$：**不稳定涨落**，非守恒成分（换道、随机加减速）

### 2.2 时空调场 → 频域：扩散的输入路径

扩散模型本质是学习数据分布，因此我们需要将时空场变换到扩散模型可处理的频域表示：

**时空间 → 频域**：对交通时空场做2D傅里叶变换（空间 $s \to k$，时间 $t \to \omega$）：

$$
\tilde{\mathbf{x}}(\omega, k) = \mathcal{F}\{\mathbf{x}(s, t)\} = \tilde{\mathbf{x}}_{\text{LWR}}(\omega, k) + \tilde{\boldsymbol{\epsilon}}_{\text{micro}}(\omega, k)
$$

**频域加噪**：扩散在频域进行，逐步叠加噪声：

$$
\tilde{\mathbf{x}}_t = \sqrt{\bar{\alpha}_t} \cdot \tilde{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_t} \cdot \tilde{\boldsymbol{\epsilon}}, \quad \tilde{\boldsymbol{\epsilon}} \sim \mathcal{N}(0, \mathbf{I})
$$

**频域去噪 → 逆变换**：去噪后的频域结果逆变换回时空：

$$
\hat{\boldsymbol{\epsilon}}_{\text{micro}}(s, t) = \mathcal{F}^{-1}\{\tilde{\boldsymbol{\epsilon}}_{\theta}(\tilde{\mathbf{x}}_t, t, \mathbf{c})\}
$$

---

### 2.3 时空图 → 图视角（TSGDiff）

核心思想：将交通时空序列转化为图结构，利用图神经网络的表达能力进行建模。

**Step 1: D-GNN空间编码**

对历史数据使用D-GNN编码N个节点的特征：

$$
Z_{\text{hist}} = \text{D-GNN\_Enc}(X_{\text{hist}}) \in \mathbb{R}^{T_{\text{hist}} \times N \times d}
$$

D算子强制空间编码沿守恒方向传播。

**Step 2: 时间步 → 图节点**

将T个时间步映射为图节点 $V = \{Z_{\text{hist}}[1], Z_{\text{hist}}[2], ..., Z_{\text{hist}}[T]\}$，每个节点携带N维空间特征。

**Step 3: 傅里叶谱建边（TSGDiff）**

基于节点特征的频率相似度构建时序图边：

$$
w_{ij} \propto \cos(\theta_i - \theta_j)
$$

其中 $\theta_i$ 为节点 $i$ 的主频率相位，边权重由相位相似度决定。这使得扩散可在时序图上进行。

---

### 2.4 物理拓扑算子

LWR PDE数值积分定义空间差分算子 $D \in \mathbb{R}^{N \times N}$：

$$
\rho_{t+1} = \rho_t + D q_t
$$

**关键洞察**：$D$ 编码路网的物理拓扑（上下游关系），当用作图卷积核时，强制消息传递沿质量守恒方向进行。

### 2.5 LWR约束的空间一致性（隐空间版本）

根据 `models/README.md` 的核心公式：

#### q通道：保持不变（不做自更新）

q 作为物理驱动力，提供空间传播信息，**不做自更新**：

$$
\mathbf{H}^{(q)} = \mathbf{H}^{(0, q)} = \text{MLP}_q(q_{i,:}) \in \mathbb{R}^d
$$

即 q 的隐表示直接来自观测，不经过空间推演层。

#### v通道的更新（LWR物理约束）

速度的更新由流量 q 通过空间差分算子 D 决定。将 LWR 物理约束提升到隐空间：

$$
\mathbf{H}^{(l+1, v)} = \sigma\left( \mathbf{H}^{(l, v)} + \mathbf{D} \cdot \mathbf{H}^{(q)} \cdot \mathbf{W}_{q \to v}^{(l)} \right)
$$

**物理语义**：
- $\mathbf{H}^{(l, v)}$: v 通道的当前状态
- $\mathbf{D} \cdot \mathbf{H}^{(q)}$: q 通道的上游-下游差分（空间梯度），**来自观测 q**
- $\cdot \mathbf{W}_{q \to v}$: 投影到 v 通道的更新方向
- 最终 v 的更新 = 自身 + q 的空间梯度贡献

#### 完整第 l+1 层

$$
\mathbf{H}^{(l+1)} = \text{LayerNorm}\left( \left[ \mathbf{H}^{(q)} ; \mathbf{H}^{(l+1, v)} \right] \right)
$$

注：q 通道在多层推演中保持不变，仅用于驱动 v 的更新。

---

### 2.5 完整流程（融合TSGDiff + SimDiff + LWR + 条件机制）

#### 训练阶段 (Training)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              训练阶段 (Training)                                  │
│                                                                                  │
│  X_hist ───┬─── X_future                                                         │
│            │                                                                     │
│            ▼                                                                     │
│  ┌─────────────────┐                                                             │
│  │ LWR_Solver      │───→ x_LWR                                                   │
│  └─────────────────┘                                                             │
│            │                                                                     │
│            ▼                                                                     │
│  Z_hist = D-GNN_Enc(X_hist) ────┐  (条件K,V)                                     │
│                                 │                                                │
│  ε_tgt = X_future - x_LWR       │                                                │
│  Z_tgt = D-GNN_Enc(ε_tgt)       │                                                │
│                                 ▼                                                │
│  Z_noisy = √(ᾱ_t)·Z_tgt + √(1-ᾱ_t)·ε                                           │
│  ε_pred = Denoiser(Z_noisy, t, Z_hist)  ← Cross-Attention(Q=Z_noisy, K,V=Z_hist) │
│                                                                                  │
│  Loss = ||ε - ε_pred||²                                                         │
└──────────────────────────────────────────────────────────────────────────────────┘
```

#### 推理阶段 (Inference + MoM)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           推理阶段 (Inference + MoM)                              │
│                                                                                  │
│  输入: X_hist                                                                    │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────────────┐                                                             │
│  │ LWR_Solver      │───→ x_LWR  (物理骨架，单次计算)                             │
│  └─────────────────┘                                                             │
│       │                                                                          │
│       ▼                                                                          │
│  Z_hist = D-GNN_Enc(X_hist)  (固定条件，缓存复用)                                 │
│       │                                                                          │
│       ├─────────────────────────────────────────────────────────────────────┐    │
│       │                     重复K次采样 (SimDiff MoM核心)                   │    │
│       │                                                                     │    │
│       │  for k = 1 to K:                                                    │    │
│       │    Z_T = randn(T_pred, N, d)  (每次独立随机初始化)                   │    │
│       │                                                                     │    │
│       │    for t = T down to 1:  (DDIM去噪)                                 │    │
│       │      ε_pred = Denoiser(Z_t, t, Z_hist)                              │    │
│       │      Z_{t-1} = DDIM_Step(Z_t, ε_pred)                               │    │
│       │                                                                     │    │
│       │    Z_0^(k) = 去噪完成                                               │    │
│       │    ε^(k) = D-GNN_Dec(Z_0^(k))                                       │    │
│       │    x^(k) = x_LWR + ε^(k)   ← 第k个生成样本                         │    │
│       │                                                                     │    │
│       └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                          │
│       ▼                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ Stage: MoM 稳健聚合 (SimDiff)                                             │   │
│  │                                                                          │   │
│  │  输入: K个样本 {x^(1), x^(2), ..., x^(K)}                                │   │
│  │                                                                          │   │
│  │  Step 1: 分组 (分成M组，每组B=K/M个)                                      │   │
│  │    Group 1: {x^(1), ..., x^(B)}   →  x̄_1 = mean(Group_1)                │   │
│  │    Group 2: {x^(B+1), ..., x^(2B)} →  x̄_2 = mean(Group_2)               │   │
│  │    ...                                                                   │   │
│  │    Group M: ...  →  x̄_M = mean(Group_M)                                 │   │
│  │                                                                          │   │
│  │  Step 2: 中位数聚合                                                        │   │
│  │    x_final = median({x̄_1, x̄_2, ..., x̄_M})                               │   │
│  │                                                                          │   │
│  │  (注：此处无需D空间平滑，物理一致性已由Encoder/Decoder保证)                │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                          │
│       ▼                                                                          │
│  输出: x_final  (最终点预测)                                                     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**条件机制（B-1：潜在空间图注意力条件化）**：
- 历史编码 $Z_{\text{hist}}$ 作为 Cross-Attention 的 K,V
- 噪声预测 $Z_{\text{noise}}$ 作为 Q
- 物理意义：学习"历史第t步对未来t'步的影响权重"

---

### 2.8 Cross-Attention数学形式

对于每个路段节点 $n \in \{1, \dots, N\}$：

$$
\begin{aligned}
Q_n &= Z_t[:, n, :] \in \mathbb{R}^{T_{\text{pred}} \times d} \quad \text{（当前噪声状态）} \\
K_n &= V_n = Z_{\text{hist}}[:, n, :] \in \mathbb{R}^{T_{\text{hist}} \times d} \quad \text{（历史条件）} \\
A_n^{\text{cross}} &= \text{Softmax}\left(\frac{Q_n K_n^\top}{\sqrt{d}}\right) \in \mathbb{R}^{T_{\text{pred}} \times T_{\text{hist}}} \\
Z_{\text{cond}, n} &= A_n^{\text{cross}} \cdot V_n \in \mathbb{R}^{T_{\text{pred}} \times d}
\end{aligned}
$$

---

### 2.7 四者角色对照

| 模块 | 核心思想 | 作用 |
|------|----------|------|
| **LWR** | 守恒方程数值积分 | 提供稳定模式骨架 |
| **D-GNN** | D算子约束的空间编码 | 强制消息沿守恒方向传播 |
| **TSGDiff** | 时间步→图节点，傅里叶谱建边 | 捕获时序依赖，建模扩散图结构 |
| **SimDiff** | Flow Matching条件生成 + MoM | 高效扩散生成 + 稳健点估计 |

---

## 3. 技术细节

### 3.1 D-GNN空间编码

$$
H^{(l+1)} = \sigma(D \cdot H^{(l)} \cdot W^{(l)})
$$

| 组件 | 说明 |
|------|------|
| $D$ | 固定矩阵，源自路网拓扑和守恒律（LWR算子） |
| $W^{(l)}$ | 可学习的通道变换 |
| $\sigma$ | 非线性激活 |

### 3.2 Cross-Attention条件化

$$
\begin{aligned}
Q &= Z_t \cdot W_Q \\
K &= Z_{\text{hist}} \cdot W_K \\
V &= Z_{\text{hist}} \cdot W_V \\
Z_{\text{cond}} &= \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
\end{aligned}
$$

### 3.3 TSGDiff图构建

时间步 → 图节点，傅里叶谱相似度建边：

$$
w_{ij} \propto \cos(\theta_i - \theta_j)
$$

### 3.4 Flow Matching扩散

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \cdot \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \cdot \boldsymbol{\epsilon}
$$

生成K个多样化样本 $\{\hat{x}^{(k)}\}$。

### 3.5 D-GNN解码

解码过程同样使用D算子约束，确保输出沿守恒方向传播：

$$
\hat{x} = D\text{-GNN\_Dec}(Z_0)
$$

### 3.6 MoM聚合

| 步骤 | 操作 |
|------|------|
| 1. 样本分组 | K个样本 → M组，每组取均值 $\{\bar{x}_m\}$ |
| 2. 中位数聚合 | $\hat{x} = \text{Median}(\{\bar{x}_m\})$ |

---

## 4. 训练细节

### 4.1 损失函数

$$
\mathcal{L} = \mathbb{E}_{z_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t, Z_{\text{hist}})\|^2\right]
$$

### 4.2 训练技巧

| 技巧 | 说明 |
|------|------|
| **历史掩码** | 随机mask部分历史时间步，强制模型从不完整历史推断 |
| **Teacher Forcing** | Cross-Attention中随机丢弃历史键值对，防止过拟合 |
| **渐进式解冻** | 训练初期冻结D-GNN_Enc，后期微调 |

---

## 5. 可行性评估

### 5.1 优势

| 方面 | 评估 |
|------|------|
| **物理一致性** | LWR骨架提供守恒约束，扩散模拟真实分布 |
| **多样性建模** | Flow Matching生成多样化未来态 |
| **计算效率** | 频域扩散 + 中位数聚合 |
| **可解释性** | 稳定/不稳定模式解耦清晰 |

### 5.2 挑战

| 问题 | 可能的解决思路 |
|------|----------------|
| LWR求解器精度 | 可学习加权组合 $\hat{x} = \beta x_{\text{LWR}} + (1-\beta) x_{\text{regression}}$ |
| 条件机制设计 | 探索cross-attention或adaptive normalization |
| D矩阵构建（一般路网） | 使用 $D = A - I$（图拉普拉斯形式） |

### 5.3 关键假设

| 假设 | 风险等级 |
|------|----------|
| 稳定+不稳定分解足够表达交通模式 | 中 |
| 扩散能有效建模涨落分布 | 中 |
| MoM能从多样性中提取稳态 | 低 |

---

## 5. 未来方向

- **动态物理图**：自适应 $D(x)$，随交通状态（自由流/拥堵流）改变
- **多尺度建模**：微观（车道）→ 中观（路段）→ 宏观（走廊）
- **物理知情扩散**：从LWR哈密顿量推导分数函数

---

## 6. 结论

PTLD方案通过**稳定模式（LWR）+ 不稳定涨落（扩散）+ MoM聚合**的三阶段流程：

1. LWR守恒方程提供确定性骨架
2. 扩散模型学习不稳定涨落分布，生成多样化未来态
3. MoM从多样性中提取稳态预测

逻辑自洽，建议从链式路网开始验证。

