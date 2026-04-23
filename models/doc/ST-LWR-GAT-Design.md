# LWR-ResDiff：基于交通流守恒律的残差扩散预测模型

> **S-LWR-GAT: Spatio-Temporal LWR-Constrained Graph Attention Network with Residual Diffusion**

---

## 摘要

本文提出 LWR-ResDiff，一种将 Lighthill-Whitham-Richards (LWR) 交通流守恒律嵌入深度扩散框架的时空预测方法。与现有"物理特征作为输入"或"物理先验作为注意力偏置"的补丁式设计不同，本方法将 LWR 方程的数值格式作为每层空间传播的核心算子，通过**相态依赖的图卷积**实现宏观守恒传播；扩散模型仅在**物理残差空间**进行去噪，学习 LWR 无法解释的微观不确定性。该设计保留了 DCRNN 的交替时空深度结构，同时确保物理参数（临界速度、温度系数）获得非消失梯度，兼具物理可解释性与数据驱动灵活性。

---

## 1. 问题背景与动机

### 1.1 现有方法的局限

当前交通预测中的物理约束引入方式主要有两类：

**类型 A：物理特征作为输入通道**

将流量、速度、密度等物理量拼接为输入向量，与原始序列一起送入 Transformer 或 GNN。此类方法仅让模型"看到"物理量，但未强制模型"服从"物理规律。物理与数据驱动部分无明确分工。

**类型 B：物理先验作为注意力偏置**

在图注意力（GAT）的 logits 中加入物理权重：

$$
\alpha_{ij} = \text{softmax}_j\left(\frac{Q_i K_j^T}{\sqrt{d}} + \lambda \log A_{ij}^{phys}\right)
$$

此类方法的根本缺陷在于：当节点嵌入已经编码了空间相关性时，$Q_i K_j^T$ 本身已很大，物理权重 $\log A_{ij}$ 对 softmax 分布的边际影响极小。数学上，当 $\text{softmax}_j \approx \delta_{ij}$（注意力已集中在正确邻居）时：

$$
\frac{\partial \log A_{ij}}{\partial \text{softmax}_i} = \text{softmax}_i (\delta_{ij} - \text{softmax}_j) \cdot \lambda \to 0
$$

**物理参数的梯度被注意力机制的"自我饱和"效应抵消，导致不可学习。**

### 1.2 我们的核心主张

> **物理传播不应是注意力的"调味料"，而应是空间聚合的"主厨"。扩散模型不应在原始信号空间去噪，而应在物理守恒律之外的残差空间去噪。**

---

## 2. LWR 方程的图结构解释

### 2.1 从守恒律到图传播

LWR 方程（一维）：

$$
\frac{\partial q}{\partial t} + \frac{\partial f(q)}{\partial x} = 0
$$

其中 $q$ 为密度，$f(q)$ 为流量函数。将道路网离散为有向图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，节点 $i$ 代表传感器/路段，应用 Godunov 有限体积格式：

$$
q_i^{t+1} = q_i^t - \frac{\Delta t}{\Delta x} \sum_{j \in \mathcal{N}(i)} \left[ f(q_i^t, q_j^t) \cdot \mathbb{1}_{i \to j} - f(q_j^t, q_i^t) \cdot \mathbb{1}_{j \to i} \right]
$$

**图神经网络视角**：上式可重写为矩阵形式

$$
\mathbf{q}^{t+1} = \mathbf{q}^t + \Delta t \cdot \mathbf{D}(\mathbf{q}^t) \cdot \mathbf{q}^t
$$

其中 $\mathbf{D}(\mathbf{q}^t)$ 是**状态依赖的图差分算子**。这启示我们：交通流的空间传播本质上是图上的**矩阵乘法**，而非消息传递中的注意力加权。

### 2.2 相态与传播方向

交通流存在明确的相态（Regime）：

- **自由流**（$v > v_c$）：信息向下游传播（车辆向前行驶）
- **拥堵流**（$v \leq v_c$）：拥堵波向上游回溢（排队向后扩散）

激波速度由 LWR 特征速度给出：

$$
v_w = \frac{df}{dq}
$$

在自由流区 $v_w > 0$，在拥堵区 $v_w < 0$。这意味着图的有效边方向会随交通状态改变。

---

## 3. 物理图核：相态驱动的有向图卷积

### 3.1 图核计算流程

**输入**：
- $q_{obs} \in \mathbb{R}^{B \times N \times P}$：真实流量观测（用于流量强度门控）
- $v_{obs} \in \mathbb{R}^{B \times N \times P}$：真实速度观测（用于相态检测）
- $A_{down}, A_{up} \in \mathbb{R}^{N \times N}$：上下游物理拓扑（固定，来自路网结构）

**Step 1：相态检测（可导版本）**

为避免硬阈值导致的梯度消失，采用带温度系数的 sigmoid：

$$
r_i = \sigma\left(\frac{1}{P}\sum_{p=1}^{P} v_{obs}(i,p) - v_c \cdot T\right), \quad \bar{v}_i = \frac{1}{P}\sum_{p=1}^{P} v_{obs}(i,p)
$$

其中 $v_c$ 为可学习临界速度，$T$ 为可学习温度系数。$r_i \to 1$ 表示自由流，$r_i \to 0$ 表示拥堵。

**Step 2：方向性加权**

$$
w_{ij} = r_i \cdot A_{ij}^{down} + (1 - r_i) \cdot A_{ij}^{up}
$$

- 自由流节点 $i$：$w_{ij}$ 激活下游边（$j$ 是 $i$ 的下游邻居）
- 拥堵节点 $i$：$w_{ij}$ 激活上游边（$j$ 是 $i$ 的上游邻居）

**Step 3：流量强度门控**

$$
s_{ij} = \frac{\min(\bar{q}_i, \bar{q}_j)}{\max(\bar{q}_i, \bar{q}_j) + \epsilon}, \quad \bar{q}_i = \frac{1}{P}\sum_{p=1}^{P} q_{obs}(i,p)
$$

$$
w'_{ij} = w_{ij} \cdot (0.5 + 0.5 \cdot s_{ij})
$$

相似流量强度的节点间权重更高，反映交通传播的同质性。

**Step 4：归一化**

$$
A_{LWR} = \text{RowSoftmax}(w') \odot A_{mask}
$$

其中 $A_{mask} = \text{clamp}(A_{down} + A_{up}, 0, 1)$ 屏蔽不存在的物理边。

### 3.2 与自适应邻接的融合

$$
A_{eff} = \alpha \cdot A_{LWR} + (1 - \alpha) \cdot A_{adp}
$$

$\alpha = \sigma(\theta_\alpha)$ 为可学习混合系数，$A_{adp} = \text{softmax}(\text{ReLU}(E_1 E_2^T))$ 为数据驱动的自适应邻接。当物理拓扑稀疏或不准时，$\alpha$ 可自动降低，由自适应邻接补充。

---

## 4. 交替时空传播层

### 4.1 设计原则：保留 DCRNN 模式

借鉴 DCRNN 的交替传播策略，每层由空间传播（LWR 图卷积）与时间交互（Patch 间注意力）串行组成：

$$
\text{层 } l: \mathbf{H}^{(l)} \xrightarrow{\text{LWR-Spatial}} \mathbf{H}_{phys}^{(l)} \xrightarrow{\text{Temporal}} \mathbf{H}^{(l+1)}
$$

**关键**：空间传播不由 Attention 主导，而由 $A_{LWR} \cdot \mathbf{H}$ 矩阵乘法主导。

### 4.2 LWR-Spatial 传播（核心算子）

对每层输入 $\mathbf{H} \in \mathbb{R}^{B \times N \times P \times d}$：

**Step 1：计算物理图核**

$$
A_{LWR}^{(l)} = \text{PCGK}(q_{obs}, v_{obs}, A_{down}, A_{up})
$$

> **注意**：$A_{LWR}^{(l)}$ 在每一层重新计算，因为前层传播可能改变节点状态，相态可能变化。

**Step 2：矩阵乘法传播（替代 Attention）**

对每个 patch $p$：

$$
\mathbf{H}_{phys}^{(l,p)} = A_{LWR}^{(l)} \cdot \mathbf{H}^{(l,p)} \in \mathbb{R}^{B \times N \times d}
$$

即：

$$
h_{phys,i}^{(l,p)} = \sum_{j \in \mathcal{N}(i)} A_{LWR,ij}^{(l)} \cdot h_j^{(l,p)}
$$

**物理意义**：节点 $i$ 的特征由其物理邻居按 LWR 权重聚合。权重由当前交通相态决定，拥堵时从上游聚合，自由流时从下游聚合。

**Step 3：轻量 FFN 修正**

$$
\mathbf{H}_{out}^{(l)} = \mathbf{H}_{phys}^{(l)} + \text{FFN}(\text{LayerNorm}(\mathbf{H}_{phys}^{(l)}))
$$

FFN 负责"物理传播后的非线性变换"，但不对传播方向做结构性改变。

### 4.3 Temporal 交互（Patch 间）

在时间维度 $P$ 上做局部自注意力（无因果掩码，适应扩散特性）：

$$
\mathbf{H}^{(l+1)} = \mathbf{H}_{out}^{(l)} + \text{TemporalAttn}(\text{LayerNorm}(\mathbf{H}_{out}^{(l)}))
$$

允许信息从"已去噪的 patch"流向"待去噪的 patch"，实现时间维度的残差修正传播。

### 4.4 多层堆叠

$$
\mathbf{H}^{(0)} = \text{PatchEmbed}(v_{cond})
$$

$$
\text{for } l = 0, \ldots, L-1:
$$

$$
A_{LWR}^{(l)} = \text{PCGK}(\cdot)
$$

$$
\mathbf{H}_{phys}^{(l)} = A_{LWR}^{(l)} \cdot \mathbf{H}^{(l)}
$$

$$
\mathbf{H}^{(l+1)} = \text{Temporal}(\text{FFN}(\mathbf{H}_{phys}^{(l)}))
$$

**与 DCRNN 的对应关系**：

| DCRNN | LWR-ResDiff | 说明 |
|-------|-------------|------|
| 空间：Diffusion Graph Conv | 空间：LWR-Graph Conv | 扩散卷积 → 相态依赖图卷积 |
| 时间：GRU | 时间：Transformer Attention | RNN → Patch 间自注意力 |
| 交替：Spatial → Temporal → 下一层 | 交替：LWR-Spatial → Temporal → 下一层 | 完全保留 |
| 多层堆叠 | 多层堆叠 | 2~4 层 |

---

## 5. 残差扩散框架

### 5.1 为什么扩散应在残差空间

交通序列可分解为：

$$
v(t) = \underbrace{v_{LWR}(t)}_{\text{宏观守恒趋势}} + \underbrace{\delta(t)}_{\text{微观不确定性}}
$$

| 特性 | 原始速度空间 | 残差空间 |
|------|-------------|----------|
| 方差 | 大（包含趋势） | 小（仅波动） |
| 分布 | 多峰（拥堵/自由流混合） | 更接近单峰高斯 |
| 物理约束 | 需要网络学习 | 已由 LWR 保证 |
| 扩散步数 | 多 | 少 |
| 可解释性 | 低 | 高（$v_{LWR}$ 可单独验证） |

**关键洞察**：扩散模型最擅长学习"均值为 0 的噪声分布"。把趋势交给 LWR，扩散只负责"去噪"，各司其职。

### 5.2 训练阶段

**Step 1：物理基线预测**

通过完整 LWR-ST 网络得到：

$$
v_{LWR} = \text{OutputProj}(\mathbf{H}^{(L)}) \in \mathbb{R}^{B \times N \times pred\_len}
$$

**Step 2：真实残差**

$$
\delta_{true} = v_{target} - v_{LWR}
$$

**Step 3：残差加噪**

$$
t \sim \text{Uniform}(0, T), \quad \epsilon \sim \mathcal{N}(0, I)
$$

$$
\delta_k = \sqrt{\bar{\alpha}_t} \cdot \delta_{true} + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon
$$

**Step 4：残差注入**

将加噪残差作为额外通道注入每层输入：

$$
\mathbf{H}^{(0)} = \text{PatchEmbed}(v_{cond}) \oplus \text{ResEmbed}(\delta_k)
$$

其中 $\oplus$ 为拼接或加法，$\text{ResEmbed}$ 为轻量投影。

**Step 5：网络预测残差**

$$
\delta_{pred} = \text{ResDenoiser}(\mathbf{H}^{(L)}, v_{LWR})
$$

**Step 6：损失函数**

$$
\mathcal{L} = \mathbb{E}_{t, \epsilon}\left[\frac{1}{\sqrt{\bar{\alpha}_t}} \|\delta_{pred} - \delta_{true}\|\right]
$$

分母为 SimDiff 的加权 MAE 形式，噪声更强的早期步骤获得更大权重。

### 5.3 推理阶段

**Step 1：确定性物理基线**

$$
v_{LWR} = \text{LWR-ST-Network}(v_{cond})
$$

（只算一次，条件历史不变）

**Step 2：残差去噪**

$$
\delta_K \sim \mathcal{N}(0, I)
$$

$$
\text{for } k = K, K-1, \ldots, 1:
$$

$$
\delta_{pred} = \text{ResDenoiser}(\delta_k, v_{LWR})
$$

$$
\delta_{k-1} = \text{DPM-Solver}(\delta_k, \delta_{pred}, k)
$$

**Step 3：MoM 集成**

$$
\delta_0 = \text{Median-of-Means}(\{\delta_0^{(1)}, \ldots, \delta_0^{(n)}\})
$$

**Step 4：最终预测**

$$
\hat{v} = v_{LWR} + \delta_0
$$

---

## 6. 梯度分析：为什么物理参数不会消失

### 6.1 梯度链

$$
\mathcal{L} = \|\delta_{pred} - \delta_{true}\|^2
$$

$$
\delta_{pred} = \text{GNN}(\mathbf{H}^{(L)}, v_{LWR}, A_{LWR}^{(L)})
$$

$$
\mathbf{H}^{(L)} = \text{Temporal}(\text{FFN}(A_{LWR}^{(L-1)} \cdot \mathbf{H}^{(L-1)}))
$$

$$
\vdots
$$

$$
A_{LWR}^{(l)} = f(v_c, T; q_{obs}, v_{obs})
$$

对 $v_c$ 求导（链式法则）：

$$
\frac{\partial v_c}{\partial \mathcal{L}} = \sum_{l=0}^{L-1} \frac{\partial \mathbf{H}^{(L)}}{\partial \mathcal{L}} \cdot \frac{\partial \mathbf{H}^{(L-1)}}{\partial \mathbf{H}^{(L)}} \cdots \frac{\partial \mathbf{H}_{phys}^{(l)}}{\partial A_{LWR}^{(l)}} \cdot \frac{\partial v_c}{\partial A_{LWR}^{(l)}}
$$

**逐项分析**：

| 项 | 表达式 | 是否为零 | 量级估计 |
|----|--------|---------|----------|
| $\partial \mathcal{L}/\partial \mathbf{H}^{(L)}$ | 由残差损失决定 | 训练初期非零 | $O(1)$ |
| $\partial \mathbf{H}^{(l+1)}/\partial \mathbf{H}^{(l)}$ | Temporal + FFN 的 Jacobian | 非零 | $O(1)$ |
| $\partial \mathbf{H}_{phys}^{(l)}/\partial A_{LWR}^{(l)}$ | $\mathbf{H}^{(l)}$（节点特征） | **非零！** | $\|h\| \sim O(1)$ |
| $\partial A_{LWR}^{(l)}/\partial v_c$ | $\sigma'(x) \cdot (-T) \cdot \bar{v}$ | **非零！** | $O(T \cdot \bar{v}) \sim O(2)$ |

### 6.2 与 Attention 偏置方案的对比

| 方案 | $\partial h/\partial A$ | 中间衰减 | 最终梯度 |
|------|-------------------------|----------|----------|
| Attention 偏置 | $\text{softmax}' \cdot \lambda \cdot (\delta_{ij} - \text{softmax}_j)$ | softmax 自我抵消 | $\sim 10^{-6}$ |
| LWR 矩阵乘法 | $h_j$（直接） | 无 | $\sim 10^{-1}$ |

**关键差异**：

- **Attention 方案**：$A$ 通过 $\log A \to \text{logits} \to \text{softmax}$ 间接影响输出，softmax 的归一化效应使梯度自我抵消
- **LWR 方案**：$A$ 直接参与矩阵乘法 $\mathbf{H}_{phys} = A \cdot \mathbf{H}$，$\partial \mathbf{H}_{phys}/\partial A = \mathbf{H}$ 是直接、无衰减的

### 6.3 多层累加效应

物理参数 $v_c$ 和 $T$ 在所有层共享。反向传播时，每层贡献一个梯度项：

$$
\frac{\partial v_c}{\partial \mathcal{L}} = \sum_{l=0}^{L-1} g^{(l)}, \quad g^{(l)} \sim O(0.1)
$$

$L=2$ 层时总梯度 $\sim O(0.2)$，$L=4$ 层时 $\sim O(0.4)$。**多层不是稀释，而是累加。**

---

## 7. 双流设计与显式分解

### 7.1 双流输入

| 流 | 输入 | 作用 | 是否参与预测输出 |
|----|------|------|----------------|
| q-Stream | $q_{obs}$（真实流量标量） | 计算 $A_{LWR}$ 的流量强度门控 | 否 |
| v-Stream | $v_{obs}$（真实速度标量）+ $v_{embed}$（速度嵌入） | 相态检测 + 特征传播载体 | 是 |

**设计原则**：q 只决定"图结构"（邻居间权重），不贡献输出特征值；v 在 q 决定的图上传播，产生预测。

### 7.2 显式分解输出

$$
\hat{v} = v_{LWR} + \delta_{pred}
$$

其中：

- $v_{LWR} = W_{LWR} \cdot \mathbf{H}^{(L)}$：物理骨架预测（可单独提取验证）
- $\delta_{pred} = W_{res} \cdot \mathbf{H}^{(L)}$：数据驱动残差修正

**监控指标**：

$$
\rho = \frac{\|\delta_{pred}\|}{\|v_{LWR}\|}
$$

- 训练初期：$\rho > 1$（物理骨架未收敛，残差主导）
- 训练后期：$\rho \in [0.1, 0.5]$（物理骨架稳定，残差微调）
- 若始终 $\rho > 2$：物理约束太弱，需增大 $\log A$ 权重或降低残差分支学习率

---

## 8. 与 DCRNN 的关系

### 8.1 结构对应

| DCRNN | LWR-ResDiff | 说明 |
|-------|-------------|------|
| 空间：Diffusion Graph Conv | 空间：LWR-Graph Conv | 扩散卷积 → 相态依赖图卷积 |
| 时间：GRU | 时间：Transformer Attention | RNN → Patch 间自注意力 |
| 交替：Spatial → Temporal → 下一层 | 交替：LWR-Spatial → Temporal → 下一层 | 完全保留 |
| 多层堆叠 | 多层堆叠 | 2~4 层 |

### 8.2 本质区别

DCRNN 的图卷积是**纯数据驱动**的扩散过程，图结构由随机游走定义。LWR-ResDiff 的图卷积是**物理驱动**的守恒过程，图结构由交通流基本图和相态检测定义。

---

## 9. 训练与推理流程

### 9.1 训练流程

```
输入: batch_x (v_cond), batch_y (v_target), batch_flow_x (q_obs)

1. 归一化 (NI)
   v_cond_norm = RevIN(v_cond)
   v_target_norm = RevIN(v_target)
   q_obs_norm = RevIN(q_obs)

2. 物理基线预测（确定性前向）
   v_embed = PatchEmbed(v_cond_norm)
   for l in range(L):
       A_LWR^(l) = PCGK(q_obs_norm, v_cond_norm, A_down, A_up)
       H_phys^(l) = A_LWR^(l) @ H^(l)
       H^(l+1) = Temporal(FFN(H_phys^(l)))
   v_LWR = OutputProj(H^(L))

3. 真实残差
   delta_true = v_target_norm - v_LWR

4. 残差加噪
   t ~ Uniform(0, T)
   delta_k = sqrt(alpha_bar[t]) * delta_true + sqrt(1 - alpha_bar[t]) * eps

5. 残差注入与去噪预测
   H^(0) = v_embed + ResEmbed(delta_k)
   for l in range(L):
       A_LWR^(l) = PCGK(...)  # 重新计算（状态可能变化）
       H_phys^(l) = A_LWR^(l) @ H^(l)
       H^(l+1) = Temporal(FFN(H_phys^(l)))
   delta_pred = ResOutputProj(H^(L))

6. 损失
   L = weighted_MAE(delta_pred, delta_true)

7. 反归一化（监控用）
   v_pred = RevIN.denorm(v_LWR + delta_pred)
```

### 9.2 推理流程

```
输入: x_enc (v_cond)

1. 预计算物理基线（只算一次！）
   v_LWR = LWR-ST-Network(x_enc)

2. 多次采样残差
   for sample_id in range(n_samples):
       delta_K = randn(B, N, pred_len)
       for k in reversed(range(K)):
           delta_pred = ResDenoiser(delta_k, v_LWR)
           delta_{k-1} = DPM-Solver(delta_k, delta_pred, k)
       all_deltas.append(delta_0)

3. MoM 集成
   delta_final = Median-of-Means(all_deltas)

4. 最终预测
   v_pred = v_LWR + delta_final
   v_pred = RevIN.denorm(v_pred)
```

---

## 10. 论文叙事建议

### 10.1 核心贡献（3 点）

1. **Residual Diffusion Space**：首次将扩散模型从原始信号空间转移到物理残差空间，LWR 负责宏观趋势，扩散负责微观不确定性。

2. **LWR-Graph Convolution as Core Operator**：用相态依赖的矩阵乘法替代 Attention 作为空间传播核心，避免物理梯度被 softmax 自我抵消。

3. **Alternating Spatio-Temporal Depth**：保留 DCRNN 的交替时空结构，每层 = LWR-Spatial → Temporal，物理约束嵌入多层传播链。

### 10.2 回应审稿人质疑

**Q: "Why not just use LWR as a standalone predictor?"**

> "LWR is a macroscopic PDE that assumes conservation and smoothness. Real traffic violates these assumptions due to sensor noise, non-conservative on-ramps, and stochastic driver behavior. Our model uses LWR as the deterministic backbone and diffusion as the stochastic corrector, combining the best of both worlds."

**Q: "Does the physical module limit model flexibility?"**

> "The physical module computes a baseline that the residual denoiser can override. If the data strongly violates LWR (e.g., accidents, extreme weather), the residual $\delta$ will be large, and the denoiser will compensate. The model is as flexible as standard diffusion, but with a better inductive bias that accelerates learning."

**Q: "How is this different from adding physical features as input?"**

> "Adding flow/speed as input channels makes the model 'see' physics but not 'obey' physics. In our design, the physical baseline $v_{LWR}$ is computed by a separate deterministic module with its own parameters ($v_c$, $T$), and the neural network is explicitly trained to predict only the residual. This architectural separation ensures physical constraints are hard-coded, not soft-suggested."

---

## 11. 实现要点

### 11.1 核心模块：LWR-Spatial Propagation

```python
class LWRGraphConv(nn.Module):
    """
    LWR 图卷积：物理矩阵乘法作为核心算子
    
    替代 GAT 的 attention，直接用 A_kernel @ H 做空间传播
    """
    def __init__(self, d_model):
        super().__init__()
        self.W_phys = nn.Linear(d_model, d_model)  # 物理分支投影
        self.beta_phys = nn.Parameter(torch.tensor(0.7))  # 物理主导系数
    
    def forward(self, h, A_kernel):
        """
        h: (B, N, P, d) 特征
        A_kernel: (B, N, N) 物理图核
        """
        B, N, P, d = h.shape
        
        # 矩阵乘法：A_kernel @ H
        # h: (B, N, P, d) -> (B*P, N, d)
        # A: (B, N, N) -> (B*P, N, N)
        h_2d = h.permute(0, 2, 1, 3).reshape(B * P, N, d)
        A_exp = A_kernel.unsqueeze(1).expand(B, P, N, N).reshape(B * P, N, N)
        
        h_phys = torch.bmm(A_exp, h_2d)  # (B*P, N, d)
        h_phys = self.W_phys(h_phys)
        
        # reshape back
        h_phys = h_phys.reshape(B, P, N, d).permute(0, 2, 1, 3)
        
        # 组合
        beta = torch.sigmoid(self.beta_phys)
        return beta * h_phys + (1 - beta) * h
```

### 11.2 梯度验证

```python
# 验证 v_critical 的梯度非零
loss.backward()

for name, p in model.named_parameters():
    if 'raw_v_critical' in name:
        grad_norm = p.grad.norm().item()
        print(f"[GRAD] {name}: grad_norm={grad_norm:.2e}")
        assert grad_norm > 1e-6, f"Gradient vanished for {name}!"
```

---

## 12. 总结

| 维度 | 现有方法（补丁式） | LWR-ResDiff |
|------|-------------------|-------------|
| 物理作用 | Attention 偏置项 | 每层矩阵乘法主导 |
| 梯度路径 | $\partial \text{softmax} \to \partial \log A$（衰减） | $\partial(A \cdot H) = H$（直接） |
| 扩散目标 | 原始速度（方差大） | 物理残差（方差小） |
| 时空结构 | 无明确交替 | 保留 DCRNN 模式 |
| 可解释性 | 弱（权重混合） | 强（$v_{LWR}$ 可单独验证） |
| 物理参数 | 不可学习 | 可学习且梯度非零 |

> **LWR 不是模型的"插件"，而是每层空间传播的"骨架"。扩散不是模型的"主体"，而是"修正器"。交替时空深度完全保留，物理与数据驱动各司其职。**

---

## 附录 A：符号表

| 符号 | 含义 |
|------|------|
| $B$ | Batch size |
| $N$ | 节点数（传感器/路段数） |
| $P$ | Patch 数（时间序列分块数） |
| $d$ | 特征维度 |
| $v_c$ | 临界速度（用于相态判断） |
| $T$ | 温度系数（sigmoid 平滑度） |
| $v$ | 速度 |
| $q$ | 流量 |
| $\mathbf{H}$ | 特征矩阵 |
| $A_{LWR}$ | LWR 物理图核 |
| $r$ | 相态指示（1=自由流，0=拥堵） |
| $\delta$ | 物理残差 |
| $\sigma(\cdot)$ | Sigmoid 函数 |
| $\odot$ | Hadamard 积 |

---

## 附录 B：监控通过标准

| 指标 | 预期范围 | 异常诊断 |
|------|---------|----------|
| $\partial v_c / \partial \mathcal{L}$ | $> 10^{-6}$ | 梯度消失，检查 SpatialGAT 是否改为矩阵乘法 |
| upstream/downstream 比 | 早高峰 >1, 平峰 <1 | 全 >1 说明拥堵频繁 |
| Free-flow ratio | 白天 0.3-0.6, 夜间 0.8+ | 全 0.5 说明 $v_c$ 设错 |
| $\|\delta\|/\|v_{LWR}\|$ | 训练后期 0.1-0.5 | 始终 >2 说明物理骨架太弱 |
| $v_c$ | 25-40 km/h | >80 或 <10 说明数据/损失有问题 |
