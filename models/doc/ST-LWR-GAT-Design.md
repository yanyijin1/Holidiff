# ST-LWR-GAT：基于交通流守恒律的频域增强时空图预测模型

> **ST-LWR-GAT: Spatio-Temporal LWR-Constrained Graph Attention Network with Frequency Decomposition**

---

## 摘要

本文提出 ST-LWR-GAT，一种将 Lighthill-Whitham-Richards (LWR) 交通流守恒律嵌入深度时空预测框架的方法，并创新性引入**频域三带分解**。模型在输入层将序列分解为低频趋势 / 中频周期 / 高频波动三路，三路独立 Patch Embedding 后共享 STLWRLayer（避免参数增加），再通过可学习的融合层动态合并。该设计使 LWR 物理约束作用于各频带特征，同时频域分解增强了模型对宏观趋势和微观波动的分别建模能力。

---

## 1. 整体架构

### 1.1 架构图

```
输入序列 (B*N, seq_len+pred_len)
           │
           ▼
    ┌──────────────┐
    │ FreqDecomposer │  频域三带分解（固定系数 EMA，无梯度）
    │  α_low=0.1    │
    │  α_mid=0.3    │
    └──────┬───────┘
           │
      ┌────┴────┬────────┐
      ▼         ▼         ▼
  Low Embed  Mid Embed  High Embed  (三路独立 Linear)
  (patch_len -> d_model) × 3
      │         │         │
      ▼         ▼         ▼
  h_low    h_mid    h_high
      │         │         │
      └────┬────┘
           ▼
    ┌──────────────┐
    │  FreqFusion   │  可学习加权融合 + GELU
    │  softmax(w)   │
    └──────┬───────┘
           │
      + Time Embedding（加法注入）
           │
           ▼
    ┌──────────────┐
    │ STLWRLayer × n│  共享（物理约束核心）
    │  PCGK → Spatial → Temporal → FFN
    └──────┬───────┘
           │
           ▼
    Output Projection → (B*N, pred_len)
```

### 1.2 关键设计原则

1. **频域分解在输入侧**：序列在 Embedding 前分解，不改变 STLWRLayer 的任何结构
2. **三路独立 Embedding**：低 / 中 / 高频各自投影到 d_model 空间，保持特性
3. **共享 STLWRLayer**：三路融合后共同经过物理层，不增加参数
4. **可学习融合权重**：频带重要性由网络自动学习
5. **固定系数 EMA**：分解使用不可学习的固定 alpha，避免 autograd 版本冲突

---

## 2. 频域三带分解

### 2.1 分解原理

输入序列 $x \in \mathbb{R}^{B \times N \times T}$，对每个节点独立做 EMA 平滑：

$$
\text{EMA}_\alpha[x]_t = \alpha \cdot x_t + (1-\alpha) \cdot \text{EMA}_{t-1}
$$

**三带提取：**

| 频带 | 提取方式 | EMA 系数 | 物理含义 |
|------|---------|---------|---------|
| 低频 (Low) | $low = \text{EMA}_{\alpha_{low}}(x)$ | $\alpha_{low}=0.1$ | 宏观趋势（慢变） |
| 中频 (Mid) | $mid = \text{EMA}_{\alpha_{mid}}(x - low)$ | $\alpha_{mid}=0.3$ | 周期成分（中等变化） |
| 高频 (High) | $high = x - low - mid$ | — | 微观波动（残差） |

**可重构性保证：** $x = low + mid + high$，零信息损失。

### 2.2 实现细节

- EMA 使用**固定系数**（不参与梯度），避免 autograd 的 in-place 版本冲突
- 系数选取依据：低频 $\alpha=0.1$（大平滑窗口 → 捕捉长周期趋势），中频 $\alpha=0.3$（中等平滑 → 捕捉周内/日内周期），高频 = 残差

### 2.3 频带融合

三路特征分别投影后，通过可学习权重融合：

$$
h_{fused} = \text{GELU}(W_l \cdot h_l) \cdot w_l + \text{GELU}(W_m \cdot h_m) \cdot w_m + \text{GELU}(W_h \cdot h_h) \cdot w_h
$$

其中 $w = \text{softmax}(W_{freq})$ 为可学习权重，初始均匀分布。

---

## 3. LWR 方程的图结构解释

### 3.1 从守恒律到图传播

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

### 3.2 相态与传播方向

交通流存在明确的相态（Regime）：

- **自由流**（$v > v_c$）：信息向下游传播（车辆向前行驶）
- **拥堵流**（$v \leq v_c$）：拥堵波向上游回溢（排队向后扩散）

---

## 4. 物理图核：相态驱动的有向图卷积

### 4.1 图核计算流程（PCGK）

**输入**：
- $q_{obs} \in \mathbb{R}^{B \times N \times P}$：真实流量观测
- $v_{obs} \in \mathbb{R}^{B \times N \times P}$：真实速度观测（用于相态检测）
- $A_{down}, A_{up} \in \mathbb{R}^{N \times N}$：上下游物理拓扑（固定）

**Step 1：相态检测**

$$
r_i = \sigma\left( \frac{1}{P}\sum_{p=1}^{P} v_{obs}(i,p) - v_c \right), \quad T=1.0 \text{ (fixed)}
$$

其中 $v_c$ 为可学习临界速度（clamp 到 $[-2, 3]$），$r_i \to 1$ 表示自由流，$r_i \to 0$ 表示拥堵。

**Step 2：方向性门控**

$$
h_{phys} = r_i \cdot h_{up} + (1 - r_i) \cdot h_{down}
$$

- 自由流节点 $i$：上游聚合主导（信息从上游来）
- 拥堵节点 $i$：下游聚合主导（拥堵波从下游来）

**Step 3：特征相似度 + 流量强度门控**

$$
c_{ij} = 0.5 + 0.3 \cdot \text{sim}(h_i, h_j) + 0.2 \cdot \frac{\min(q_i, q_j)}{\max(q_i, q_j) + \epsilon}
$$

**Step 4：最终边权重**

$$
W_{up} = A_{up} \odot c, \quad W_{down} = A_{down} \odot c
$$

> **关键：无 softmax，无行归一化**。物理边权重直接由流量强度和特征相似度决定，保证梯度路径直接无衰减。

### 4.2 与自适应邻接的融合（可选）

$$
W_{up}^{eff} = \alpha \cdot W_{up} + (1 - \alpha) \cdot A_{adp}
$$

---

## 5. STLWRLayer：交替时空传播

### 5.1 层结构

每层 = **PCGK → LWRSpatial → Temporal → FFN**：

```
Input H^(l)  ──► PCGK ──► (W_up, W_down, regime)
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
         LWRSpatialConv ──► LayerNorm ──► Dropout
              │                         │
              └────────► Temporal ──► LayerNorm ──► Dropout
                               │
                               ▼
                          FFN (+残差) ──► H^(l+1)
```

### 5.2 LWRSpatialConv

对每层输入 $\mathbf{H} \in \mathbb{R}^{B \times N \times P \times d}$：

**Step 1：上/下游聚合**

$$
h_{up} = \text{norm}(W_{up}) \cdot \mathbf{H}, \quad h_{down} = \text{norm}(W_{down}) \cdot \mathbf{H}
$$

**Step 2：相态门控**

$$
h_{phys} = r \cdot h_{up} + (1 - r) \cdot h_{down}
$$

**Step 3：残差连接**

$$
h_{out} = \mathbf{H} + \beta \cdot h_{phys}, \quad \beta = \sigma(\theta_\beta)
$$

### 5.3 TemporalAttention

在 Patch 维度 $P$ 上做多头自注意力（无因果掩码，适应扩散特性）：

$$
\mathbf{H}_{temp} = \text{MHA}(Q, K, V), \quad Q=K=V=\text{LN}(h_{phys})
$$

### 5.4 与 DCRNN 的对应关系

| DCRNN | ST-LWR-GAT | 说明 |
|-------|------------|------|
| 空间：Diffusion Graph Conv | 空间：LWR-Spatial Conv | 随机游走 → 相态依赖图卷积 |
| 时间：GRU | 时间：Transformer Attention | RNN → Patch 间多头注意力 |
| 交替：Spatial → Temporal → 下一层 | 交替：LWR-Spatial → Temporal → 下一层 | 完全保留 |
| 多层堆叠 | 多层堆叠 | 2~4 层 |

---

## 6. 训练与推理流程

### 6.1 训练流程

```
输入: batch_x (条件速度), batch_flow_x (流量), batch_y (目标速度)

1. 归一化 (RevIN)
   v_cond_norm = RevIN(batch_x)
   v_target_norm = RevIN(batch_y)
   q_obs_norm = RevIN(batch_flow_x)

2. 频域分解（输入侧）
   low, mid, high = FreqDecomposer([v_cond; v_target])

3. Patch Embedding（三路独立）
   h_low = input_proj_low(low_patches)
   h_mid = input_proj_mid(mid_patches)
   h_high = input_proj_high(high_patches)

4. 频带融合
   h = FreqFusion(h_low, h_mid, h_high)

5. 时间嵌入注入
   h = h + TimeEmbedding(timesteps)

6. STLWRLayer 前向（n 层）
   for l in range(L):
       W_up^(l), W_down^(l), regime^(l) = PCGK(q_obs_norm)
       h = STLWRLayer^(l)(h, W_up^(l), W_down^(l), regime^(l))

7. 输出投影
   v_pred = OutputProj(h[:, P_cond:])

8. 损失
   L = MSE(v_pred, v_target_norm)
```

### 6.2 推理流程

```
输入: x_enc (条件速度)

1. 预计算物理基线（只算一次！）
   v_embed = PatchEmbed(FreqDecomp(x_enc))
   for l in range(L):
       h = STLWRLayer^(l)(h)
   v_pred = OutputProj(h)

2. DPM-Solver 残差去噪（可选项）

3. 反归一化
   v_pred = RevIN.denorm(v_pred)
```

---

## 7. 梯度分析：为什么物理参数不会消失

### 7.1 梯度链

$$
\mathcal{L} = \|v_{pred} - v_{true}\|^2
$$

$$
\frac{\partial \mathcal{L}}{\partial v_c} = \frac{\partial \mathcal{L}}{\partial v_{pred}} \cdot \frac{\partial v_{pred}}{\partial h^{(L)}} \cdot \frac{\partial h^{(L)}}{\partial h_{phys}^{(L-1)}} \cdots \frac{\partial h_{phys}}{\partial r} \cdot \frac{\partial r}{\partial v_c}
$$

**逐项分析**：

| 项 | 表达式 | 是否为零 |
|----|--------|---------|
| $\partial \mathcal{L}/\partial v_{pred}$ | 由 MSE 损失决定 | 训练初期非零，$O(1)$ |
| $\partial h_{phys}/\partial r$ | $h_{up} - h_{down}$ | 非零，$O(1)$ |
| $\partial r/\partial v_c$ | $\sigma'(x) \cdot (-1)$ | 非零，$O(1)$ |

**关键差异**：

- **Attention 方案**：$A$ 通过 $\log A \to \text{logits} \to \text{softmax}$ 间接影响，梯度被 softmax 归一化抵消
- **LWR 方案**：$v_c \to r \to h_{phys}$ 直接链式传播，无中间非线性归一化

### 7.2 与 Attention 偏置方案的对比

| 方案 | $\partial h/\partial A$ | 中间衰减 | 最终梯度 |
|------|-------------------------|----------|----------|
| Attention 偏置 | $\text{softmax}' \cdot \lambda$ | softmax 自我抵消 | $\sim 10^{-6}$ |
| LWR 矩阵乘法 | $h_{up} - h_{down}$ | 无 | $\sim 10^{-1}$ |

---

## 8. 实验结果

### 8.1 配置

- 数据集：METR-LA（交通速度）
- seq_len=96, pred_len=12
- d_model=64, n_layers=2, n_heads=4
- 频域分解：$\alpha_{low}=0.1$, $\alpha_{mid}=0.3$

### 8.2 消融结果

| 实验 | MSE | MAE | 说明 |
|------|------|------|------|
| **Freq Decomp (三频带)** | **0.779** | **0.622** | 当前最佳 |
| dual_head (vc0) | 0.870 | 0.668 | 无频域分解 |
| dual_head (d96) | 0.890 | 0.682 | 无频域分解 |

频域三带分解使 MSE 降低 **~12%**，MAE 降低 **~7%**，验证了分解策略的有效性。

---

## 9. 附录：监控指标

| 指标 | 预期范围 | 异常诊断 |
|------|---------|---------|
| regime (speed, normed) | 0.3-0.9 | 全 0.5 说明 $v_c$ 未收敛 |
| $v_c$ | $-2 \leq v_c \leq 3$ | 接近边界说明需要调整 clamp 范围 |
| Freq weights | 动态学习 | 观察三路融合比例变化 |
| train loss | 持续下降 | loss 不降检查学习率 |

---

## 附录 A：符号表

| 符号 | 含义 |
|------|------|
| $B$ | Batch size |
| $N$ | 节点数（传感器/路段数） |
| $P$ | Patch 数（时间序列分块数） |
| $d$ | 特征维度 |
| $v_c$ | 临界速度（用于相态判断） |
| $\alpha_{low}, \alpha_{mid}$ | EMA 平滑系数（固定） |
| $v$ | 速度 |
| $q$ | 流量 |
| $\mathbf{H}$ | 特征矩阵 |
| $W_{up}, W_{down}$ | 上/下游物理边权重 |
| $r$ | 相态指示（1=自由流，0=拥堵） |
| $\sigma(\cdot)$ | Sigmoid 函数 |
| $\odot$ | Hadamard 积 |
