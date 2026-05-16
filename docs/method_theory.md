# HoliDiff 方法：理论基础与模块定义

> 本文将交通预测重新表述为**微观实现**与**宏观交通流估计**的联合建模问题。给定相同的历史条件，未来交通存在多种等价的微观实现（跟驰扰动、局部事件、测量噪声及节假日诱发的结构性变化），未来状态应被理解为条件分布 $p(\mathbf{Y}\,|\,\mathbf{X}, \mathbf{A})$ 的一次采样，而非确定性输出。

---

## 一、总体框架

**输入**：历史交通状态 $\mathbf{X}\in\mathbb{R}^{N\times T_{\mathrm{in}}\times C}$、路网拓扑 $\mathcal{G}=(\mathcal{V}, \mathcal{E}, \mathbf{A})$、节假日条件 $c$。

**输出**：未来交通状态预测 $\hat{\mathbf{Y}}\in\mathbb{R}^{N\times T_{\mathrm{out}}\times C}$。

**三个核心模块**（单向递进）：

1. **频率解耦**：将历史状态分解为多频带分量，分别处理不同时间尺度上的演化规律；
2. **微观实现生成**：通过扩散模型对未来状态的条件分布进行建模，单次前向传播输出一条微观实现。该模块的时空结构受 LWR 方程约束，确保微观不确定性的传播服从真实路网的宏观规律；
3. **宏观交通流估计**：通过多次微观实现的聚合，从节假日增大的不确定性中提取稳健宏观交通流状态。

---

## 二、频率解耦

### 2.1 动机

节假日诱发的交通分布漂移同时影响多个时间尺度：
- **低频分量**：整体流量基线的抬升；
- **中频分量**：通勤节律的改变；
- **高频分量**：局部拥堵与异常峰值的出现。

为了显式建模不同尺度上的演化差异，引入频率解耦模块。

### 2.2 数学形式

对每个节点的时间序列执行快速傅里叶变换（FFT）：

$$
\mathbf{X}_i(\omega) = \mathrm{FFT}(\mathbf{X}_i), \quad \forall i \in \mathcal{V}.
$$

通过可学习的频带掩码将频谱划分为 $B$ 个子带：

$$
\mathbf{X}^{(b)}_i = \mathrm{IFFT}\bigl( \mathbf{M}^{(b)} \odot \mathbf{X}_i(\omega) \bigr), \quad b=1,\dots,B,
$$

其中 $\mathbf{M}^{(b)}$ 为第 $b$ 个频带的可学习掩码。

各频带分量独立通过后续模块，最终融合为完整状态估计：

$$
\hat{\mathbf{Y}} = \sum_{b=1}^{B} \alpha_b \cdot \hat{\mathbf{Y}}^{(b)},
$$

其中 $\alpha_b$ 为频带聚合权重。

---

## 三、微观实现生成

### 3.1 定义

**微观实现（micro-realization）**：扩散模型单次前向传播输出的未来状态样本 $\hat{\mathbf{Y}}^{(k)}$，对应条件分布 $p(\mathbf{Y}\,|\,\mathbf{X}, \mathbf{A})$ 的一次采样，即一种可能的未来交通场景。

### 3.2 前向扩散过程

对未来状态逐步注入高斯噪声，生成第 $t$ 阶噪声样本：

$$
\mathbf{Y}^{(t)} = \sqrt{\bar{\alpha}_t}\,\mathbf{Y} + \sqrt{1-\bar{\alpha}_t}\,\mathbf{z}, \quad \mathbf{z}\sim\mathcal{N}(\mathbf{0}, \mathbf{I}),
$$

其中 $\bar{\alpha}_t$ 为噪声调度参数。随着 $t$ 增大，未来结构逐渐被噪声淹没。

### 3.3 反向扩散估计

训练条件网络 $\mathcal{D}_\theta$，从噪声中恢复未来状态：

$$
\hat{\mathbf{Y}} = \mathcal{D}_\theta(\mathbf{X}, \mathbf{Y}^{(t)}, t; \mathbf{A}).
$$

由于扩散过程的随机性，多次独立采样将产生不同的微观实现，这些实现围绕宏观状态波动。

### 3.4 LWR 先验约束的时空结构

**核心约束来源**：LWR 方程的有限波速传播规律——交通波沿路网以有限速度传播，不会违反时间因果，不会在非邻接断面间瞬时跳跃。

**实现方式**：在 $\mathcal{D}_\theta$ 中引入拓扑因果注意力掩码。

将历史与未来序列拼接为时空 token 序列 $\mathbf{H}\in\mathbb{R}^{N\times L\times d}$，其自注意力分数定义为：

$$
S\bigl[(i,\tau), (j,\tau')\bigr] = \frac{(W_q \mathbf{H}_{i,\tau})^\top (W_k \mathbf{H}_{j,\tau'})}{\sqrt{d}} + \mathbf{M}\bigl[(i,\tau), (j,\tau')\bigr],
$$

其中掩码 $\mathbf{M}$ 同时编码拓扑与因果约束：

$$
\mathbf{M}\bigl[(i,\tau), (j,\tau')\bigr] = 
\begin{cases}
    0, & \text{if } A_{ij}=1 \text{ and } \tau' \leq \tau, \\
    -\infty, & \text{otherwise}.
\end{cases}
$$

**物理意义**：节点 $i$ 在时刻 $\tau$ 的状态更新，只能聚合来自拓扑邻接节点 $j$（$A_{ij}=1$）在不晚于 $\tau$ 的时刻的信息。上述约束将 LWR 的宏观传播规律硬编码进微观实现生成器，确保单次采样的传播轨迹服从真实路网的物理规律。

估计器的层内更新为：

$$
\mathbf{H}^{(l)} = \mathbf{H}^{(l-1)} + \mathrm{Softmax}(S^{(l)}) V^{(l)} + \mathrm{FFN}(\mathbf{H}^{(l-1)}).
$$

经过 $M$ 层后，通过反 Patch 映射得到未来状态估计 $\hat{\mathbf{Y}}\in\mathbb{R}^{N\times T_{\mathrm{out}}\times C}$。

---

## 四、宏观交通流估计

### 4.1 定义

**宏观交通流**：在交通流理论中，断面流量本质上是微观车辆行为（跟驰、换道、加减速）的统计平均。宏观交通流状态是多次微观实现的稳健中心。

### 4.2 问题

单次前向传播输出的是一条微观实现 $\hat{\mathbf{Y}}^{(k)}$。由于节假日期间微观不确定性增大，不同采样下的实现结果存在显著发散：部分采样可能因随机噪声而偏离宏观传播规律（如突发拥堵导致的异常峰值）。若直接采用单次采样作为最终预测，模型在节假日场景下的稳定性将严重下降。

### 4.3 数学形式

通过对多次微观实现的聚合，得到符合 LWR 传播规律的稳健断面流量估计。执行 $K$ 组采样，每组 $M$ 条微观实现轨迹，取组内均值后求中位数：

$$
\hat{\mathbf{Y}}_{\mathrm{macro}} = \mathrm{Median}\left( \left\{ \frac{1}{M}\sum_{m=1}^{M} \hat{\mathbf{Y}}^{(k,m)} \right\}_{k=1}^{K} \right).
$$

**物理意义**：Median-of-Means 不是简单的异常值剔除，而是从多种微观实现中估计最可能的宏观交通流。在节假日期间，尽管微观实现的发散程度增大，该机制仍能通过中位数将估计结果稳定于 LWR 传播规律所约束的宏观交通流区域。

### 4.4 最终预测

最终预测即为宏观交通流估计的输出：

$$
\hat{\mathbf{Y}} = \hat{\mathbf{Y}}_{\mathrm{macro}}.
$$

---

## 五、训练目标

总体损失函数为微观实现恢复误差：

$$
\mathcal{L} = \bigl\| \mathcal{D}_\theta(\mathbf{X}, \mathbf{Y}^{(t)}, t; \mathbf{A}) - \mathbf{Y} \bigr\|_2^2.
$$

---

## 六、模块对照表（用于代码命名）

| 论文术语 | 代码建议命名 | 说明 |
|---------|------------|------|
| 频率解耦 | `FrequencyDisentangler` | 频带分解与融合 |
| 微观实现生成器 | `MicroRealizationGenerator` | 基于扩散的条件采样网络 |
| LWR 先验约束 | `LWRCausalMask` / `TopologyCausalAttention` | 拓扑因果注意力掩码 |
| 宏观交通流估计 | `MacroTrafficFlowEstimator` | Median-of-Means 聚合器 |
| 微观实现 | `micro_realization` | 单次采样输出变量名 |
| 宏观估计 | `macro_flow_estimate` | 最终聚合输出变量名 |
