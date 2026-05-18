# Denoiser 结构修改渐进式实验方案（Step-by-Step）

> **定位**：从最小侵入到核心算子，逐层验证 Denoiser 保守性的结构性主因  
> **基线**：adaptive eta 后处理（总 MAE=0.2281，CG_MAE=30.83）  
> **原则**：一次只改一个组件，10 epoch 快速验证，有效果再拓展  
> **目标**：找到能让 CG_MAE 突破 30.5 的最小修改

---

## 零、实验基线与对照组

### 0.1 基线配置（所有实验共用）

| 配置项 | 值 | 说明 |
|--------|----|------|
| 数据集 | Fujian-30 | 15min 粒度 |
| 模型 | SimDiff | 原始架构 |
| 训练 epoch | 10（快速验证）/ 30（完整验证） | 先 10 epoch 看趋势 |
| Batch size | 32 | 不变 |
| Learning rate | 0.0001 | 不变 |
| Optimizer | Adam | 不变 |
| 后处理 | adaptive eta=0.5 | 保留最优外部补丁 |

### 0.2 评估指标（必须全部记录）

| 指标 | 公式 | 目标 |
|------|------|------|
| 总 MAE | $\frac{1}{NT}\sum_{n,t} \|\hat{Y}_{n,t} - Y_{n,t}^*\|$ | 不劣化 > 0.2300 |
| CG_MAE | 拥堵相样本平均 MAE | **< 30.5（生死线）** |
| FF_MAE | 自由流样本平均 MAE | 不劣化 > 15% |
| TPR_CG | 拥堵相趋势保持率 | 向 0.473 靠近 |
| 采样分散度 | $\frac{1}{N}\sum_n \text{std}_k(\hat{Y}_n^{(k)})$ | **增大**（证明打破保守性） |
| 激波覆盖率 | 真值落在 $[\min_k \hat{Y}^{(k)}, \max_k \hat{Y}^{(k)}]$ 内的比例 | **提高** |

---

## Step 1：Trend-Aware PatchEmbed（最小侵入）

### 1.1 修改内容

只改 PatchEmbed 层，其他全部冻结。

**原始**：
$$z_0^{(i)} = W_p \cdot \underbrace{\frac{1}{P}\sum_{j=0}^{P-1} x_{iP+j}}_{\mu^{(i)}}$$

**修改后**：
$$\mu^{(i)} = \frac{1}{P}\sum_{j=0}^{P-1} x_{iP+j}, \quad \tau^{(i)} = \frac{x_{(i+1)P-1} - x_{iP}}{P - 1}$$

$$z_0^{(i)} = W_\mu \cdot \mu^{(i)} + W_\tau \cdot \tau^{(i)}$$

其中 $W_\mu, W_\tau \in \mathbb{R}^{d \times d}$ 是新投影矩阵。

### 1.2 实验配置

```
对照组：原始 PatchEmbed（单通道均值）
实验组 A：双通道，dim 平分（mu: d/2, tau: d/2）
实验组 B：双通道，dim 不变（mu: d, tau: d，总输入 2d 经线性压缩到 d）
```

### 1.3 训练策略

- 加载原始 SimDiff 预训练权重（除 PatchEmbed 外全部冻结）
- 只训练 $W_\mu, W_\tau$ 和后续 LayerNorm 的统计量
- 10 epoch 快速验证
- 若 CG_MAE 趋势下降（< 31.0），解冻全部权重跑满 30 epoch

### 1.4 判断标准

| 结果 | CG_MAE（10 epoch） | 决策 |
|------|-------------------|------|
| **有效** | < 31.0 且趋势下降 | 跑满 30 epoch，若 < 30.5 则成功 |
| **边际** | 30.8-31.2 | 跑满 30 epoch，同时准备 Step 2 |
| **无效** | > 31.5 | **立即终止**，进入 Step 2 |

---

## Step 2：PatchEmbed + 下游权重微调（拓展）

### 2.1 前提

Step 1 有效（CG_MAE < 31.0），但 30 epoch 后仍未突破 30.5。

### 2.2 修改内容

在 Step 1 基础上，解冻 Transformer 前 2 层权重，让模型学会利用新的双通道输入。

**冻结策略**：
- 解冻：PatchEmbed + Transformer Layer 1-2 + 输出投影
- 冻结：Transformer Layer 3+（深层语义已固化，不改）

### 2.3 判断标准

| 结果 | CG_MAE（30 epoch） | 决策 |
|------|-------------------|------|
| **成功** | < 30.5 | 论文聚焦 PatchEmbed + 浅层微调 |
| **边际** | 30.5-30.7 | 进入 Step 3（核心算子修改） |
| **退化** | > 30.8 | 回退到 Step 1 最优 checkpoint |

---

## Step 3：Trend-Conditional Attention（核心算子）

### 3.1 前提

Step 1-2 均无法让 CG_MAE < 30.5，证明保守性主因不在输入层。

### 3.2 修改内容

改核心 Attention 机制，引入趋势相位偏置。

**原始**：
$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

**修改后**：
$$A = \text{softmax}\left(\frac{QK^T + \lambda \cdot \phi \cdot M_{\text{causal}}}{\sqrt{d_k}}\right)$$

其中：
- $\phi = \text{sign}(s^{\text{hist}}) \cdot \sigma\left(\frac{|s^{\text{hist}}| - \mu}{\sigma_{\text{train}}} \cdot \beta\right)$，历史趋势相位
- $M_{\text{causal}}[i,j] = \frac{j-i}{L}$（当 $j>i$），否则 $0$
- $\lambda \in \{0.1, 0.5, 1.0\}$：偏置强度

### 3.3 实验配置

```
对照组：原始 Attention（lambda=0）
实验组 A：lambda=0.1（弱偏置）
实验组 B：lambda=0.5（中等偏置）
实验组 C：lambda=1.0（强偏置）
```

### 3.4 训练策略

- **完整重训 30 epoch**（Attention 是核心算子，必须从头学习）
- 不继承任何预训练权重（避免保守性固化）
- 或：继承 Embedding 层，随机初始化 Attention 权重

### 3.5 判断标准

| 结果 | CG_MAE（30 epoch） | 决策 |
|------|-------------------|------|
| **成功** | < 30.5 | 论文聚焦 Trend-Conditional Attention |
| **边际** | 30.5-30.7 | 进入 Step 4（损失函数修改） |
| **退化/不稳定** | > 30.8 或 loss nan | 调小 lambda，或改用 soft 偏置 |

---

## Step 4：LWR Conservation Loss（根本修改）

### 4.1 前提

Step 3 仍无法突破 30.5，证明保守性根植于 MSE 损失本身。

### 4.2 修改内容

训练目标加入物理守恒约束。

**原始损失**：
$$\mathcal{L}_{\text{MSE}} = \mathbb{E}\left[\|\epsilon - \epsilon_\theta\|^2\right]$$

**修改后**：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \gamma \cdot \mathcal{L}_{\text{conserv}}$$

守恒损失（离散 LWR）：
$$\mathcal{L}_{\text{conserv}} = \frac{1}{NT}\sum_{n,t} \left(\hat{q}_{n,t+1} - \hat{q}_{n,t} + \sum_{m \in \mathcal{N}_n} w_{m,n}(\hat{q}_{m,t} - \hat{q}_{n,t})\right)^2$$

其中 $w_{m,n}$ 是基于路段距离的权重。

### 4.3 实验配置

```
对照组：纯 MSE（gamma=0）
实验组 A：gamma=0.01（弱物理约束）
实验组 B：gamma=0.1（中等约束）
实验组 C：gamma=1.0（强约束）
```

### 4.4 判断标准

| 结果 | CG_MAE（30 epoch） | 决策 |
|------|-------------------|------|
| **成功** | < 30.5 | 论文聚焦物理引导扩散 |
| **边际** | 30.5-30.7 | 结合 Step 3（Attention 偏置 + 物理损失） |
| **退化** | > 30.8 | 论文转为诊断性发现：扩散模型在激波预测中的固有限制 |

---

## 五、总体决策树

```
开始
  │
  ▼
Step 1: Trend-Aware PatchEmbed（10 epoch）
  │
  ├── CG_MAE < 31.0 且趋势下降 ──→ 跑满 30 epoch
  │       │
  │       ├── CG_MAE < 30.5 ──→ 成功，论文聚焦 PatchEmbed
  │       │
  │       └── CG_MAE > 30.5 ──→ Step 2（浅层微调）
  │               │
  │               ├── CG_MAE < 30.5 ──→ 成功
  │               │
  │               └── CG_MAE > 30.5 ──→ Step 3
  │
  └── CG_MAE > 31.5 ──→ 立即进入 Step 3
          │
          ▼
Step 3: Trend-Conditional Attention（30 epoch）
  │
  ├── CG_MAE < 30.5 ──→ 成功，论文聚焦 Attention
  │
  └── CG_MAE > 30.5 ──→ Step 4
          │
          ▼
Step 4: LWR Conservation Loss（30 epoch）
  │
  ├── CG_MAE < 30.5 ──→ 成功，论文聚焦物理损失
  │
  └── CG_MAE > 30.5 ──→ 论文转为诊断性发现
```

---

## 六、每步必须记录的数据

### 6.1 训练日志

- 每 epoch 的 train loss / val loss
- 每 epoch 的 CG_MAE（验证集）
- 每 epoch 的 FF_MAE（验证集）
- 学习率变化

### 6.2 测试诊断

- 总 MAE / CG_MAE / FF_MAE / HOL_CG
- TPR / TPR_CG / true_cong_TPR
- 采样分散度（10 条采样的标准差）
- 激波覆盖率（真值在采样 min-max 内的比例）
- eta 分布（若保留 adaptive eta 后处理）

### 6.3 可视化

- 激波样本的采样轨迹图（如 Sample 2475/3817 格式）
- CG_MAE 随 epoch 变化曲线
- 采样分散度随 epoch 变化曲线

---

## 七、时间规划

| 步骤 | 预计时间 | 产出 |
|------|---------|------|
| Step 1（10 epoch） | 2-4 小时 | 判断 PatchEmbed 是否主因 |
| Step 1（30 epoch，若有效） | 6-8 小时 | 完整结果 |
| Step 2（30 epoch，若需） | 6-8 小时 | 浅层微调结果 |
| Step 3（30 epoch，若需） | 8-12 小时 | Attention 修改结果 |
| Step 4（30 epoch，若需） | 8-12 小时 | 物理损失结果 |

---

## 八、论文叙事（Step-by-Step 诊断）

### Paragraph 1：问题定义与基线

> 在 Fujian-30 数据集上，SimDiff 原始模型总 MAE=0.2296，CG_MAE=30.88。后处理校正（adaptive $\eta$）将总 MAE 降至 0.2281，但 CG_MAE 仅降至 30.83，无法突破 30.5。这表明外部加法校正已达上限，必须修改 Denoiser 内部结构。

### Paragraph 2：渐进式诊断框架

> 我们提出四步递进式结构修改：Step 1 验证输入编码（Trend-Aware PatchEmbed），Step 2 验证浅层适配，Step 3 验证核心算子（Trend-Conditional Attention），Step 4 验证训练目标（LWR Conservation Loss）。每步独立验证，10 epoch 快速筛选，有效果再拓展，无效则进入下一步，精确定位保守性的结构性主因。

### Paragraph 3：预期贡献

> 该框架不仅旨在找到最优修改方案，更旨在回答一个基础问题：扩散模型在交通流激波预测中的保守性，究竟源于输入表示不足、注意力机制平滑、还是损失函数均值陷阱？每一步的负结果同样具有方法论价值。

---

*文档结束*
