# SimDiff Denoiser 物理注入渐进式实验方案

> **定位**：从诊断结论出发，逐步向 SimDiff Transformer Denoiser 注入物理机制  
> **核心诊断**：采样系统性低估源于 Denoiser 的三重保守性——Softmax 均值回归 + MSE 条件均值陷阱 + Patch 趋势抹平  
> **实验策略**：控制变量，每一步只改一个组件，精确追踪哪一步带来提升  
> **原则**：不提中间尝试，从 SimDiff 原始公式出发

---

## 一、诊断结论回顾

从采样可视化与聚合器实验得出核心结论：

> **所有扩散采样呈现系统性保守偏差 $\delta > 0$（真值高于采样均值），且偏差方向与历史趋势 $s^{\text{hist}}$ 正相关。**

这意味着：
- 聚合器（MoM/Mean/Weighted）无法修复 $\delta$，因为所有采样同向偏移
- 必须从 Denoiser 源头打破保守性
- 需要精确定位：Denoiser 的哪个组件是保守性的主因

---

## 二、SimDiff 原始 Denoiser 公式

### 2.1 输入 Patch 化

$$z_0 = \text{PatchEmbed}(x_t) + \text{PosEmbed} + \text{TimeEmbed}$$

其中 PatchEmbed 通常为线性投影或简单平均：
$$z_0^{(i)} = W_p \cdot \frac{1}{P}\sum_{j=0}^{P-1} x_{t, iP+j}$$

### 2.2 Transformer Block（标准）

$$z' = z + \text{MSA}(\text{LN}(z))$$
$$z_{\text{out}} = z' + \text{FFN}(\text{LN}(z'))$$

### 2.3 多头自注意力

$$\text{MSA}(z) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 2.4 训练目标

$$\mathcal{L} = \mathbb{E}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

---

## 三、渐进式注入策略（4 Steps）

### 总体原则

- **Step 1-2**：推理时修改（不改训练），验证"已有模型能否被校正"
- **Step 3-4**：训练时修改（需重新训练），验证"从源头修复"
- 每一步独立评估，不累积修改

---

## Step 1：Trend-Biased Attention（趋势偏置注意力）

### 3.1.1 修改点

只改 Attention Score，不改架构、不改训练。

原始：
$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

修改后：
$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \underbrace{\lambda \cdot s^{\text{hist}} \cdot M_{\text{trend}}}_{\text{趋势偏置}}\right)$$

其中：
- $s^{\text{hist}} = \frac{x_{T} - x_{1}}{T-1}$：历史窗口的全局趋势斜率（标量，节点级）
- $M_{\text{trend}} \in \mathbb{R}^{L \times L}$：因果趋势掩码
  - $M_{\text{trend}}[i,j] = \frac{j - i}{L}$ 当 $j > i$（未来位置越远，偏置越强）
  - $M_{\text{trend}}[i,j] = 0$ 当 $j \leq i$（不回顾过去）
- $\lambda > 0$：偏置强度（超参，建议初值 0.1-1.0）

### 3.1.2 物理意义

当 $s^{\text{hist}} > 0$（历史上升趋势）：
- 注意力被迫增强对**远端未来 patch** 的关注
- 打破 softmax 的均匀回顾，让模型"相信趋势会延续"

当 $s^{\text{hist}} < 0$（历史下降趋势）：
- 注意力被迫增强对**近端未来 patch** 的关注（或远端负偏置）
- 让模型"相信趋势会继续下跌"

### 3.1.3 实验配置

```
对照组：原始 SimDiff（lambda=0）
实验组：Trend-Bias lambda in {0.1, 0.5, 1.0, 2.0}
```

**注意**：此步骤**不需要重新训练模型**。在推理阶段，加载原始权重，只修改 attention score 计算。

### 3.1.4 预期效果

- 若有效：CG_MAE 下降，TPR 上升，且效果随 lambda 增大而增强（直到过冲）
- 若无效：说明注意力偏置不足以克服训练阶段植入的保守性，必须修改训练目标

---

## Step 2：Historical Trend Residual（历史趋势残差校正）

### 3.2.1 修改点

不改 Denoiser 内部，只在**输出层后**加残差校正。

原始输出：
$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$$

修改后：
$$\hat{x}_0^{\text{corrected}} = \hat{x}_0 + \underbrace{\eta \cdot s^{\text{hist}} \cdot \Delta t}_{\text{趋势外推校正}}$$

其中：
- $\eta \in [0,1]$：校正系数（超参）
- $\Delta t$：预测步长（与 pred_len 成正比）
- $s^{\text{hist}}$：历史趋势斜率

### 3.2.2 物理意义

直接对 Denoiser 输出做**一阶泰勒外推**：
$$x(t + \Delta t) \approx x(t) + \frac{dx}{dt} \cdot \Delta t$$

当 Denoiser 因为 MSE 训练而保守估计时，外推项强制预测向趋势方向偏移。

### 3.2.3 实验配置

```
对照组：原始 SimDiff（eta=0）
实验组：Trend-Residual eta in {0.1, 0.3, 0.5, 0.8, 1.0}
```

**不需要重新训练**，纯推理时后处理。

### 3.2.4 预期效果

- 若有效：CG_MAE 显著下降，但可能整体 MAE 微升（自由流被过度外推）
- 若无效：说明历史斜率 $s^{\text{hist}}$ 不是正确的校正信号，或 Denoiser 保守性过强

---

## Step 3：Intra-Patch Trend Encoding（Patch 内趋势编码）

### 3.3.1 修改点

改 PatchEmbed，保留 patch 内部的趋势信息。

原始 PatchEmbed（平均池化）：
$$z_0^{(i)} = W_p \cdot \text{Mean}(x_{iP:(i+1)P})$$

修改后（均值+斜率双通道）：
$$\mu^{(i)} = \text{Mean}(x_{iP:(i+1)P})$$
$$\tau^{(i)} = \frac{x_{(i+1)P-1} - x_{iP}}{P-1}$$
$$z_0^{(i)} = W_p^{(\mu)} \cdot \mu^{(i)} + W_p^{(\tau)} \cdot \tau^{(i)}$$

其中 $\tau^{(i)}$ 是 patch 内部的线性趋势斜率。

### 3.3.2 物理意义

原始 PatchEmbed 把"从 200 涨到 600"的激波段压缩成一个均值 400 的向量，**丢失了"正在快速上升"的动态信息**。

双通道编码显式保留：
- $\mu^{(i)}$：patch 的平均水平
- $\tau^{(i)}$：patch 的变化方向与速度

Transformer 可以同时关注"当前多高"和"当前多快"。

### 3.3.3 实验配置

```
对照组：原始 PatchEmbed（单通道）
实验组：Trend-Aware PatchEmbed（双通道，dim 不变或各 dim/2）
```

**需要重新训练**（因为输入维度/投影矩阵改变）。

### 3.3.4 预期效果

- 若有效：激波区域的采样分散度增大（部分采样敢追高/追低），CG_MAE 下降
- 若无效：说明 Patch 化不是主因，或模型未学会利用 $\tau^{(i)}$

---

## Step 4：LWR Conservation Loss（物理守恒损失）

### 3.4.1 修改点

不改模型架构，只改训练目标，加入 LWR 守恒约束。

原始损失：
$$\mathcal{L}_{\text{MSE}} = \|\epsilon - \epsilon_\theta\|^2$$

修改后（多目标）：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{LWR}} \cdot \mathcal{L}_{\text{conserv}}$$

守恒损失定义：
$$\mathcal{L}_{\text{conserv}} = \left\|\frac{\partial \hat{q}}{\partial t} + \frac{\partial \hat{f}(q)}{\partial x}\right\|^2$$

离散形式（时空图）：
$$\mathcal{L}_{\text{conserv}} = \sum_{n,t} \left( \hat{q}_{n,t+1} - \hat{q}_{n,t} + \sum_{m \in \mathcal{N}_n} w_{m,n} (\hat{q}_{m,t} - \hat{q}_{n,t}) \right)^2$$

其中 $w_{m,n}$ 是基于路段距离的权重。

### 3.4.2 物理意义

MSE 损失让模型学习"统计上最可能的值"，但在激波处统计均值是错的。

LWR 损失强制模型学习"物理上守恒的演化"：
- 拥堵形成时，流量下降必须伴随上游到下游的传播
- 拥堵消散时，流量回升必须有明确的恢复波前
- 模型不能随意输出 flat 的保守预测，必须满足守恒律的时空结构

### 3.4.3 实验配置

```
对照组：原始 SimDiff（lambda_LWR=0）
实验组：LWR-Loss lambda_LWR in {0.01, 0.1, 1.0}
```

**需要重新训练**。

### 3.4.4 预期效果

- 若有效：采样不再全部保守，而是呈现与激波传播方向一致的分化
- 此时聚合器（如 DWA）将复活，因为采样终于有了结构性差异

---

## 四、实验评估矩阵

每一步实验必须记录：

| 指标 | 说明 | 目标 |
|------|------|------|
| 整体 MAE | 全样本平均 | 不劣化 > 5% |
| CG_MAE | 拥堵相 MAE | **显著下降**（< 30.98） |
| HOL_CG | 节假日-拥堵 MAE | **显著下降** |
| FF_MAE | 自由流 MAE | 不劣化 > 5% |
| TPR_CG | 拥堵相趋势保持率 | **向 0.473 靠近** |
| 采样分散度 | 10 条采样的 std | **增大**（证明打破保守性） |
| 激波区域采样覆盖率 | 真值落在 [min, max] 内的比例 | **提高** |

---

## 五、执行顺序与决策树

```
Week 1（推理时修改，无需重训）
├── Step 1: Trend-Biased Attention
│   └── 若 CG_MAE < 30.98 → 记录最优 lambda，进入 Step 2
│   └── 若 CG_MAE > 30.98 → 进入 Step 2（继续测试）
│
└── Step 2: Historical Trend Residual
    └── 若 CG_MAE < 30.98 → 找到轻量后处理方案，论文故事为"推理时校正"
    └── 若 CG_MAE > 30.98 → 进入 Week 2（必须改训练）

Week 2（训练时修改，需重训）
├── Step 3: Intra-Patch Trend Encoding
│   └── 若有效 → 论文故事为"Patch 化改进"
│   └── 若无效 → 进入 Step 4
│
└── Step 4: LWR Conservation Loss
    └── 若有效 → 论文故事为"物理引导扩散"
    └── 若无效 → 转向生成器 Guidance 方案
```

---

## 六、论文叙事线（基于渐进诊断）

### Paragraph 1：现象

> SimDiff 在交通流拥堵相出现系统性低估：所有采样聚集在历史均值附近，无法追踪激波峰值（Gap 达 226-308 veh/15min）。聚合器实验（MoM/Mean/Weighted）证明：问题不在聚合阶段，而在生成阶段——所有采样同向偏移。

### Paragraph 2：根因定位

> 我们提出三重保守性假说：(1) Softmax 注意力的凸组合平滑效应；(2) MSE 训练的条件均值陷阱；(3) Patch 平均池化对趋势信息的抹平。为精确定位主因，设计四步渐进注入实验。

### Paragraph 3：方法框架

> 从推理时轻量修改（Step 1-2：注意力偏置与趋势残差）到训练时结构修改（Step 3-4：Patch 趋势编码与 LWR 守恒损失），每一步独立验证，控制变量。

### Paragraph 4：预期贡献

> 实验将明确回答：交通流激波预测的保守性是否可通过轻量后处理修复，或必须从训练阶段引入物理约束。这为扩散模型在交通领域的适配提供了系统性的诊断方法论。

---

## 七、风险与备选

| 风险 | 判断标准 | 备选 |
|------|---------|------|
| Step 1-2 均无效 | CG_MAE 无改善 | 说明保守性已固化在训练权重中，必须重训（Step 3-4） |
| Step 3 有效但 Step 4 更优 | 两者都改善 | 论文聚焦 Step 4（物理损失），Step 3 作为消融 |
| Step 3-4 均无效 | 重训后 CG_MAE 仍 > 30.98 | 转向生成器 Guidance（Classifier-Free Guidance with Trend） |
| 自由流劣化严重 | FF_MAE 上升 > 10% | 加入自由流保护门控：当 $|s^{\text{hist}}| < \epsilon$ 时关闭所有校正 |

---

## 八、本周执行清单

### Day 1-2（Step 1）
- [ ] 实现 Trend-Biased Attention（修改 attention score，约 10 行）
- [ ] 测试 lambda in {0.1, 0.5, 1.0, 2.0}
- [ ] 记录完整评估矩阵

### Day 3-4（Step 2）
- [ ] 实现 Historical Trend Residual（输出层后处理，约 5 行）
- [ ] 测试 eta in {0.1, 0.3, 0.5, 0.8, 1.0}
- [ ] 记录完整评估矩阵

### Day 5（决策）
- [ ] 对比 Step 1-2 结果
- [ ] 若均无效：准备 Step 3-4 的重训代码
- [ ] 若任一有效：精细化搜索超参，准备论文图表

---

*文档结束*
