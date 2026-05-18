# 阶段性转向：从外部补丁到 Denoiser 结构修改

> **定位**：后处理天花板确认 + Denoiser 最小侵入性修改方案  
> **核心判断**：eta 后处理总 MAE=0.2281 是外部补丁上限，要突破 CG_MAE 必须修改模型内部算子  
> **原则**：一次只改一个组件，控制变量，用数学公式定义修改，不写代码

---

## 一、阶段性结论：外部补丁的天花板

### 1.1 已验证的边界

| 方向 | 总 MAE | CG_MAE | 本质 | 评价 |
|------|--------|--------|------|------|
| SimDiff 原始 | 0.2296 | 30.88 | 基线 | 保守 |
| 固定 eta=1.0 | $\sim$0.2288 | 30.69 | 外部加法校正 | 后处理最优 |
| adaptive eta v1 | **0.2281** | 30.83 | 外部特征驱动校正 | 总 MAE 最优 |

**关键事实**：
- adaptive eta 总 MAE=0.2281 是外部补丁的全局最优
- 但其 CG_MAE=30.83 甚至不如固定 eta=1.0 的 30.69
- 说明外部校正做的是"全局折中"：牺牲拥堵相精度换取自由流精度
- **没有任何外部后处理能让 CG_MAE < 30.5**

### 1.2 为什么必须改结构

外部后处理的数学极限：

$$\hat{Y}_{\text{corrected}} = \hat{Y}_{\text{pred}} + \eta(s) \cdot s \cdot \Delta t$$

- $\hat{Y}_{\text{pred}}$ 是 Denoiser 输出的条件均值
- 如果条件均值本身系统性偏离真值 $\delta$，外部加法只能补偿部分 $\delta$
- 当 $\eta$ 过大时，自由流被污染；当 $\eta$ 过小时，拥堵相补偿不足
- **加法校正的上限由 Denoiser 的采样分布决定，无法超越**

要降低 CG_MAE，必须让 Denoiser 本身产生**不那么保守的采样分布**——即减小 $\delta$，而非在 $\delta$ 存在后补偿。

---

## 二、Denoiser 保守性的结构性根因

SimDiff Denoiser 由三个组件构成，每个组件都贡献保守性：

### 根因 1：PatchEmbed 的趋势抹平

原始 PatchEmbed：

$$z^{(i)} = W_p \cdot \frac{1}{P}\sum_{j=0}^{P-1} x_{iP+j}$$

**问题**：一个 patch 内可能包含激波段（如 $200 \to 600$），平均池化后变成 $400$。Transformer 看不到"正在飙升"，只能看到"这一段均值较高"。

**后果**：模型对激波的感知是静态的、片段化的，缺乏动态趋势信息。

### 根因 2：Softmax Attention 的凸组合平滑

原始 Attention Score：

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

**问题**：softmax 输出是 $V$ 的凸组合（加权平均）。当历史出现激波峰值时，attention 把峰值与周围正常值平均，输出被拉回到历史分布的常见范围。

**后果**：所有隐层表示都被"均值回归"效应驯化，没有任何位置敢突破历史统计范围。

### 根因 3：MSE 损失的条件均值陷阱

原始训练目标：

$$\mathcal{L} = \mathbb{E}\left[\|\epsilon - \epsilon_\theta\|^2\right]$$

**问题**：MSE 最优解是条件期望 $\mathbb{E}[x_0 \mid x_t]$。在激波处，条件分布 $p(x_0 \mid x_t)$ 是偏态的（拥堵形成时真值可能继续下跌，也可能反弹），而均值天然偏向保守（被反弹分支拉高）。

**后果**：模型被训练成"统计上最安全的估计"，而非"物理上最可能的演化"。

---

## 三、最小侵入性修改方案（两步递进）

### 原则

- 一次只改一个组件
- 其他所有权重、超参保持不变
- 每个方案独立验证，不累积
- 用 10 epoch 快速验证收敛趋势，再决定是否跑满 30 epoch

---

## 方案一：Trend-Aware PatchEmbed（最小侵入，只改输入层）

### 3.1 修改公式

原始（单通道均值）：

$$\mu^{(i)} = \frac{1}{P}\sum_{j=0}^{P-1} x_{iP+j}, \quad z_0^{(i)} = W_p \cdot \mu^{(i)}$$

修改后（双通道：均值 + 趋势）：

$$\mu^{(i)} = \frac{1}{P}\sum_{j=0}^{P-1} x_{iP+j}, \quad \tau^{(i)} = \frac{x_{(i+1)P-1} - x_{iP}}{P - 1}$$

$$z_0^{(i)} = W_\mu \cdot \mu^{(i)} + W_\tau \cdot \tau^{(i)}$$

其中 $\tau^{(i)}$ 是第 $i$ 个 patch 内部的线性趋势斜率。

### 3.2 物理意义

- $\mu^{(i)}$：patch 的"静态水平"（模型原本就能看到的）
- $\tau^{(i)}$：patch 的"动态方向"（模型原本看不到的）

激波上升沿的 patch：$\mu=400$（中等），$\tau=+200$（飙升）

自由流稳态的 patch：$\mu=1200$（高），$\tau=+5$（平缓）

Transformer 同时接收"当前多高"和"当前多快"，能区分"高但平缓"与"中等但飙升"。

### 3.3 预期效果

- 激波区域的采样分散度增大（部分采样敢追高/追低）
- 此时后处理 $\eta=1.0$ 可能复活（因为采样有了结构性分化）
- CG_MAE 有望突破 30.69

### 3.4 验证设计

- 对照组：原始 SimDiff（单通道 PatchEmbed）
- 实验组：Trend-Aware PatchEmbed（双通道，dim 不变或各 dim/2）
- 训练：继承其他权重，只重新初始化 $W_\mu$ 和 $W_\tau$，跑 10 epoch
- 评估：CG_MAE、采样分散度、TPR_CG

---

## 方案二：Trend-Conditional Attention（中等侵入，改核心算子）

### 3.5 修改公式

原始 Attention：

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

修改后（趋势条件偏置）：

$$A = \text{softmax}\left(\frac{QK^T + \lambda \cdot \phi \cdot M_{\text{causal}}}{\sqrt{d_k}}\right)$$

其中：
- $\phi \in [-1, 1]$：由历史趋势强度归一化得到的相位标量
- $M_{\text{causal}} \in \mathbb{R}^{L \times L}$：因果趋势掩码
  - $M_{\text{causal}}[i,j] = \dfrac{j - i}{L}$  当 $j > i$（未来位置越远，偏置越强）
  - $M_{\text{causal}}[i,j] = 0$  当 $j \leq i$（不回顾过去）
- $\lambda > 0$：偏置强度（超参，建议初值 $0.1$-$1.0$）

### 3.6 物理意义

当 $\phi > 0$（历史上升趋势）：
- attention 被迫增强对远端未来 patch 的关注
- 打破 softmax 的均匀回顾，让模型"相信趋势会延续"

当 $\phi < 0$（历史下降趋势）：
- attention 被迫增强对近端未来 patch 的关注（或远端负偏置）
- 让模型"相信趋势会继续下跌"

当 $\phi \approx 0$（稳态）：
- 偏置项趋近于 $0$，退化为原始 attention

### 3.7 与之前推理注入的本质区别

之前 Step 1 在推理时注入趋势偏置，对已训练权重无效，因为：
- 权重已固化在"保守模式"，偏置不足以克服固化映射
- 相当于在近视眼镜上加放大镜，镜片本身仍是近视的

现在是在**训练阶段**引入偏置：
- 权重从头学习利用趋势信息
- attention 机制学会"何时该看远端，何时该看近端"
- 这是结构性适应，而非外部补丁

### 3.8 验证设计

- 对照组：原始 SimDiff（$\lambda=0$）
- 实验组：Trend-Conditional Attention（$\lambda \in \{0.1, 0.5, 1.0\}$）
- 训练：完整重训 30 epoch（因为 attention 是核心算子，权重需重新学习）
- 评估：CG_MAE、采样分散度、TPR_CG、总 MAE

---

## 四、执行优先级

```
Week 1
├── 方案一：Trend-Aware PatchEmbed（10 epoch 快速验证）
│   └── 若 CG_MAE < 30.5 → 最小侵入方案成功，论文聚焦输入编码改进
│   └── 若 CG_MAE > 30.8 → 输入编码不是主因，进入方案二
│
Week 2
└── 方案二：Trend-Conditional Attention（30 epoch 完整训练）
    └── 若 CG_MAE < 30.5 → 核心算子改进成功，论文聚焦趋势感知注意力
    └── 若 CG_MAE > 30.8 → attention 偏置不是主因，进入方案三（MSE 损失修改）
```

---

## 五、论文叙事（结构修改导向）

### Paragraph 1：外部补丁的天花板

> 我们通过系统实验验证了推理阶段趋势校正的有效性边界：固定 $\eta=1.0$ 将总 MAE 降至 0.2281，但 CG_MAE 仅能从 30.88 降至 30.69，无法突破 30.5。自适应校正网络（adaptive $\eta$）虽然总 MAE 最优，但其"自适应"退化为全局折中（FF 与 CG 的 $\eta$ 差异仅 0.01），无法精确识别激波相位。这表明：外部加法校正的上限由 Denoiser 的采样分布决定，要彻底修复激波保守性，必须修改模型内部算子。

### Paragraph 2：保守性的三重结构性根因

> 诊断分析揭示 SimDiff Denoiser 的三重保守性机制：(1) PatchEmbed 的平均池化抹平 patch 内趋势，使激波段丢失动态信息；(2) Softmax Attention 的凸组合平滑效应，将峰值与周围值平均，输出被拉回历史常见范围；(3) MSE 训练目标的最优解是条件均值，而激波处的偏态分布使均值天然保守。

### Paragraph 3：最小侵入修改方案

> 基于上述诊断，我们提出两步递进式结构修改：第一步，Trend-Aware PatchEmbed，将 patch 内线性趋势斜率 $\tau$ 显式编码为第二通道，让 Transformer 同时感知"当前多高"与"当前多快"；第二步，Trend-Conditional Attention，在训练阶段引入历史趋势相位 $\phi$ 对 attention score 的因果偏置，让模型学会"激波延续时看远端，稳态时均匀看"。两步修改独立验证，控制变量，精确定位保守性的主因。

---

## 六、风险与备选

| 风险 | 判断标准 | 备选 |
|------|---------|------|
| PatchEmbed 有效但幅度小 | CG_MAE 降至 30.5-30.6 | 结合方案二，双重修改 |
| Attention 偏置导致训练不稳定 | loss 发散或 nan | 调小 $\lambda$，或改用 $\tanh$ 替代 softmax 前的偏置 |
| 两者均无效 | CG_MAE 仍 > 30.8 | 进入根本修改：MSE 损失 $\to$ 多模态/物理损失 |
| 总 MAE 劣化 | 总 MAE > 0.2300 | 回退到后处理最优配置（$\eta=1.0$）作为最终方案 |

---

*文档结束*
