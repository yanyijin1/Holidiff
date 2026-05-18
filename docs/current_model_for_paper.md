# Current Model Summary for Paper

> 目标：整理当前保留模型，突出可写入论文的公式、模块与物理含义。  
> 当前保留设置：**Adaptive Historical Trend Residual（init = 0.5）**，其总 MAE 从 0.2296 下降到 0.2281。

---

## 1. 基础预测框架

交通流预测的核心任务，是根据一段历史观测恢复未来演化轨迹。对于扩散式预测器而言，这一过程可以理解为：先从噪声空间生成未来候选，再逐步去噪得到最终交通流序列。

设历史输入为
\[
X^{\mathrm{hist}} \in \mathbb{R}^{B \times L \times N},
\]
其中 \(B\) 为 batch size，\(L\) 为历史长度，\(N\) 为节点数。

目标是预测未来窗口
\[
X^{\mathrm{fut}} \in \mathbb{R}^{B \times H \times N},
\]
其中 \(H\) 为预测长度。

当前模型记为
\[
\hat{X}^{0} = f_{\theta}(X^{\mathrm{hist}}, t),
\]
其中 \(f_{\theta}\) 是基于扩散去噪器的生成预测器，\(t\) 为扩散时间步。

---

## 2. Patch Tokenization

交通流序列具有局部时段模式，例如短时上升、平台维持与回落过程。将长序列切分为 patch，可以把这些局部演化单元作为更稳定的建模对象，从而减轻逐点建模的噪声敏感性。

对历史序列与未来噪声序列进行 patch 化。设 patch 长度为 \(P\)，stride 为 \(S\)。

历史 patch 表示为
\[
T^{\mathrm{hist}} = \mathrm{Unfold}(X^{\mathrm{hist}}; P, S),
\]
未来 patch 表示为
\[
T^{\mathrm{fut}} = \mathrm{Unfold}(X^{\mathrm{fut}}; P, S).
\]

拼接后得到总 token 序列
\[
T = [T^{\mathrm{hist}}; T^{\mathrm{fut}}].
\]

每个 patch 通过线性映射投影到隐空间：
\[
Z = W_{\mathrm{in}} T + b_{\mathrm{in}}.
\]

同时加入时间 token：
\[
Z^{(0)} = [z_t; Z].
\]

---

## 3. Denoiser Backbone

交通流在不同站点、不同时间片之间存在复杂的相关结构，因此去噪器需要同时建模局部时序依赖与未来演化的一致性。当前 backbone 的作用就是在 token 空间中恢复这些结构，并输出原始未来预测。

当前去噪器由多层 token attention 组成，基本块形式可写为
\[
Z' = \mathrm{Attn}(Z),
\]
\[
Z'' = \mathrm{FFN}(Z').
\]

最终 flatten 后通过输出层映射到预测窗口：
\[
\hat{X}^{0} = W_{\mathrm{out}} \cdot \mathrm{Flatten}(Z'').
\]

这对应扩散预测中的 \(x_0\) 重建项，即未来交通流预测的原始输出。

---

## 4. Historical Trend Residual

在交通流场景中，主干生成器往往能恢复整体形状，但对拥堵形成或消散阶段的斜率延续仍可能偏保守。因此，我们在输出端显式引入历史趋势残差，用一个低成本的外推项补偿这种系统性低估。

### 4.1 历史趋势斜率

这一模块首先从历史窗口中提取最直接的物理线索：趋势方向、波动强度与当前水平。它们分别对应“是否在上升/下降”“当前是否稳定”“当前处于什么流量区间”。

对每个样本、每个节点定义历史趋势斜率：
\[
s^{\mathrm{hist}}_{b,n} = \frac{X^{\mathrm{hist}}_{b,L,n} - X^{\mathrm{hist}}_{b,1,n}}{L-1}.
\]

同时定义历史波动率：
\[
\sigma^{\mathrm{hist}}_{b,n} = \mathrm{Std}(X^{\mathrm{hist}}_{b,:,n}).
\]

以及历史末端水平：
\[
\ell^{\mathrm{hist}}_{b,n} = X^{\mathrm{hist}}_{b,L,n}.
\]

---

## 5. Adaptive Eta Parameterization

不同交通状态对趋势外推的需求并不一致：自由流稳态通常只需很弱校正，而拥堵形成或强波动阶段往往需要更强的趋势延续。因此，校正强度不再设为全局常数，而是由当前历史状态自适应决定。

当前保留方案为样本/节点级自适应校正系数：
\[
\eta_{b,n} = g_{\phi} \big(
|s^{\mathrm{hist}}_{b,n}|,
\sigma^{\mathrm{hist}}_{b,n},
\ell^{\mathrm{hist}}_{b,n}
\big).
\]

其中 \(g_{\phi}\) 是一个两层 MLP：
\[
g_{\phi}(u) = \eta_{\max} \cdot \sigma\Big(W_2 \, \mathrm{ReLU}(W_1 u + b_1) + b_2\Big),
\]
其中：
- \(u \in \mathbb{R}^{3}\)
- \(\sigma(\cdot)\) 为 sigmoid
- \(\eta_{\max}\) 为最大校正强度

因此
\[
\eta_{b,n} \in [0, \eta_{\max}].
\]

当前实现中：
- hidden dim = 16
- 输入维度 = 3
- 输出经 sigmoid 后映射到 \([0, \eta_{\max}]\)

---

## 6. Residual Correction Formula

有了自适应 \(\eta\) 后，模型就可以将历史趋势转化为未来预测中的显式补偿项。该补偿项沿预测 horizon 线性累积，对交通流中的持续上升或持续回落过程尤其有效。

对未来第 \(h\) 个预测步，残差校正写为：
\[
\Delta X_{b,h,n} = \eta_{b,n} \cdot s^{\mathrm{hist}}_{b,n} \cdot h.
\]

因此最终预测为：
\[
\hat{X}^{\mathrm{corr}}_{b,h,n} = \hat{X}^{0}_{b,h,n} + \eta_{b,n} \cdot s^{\mathrm{hist}}_{b,n} \cdot h.
\]

矩阵形式写为：
\[
\hat{X}^{\mathrm{corr}} = \hat{X}^{0} + \eta \odot s^{\mathrm{hist}} \odot \tau,
\]
其中：
- \(\eta \in \mathbb{R}^{B \times N}\)
- \(s^{\mathrm{hist}} \in \mathbb{R}^{B \times N}\)
- \(\tau = [1,2,\dots,H] \in \mathbb{R}^{H}\)
- \(\odot\) 表示 broadcast 后的逐元素乘法

这本质上是一个**节点级、样本级、一阶趋势外推修正项**。

---

## 7. Physical Interpretation

从交通流机理上看，该模块等价于在数据驱动预测之外，再加入一个“局部趋势延续”的显式先验。它不改变主干网络的表示方式，而是用物理上更易解释的形式纠正未来相位中的偏保守输出。

该模块的物理含义是：

- 若历史斜率 \(s^{\mathrm{hist}} > 0\)，则说明局部交通流处于上升趋势；
- 若 \(s^{\mathrm{hist}} < 0\)，则说明局部交通流处于下降趋势；
- \(\eta\) 决定这种趋势延续应被强化到何种程度。

因此：
\[
\eta \approx 0
\]
表示几乎不校正；而
\[
\eta \gg 0
\]
表示对趋势延续进行显式外推。

在交通场景下，这意味着：
- 自由流稳态样本应学习到较小 \(\eta\)
- 拥堵形成/消散样本应学习到较大 \(\eta\)

---

## 8. Why This Module Is Kept

对于当前模型阶段，我们需要保留的是一种既有效、又容易解释、同时改动足够小的增强方式。Adaptive historical trend residual 满足这三个条件，因此适合作为当前版本的保留模块。

保留该模块的原因是：

1. **不改动主干生成器结构**  
   仅在输出层后增加显式物理校正，代价小。

2. **直接针对系统性低估问题**  
   若 Denoiser 对拥堵相峰值存在保守偏差，则外部趋势残差可直接补偿。

3. **具有明确物理解释**  
   校正项直接来源于历史斜率，而不是黑盒偏置。

4. **实验上已验证有效**  
   当前保留版本在总 MAE 上优于原始 baseline：
   \[
   0.2296 \rightarrow 0.2281.
   \]
