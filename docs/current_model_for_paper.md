# Current Model Summary for Paper

> 目标：整理当前保留模型，突出可写入论文的公式、模块与物理含义。  
> 当前保留设置：**Trend-Aware PatchEmbed (concat) + fixed Historical Trend Residual (\(\eta=0.5\))**，其总 MAE 从 0.2296 下降到 0.2276。

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

当前保留的输入编码不是原始单通道 patch projection，而是 **Trend-Aware PatchEmbed (concat)**。对每个 patch，额外提取局部线性斜率：
\[
\tau^{(i)} = \frac{x_{(i+1)P-1} - x_{iP}}{P-1}.
\]

然后将原 patch 值与趋势斜率拼接后投影：
\[
Z = W_{\mathrm{concat}} [T; \tau] + b_{\mathrm{concat}}.
\]

同时加入时间 token：
\[
Z^{(0)} = [z_t; Z].
\]

这样做的核心目的，是让模型不仅看到某一段“数值高低”，也能看到该 patch 内部“正在上升还是下降”，从而缓解局部激波被均值化后的趋势湮灭。

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

## 5. Fixed Eta Parameterization

为了保持趋势残差校正的结构简单且稳定，当前模型不再对 \(\eta\) 进行额外学习，而是将其定义为一个固定标量参数。于是，趋势校正模块可写为：

\[
\mathcal{R}_{\eta}(s^{\mathrm{hist}}, h) = \eta \cdot s^{\mathrm{hist}} \cdot h,
\qquad \eta \in \mathbb{R}_{+}.
\]

在最终保留模型中，\(\eta\) 取常数：
\[
\eta = 0.5.
\]

因此，相比自适应参数化，这里保留的是一个**固定系数的趋势外推算子**。它不引入新的学习参数，也不改变主干网络结构，只通过一个显式常数控制历史趋势向未来的传播强度。

从模型角度看，这一设定的好处是：
- 保持结构最小化；
- 保持校正项可解释；
- 避免额外自由度干扰主干生成器。

---

## 6. Residual Correction Formula

在固定 \(\eta\) 的设定下，历史趋势残差校正的定义非常直接。对未来第 \(h\) 个预测步，校正项写为：
\[
\Delta X_{b,h,n} = \mathcal{R}_{\eta}(s^{\mathrm{hist}}_{b,n}, h) = \eta \cdot s^{\mathrm{hist}}_{b,n} \cdot h.
\]

因此最终预测为：
\[
\hat{X}^{\mathrm{corr}}_{b,h,n} = \hat{X}^{0}_{b,h,n} + \mathcal{R}_{\eta}(s^{\mathrm{hist}}_{b,n}, h).
\]

代入固定系数形式，可得：
\[
\hat{X}^{\mathrm{corr}}_{b,h,n} = \hat{X}^{0}_{b,h,n} + \eta \cdot s^{\mathrm{hist}}_{b,n} \cdot h,
\qquad \eta = 0.5.
\]

矩阵形式写为：
\[
\hat{X}^{\mathrm{corr}} = \hat{X}^{0} + \eta \cdot s^{\mathrm{hist}} \odot \tau,
\]
其中：
- \(s^{\mathrm{hist}} \in \mathbb{R}^{B \times N}\)
- \(\tau = [1,2,\dots,H] \in \mathbb{R}^{H}\)
- \(\odot\) 表示 broadcast 后的逐元素乘法

该模块本质上对应一个**固定参数的一阶趋势延续算子**，用显式外推项补偿扩散模型的保守估计。

---

## 7. Physical Interpretation

从交通流机理上看，该模块等价于在数据驱动预测之外，再加入一个“局部趋势延续”的显式先验。它不改变主干网络的表示方式，而是用物理上更易解释的形式纠正未来相位中的偏保守输出。

该模块的物理含义是：

- 若历史斜率 \(s^{\mathrm{hist}} > 0\)，则说明局部交通流处于上升趋势；
- 若 \(s^{\mathrm{hist}} < 0\)，则说明局部交通流处于下降趋势；
- 固定的 \(\eta=0.5\) 决定这种趋势延续被补偿到何种程度。

因此，该方案不是让模型额外学习一个复杂校正器，而是通过一个稳定的一阶外推系数，在不破坏主干生成结构的前提下，对拥堵相峰值低估做适度补偿。

---

## 8. Why This Module Is Kept

对于当前模型阶段，我们需要保留的是一种既有效、又容易解释、同时改动足够小的增强方式。Trend-Aware PatchEmbed (concat) 与 fixed historical trend residual 满足这三个条件，因此适合作为当前版本的保留模块。

保留该模块的原因是：

1. **不改动主干生成器结构**  
   仅在输入编码与输出层后增加显式趋势信息和物理校正，代价小。

2. **直接针对系统性低估问题**  
   Trend-Aware PatchEmbed 缓解局部趋势抹平，固定残差校正直接补偿拥堵相峰值低估。

3. **具有明确物理解释**  
   输入端显式编码 patch 内斜率，输出端显式沿历史趋势方向做一阶外推。

4. **实验上已验证有效**  
   当前最终保留版本在总 MAE、CG_MAE 与 TPR_CG 上都优于原始 baseline。

---

## 9. Main Experimental Results

当前保留模型的核心实验结果如下。

### 9.1 PatchEmbed 消融

| 变体 | Epoch | 总 MAE | CG_MAE | TPR_CG |
|------|------:|-------:|-------:|-------:|
| 原始 PatchEmbed | 30 | 0.2296 | 30.8779 | 0.4519 |
| Trend-Aware add | 30 | 0.2286 | 30.7527 | 0.4512 |
| Trend-Aware concat | 30 | **0.2279** | **30.6623** | **0.4552** |

Trend-Aware PatchEmbed（concat）显著优于 add，并将 CG_MAE 从 30.88 降至 30.66。

### 9.2 固定趋势残差校正

| \(\eta\) | 总 MAE | FF_MAE | CG_MAE | TPR_CG |
|----------:|-------:|-------:|-------:|-------:|
| 0.0 | 0.2279 | 13.7777 | 30.6623 | 0.4552 |
| 0.3 | 0.2284 | 13.8420 | 30.8726 | 0.4510 |
| 0.5 | **0.2276** | 13.7973 | **30.5635** | **0.4742** |
| 1.0 | 0.2288 | 13.8933 | 30.8652 | 0.4544 |

固定 \(\eta=0.5\) 在综合指标上最优，是当前最终保留设置。

### 9.3 联合最优配置

| 配置 | 总 MAE | FF_MAE | CG_MAE | TPR_CG |
|------|-------:|-------:|-------:|-------:|
| SimDiff 原始 | 0.2296 | 13.8832 | 30.8779 | 0.4519 |
| Trend-Aware concat | 0.2279 | 13.7777 | 30.6623 | 0.4552 |
| concat + \(\eta=0.5\) | **0.2276** | 13.7973 | **30.5635** | **0.4742** |

因此，当前论文版本的最终模型应定义为：

> **Trend-Aware PatchEmbed (concat) + fixed Historical Trend Residual (\(\eta=0.5\))**。
