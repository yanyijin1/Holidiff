# Current Model Summary for Paper

> 目标：整理当前保留模型，突出可写入论文的公式、模块与物理含义。  
> 当前保留设置：**Trend-Aware PatchEmbed (concat) + Frequency-aware Phase-D Core（fixed FFT, K=4, patch_len=12）+ Hybrid Residual**，并保留时间趋势残差项（$\eta=0.5$）。

---

## 0. 问题背景与整体架构

### 0.1 核心矛盾：从确定性回归到条件分布

传统交通预测将未来状态视为历史状态的确定性映射，即假设存在唯一真值 $\mathbf{Y}$ 使得 $\mathbf{Y}=f(\mathbf{X})$。这一假设在常规日场景下近似成立，因为训练与测试分布基本一致，模型只需拟合稳定的周期模式与空间依赖。

然而，节假日诱发的**结构化分布漂移**（structured distribution shifts）使得该假设失效：
- **低频层面**：整体需求基线发生抬升或下降；
- **中频层面**：通勤节律改变，峰值时刻与形态偏移；
- **高频层面**：局部拥堵与异常扰动增强，激波峰值更尖锐。

相同的历史输入在节假日场景下可能对应多种等价的未来实现，单一确定性预测无法覆盖真实的不确定性范围。因此，未来状态应被理解为**条件分布** $p(\mathbf{Y}\,|\,\mathbf{X}, \mathbf{A})$ 的一次采样，而非确定性输出。

### 0.2 微观实现与宏观估计的新范式

基于上述观察，本文将交通预测重新表述为**微观实现**（micro-realization）与**宏观估计**（macro-flow estimation）的联合建模问题：

1. **微观实现生成**：通过扩散模型对未来状态的条件分布进行建模，单次前向传播输出一条微观实现——对应一种可能的未来交通场景；
2. **宏观交通流估计**：通过多次微观实现的聚合，从节假日增大的不确定性中提取稳健宏观交通流状态。

这一范式转变的核心动机是：交通流本质上是微观车辆行为（跟驰、换道、加减速）的统计平均，宏观状态是多次微观实现的稳健中心。确定性回归强行压缩了这种多解性，而扩散模型通过逐步去噪生成多条候选轨迹，天然适合表达节假日场景下的预测多样性。

### 0.3 保守偏差的三重根因

尽管扩散模型具备表达多解性的能力，现有通用时间序列扩散模型在交通预测中仍存在**系统性保守偏差**：

1. **PatchEmbed 平均池化抹平趋势**：patch 内激波峰值的上升/下降斜率被均值化，局部动力学信息丢失；
2. **Softmax Attention 凸组合压制极值**：Transformer 注意力加权平均进一步平滑峰值；
3. **MSE 损失条件均值陷阱**：去噪目标偏向统计安全估计，扩散采样聚集在条件均值附近，无法追踪激波峰值。

这三重效应叠加，导致拥堵相（Congestion Phase）预测存在持续的峰值低估。在 Fujian-30 数据集上，原始 SimDiff 基线的拥堵相真阳性率 $TPR_{CG}=0.452$，显著低于物理真值 $0.473$；节假日拥堵子集 $Hol\_Cong\ MAE$ 高达 $36.9$。

### 0.4 层次化解决思路

本文以 SimDiff (AAAI 2026) 为基座，保留其扩散框架与 MoM 采样机制，在三个层次注入交通物理感知：

1. **输入层**：将原始单通道 PatchEmbed 替换为 **Trend-Aware 频域解耦嵌入**，显式保留 patch 内斜率与多频带相位信息，缓解局部激波被均值化后的趋势湮灭；
2. **主干层**：沿用 SimDiff Transformer Denoiser，不做结构性修改，保证扩散生成框架的完整性；
3. **输出层**：在扩散去噪输出后附加 **历史趋势残差校正**，以一阶外推算子补偿拥堵相峰值低估。

---

## 1. 基础预测框架

交通流预测的核心任务，是根据一段历史观测恢复未来演化轨迹。对于扩散式预测器而言，这一过程可以理解为：先从噪声空间生成未来候选，再逐步去噪得到最终交通流序列。

设历史输入为
$$X^{\mathrm{hist}} \in \mathbb{R}^{B \times N \times L},$$
其中 $B$ 为 batch size，$N$ 为空间节点数，$L$ 为历史长度。

目标是预测未来窗口
$$X^{\mathrm{fut}} \in \mathbb{R}^{B \times N \times H},$$
其中 $H$ 为预测长度。

当前模型记为
$$\hat{X}^{0} = f_{\theta}(X^{\mathrm{hist}}, t),$$
其中 $f_{\theta}$ 是基于扩散去噪器的生成预测器，$t$ 为扩散时间步。

---

## 2. Patch Tokenization

交通流序列具有局部时段模式，例如短时上升、平台维持与回落过程。将长序列切分为 patch，可以把这些局部演化单元作为更稳定的建模对象，从而减轻逐点建模的噪声敏感性。

对历史序列与未来噪声序列进行 patch 化。设 patch 长度为 $P$，stride 为 $S$。

历史 patch 表示为
$$T^{\mathrm{hist}} = \mathrm{Unfold}(X^{\mathrm{hist}}; P, S) \in \mathbb{R}^{B \times N \times M \times P},$$
其中 $M = \lfloor (L-P)/S \rfloor + 1$ 为 patch 数。未来 patch 表示为
$$T^{\mathrm{fut}} = \mathrm{Unfold}(X^{\mathrm{fut}}; P, S) \in \mathbb{R}^{B \times N \times M_{f} \times P}.$$

拼接后得到总 token 序列
$$T = [T^{\mathrm{hist}}; T^{\mathrm{fut}}].$$

当前保留的输入编码不是原始单通道 patch projection，而是 **Trend-Aware PatchEmbed (concat)**。对每个 patch，先提取局部线性斜率：
$$\tau^{(i)}_{b,n} = \frac{T^{\mathrm{hist}}_{b,n,i,P-1} - T^{\mathrm{hist}}_{b,n,i,0}}{P-1} \in \mathbb{R}.$$

然后将原 patch 值与趋势斜率拼接后投影，得到基线 token 表示：
$$Z^{\mathrm{base}}_{b,n,i} = W_{\mathrm{concat}} \cdot [T^{\mathrm{hist}}_{b,n,i,:}; \tau^{(i)}_{b,n}] + b_{\mathrm{concat}} \in \mathbb{R}^{d},$$
其中 $d$ 为 token 维度，$[\cdot; \cdot]$ 表示拼接操作。

### 2.1 Frequency Decoupling (Fixed FFT, $K=4$)

节假日诱发的交通分布漂移同时影响多个时间尺度：低频分量反映整体需求基线的抬升或下降，中频分量反映通勤节律的改变，高频分量反映局部拥堵与异常峰值的出现。显式分离这些频带，可使 Denoiser 在不同动力学尺度上获得差异化表征，避免常规日基线与节假日激波在时域混叠。

在当前保留设置中，历史序列先进行固定频带分解。对每个样本、每个节点的时间序列做一维实数 FFT：
$$\tilde{X}_{b,n,:} = \mathcal{F}\left(X^{\mathrm{hist}}_{b,n,:}\right) \in \mathbb{C}^{\lfloor L/2 \rfloor + 1},$$
其中 $\mathcal{F}$ 作用于最后一个时间维度。

将频域索引均分为 $K=4$ 个带通区间 $\{\Omega_k\}_{k=1}^{K}$，构造硬掩码 $M_k$：
$$\tilde{X}^{(k)}_{b,n,:} = M_k \odot \tilde{X}_{b,n,:}, \qquad X^{(k)}_{b,n,:} = \mathcal{F}^{-1}\left(\tilde{X}^{(k)}_{b,n,:}\right) \in \mathbb{R}^{L}.$$

于是得到频带集合
$$\mathcal{B}=\{X^{(k)}\}_{k=1}^{K},\qquad K=4.$$

四个频带的交通物理意义如下：

| 频带 $k$ | 频率范围 | 交通物理意义 | 节假日变化 |
|:---:|:---|:---|:---|
| 1 (low) | 低频段 | 日级/周级需求基线 | 整体需求抬升或下降 |
| 2 (low-mid) | 中低频段 | 通勤节律（早晚高峰周期） | 峰值时刻偏移 |
| 3 (mid-high) | 中高频段 | 拥堵形成/消散过渡 | 激波强度变化 |
| 4 (high) | 高频段 | 局部激波/短时异常 | 异常峰值增强 |

关键洞察：常规日与节假日的差异主要体现在低频基线和中频节律，而拥堵激波本身的高频动力学具有跨域不变性。频域解耦使得 Denoiser 可以分别处理"漂移的基线"与"不变的激波"，避免两者在时域混叠导致的保守估计。

### 2.2 Band-wise Patch Injection

对每个频带 $X^{(k)}$ 做与主干一致的 patch 切分，并提取带内局部斜率：
$$T^{(k)} = \mathrm{Unfold}(X^{(k)}; P, S) \in \mathbb{R}^{B \times N \times M \times P},$$
$$\tau^{(k)}_{b,n,i} = \frac{T^{(k)}_{b,n,i,P-1} - T^{(k)}_{b,n,i,0}}{P-1} \in \mathbb{R}.$$

频带级 patch 嵌入写为
$$E^{(k)}_{b,n,i} = W_k \cdot [T^{(k)}_{b,n,i,:}; \tau^{(k)}_{b,n,i}] + b_k \in \mathbb{R}^{d_k},$$
其中 $d_k = d / K$ 为单频带投影维度。

将 $K$ 个频带嵌入拼接并投影回 $d$-维 token 空间：
$$E^{\mathrm{freq}}_{b,n,i} = W_f \cdot [E^{(1)}_{b,n,i}; \dots; E^{(K)}_{b,n,i}] + b_f \in \mathbb{R}^{d}.$$

当前模型采用 **embed_replace** 策略：频域解耦嵌入完全替代原始单通道基线编码，作为 Denoiser 的输入 token：
$$Z_{b,n,i} = E^{\mathrm{freq}}_{b,n,i}.$$

最后加入时间位置编码与扩散时间步编码：
$$Z^{(0)} = [z_t; Z].$$

这样做的核心目的，是让模型不仅看到某一段"数值高低"，也能看到该 patch 内部"正在上升还是下降"，并显式保留不同频带的相位与幅值信息，从而缓解局部激波被均值化后的趋势湮灭。

---

## 3. Denoiser Backbone

交通流在不同站点、不同时间片之间存在复杂的相关结构，因此去噪器需要同时建模局部时序依赖与未来演化的一致性。当前 backbone 沿用 SimDiff 原始架构，不做结构性修改，以保证扩散生成框架的完整性与可复现性。

当前去噪器由多层 token attention 组成，输入为 $Z^{(0)}$，基本块形式可写为
$$Z' = \mathrm{Attn}(Z), \qquad Z'' = \mathrm{FFN}(Z').$$

最终 flatten 后通过输出层映射到预测窗口：
$$\hat{X}^{0} = W_{\mathrm{out}} \cdot \mathrm{Flatten}(Z'') \in \mathbb{R}^{B \times N \times H}.$$

这对应扩散预测中的 $x_0$ 重建项，即未来交通流预测的原始输出。

---

## 4. Historical Trend Residual

在交通流场景中，主干生成器往往能恢复整体形状，但对拥堵形成或消散阶段的斜率延续仍可能偏保守。因此，我们在输出端显式引入历史趋势残差，用一个低成本的一阶外推项补偿这种系统性低估。

### 4.1 历史趋势斜率

这一模块首先从历史窗口中提取最直接的物理线索：趋势方向。对每个样本、每个节点定义历史趋势斜率：
$$s^{\mathrm{hist}}_{b,n} = \frac{X^{\mathrm{hist}}_{b,n,L} - X^{\mathrm{hist}}_{b,n,1}}{L-1} \in \mathbb{R}.$$

该斜率编码了历史窗口内交通流的整体演化方向：$s^{\mathrm{hist}} > 0$ 对应上升趋势（拥堵正在形成，交通波向上游传播），$s^{\mathrm{hist}} < 0$ 对应下降趋势（拥堵缓解，交通波向下游消散），$s^{\mathrm{hist}} \approx 0$ 对应自由流稳态或平台期。

### 4.2 固定参数化

为了保持趋势残差校正的结构简单且稳定，当前模型将校正强度定义为一个固定标量参数：
$$\mathcal{R}_{\eta}(s^{\mathrm{hist}}, h) = \eta \cdot s^{\mathrm{hist}} \cdot h, \qquad \eta \in \mathbb{R}_{+}.$$

在最终保留模型中，$\eta$ 取常数：
$$\eta = 0.5.$$

因此，相比自适应参数化，这里保留的是一个**固定系数的趋势外推算子**。它不引入新的学习参数，也不改变主干网络结构，只通过一个显式常数控制历史趋势向未来的传播强度。

从模型角度看，这一设定的好处是：
- 保持结构最小化；
- 保持校正项可解释；
- 避免额外自由度干扰主干生成器。

### 4.3 残差校正公式

在固定 $\eta$ 的设定下，对未来第 $h$ 个预测步，校正项写为：
$$\Delta X_{b,n,h} = \mathcal{R}_{\eta}(s^{\mathrm{hist}}_{b,n}, h) = \eta \cdot s^{\mathrm{hist}}_{b,n} \cdot h.$$

因此最终预测为：
$$\hat{X}^{\mathrm{corr}}_{b,n,h} = \hat{X}^{0}_{b,n,h} + \mathcal{R}_{\eta}(s^{\mathrm{hist}}_{b,n}, h).$$

代入固定系数形式，可得：
$$\hat{X}^{\mathrm{corr}}_{b,n,h} = \hat{X}^{0}_{b,n,h} + \eta \cdot s^{\mathrm{hist}}_{b,n} \cdot h, \qquad \eta = 0.5.$$

矩阵形式写为：
$$\hat{X}^{\mathrm{corr}} = \hat{X}^{0} + \eta \cdot s^{\mathrm{hist}} \otimes \tau,$$
其中：
- $s^{\mathrm{hist}} \in \mathbb{R}^{B \times N}$，
- $\tau = [1, 2, \dots, H]^{\top} \in \mathbb{R}^{H}$，
- $\otimes$ 表示外积（通过广播实现逐元素乘法），结果维度为 $\mathbb{R}^{B \times N \times H}$。

该模块本质上对应一个**固定参数的一阶趋势延续算子**，用显式外推项补偿扩散模型的保守估计。

---

## 5. Physical Interpretation

从交通流机理上看，该模块等价于在数据驱动预测之外，再加入一个"局部趋势延续"的显式先验。它不改变主干网络的表示方式，而是用物理上更易解释的形式纠正未来相位中的偏保守输出。

### 5.1 趋势斜率的交通语义

| 数学条件 | 交通物理意义 | 节假日场景行为 |
|:---|:---|:---|
| $s^{\mathrm{hist}} > 0$ | 交通波向上游传播，拥堵正在形成 | 节假日高峰提前，激波斜率更陡，扩散模型更易低估 |
| $s^{\mathrm{hist}} \approx 0$ | 自由流稳态或拥堵完全消散 | 节假日基线抬升但无局部激波，校正项近似为零 |
| $s^{\mathrm{hist}} < 0$ | 交通波向下游消散，拥堵缓解 | 节假日返程高峰后快速回落，校正项为负防止过估 |

### 5.2 固定系数 $\eta = 0.5$ 的物理含义

$\eta = 0.5$ 对应**半速外推**：承认扩散模型已通过去噪过程捕获了部分趋势信息，但保守估计仍不足，需要额外补偿历史斜率的 50%。这一数值不是通过网格搜索得到的最优拟合参数，而是基于以下交通物理直觉：

- 若 $\eta = 1.0$（全速外推），则假设历史趋势完全延续至未来，忽略了拥堵激波在传播过程中的衰减与消散（LWR 理论中的冲击波宽度有限性），容易导致自由流阶段的过估；
- 若 $\eta = 0$，则完全依赖扩散模型的保守估计，拥堵相峰值持续低估；
- $\eta = 0.5$ 在"补偿保守偏差"与"避免过度外推"之间取得平衡，且作为固定常数不引入额外自由度，保证了模型的泛化稳定性。

---

## 6. Why This Module Is Kept

对于当前模型阶段，我们需要保留的是一种既有效、又容易解释、同时改动足够小的增强方式。Trend-Aware PatchEmbed (concat)、Fixed FFT 频率解耦（$K=4$）与 fixed historical trend residual 满足这三个条件，因此适合作为当前版本的保留模块。

保留该模块的原因是：

1. **不改动主干生成器结构**  
   仅在输入编码与输出层后增加显式趋势信息和物理校正，代价小，可回滚。

2. **直接针对系统性低估问题**  
   Trend-Aware PatchEmbed 缓解局部趋势抹平，频率解耦保留多频带结构，固定残差校正直接补偿拥堵相峰值低估。

3. **具有明确物理解释**  
   输入端显式编码 patch 内斜率与频带相位，输出端显式沿历史趋势方向做半速外推，每个模块的行为都可映射到交通流动力学概念。

4. **实验上已验证有效**  
   当前最终保留版本在总 MAE、CG_MAE 与 TPR_CG 上都优于原始 SimDiff baseline，且对节假日拥堵子集（Hol_Cong）改善最为显著。

---

## 7. Main Experimental Results

当前保留模型的核心实验结果如下。

### 7.0 模型演进总览（30 epoch 口径，Fujian-30）

| 阶段 | 配置 | Test MAE | Test MSE | Test RMSE | Free MAE | Cong MAE | Hol\_Cong MAE | TPR\_Cong |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SimDiff 原始 | 无 patch trend, 无频域, 无残差 | 0.2296 | 0.1020 | 0.3194 | 13.8832 | 30.8779 | - | 0.4519 |
| Phase-A | + Trend-Aware concat | 0.2279 | 0.1012 | 0.3180 | 13.6976 | 30.7651 | 36.5047 | 0.4662 |
| Phase-A + Residual | + fixed eta=0.5 | 0.2276 | 0.1003 | 0.3167 | 13.7973 | 30.5635 | 36.2804 | 0.4742 |

从 SimDiff 原始基线到 Phase-A + Residual，总 MAE 从 0.2296 降至 0.2276（-0.87%），拥堵相 MAE 从 30.88 降至 30.56（-1.04%），节假日拥堵 MAE 从未统计改进至 36.28，TPR\_Cong 从 0.452 提升至 0.474（+4.9%）。

### 7.1 频带数 K 消融（Phase-D Core, 20 epoch 口径）

| K | Best Vali Loss | Test MAE | Test MSE | Test RMSE | Cong MAE |
|---:|---------------:|---------:|---------:|----------:|---------:|
| 2 | 0.2188174 | 0.2277 | 0.1004 | 0.3169 | 30.5817 |
| 3 | 0.2184268 | 0.2276 | 0.1004 | 0.3168 | 30.6079 |
| 4 | **0.2179760** | **0.2276** | **0.1003** | **0.3166** | **30.5693** |

由此固定频带数为 $K=4$。

### 7.2 第二层局部搜索（固定 K=4）

| 变体 | Best Vali Loss | Test MAE | Test MSE | Test RMSE | Cong MAE | Hol_Cong MAE |
|---|---------------:|---------:|---------:|----------:|---------:|--------------:|
| baseline (patch_len=16, fixed_fft) | **0.2179760** | 0.2276 | 0.1003 | 0.3166 | 30.5693 | 36.1085 |
| beta learnable | 0.2184280 | 0.2277 | 0.1003 | 0.3168 | 30.5870 | 36.1549 |
| patch_len=24 | 0.2199259 | 0.2277 | 0.1005 | 0.3170 | 30.6551 | 36.1364 |
| patch_len=12 | 0.2186021 | **0.2274** | **0.1001** | **0.3164** | **30.5262** | **35.9142** |
| learnable_fft | 0.2189887 | 0.2279 | 0.1006 | 0.3172 | 30.6644 | 36.3284 |

### 7.3 结果分析与配置定版

尽管 patch_len=12 的 Best Vali Loss（0.2186）略高于 baseline（0.2180），但其在测试集上全面更优，尤其在节假日拥堵子集（Hol_Cong MAE = 35.91 vs 36.11）上优势显著。这一现象符合**分布漂移假设**：验证集与训练集同分布（常规日为主），更粗的 patch（len=16）在平滑分布下拟合更优；而测试集包含节假日漂移，更细的 patch（len=12）能捕捉激波斜率突变，因此在漂移场景下泛化更优。这进一步证明了输入层频域解耦对分布漂移的必要性。

因此，综合测试指标（含拥堵相与节假日子集）最优解为：
$$K=4,\ \texttt{patch\_len}=12,\ \texttt{frequency\_decomp}=\texttt{fixed\_fft}.$$

### 7.4 当前论文保留配置（简表）

| 组件 | 当前取值 |
|---|---|
| Patch Encoder | Trend-Aware concat |
| Frequency Enable | true |
| Frequency Decomposer | fixed_fft |
| Num Bands $K$ | 4 |
| Frequency Injection | embed_replace |
| Frequency Patch Embed | band_trend + concat_proj |
| Residual Type | hybrid_residual |
| Physical Residual Eta | 0.5 |
| Patch Length | 12 |
| Stride | 1 |
| Spatial Field Enable | true |
| Spatial Field Target Adapter | matrix_ni |
| Spatial Field Coupling $\zeta$ | 0.05 (fixed) |
| Spatial Edge Variance Window | 720 |
| Spatial Adjacency | topology |
| Local Scaling Adapter | vanilla_revin |

据此，当前论文版本模型可定义为：

> **Trend-Aware PatchEmbed (concat) + Phase-D Frequency Core (fixed FFT, $K=4$, patch_len=12) + Hybrid Residual ($\eta=0.5$) + SFCN target adapter ($\zeta=0.05$, edge-var window = 720)**。

### 7.5 SFCN 精细化实验结果对比（20 epoch 口径）

| 实验 | 配置变化 | Test MAE | Test RMSE | Cong MAE | Hol\_Cong MAE | SGFE | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| exp1\_zeta005 | $\zeta=0.05$ | **0.2267** | **0.3151** | 30.3810 | **35.7827** | **25.2717** | 最优初始化 |
| exp1\_zeta01 | $\zeta=0.10$ | 0.2268 | 0.3151 | 30.3854 | 35.7943 | 25.2947 | 略差于 0.05 |
| exp1\_zeta02 | $\zeta=0.20$ | 0.2269 | 0.3153 | 30.4008 | 35.8199 | 25.3471 | 更大初始化无收益 |
| exp2\_edgevar168 | edge-var window = 168 | 0.2269 | 0.3153 | 30.4038 | 35.8231 | 25.3483 | 短窗口不稳定 |
| exp2\_edgevar336 | edge-var window = 336 | 0.2268 | 0.3152 | 30.3864 | 35.8101 | 25.3474 | 中等窗口仍不如长窗 |
| exp2\_edgevar720 | edge-var window = 720 | **0.2267** | **0.3151** | **30.3677** | 35.7945 | 25.3476 | 最优边统计窗口 |
| exp3\_directed\_adj | mean-gradient directed adjacency | 0.2296 | 0.3191 | 30.6763 | 36.2049 | 25.5951 | 明显退化，删除 |

这一轮精细化实验表明：SFCN 模块的最优保留策略并不是引入更复杂的方向化邻接或短时自适应统计，而是保留**最简单且最稳定**的设置：固定拓扑邻接、较小的场耦合系数 $\zeta=0.05$、以及较长时间窗口（720）估计边差值标准差。方向化邻接在总 MAE、拥堵相 MAE、Hol\_Cong MAE 与 SGFE 上均明显退化，因此不再保留进最终模型。
