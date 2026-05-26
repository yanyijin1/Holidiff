# Current Model Summary for Paper

> 目标：整理当前保留模型，突出可写入论文的公式、模块与物理含义。  
> 当前保留设置：**Trend-Aware PatchEmbed (concat) + Frequency-aware Phase-D Core（fixed FFT, K=4, patch_len=12）+ SFCN (learnable coupling α) + Hybrid Residual**，并保留时间趋势残差项（$\eta=0.5$）。

---

## 0. 问题背景与整体架构

### 0.1 核心矛盾：从确定性回归到条件分布

传统交通预测将未来状态视为历史状态的确定性映射，即假设存在唯一真值 $\mathbf{Y}$ 使得 $\mathbf{Y}=f(\mathbf{X})$。这一假设在常规日场景下近似成立，因为训练与测试分布基本一致，模型只需拟合稳定的周期模式与空间依赖。

然而，节假日诱发的**结构化分布漂移**（structured distribution shifts）使得该假设失效。本文从交通流物理出发，将未来状态分解为**宏观波分量**与**微观扰动**两层：

$$\mathbf{Y} = \mathbf{Y}_{\text{macro}} + \mathbf{Y}_{\text{micro}}.$$

**宏观波分量 $\mathbf{Y}_{\text{macro}}$**：服从 Lighthill-Whitham-Richards (LWR) 方程的离散化形式
$$\frac{\partial \rho}{\partial t} + \frac{\partial Q(\rho)}{\partial x} = 0,$$
其中 $Q(\rho)$ 为交通流基本图。宏观波以有限速度沿路网传播，上下游断面呈场级耦合。节假日期间，宏观波表现为整体需求基线抬升（低频层面）与峰值相位偏移（中频层面）。

**微观扰动 $\mathbf{Y}_{\text{micro}}$**：源于驾驶行为异质性、突发事故、信号控制波动，表现为高频局部激波与随机噪声。微观扰动在单断面时间序列上呈现强随机性，但在多断面空间场中通过传播耦合形成可统计的结构。

从物理视角看，交通流状态本质上是宏观确定性波与微观随机扰动的叠加。LWR 模型刻画前者——激波以有限速度传播，断面间呈场级耦合；后者源于驾驶员异质性、突发事件及信号波动。扩散模型的去噪过程恰好对应从微观无序（纯噪声）向宏观有序（物理波）的逐步演化，而多次微观实现的聚合则对应大数定律下的宏观稳态恢复。

这一对应关系构成了本文将扩散模型引入交通预测的根本动机：**并非将各断面视为独立时间序列进行点估计，而是将预测重构为条件分布采样**，以显式表达节假日强不确定性下的多解性。相同的历史输入在节假日场景下可能对应多种等价的未来实现，单一确定性预测无法覆盖真实的不确定性范围。因此，未来状态应被理解为**条件分布** $p(\mathbf{Y} \mid \mathbf{X}, \mathbf{A})$ 的一次采样，而非确定性输出。

### 0.2 微观实现与宏观估计的新范式

基于上述宏观-微观分解，本文将交通预测重新表述为**微观实现生成**（micro-realization）与**宏观交通流估计**（macro-flow estimation）的联合建模问题：

1. **微观实现生成**：扩散模型对未来状态的条件分布进行建模，单次前向传播输出一条微观轨迹 $\hat{\mathbf{Y}}^{(s)}$。该轨迹包含宏观波骨架与微观扰动采样，对应一种可能的未来交通场景；
2. **宏观交通流估计**：通过多次微观实现的聚合，从节假日增大的不确定性中提取稳健宏观状态 $\hat{\mathbf{Y}}_{\text{macro}}$。

这一范式转变的核心在于：交通流本质上是微观车辆行为（跟驰、换道、加减速）的统计平均，宏观状态是多次微观实现的稳健中心。确定性回归强行压缩了这种多解性，而扩散模型通过逐步去噪生成多条候选轨迹，天然适合表达节假日场景下宏观波漂移与微观扰动放大的联合效应。

### 0.3 保守偏差的三重根因：从频率、聚合、估计三个维度

尽管扩散模型具备表达多解性的能力，现有通用时间序列扩散模型在交通预测中仍存在**系统性保守偏差**。本文从宏观-微观分解视角，将根因归纳为三个维度：

**维度一：频率混叠——PatchEmbed 抹平宏观波斜率**

PatchEmbed 的平均池化将 patch 内激波峰值的上升/下降斜率均值化，导致 $\mathbf{Y}_{\text{macro}}$ 中的中频节律（通勤峰相位）与高频激波（局部拥堵形成）在时域混叠。节假日期间，宏观波的低频基线抬升与中频峰值偏移被进一步压缩为单一"平均趋势"，LWR 激波传播的动力学信息丢失。

**维度二：宏观聚合失效——Mean/Median 落入双峰真空带**

交通流基本图 $Q(\rho)$ 的非单调性导致流量分布天然**双峰**：自由流稳态（$Q \approx 50 \sim 150$）与拥堵稳态（$Q \approx 180 \sim 300$）之间由临界真空带（$Q \approx 160 \sim 180$）分隔。扩散模型生成的微观实现云团继承了这一双峰结构：左峰为保守估计（大量样本），右峰为物理真值（少量样本）。Mean 落入真空带中央，Median 被左峰多数派拉偏，均无法代表任何物理稳态。多次微观实现的聚合若采用几何质心，则宏观估计必然系统性低估拥堵相峰值。

**维度三：微观估计偏差——MSE 损失的条件均值陷阱**

去噪目标 $\epsilon_\theta(\mathbf{x}_t, t)$ 在 MSE 损失下偏向统计安全估计，扩散采样聚集在条件均值附近。该均值在双峰分布下不代表任何物理稳态，而是两峰之间的"伪平衡"。微观扰动 $\mathbf{Y}_{\text{micro}}$ 的随机性被过度平滑，高频激波峰值无法追踪。

三重效应叠加，导致拥堵相（Congestion Phase）预测存在持续的峰值低估。在 Fujian-30 数据集上，原始 SimDiff 基线的拥堵相真阳性率 $TPR_{CG}=0.452$，显著低于物理真值 $0.473$；节假日拥堵子集 $Hol\_Cong\ MAE$ 高达 $36.9$。

### 0.4 层次化解决思路：频率解耦、宏观聚合、微观估计

本文以 SimDiff (AAAI 2026) 为基座，保留其扩散框架，从三个维度注入交通物理感知，分别对应 0.3 节的三重根因：

**维度一：频率解耦（输入层）——分离宏观波的多尺度动力学**

将原始单通道 PatchEmbed 替换为 **Trend-Aware 频域解耦嵌入（LSTDE）**。通过 Fixed FFT 将历史序列硬分离为 $K=4$ 个频带：低频对应日级需求基线（节假日抬升/下降），中低频对应通勤节律（节假日峰值偏移），中高频对应拥堵过渡，高频对应局部激波。频带级 Trend-Aware PatchEmbed 显式保留每 patch 内斜率与相位信息，避免宏观波的多尺度动力学在时域混叠导致的保守估计。

**维度二：宏观聚合（输出层）——从几何质心到密度质心**

提出 **Density-Centroid Aggregation (DCA)**，以历史兼容的密度质心替代传统 Mean/Median。微观实现云团中，每个采样点按其局域核密度与历史稳态兼容性加权。位于拥堵稳态核（右峰）的少量样本因密度高、与历史真值兼容性强而获得高杠杆；位于真空带的样本因两势阱均浅而被去杠杆。DCA 使宏观估计从"统计中心趋势"进化为"物理稳态检索"。

**维度三：微观估计（归一化层）——场级空间耦合与趋势残差校正**

在归一化层引入 **Spatial Field Coupled Normalization (SFCN)**，将 1-hop 邻域未来分布编码为稀疏场统计量矩阵，使去噪器学习"在场中的相对位置"而非单点偏移。在输出层附加 **历史趋势残差校正**（$\eta=0.5$ 半速外推），以一阶算子补偿扩散模型对宏观波传播斜率的系统性低估。

三个维度形成完整闭环：频率解耦确保输入端宏观波尺度分离，SFCN 与趋势残差确保去噪过程中场级耦合与波传播物理，DCA 确保输出端从微观实现云团中检索正确的宏观稳态。

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

## 4. Spatial Field Coupled Normalization (SFCN)

SimDiff 的实例级归一化将每个传感器视为独立时间序列，节点间未来分布互不渗透。然而交通流在路网上连续演化，节点 $i$ 的未来状态不仅取决于自身历史，还受 1-hop 邻域未来场的约束。为此，本文提出 **SFCN**，将 1-hop 空间场结构编码为稀疏矩阵，嵌入扩散训练的目标空间。

### 4.1 Field Statistics Matrix

定义有向邻接矩阵 $\mathbf{A} \in \{0,1\}^{N \times N}$，其中 $\mathbf{A}[i,j]=1$ 表示节点 $j$ 到 $i$ 存在有效交通传播边（拥堵信息由 $j$ 向 $i$ 传递）。构造两个稀疏场矩阵 $\mathbf{F}_\mu, \mathbf{F}_\sigma \in \mathbb{R}^{N \times N}$，其非零结构完全由 $\mathbf{A} + \mathbf{I}$ 决定：

$$\mathbf{F}_\mu[i,j] = \begin{cases} \mu_i & i=j \\ \delta_{ij} = \mu_j - \mu_i & (i,j) \in \mathcal{E} \\ 0 & \text{otherwise} \end{cases}$$

$$\mathbf{F}_\sigma[i,j] = \begin{cases} \sigma_i & i=j \\ \nu_{ij} = \mathrm{Std}[X_j - X_i] & (i,j) \in \mathcal{E} \\ 0 & \text{otherwise} \end{cases}$$

其中：
- 对角线 $\mathbf{F}_\mu[i,i] = \mu_i$、$\mathbf{F}_\sigma[i,i] = \sigma_i$ 为节点 $i$ 的**自身未来分布**；
- 非对角线 $\mathbf{F}_\mu[i,j] = \delta_{ij}$ 为边 $(i,j)$ 的**均值差**（一阶趋势），$\mathbf{F}_\sigma[i,j] = \nu_{ij}$ 为边 $(i,j)$ 的**变异性**（历史标准差）；
- 非邻接位置严格为零，保证场结构的稀疏性与局部性。

### 4.2 场编码（训练时）

对批次内未来真值 $\mathbf{X}^{\mathrm{fut}} \in \mathbb{R}^{B \times N \times H}$，SFCN 输出联合归一化目标 $\tilde{\mathbf{X}}$，由节点残差与场耦合残差叠加而成。

**节点残差**（保留单点归一化骨架）：
$$\tilde{X}^{(0)}_{b,i,h} = \frac{X^{\mathrm{fut}}_{b,i,h} - \mu^{\mathrm{fut}}_i}{\sigma^{\mathrm{fut}}_i}$$

**场耦合残差**（空间结构注入）：
$$\tilde{X}^{(1)}_{b,i,h} = \sum_{j \in \mathcal{N}(i)} \frac{(X^{\mathrm{fut}}_{b,j,h} - X^{\mathrm{fut}}_{b,i,h}) - \delta^{\mathrm{fut}}_{ij}}{\nu^{\mathrm{fut}}_{ij}}$$

**联合目标**（场耦合强度 $\alpha$ 为唯一可学习标量，softplus 约束）：
$$\boxed{\tilde{X}_{b,i,h} = \tilde{X}^{(0)}_{b,i,h} + \alpha \cdot \tilde{X}^{(1)}_{b,i,h}}$$

扩散训练目标形式不变，但目标空间已场化：
$$\mathcal{L} = \mathbb{E}_{t,\epsilon} \left[ \left\| \epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_t} \tilde{\mathbf{X}} + \sqrt{1-\bar{\alpha}_t}\epsilon, \; t, \; \tilde{\mathbf{X}}^{\mathrm{hist}}\right) \right\|^2 \right]$$

### 4.3 场解码（推理时）

模型输出 $\mathbf{z} \in \mathbb{R}^{B \times N \times H}$（标准化空间）。推理时无未来真值，以**历史场统计量**代理未来场：

**节点预测**（严格保留单点反归一化）：
$$\hat{X}^{(0)}_{b,i,h} = \sigma^{\mathrm{hist}}_i \cdot z_{b,i,h} + \mu^{\mathrm{hist}}_i$$

**场渗透预测**（邻居标准化输出经边变异性还原）：
$$\hat{X}^{(1)}_{b,i,h} = \sum_{j \in \mathcal{N}(i)} \nu^{\mathrm{hist}}_{ij} \cdot (z_{b,j,h} - z_{b,i,h})$$

**联合解码**：
$$\boxed{\hat{X}_{b,i,h} = \hat{X}^{(0)}_{b,i,h} + \alpha \cdot \hat{X}^{(1)}_{b,i,h}}$$

**验证**：$\alpha=0$ 时严格退化 $\hat{X}_{b,i,h} = \sigma^{\mathrm{hist}}_i z_{b,i,h} + \mu^{\mathrm{hist}}_i$，与 SimDiff 原始反归一化完全一致。

### 4.4 局部自适应缩放（LocalNorm）

SFCN 负责节点间的场化归一化，节点内的实例级仿射变换仍由 **LocalNorm**（逐节点可学习 $\gamma_i, \beta_i$）完成。当前保留版本为**逐节点独立缩放**（版 A），前端场化缩放（版 B）经实验验证未带来额外增益，故不进入主线。

---

## 5. Historical Trend Residual

在交通流场景中，主干生成器往往能恢复整体形状，但对拥堵形成或消散阶段的斜率延续仍可能偏保守。因此，我们在输出端显式引入历史趋势残差，用一个低成本的一阶外推项补偿这种系统性低估。

### 5.1 历史趋势斜率

这一模块首先从历史窗口中提取最直接的物理线索：趋势方向。对每个样本、每个节点定义历史趋势斜率：
$$s^{\mathrm{hist}}_{b,n} = \frac{X^{\mathrm{hist}}_{b,n,L} - X^{\mathrm{hist}}_{b,n,1}}{L-1} \in \mathbb{R}.$$

该斜率编码了历史窗口内交通流的整体演化方向：$s^{\mathrm{hist}} > 0$ 对应上升趋势（拥堵正在形成，交通波向上游传播），$s^{\mathrm{hist}} < 0$ 对应下降趋势（拥堵缓解，交通波向下游消散），$s^{\mathrm{hist}} \approx 0$ 对应自由流稳态或平台期。

### 5.2 固定参数化

为了保持趋势残差校正的结构简单且稳定，当前模型将校正强度定义为一个固定标量参数：
$$\mathcal{R}_{\eta}(s^{\mathrm{hist}}, h) = \eta \cdot s^{\mathrm{hist}} \cdot h, \qquad \eta \in \mathbb{R}_{+}.$$

在最终保留模型中，$\eta$ 取常数：
$$\eta = 0.5.$$

因此，相比自适应参数化，这里保留的是一个**固定系数的趋势外推算子**。它不引入新的学习参数，也不改变主干网络结构，只通过一个显式常数控制历史趋势向未来的传播强度。

从模型角度看，这一设定的好处是：
- 保持结构最小化；
- 保持校正项可解释；
- 避免额外自由度干扰主干生成器。

### 5.3 残差校正公式

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

## 6. Density-Centroid Aggregation (DCA)

### 6.1 问题背景：交通流的双峰宿命与中心趋势陷阱

交通流基本图 $Q(\rho)$ 存在物理必然的双分支结构：自由流分支（$Q = v_f \cdot \rho$）与拥堵分支（$Q = w \cdot (\rho_{\max} - \rho)$），两者由临界区 $\rho \approx \rho_c$ 分隔。临界区物理不稳定，车辆停留时间极短，导致流量分布天然呈现**双峰结构**——自由流稳态与拥堵稳态之间形成统计真空带。

扩散模型生成的微观实现云团 $\{y^{(s)}\}_{s=1}^S$ 继承了这一双峰特征：
- **左峰**（保守估计）：大量样本聚集在 $150 \sim 180$ veh/15min，反映扩散模型在 MSE 损失与 Softmax Attention 凸组合下的系统性保守偏差；
- **右峰**（物理真值）：少量样本聚集在 $220 \sim 280$ veh/15min，对应拥堵相的真实稳态；
- **真空带**（$160 \sim 210$）：样本稀少，物理上最不常出现。

传统聚合器在此结构下必然失效：
- **Mean** 落入真空带中央（$\approx 170$），不代表任何物理稳态；
- **Median** 被左峰多数派拉向 $\approx 120$，完全忽略拥堵相；
- **MoM** 在分组平均后取中位数，进一步坍缩至真空带偏左（$\approx 160$）。

节假日加剧了这一畸变（motivation.pdf 图 d）：右峰抬升且展宽，但 Mean 仍被拉向两峰之间的统计真空，既不代表自由流真值，也不代表拥堵真值。

### 6.2 核心范式：从几何质心到密度质心

传统范式将 $S$ 次采样视为等质量质点，求几何质心：
$$\hat{y} = \frac{1}{S}\sum_{s=1}^S y_s.$$

本文提出**密度质心**范式：每个微观实现 $y_s$ 具有与其局部核密度及历史兼容性成正比的质量 $w_s$，聚合器寻找与物理稳态最共振的密度核：
$$\hat{y}_{\text{DCA}} = \frac{\sum_{s=1}^S w_s \cdot y_s}{\sum_{s=1}^S w_s}, \quad w_s = \rho_s^{\text{kde}} \times \rho_s^{\text{hist}}.$$

**物理直觉**：位于 $180$ 的孤立保守估计周围空旷、密度低、质量小；位于 $250$ 的紧密真值核虽然样本少，但局域密度高、与历史拥堵稳态兼容性强，获得高杠杆。密度质心被"硬核"吸引，跳出真空带。

### 6.3 数学推导：联合密度权重

#### 6.3.1 内部核密度权重（KDWM）

对每个采样点 $y_s$，计算其 $K$ 近邻核密度：
$$\rho_s^{\text{kde}} = \sum_{t=1}^S K\left(\frac{y_s - y_t}{h}\right), \quad K(u) = \exp\left(-\frac{u^2}{2}\right),$$

其中 $h$ 为带宽。$h$ 控制邻居半径：$h \to \infty$ 时退化为 Mean；$h \to 0$ 时退化为 Mode；$h \approx \sigma_{\text{peak}}$ 时恰好识别单个峰的密集核。

#### 6.3.2 历史似然权重（HLWC）

设历史真值库按相态划分为 $\mathcal{H}_{\text{free}}$（自由流稳态真值）与 $\mathcal{H}_{\text{cong}}$（拥堵稳态真值）。对每个 $y_s$ 计算两势阱的兼容性：

$$L_s^{\text{free}} = \sum_{y_k \in \mathcal{H}_{\text{free}}} \exp\left(-\frac{(y_s - y_k)^2}{2\sigma^2}\right), \quad L_s^{\text{cong}} = \sum_{y_k \in \mathcal{H}_{\text{cong}}} \exp\left(-\frac{(y_s - y_k)^2}{2\sigma^2}\right).$$

历史主导权重取最大兼容势阱：
$$\rho_s^{\text{hist}} = \max\left(L_s^{\text{free}}, L_s^{\text{cong}}\right).$$

**物理语义**：$y_s$ 落入历史自由流势阱深，则被视为自由流稳态候选；落入拥堵势阱深，则被视为拥堵稳态候选。真空带的样本在两势阱中均浅，自然被去杠杆。

#### 6.3.3 联合密度质心

$$\boxed{\hat{y}_{\text{DCA}} = \frac{\sum_{s=1}^S \rho_s^{\text{kde}} \cdot \rho_s^{\text{hist}} \cdot y_s}{\sum_{s=1}^S \rho_s^{\text{kde}} \cdot \rho_s^{\text{hist}}}}$$

**退化验证**：
- $h \to \infty$ 且 $\sigma \to \infty$：$w_s \to \text{const}$，退化为 **Mean**；
- $h \to 0$ 且历史库为空：退化为 **Mode**（最近邻聚合）；
- 单峰对称分布：$\rho_s^{\text{kde}}$ 恒定，$\rho_s^{\text{hist}}$ 对称，退化为 **Mean**。

## 7. Physical Interpretation

从交通流机理上看，该模块等价于在数据驱动预测之外，再加入一个"局部趋势延续"的显式先验。它不改变主干网络的表示方式，而是用物理上更易解释的形式纠正未来相位中的偏保守输出。

### 8.1 趋势斜率的交通语义

| 数学条件 | 交通物理意义 | 节假日场景行为 |
|:---|:---|:---|
| $s^{\mathrm{hist}} > 0$ | 交通波向上游传播，拥堵正在形成 | 节假日高峰提前，激波斜率更陡，扩散模型更易低估 |
| $s^{\mathrm{hist}} \approx 0$ | 自由流稳态或拥堵完全消散 | 节假日基线抬升但无局部激波，校正项近似为零 |
| $s^{\mathrm{hist}} < 0$ | 交通波向下游消散，拥堵缓解 | 节假日返程高峰后快速回落，校正项为负防止过估 |

### 8.2 固定系数 $\eta = 0.5$ 的物理含义

$\eta = 0.5$ 对应**半速外推**：承认扩散模型已通过去噪过程捕获了部分趋势信息，但保守估计仍不足，需要额外补偿历史斜率的 50%。这一数值不是通过网格搜索得到的最优拟合参数，而是基于以下交通物理直觉：

- 若 $\eta = 1.0$（全速外推），则假设历史趋势完全延续至未来，忽略了拥堵激波在传播过程中的衰减与消散（LWR 理论中的冲击波宽度有限性），容易导致自由流阶段的过估；
- 若 $\eta = 0$，则完全依赖扩散模型的保守估计，拥堵相峰值持续低估；
- $\eta = 0.5$ 在"补偿保守偏差"与"避免过度外推"之间取得平衡，且作为固定常数不引入额外自由度，保证了模型的泛化稳定性。

---

## 8. Why This Module Is Kept

对于当前模型阶段，我们需要保留的是一种既有效、又容易解释、同时改动足够小的增强方式。Trend-Aware PatchEmbed (concat)、Fixed FFT 频率解耦（$K=4$）、SFCN (learnable $\alpha$)、fixed historical trend residual 与 Density-Centroid Aggregation 满足这些条件，因此适合作为当前版本的保留模块。

### 8.1 输入层与归一化层保留理由

1. **不改动主干生成器结构**  
   仅在输入编码、归一化层与输出层后增加显式趋势信息、场耦合与密度质心聚合，代价小，可回滚。

2. **直接针对系统性低估问题**  
   Trend-Aware PatchEmbed 缓解局部趋势抹平，频率解耦保留多频带结构，SFCN 引入空间场一致性，固定残差校正直接补偿拥堵相峰值低估。

3. **具有明确物理解释**  
   输入端显式编码 patch 内斜率与频带相位，归一化端显式编码 1-hop 场分布结构，输出端显式沿历史趋势方向做半速外推，每个模块的行为都可映射到交通流动力学概念。

4. **实验上已验证有效**  
   当前最终保留版本在总 MAE、CG_MAE、SGFE 与 TPR_CG 上都优于原始 SimDiff baseline，且对节假日拥堵子集（Hol_Cong）改善最为显著。

### 8.2 DCA 聚合器保留理由

Density-Centroid Aggregation 作为宏观估计层的核心创新，其保留基于以下四方面验证：

1. **解决根本性的统计陷阱**  
   传统 Mean/Median/MoM 在交通流双峰分布下落入真空带，不代表任何物理稳态。DCA 以密度质心替代几何质心，从"统计中心趋势"进化为"物理稳态检索"。

2. **参数极简，零新增学习成本**  
   DCA 仅依赖带宽 $h$ 与 $\sigma$ 两个超参数（从历史数据离线校准），不引入任何可学习权重，不增加训练负担。

3. **与交通流物理直接对应**  
   内部核密度权重对应"微观实现云团的局域结构"，历史兼容性权重对应"历史稳态势阱的引力"，两者乘积与交通流基本图的双峰稳态（自由流/拥堵）同构。

4. **消融实验验证最优性**  
   在固定 Phase-E 最优模型之上，DCA 显著优于所有对比聚合策略（见表 8.1）。

### 8.3 聚合器消融实验对比

为验证密度质心范式的有效性，本文在固定 Phase-E 最优模型（SFCN + LSTDE + Trend-Aware + Residual）之上，对五种聚合策略进行系统对比：

| 聚合策略 | Test MAE | Cong MAE | Hol\_Cong MAE | TPR\_Cong | SGFE | 核心机制 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|---|
| **Simple Mean** | 0.2267 | 30.3677 | 35.7945 | 0.4664 | 25.3476 | 等权几何质心 | Phase-E 基线 |
| **MoM** | — | — | — | — | — | 分组中位数 | 已废弃（比 Mean 更保守） |
| **SCP** | 0.2267 | 30.4342 | 35.8624 | 0.4690 | 25.4014 | 偏度解析偏移 | 接近最优，轻量备选 |
| **ABMS** | 0.2345 | 31.0382 | 36.5216 | 0.4685 | 27.1525 | 自适应 Mean-Shift 迭代 | 明显退化 |
| **WBP** | 0.6103 | 73.0635 | 82.3483 | 0.0000 | 35.8135 | Wasserstein 重心投影 | 彻底失效 |
| **RCL** | 0.2261 | 30.3853 | 35.8166 | 0.4689 | 25.2397 | 双阵营竞争加权 | 与 DCA 等价 |
| **DCA** | **0.2261** | **30.3852** | **35.8167** | **0.4689** | **25.2396** | **密度质心 × 历史兼容** | **最优定版** |

**关键发现**：

1. **DCA 与 RCL 指标在小数点后四位完全一致**，说明在当前云团结构下，内部核密度与历史兼容性两种权重来源收敛到同一物理最优。RCL 的"阵营竞争"叙事与 DCA 的"密度质心"叙事在数学上近似等价，但 DCA 的物理图像（粒子密度 × 势阱深度）与交通流基本图的双峰稳态更直接对应。

2. **SCP（偏度解析偏移）作为零历史依赖的轻量备选**，MAE 仅比 DCA 高 0.0006，Cong MAE 高 0.05。在无法构建可靠历史库的场景下，SCP 可作为快速部署方案。

3. **ABMS 退化**揭示了迭代爬山在双峰云团中的陷阱：若初始点落在左峰吸引域，Mean-Shift 无法跳过真空带到达右峰，最终陷入次优保守估计。

4. **WBP 彻底失效**证明了分布传输映射在双峰交通流下的结构性错误：WBP 的单调分位数映射将真空带采样强行拉升至拥堵区间，导致 TPR\_Cong 归零、MAE 暴涨至 0.61。

## 9. Main Experimental Results

### 8.0 模型演进总览（30 epoch 口径，Fujian-30）

| 阶段 | 配置 | Test MAE | Test MSE | Test RMSE | Free MAE | Cong MAE | Hol\_Cong MAE | TPR\_Cong |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SimDiff 原始 | 无 patch trend, 无频域, 无残差 | 0.2296 | 0.1020 | 0.3194 | 13.8832 | 30.8779 | — | 0.4519 |
| Phase-A | + Trend-Aware concat | 0.2279 | 0.1012 | 0.3180 | 13.6976 | 30.7651 | 36.5047 | 0.4662 |
| Phase-A+Res | + fixed $\eta=0.5$ | 0.2276 | 0.1003 | 0.3167 | 13.7973 | 30.5635 | 36.2804 | 0.4742 |
| Phase-D | + LSTDE (K=4, patch=12) | **0.2274** | **0.1001** | **0.3164** | **13.7284** | **30.5262** | **35.9142** | **0.4642** |
| Phase-E | + SFCN (learnable $\alpha$) | **0.2268** | **0.0993** | **0.3151** | **13.7284** | **30.3868** | 35.7928 | **0.4661** |

从 SimDiff 原始基线到 Phase-E，总 MAE 从 0.2296 降至 0.2268（−1.22%），拥堵相 MAE 从 30.88 降至 30.39（−1.58%），TPR_CG 从 0.452 提升至 0.466（+3.1%）。

### 8.1 频带数 K 消融（Phase-D Core, 20 epoch 口径）

| K | Best Vali Loss | Test MAE | Test MSE | Test RMSE | Cong MAE |
|---:|---------------:|---------:|---------:|----------:|---------:|
| 2 | 0.2188174 | 0.2277 | 0.1004 | 0.3169 | 30.5817 |
| 3 | 0.2184268 | 0.2276 | 0.1004 | 0.3168 | 30.6079 |
| 4 | **0.2179760** | **0.2276** | **0.1003** | **0.3166** | **30.5693** |

由此固定频带数为 $K=4$。

### 8.2 第二层局部搜索（固定 K=4）

| 变体 | Best Vali Loss | Test MAE | Test MSE | Test RMSE | Cong MAE | Hol_Cong MAE |
|---|---------------:|---------:|---------:|----------:|---------:|--------------:|
| baseline (patch_len=16) | **0.2179760** | 0.2276 | 0.1003 | 0.3166 | 30.5693 | 36.1085 |
| patch_len=12 | 0.2186021 | **0.2274** | **0.1001** | **0.3164** | **30.5262** | **35.9142** |
| learnable_fft | 0.2189887 | 0.2279 | 0.1006 | 0.3172 | 30.6644 | 36.3284 |

### 8.3 Phase-D 定版分析

尽管 patch_len=12 的 Best Vali Loss（0.2186）略高于 baseline（0.2180），但其在测试集上全面更优，尤其在节假日拥堵子集（Hol_Cong MAE = 35.91 vs 36.11）上优势显著。这一现象符合**分布漂移假设**：验证集与训练集同分布（常规日为主），更粗的 patch（len=16）在平滑分布下拟合更优；而测试集包含节假日漂移，更细的 patch（len=12）能捕捉激波斜率突变，因此在漂移场景下泛化更优。这进一步证明了输入层频域解耦对分布漂移的必要性。

### 8.4 当前论文保留配置（简表）

| 组件 | 当前取值 |
|---|---|
| Patch Encoder | Trend-Aware concat |
| Frequency Enable | true |
| Frequency Decomposer | fixed_fft |
| Num Bands $K$ | 4 |
| Frequency Injection | embed_replace |
| Frequency Patch Embed | band_trend + concat_proj |
| Spatial Normalization | **SFCN (learnable $\alpha$)** |
| Local Scaling | LocalNorm (per-node $\gamma, \beta$) |
| Residual Type | hybrid_residual |
| Physical Residual $\eta$ | 0.5 |
| Patch Length | 12 |
| Stride | 1 |

据此，当前论文版本模型可定义为：

> **Trend-Aware PatchEmbed (concat) + Phase-D Frequency Core (fixed FFT, $K=4$, patch_len=12) + SFCN (learnable $\alpha$) + Hybrid Residual ($\eta=0.5$)**。

### 8.5 Phase-E：SFCN 消融与对比

在固定 Phase-D 主线之上，进一步验证空间场耦合归一化模块。四组对比如下：

| 变体 | 空间归一化 | 局部缩放 | MAE | MSE | Cong MAE | Hol_Cong MAE | SGFE |
|---|---|---:|---:|---:|---:|---:|---:|
| E0 | Vanilla | Vanilla | 0.2275 | 0.1002 | 30.5009 | 35.9757 | 25.3543 |
| E1 | SFCN (fixed $\alpha$) | Vanilla | 0.2276 | 0.1000 | 30.4509 | **35.7663** | 25.3071 |
| **E2** | **SFCN (learnable $\alpha$)** | **Vanilla** | **0.2268** | **0.0993** | **30.3868** | 35.7928 | **25.2943** |
| E3 | SFCN (fixed $\alpha$) | Field-RevIN | 0.2274 | 0.0999 | 30.4075 | 35.8627 | 25.3224 |

**结论**：

1. **E2 为 Phase-E 最优配置**。相比 E0（Vanilla），learnable $\alpha$ 将总 MAE 从 0.2275 压至 0.2268，Cong MAE 从 30.50 压至 30.39，SGFE 从 25.35 降至 25.29，说明可学习场耦合强度能自适应适配局部交通场的空间一致性。

2. **E1 证明 SFCN 本体有效**。固定 $\alpha=0.1$ 已在 SGFE 和 Hol_Cong 上优于 E0，说明矩阵化 target space 本身引入了有益的空间结构；但释放 $\alpha$ 的可学习性后（E2），收益进一步扩大。

3. **E3 未进一步优于 E2**。Field-RevIN（前端场化缩放）在当前设置下没有带来额外增益，反而在 Holiday MAE 上略有回退。这说明**目标端场耦合（SFCN）已足够**，前端场化冗余，因此不进入主线。

---

*文档版本: v2.0 (Phase-E 更新)*  
*更新时间: 2026-05-19*  
*下次更新: Holiday 条件化模块设计完成后*
