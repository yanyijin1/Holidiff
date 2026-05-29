# Current Model Summary for Paper（论文终版对齐版）

> 目标：将当前代码实现收敛到论文终版 HoliDiff 主线，明确哪些模块进入论文、哪些只作为消融或 legacy 代码保留。  
> 论文主线：**Localized Spectral-Temporal Decoupling Embedding (LSTDE) + Spatial Field Coupled Normalization (SFCN) + Conditional Diffusion Micro-Realization Generation + Hybrid Trend Residual Correction + Density-Centroid Aggregation (DCA)**。

---

## 0. 论文终版核心设定

### 0.1 核心问题

本文研究的是**节假日诱导的结构化交通流分布漂移**。节假日并不是简单改变平均流量，而是同时改变：

- 日内需求基线；
- 峰值相位与拥堵持续时间；
- 节点间影响强度；
- 频带能量分布；
- 自由流、拥堵与过渡状态的边缘分布结构。

因此，HoliDiff 不应被写成一个“带节假日标签的普通预测模型”，而应被写成：

> 在节假日结构化漂移下，先生成多个可能的未来交通微观实现，再从采样点云中恢复具有交通状态代表性的宏观流量预测。

### 0.2 最终保留模块

| 模块 | 论文名称 | 代码建议名称 | 是否进入主线 | 作用 |
|---|---|---|---|---|
| 频域分解 + 趋势 patch 嵌入 | LSTDE | `FixedFFTDecomposer` + `TrendAwarePatchEmbedding` | 是 | 分离需求基线、峰值节律、局部扰动 |
| 空间场耦合归一化 | SFCN | `SpatialFieldCoupledNorm` | 是 | 将 1-hop 拓扑邻域分布结构引入扩散目标空间 |
| 条件扩散生成 | Micro-realization generation | `ConditionalDenoiser` / `DiffusionSampler` | 是 | 多次采样得到微观交通实现 |
| 趋势残差校正 | Hybrid Trend Residual Correction | `HybridTrendResidual` | 是 | 补偿扩散预测对峰值斜率的保守估计 |
| 密度质心聚合 | DCA | `DensityCentroidAggregator` | 是 | 从微观实现云团恢复宏观交通流状态 |

### 0.3 不进入主模型的内容

以下内容不应出现在 HoliDiff 主模型 forward 路径中：

- `holiday_flag` / holiday condition 作为模型输入；
- `aggregation_factory` 中的大量可插拔聚合器；
- ABMS、WBP、RCL 等早期聚合器；
- spectral gate、conditioner、modulator 等早期频域实验模块；
- 复杂动态图学习、连续 distance/cost 边权、OSM 重构边权；
- 高维轨迹聚类式聚合。

其中，`holiday label` 仍然保留在 **data/eval/metrics** 中，用于：

- 跨数据集节假日漂移分析；
- Fujian-30 节假日细粒度评估；
- Hol-MAE、Hol-CG MAE、HDR、PTE 等指标计算。

它不作为 HoliDiff 主模型训练阶段的监督或条件输入。

---

## 1. 问题定义

设高速路交通网络为

\[
G=(V,E,A), \qquad A \in \{0,1\}^{N\times N},
\]

其中 \(N=|V|\) 为传感器节点数，\(A\) 为二值邻接矩阵。给定历史窗口

\[
X \in \mathbb{R}^{B\times N\times L},
\]

目标是预测未来窗口

\[
Y \in \mathbb{R}^{B\times N\times H}.
\]

节假日标签 \(c\in\{0,1\}\) 仅用于分布漂移分析与评估，不作为模型主输入。本文关注的条件分布漂移为

\[
P(Y\mid X,G,c=1) \neq P(Y\mid X,G,c=0).
\]

HoliDiff 将预测过程解释为：

\[
\{\hat{Y}^{(s)}\}_{s=1}^{S} \sim p_\theta(Y\mid X,G),
\]

其中每个 \(\hat{Y}^{(s)}\) 是同一历史条件下的一种**微观交通实现**。最终预测不是简单平均，而是通过 DCA 得到宏观流状态估计：

\[
\hat{Y}_{\mathrm{macro}} = \mathrm{DCA}\left(\{\hat{Y}^{(s)}\}_{s=1}^{S}; X\right).
\]

---

## 2. HoliDiff 主流程

最终主模型应保持一条清晰路径：

```text
X, A
  └── LSTDE
        ├── Fixed FFT decomposition
        └── Trend-aware patch embedding
  └── SFCN field statistics
        ├── training: future-field encoding
        └── inference: history-field decoding
  └── Conditional diffusion denoising
        └── S samples of micro-realizations
  └── Hybrid trend residual correction
  └── DCA macro-flow aggregation
        └── final prediction
```

对应代码主路径建议为：

```python
class HoliDiff(nn.Module):
    def __init__(self, cfg):
        self.decomposer = FixedFFTDecomposer(num_bands=cfg.num_bands)
        self.patch_embed = TrendAwarePatchEmbedding(patch_len=cfg.patch_len)
        self.sfcn = SpatialFieldCoupledNorm(alpha=cfg.sfcn.alpha)
        self.denoiser = ConditionalDenoiser(cfg)
        self.diffusion = DiffusionSampler(cfg)
        self.residual = HybridTrendResidual(eta=cfg.residual.eta)
        self.aggregator = DensityCentroidAggregator(cfg.dca)

    def forward_train(self, x, y, adj):
        cond = self.encode_history(x)
        y_field = self.sfcn.encode(y, adj)
        loss = self.diffusion.training_loss(y_field, cond)
        return loss

    @torch.no_grad()
    def sample(self, x, adj, num_samples):
        cond = self.encode_history(x)
        z_samples = self.diffusion.sample(cond, num_samples)
        y_samples = self.sfcn.decode(z_samples, x, adj)
        y_samples = self.residual(y_samples, x)
        y_hat = self.aggregator(y_samples, x)
        return y_hat, y_samples
```

主路径中不再保留可插拔 factory，不再在模型内部根据字符串动态切换 ABMS/WBP/RCL/MoM 等聚合器。

---

## 3. LSTDE：局部谱时序解耦嵌入

### 3.1 目的

节假日分布漂移同时发生在多个时间尺度：

- 低频：需求基线变化；
- 中频：峰值节律与相位偏移；
- 高频：局部拥堵、短时突增和激波扰动。

LSTDE 的作用是将这些混合在原始时域中的变化解耦，使去噪器能够分别接收不同尺度的条件信息。

### 3.2 Fixed FFT decomposition

对每个节点的历史序列执行一维实数 FFT：

\[
\tilde{X}_{b,n,:}=\mathcal{F}(X_{b,n,:}).
\]

将频域索引划分为 \(K\) 个固定频带，当前论文终版取：

\[
K=4.
\]

第 \(k\) 个频带为

\[
\tilde{X}^{(k)}_{b,n,:}=M_k\odot \tilde{X}_{b,n,:},
\qquad
X^{(k)}_{b,n,:}=\mathcal{F}^{-1}(\tilde{X}^{(k)}_{b,n,:}).
\]

得到频带集合：

\[
\mathcal{B}=\{X^{(k)}\}_{k=1}^{K}.
\]

### 3.3 Trend-aware patch embedding

对每个频带序列切分 patch。设 patch 长度为 \(P\)，当前论文终版取：

\[
P=12.
\]

第 \(k\) 个频带的 patch 表示为

\[
T^{(k)}=\mathrm{Unfold}(X^{(k)};P,S).
\]

对每个 patch 提取局部趋势斜率：

\[
\tau^{(k)}_{b,n,m}
=
\frac{T^{(k)}_{b,n,m,\mathrm{end}}-T^{(k)}_{b,n,m,\mathrm{start}}}{P-1}.
\]

将 patch 原始值与斜率拼接后投影：

\[
E^{(k)}_{b,n,m}
=
W_k\left[T^{(k)}_{b,n,m,:};\tau^{(k)}_{b,n,m}\right]+b_k.
\]

融合所有频带：

\[
Z_{b,n,m}=F_{\mathrm{fuse}}(E^{(1)},\ldots,E^{(K)}).
\]

代码中对应：

```text
frequency/decompose.py      -> FixedFFTDecomposer
frequency/patch_embed.py    -> TrendAwarePatchEmbedding
```

不再进入主线的早期频域模块：

```text
frequency/conditioner.py
frequency/modulator.py
frequency/fusion.py
frequency/spectral_gate.py
```

这些模块若暂时需要保留，应移动到 `legacy/frequency/`。

---

## 4. SFCN：空间场耦合归一化

### 4.1 目的

SFCN 不是学习复杂边权，也不是使用 distance/cost 作为连续边权。它只利用二值邻接矩阵 \(A\) 定义 1-hop 拓扑邻域，将节点自身分布与邻域相对差分结构编码到扩散目标空间。

所有数据集统一使用二值邻接矩阵：

\[
A_{ij}=1 \quad \text{if an edge from sensor } i \text{ to sensor } j \text{ is provided.}
\]

自环在数据图中去除；SFCN 内部需要节点自身统计量时通过对角统计项处理。

### 4.2 场统计量

构造场统计量集合

\[
\Theta_{\mathrm{field}}=\{F_\mu,F_\sigma,\alpha\}.
\]

其中 \(F_\mu,F_\sigma\in\mathbb{R}^{N\times N}\) 的非零结构由 \(A\) 与对角项共同决定：

\[
F_\mu[i,j]
=
\begin{cases}
\mu_i, & i=j,\\
\delta_{ij}=\mu_j-\mu_i, & (i,j)\in E,\\
0, & \text{otherwise},
\end{cases}
\]

\[
F_\sigma[i,j]
=
\begin{cases}
\sigma_i, & i=j,\\
\nu_{ij}=\mathrm{Std}(X_j-X_i), & (i,j)\in E,\\
0, & \text{otherwise}.
\end{cases}
\]

### 4.3 训练时场编码

训练阶段使用未来真值统计量构造归一化目标。节点残差为

\[
\tilde{Y}^{(0)}_{b,i,h}
=
\frac{Y_{b,i,h}-\mu^{\mathrm{fut}}_i}{\sigma^{\mathrm{fut}}_i}.
\]

场耦合残差使用度归一化邻接权重：

\[
\tilde{Y}^{(1)}_{b,i,h}
=
\sum_{j\in\mathcal{N}(i)}
\frac{A_{ij}}{\mathrm{deg}(i)}
\cdot
\frac{(Y_{b,j,h}-Y_{b,i,h})-\delta^{\mathrm{fut}}_{ij}}{\nu^{\mathrm{fut}}_{ij}}.
\]

联合场化目标为

\[
\tilde{Y}_{b,i,h}
=
\tilde{Y}^{(0)}_{b,i,h}
+
\alpha\cdot \tilde{Y}^{(1)}_{b,i,h}.
\]

其中 \(\alpha>0\) 为场耦合强度。当前代码可使用 learnable \(\alpha\)，并通过 softplus 或 clamp 保证非负。

### 4.4 推理时场解码

推理阶段未来统计量不可知，因此使用历史观测统计量代理未来场结构。节点预测为

\[
\hat{Y}^{(0)}_{b,i,h}
=
\sigma^{\mathrm{hist}}_i z_{b,i,h}+\mu^{\mathrm{hist}}_i.
\]

场渗透项为

\[
\hat{Y}^{(1)}_{b,i,h}
=
\sum_{j\in\mathcal{N}(i)}
\frac{A_{ij}}{\mathrm{deg}(i)}
\cdot
\nu^{\mathrm{hist}}_{ij}(z_{b,j,h}-z_{b,i,h}).
\]

最终解码为

\[
\hat{Y}_{b,i,h}
=
\hat{Y}^{(0)}_{b,i,h}
+
\alpha\cdot \hat{Y}^{(1)}_{b,i,h}.
\]

当 \(\alpha=0\) 时，SFCN 退化为单点归一化/反归一化。

代码中对应：

```text
micro/sfcn.py -> SpatialFieldCoupledNorm
```

---

## 5. Conditional Diffusion Micro-Realization Generation

扩散模型在场化空间中学习条件未来分布。令 \(x_0=\tilde{Y}\)，前向扩散为

\[
q(x_t\mid x_0)
=
\mathcal{N}\left(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I\right).
\]

去噪网络学习噪声估计：

\[
\mathcal{L}
=
\mathbb{E}_{t,\epsilon}
\left[\left\|\epsilon-
\epsilon_\theta(x_t,t,Z)
\right\|_2^2\right].
\]

训练后，对同一历史输入执行 \(S\) 次采样：

\[
\{\hat{Y}^{(s)}\}_{s=1}^{S}, \qquad s=1,\ldots,S.
\]

当前论文终版推荐：

\[
S=10.
\]

每一条 \(\hat{Y}^{(s)}\) 被解释为一个微观交通实现，而不是普通可交换统计样本。

---

## 6. Hybrid Trend Residual Correction

### 6.1 目的

扩散模型容易产生均值回归式预测，在峰值上升沿或拥堵形成阶段偏保守。趋势残差校正用于补偿这种斜率低估。

### 6.2 时间趋势残差

历史趋势斜率为

\[
s^{\mathrm{hist}}_{b,n}
=
\frac{X_{b,n,L}-X_{b,n,1}}{L-1}.
\]

时间趋势外推为

\[
\Delta Y^{\mathrm{time}}_{b,n,h}
=
\eta\cdot s^{\mathrm{hist}}_{b,n}\cdot h,
\qquad h=1,\ldots,H.
\]

当前论文终版默认：

\[
\eta=0.5.
\]

### 6.3 频带趋势残差

若代码中已经能够从 LSTDE 输出频带级历史斜率，可进一步构造频带趋势项：

\[
\Delta Y^{\mathrm{freq}}_{b,n,h}
=
\sum_{k=1}^{K}\beta_k\eta_k s^{\mathrm{hist},(k)}_{b,n}\cdot h.
\]

最终混合残差为

\[
\Delta Y
=
\lambda\Delta Y^{\mathrm{time}}
+(1-\lambda)\Delta Y^{\mathrm{freq}}.
\]

校正后的采样实现为

\[
\hat{Y}^{(s)}\leftarrow \hat{Y}^{(s)}+\Delta Y.
\]

如果当前代码暂时只实现时间趋势残差，则可视为 \(\lambda=1\) 的简化版本，但文档和命名仍建议保留 `HybridTrendResidual` 接口，便于与论文公式一致。

代码中对应：

```text
frequency/residual.py -> HybridTrendResidual
```

---

## 7. DCA：密度质心聚合

### 7.1 目的

多次扩散采样得到的是微观实现云团。节假日拥堵场景下，样本可能形成自由流、拥堵和过渡状态附近的多峰结构。简单 Mean / Median / MoM 可能落入状态之间的低密度区域，得到物理含义不明确的中间预测。

DCA 的目标是从微观实现云团中恢复更符合交通状态结构的宏观流量预测。

### 7.2 逐点聚合粒度

DCA 沿样本轴逐点作用于每个时空位置 \((b,h,n)\)，而不是对完整高维轨迹聚类。

给定

\[
\left\{\hat{Y}^{(s)}_{b,h,n}\right\}_{s=1}^{S},
\]

分别计算点云内部密度权重和历史稳态相容权重。

### 7.3 点云内部密度权重

\[
\rho^{\mathrm{cloud}}_{b,h,n,s}
=
\frac{1}{S}\sum_{s'=1}^{S}
\exp\left(
-
\frac{(\hat{Y}^{(s)}_{b,h,n}-\hat{Y}^{(s')}_{b,h,n})^2}{2h_{\mathrm{kde}}^2}
\right).
\]

该权重使聚合中心偏向采样点云中的密集区域。

### 7.4 历史稳态相容权重

对每个节点 \(n\)，根据历史上下文中位数划分两个历史稳态子集：

\[
S_{n,\mathrm{free}}
=
\{y\in X^{\mathrm{hist}}_n \mid y\le m_n\},
\qquad
S_{n,\mathrm{cong}}
=
X^{\mathrm{hist}}_n\setminus S_{n,\mathrm{free}},
\]

其中

\[
m_n=\mathrm{median}(X^{\mathrm{hist}}_n).
\]

分别计算采样值与两类历史状态的核平滑相容度：

\[
\ell^{\mathrm{free}}_{b,h,n,s}
=
\sum_{y\in S_{n,\mathrm{free}}}
\exp\left(
-
\frac{(\hat{Y}^{(s)}_{b,h,n}-y)^2}{2h_{\mathrm{hist}}^2}
\right),
\]

\[
\ell^{\mathrm{cong}}_{b,h,n,s}
=
\sum_{y\in S_{n,\mathrm{cong}}}
\exp\left(
-
\frac{(\hat{Y}^{(s)}_{b,h,n}-y)^2}{2h_{\mathrm{hist}}^2}
\right).
\]

论文终版采用二者之和，而不是 max：

\[
\rho^{\mathrm{hist}}_{b,h,n,s}
=
\ell^{\mathrm{free}}_{b,h,n,s}
+
\ell^{\mathrm{cong}}_{b,h,n,s}.
\]

### 7.5 密度质心估计

联合权重为

\[
w_{b,h,n,s}
=
\rho^{\mathrm{cloud}}_{b,h,n,s}
\cdot
\rho^{\mathrm{hist}}_{b,h,n,s}.
\]

最终输出为

\[
\hat{Y}_{\mathrm{macro},b,h,n}
=
\frac{\sum_{s=1}^{S}w_{b,h,n,s}\hat{Y}^{(s)}_{b,h,n}}{\sum_{s=1}^{S}w_{b,h,n,s}}.
\]

代码中对应：

```text
macro/dca_aggregator.py -> DensityCentroidAggregator
```

---

## 8. 聚合器代码边界

### 8.1 主线聚合器

只保留 DCA 作为 HoliDiff 主模型默认聚合器：

```text
macro/dca_aggregator.py
```

### 8.2 论文消融用聚合器

Mean、Median、MoM 只作为实验对照，不进入 HoliDiff 主路径：

```text
macro/simple_aggregators.py
  - mean_aggregate
  - median_aggregate
  - mom_aggregate
  - kde_mode_aggregate
```

这些聚合器用于：

- `w/o DCA (Mean)`；
- `w/o DCA (Median)`；
- `w/o DCA (MoM)`；
- Table IX 聚合策略对比。

### 8.3 非论文主线聚合器

以下聚合器不写入主论文方法，不进入主模型路径：

```text
ABMS
WBP
RCL
```

如果需要保留实验记录，移动到：

```text
legacy/aggregation/
```

### 8.4 SCP 的位置

`SCP` 可以作为轻量工程备选暂时保留，但当前论文主线不依赖它。建议放在：

```text
macro/scp_aggregator.py
```

并在文档中标注：

> SCP is an engineering backup and is not part of the main HoliDiff method unless explicitly added to appendix experiments.

---

## 9. 实验脚本与论文表图对应

`exp/` 只负责训练、评估和导出结果，不负责画图。  
`plot/` 只负责读取 `outputs/` 中的结果并画图，不重新训练模型。

### 9.1 exp 目录

```text
exp/
  exp01_shift_score.py
  exp02_train_holidiff.py
  exp03_eval_overall.py
  exp04_eval_holiday_fujian.py
  exp05_ablation_modules.py
  exp06_eval_aggregation.py
  exp07_sensitivity_efficiency.py
  exp08_export_prediction_cases.py
  exp09_export_sampling_cloud.py
```

| 脚本 | 论文目标 | 输出 |
|---|---|---|
| `exp01_shift_score.py` | Fig. 3 | `outputs/exp01_shift_score/shift_scores.csv` |
| `exp02_train_holidiff.py` | 主模型 checkpoint | `checkpoints/paper/{dataset}/best.pt` |
| `exp03_eval_overall.py` | Table IV | `outputs/exp03_eval_overall/table_overall.csv` |
| `exp04_eval_holiday_fujian.py` | Table V / VI | `table_holiday.csv`, `table_peak.csv` |
| `exp05_ablation_modules.py` | Table VII | `table_ablation.csv` |
| `exp06_eval_aggregation.py` | Table IX | `table_aggregation.csv` |
| `exp07_sensitivity_efficiency.py` | Table X / XI | `table_sensitivity.csv`, `table_efficiency.csv` |
| `exp08_export_prediction_cases.py` | Fig. 4 数据 | `cases_regular_holiday.npz` |
| `exp09_export_sampling_cloud.py` | Fig. 6 数据 | `sampling_cloud_case.npz` |

### 9.2 plot 目录

```text
plot/
  plot01_shift_score.py
  plot02_prediction_cases.py
  plot03_frequency_decomposition.py
  plot04_sampling_kde.py
  plot05_sensitivity_efficiency.py
```

| 脚本 | 输入 | 输出 |
|---|---|---|
| `plot01_shift_score.py` | `shift_scores.csv` | `figures/fig3_shift_score.pdf` |
| `plot02_prediction_cases.py` | `cases_regular_holiday.npz` | `figures/fig4_prediction_cases.pdf` |
| `plot03_frequency_decomposition.py` | frequency case outputs | `figures/fig5_frequency_decomposition.pdf` |
| `plot04_sampling_kde.py` | `sampling_cloud_case.npz` | `figures/fig6_sampling_kde.pdf` |
| `plot05_sensitivity_efficiency.py` | sensitivity / efficiency csv | sensitivity figure |

---

## 10. 配置文件建议

论文主配置只保留少量可复现 YAML：

```text
configs/
  paper/
    fujian30_holidiff.yaml
    pems03_holidiff.yaml
    pems04_holidiff.yaml
    pems08_holidiff.yaml

  ablation/
    fujian30_wo_lstde.yaml
    fujian30_wo_trend.yaml
    fujian30_wo_sfcn.yaml
    fujian30_agg_mean.yaml
    fujian30_agg_median.yaml
    fujian30_agg_mom.yaml
```

主配置推荐：

```yaml
model:
  name: HoliDiff
  use_holiday_flag: false

lstde:
  enabled: true
  decomposer: fixed_fft
  num_bands: 4
  patch_len: 12
  patch_embed: trend_aware_concat

sfcn:
  enabled: true
  alpha: learnable
  init_alpha: 0.05

residual:
  enabled: true
  type: hybrid_trend_residual
  eta: 0.5
  lambda_time: 1.0   # if frequency residual is not yet implemented

aggregation:
  type: dca
  num_samples: 10

diffusion:
  steps: 100
```

说明：

- `use_holiday_flag: false` 明确论文主模型不使用节假日条件输入；
- `lambda_time: 1.0` 表示当前实现若仅保留时间趋势残差，则是 Hybrid Residual 的简化版本；
- 后续若补充频带趋势残差，可将 `lambda_time` 调整为 \([0,1]\) 内的混合系数。

---

## 11. 清理优先级

### 11.1 第一阶段：主路径固定

优先修改：

```text
models/HoliDiff.py
macro/aggregation_factory.py
exp/ entry scripts
```

目标：HoliDiff 主路径只包含：

```text
LSTDE -> SFCN -> Conditional Diffusion -> Hybrid Residual -> DCA
```

### 11.2 第二阶段：模块目录收敛

建议最终目录：

```text
models/
  holidiff.py
  diffusion.py
  denoiser.py

  frequency/
    decompose.py
    patch_embed.py
    residual.py

  micro/
    sfcn.py

  macro/
    dca_aggregator.py
    scp_aggregator.py
    simple_aggregators.py
```

### 11.3 第三阶段：legacy 隔离

移动到 legacy：

```text
legacy/frequency/
  conditioner.py
  modulator.py
  fusion.py
  spectral_gate.py

legacy/aggregation/
  abms.py
  wbp.py
  rcl.py

legacy/configs/
legacy/exp/
```

---

## 12. 当前文档相对旧版的关键修正

1. 删除“以 SimDiff 为基座”的强叙事，SimDiff 只作为重要 baseline，不作为论文方法定义的必要部分。
2. 删除“严格服从 LWR 方程”或“显式求解基本图”的过强物理表述，改为拓扑传播一致性与宏观状态结构感知。
3. 删除固定流量区间、TPR、SGFE、Phase-A/D/E 等阶段性实验叙事，避免与论文当前实验表格不一致。
4. 修正 SFCN 公式：加入度归一化邻接权重，并区分训练时未来统计量与推理时历史统计量。
5. 修正 DCA 公式：历史相容权重采用 \(\ell_{free}+\ell_{cong}\)，不是 max。
6. 明确 DCA 是逐点聚合，不是高维轨迹聚类。
7. 明确 holiday label 不进入模型主路径，仅用于漂移分析与评估。
8. 明确 SCP、ABMS、WBP、RCL 均不是论文主方法；MoM 仅作为聚合消融 baseline。
9. 将 exp 与 plot 分离：exp 产出 csv/npz/checkpoint，plot 只读取结果画图。

