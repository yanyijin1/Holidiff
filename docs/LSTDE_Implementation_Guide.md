# LSTDE: Learnable Spectral-Trend Disentangled Embedding

> 版本：v1.0 (基础标量门控版)  
> 定位：SimDiff PatchEmbed 替换模块，用于 HoliDiff 拥堵相预测优化  
> 核心假设：拥堵/自由流/节假日的信号分布在不同频带，应差异化编码与校正  
> 预期目标：CG_MAE < 30.56（击败当前最优），FF_MAE 不劣化（< 13.80）

---

## 一、背景与动机

### 1.1 当前瓶颈
HoliDiff 当前最优配置（Trend-Aware PatchEmbed + fixed eta=0.5）已触及轻量修改天花板：
- CG_MAE = 30.56，TPR_CG = 0.474
- 保守性根因：单通道输入层无法区分自由流基线与拥堵激波的不同动力学尺度

### 1.2 频率解耦的物理直觉
交通流信号在多时间尺度上同时重构：

| 频带 | 时间尺度 | 物理意义 | 节假日变化 |
|------|---------|---------|-----------|
| 低频 | > 2h | 整体需求基线 | 基线抬升或下降 |
| 中频 | 30min–2h | 通勤/出行节律 | 峰值时刻偏移 |
| 高频 | < 30min | 局部拥堵/扰动 | 异常峰值增强 |

关键假设：
- 自由流稳态：信号集中在低频 → 保守估计
- 拥堵形成/消散：信号集中在中高频 → 激进估计
- 节假日 vs 常规日：低频基线和中频节律不同 → 可学习分离

### 1.3 与现有工作的关系
- **FreDN (AAAI 2026)**：提出可学习频域门控替代固定 FFT 掩码，解决 spectral entanglement。LSTDE 将双分支 (trend/seasonal) 扩展为三频带 (low/mid/high)。
- **DualCast (IJCAI 2025)**：在时域显式解耦交通 periodic/aperiodic 分量。LSTDE 在频域完成互补解耦。
- **FEDformer (ICML 2022)**：MOEDecomp + 频域稀疏注意力。LSTDE 保留全谱信息，不做随机采样。
- **TFDNet (PR 2025)**：时频双分支独立 block。LSTDE 将频带差异压缩到 PatchEmbed 层，更轻量。

---

## 二、参考文献与代码仓库

| 工作 | 会议/期刊 | 核心借鉴点 | 代码仓库 |
|------|----------|-----------|---------|
| **FreDN** | AAAI 2026 | 可学习频域门控 (Frequency Disentangler) | https://github.com/An-z-d/FreDN |
| **DualCast** | IJCAI 2025 | 交通时域解耦 + cross-time attention | https://github.com/suzy0223/DualCast |
| **FEDformer** | ICML 2022 | MOEDecomp 趋势-季节分解 | https://github.com/MAZiqing/FEDformer |
| **TFDNet** | PR 2025 | 时频双分支独立 block | https://github.com/YuxiaoLuo0013/TFDNet |
| **TimesNet** | ICLR 2023 | 1D→2D 多尺度重构 | https://github.com/thuml/TimesNet |

---

## 三、数学符号表

| 符号 | 维度 | 含义 |
|------|------|------|
| $\mathbf{X}$ | $\mathbb{R}^{B \times N \times T}$ | 输入历史序列 |
| $B$ | scalar | batch size |
| $N$ | scalar | 空间节点数 |
| $T$ | scalar | 历史时间步长 (当前 = 9, 对应 3 patch × 3) |
| $K$ | scalar | 频带数 (固定 = 3: low/mid/high) |
| $L$ | scalar | RFFT 频率点数 = $\lfloor T/2 \rfloor + 1$ |
| $P$ | scalar | patch_size (当前 = 3) |
| $M$ | scalar | patch 数 = $T/P$ (当前 = 3) |
| $d_{\text{model}}$ | scalar | 模型维度 (当前 = 128) |
| $d_k$ | scalar | 单频带投影维度 = $d_{\text{model}}/K$ |
| $\widehat{\mathbf{X}}$ | $\mathbb{C}^{B \times N \times L}$ | RFFT 频谱 |
| $\mathbf{G}$ | $\mathbb{R}^{L \times K}$ | 可学习门控参数 |
| $\mathbf{W}$ | $\mathbb{R}^{L \times K}$ | softmax 归一化频带权重 |
| $\mathbf{X}^{(k)}$ | $\mathbb{R}^{B \times N \times T}$ | 第 $k$ 个频带子序列 |
| $\mu_i^{(k)}$ | $\mathbb{R}^{B \times N}$ | 第 $k$ 频带第 $i$ 个 patch 均值 |
| $\tau_i^{(k)}$ | $\mathbb{R}^{B \times N}$ | 第 $k$ 频带第 $i$ 个 patch 斜率 |
| $\mathbf{z}_i^{(k)}$ | $\mathbb{R}^{B \times N \times d_k}$ | 第 $k$ 频带第 $i$ 个 patch 嵌入 |
| $\eta_k$ | $\mathbb{R}$ | 第 $k$ 频带趋势调制强度 (可学习) |
| $\mathbf{Z}_0$ | $\mathbb{R}^{B \times N \times M \times d_{\text{model}}}$ | 最终融合嵌入，输入 SimDiff Denoiser |
| $\mathbf{Y}_{\text{pred}}$ | $\mathbb{R}^{B \times N \times \text{pred\_len}}$ | 扩散模型原始预测 |
| $\mathbf{Y}_{\text{final}}$ | $\mathbb{R}^{B \times N \times \text{pred\_len}}$ | 频带级趋势校正后最终输出 |
| $\beta_k$ | $\mathbb{R}$ | 后处理校正频带权重 |

---

## 四、核心公式

### 4.1 可学习频谱门控 (Learnable Spectral Gating)

**Step 1: RFFT**
$$
\widehat{\mathbf{X}} = \text{RFFT}(\mathbf{X}, \text{dim}=-1) \in \mathbb{C}^{B \times N \times L}
$$

**Step 2: 可学习门控矩阵**
引入实值参数矩阵：
$$
\mathbf{G} \in \mathbb{R}^{L \times K}
$$
通过 softmax 得到归一化频带分配权重：
$$
\mathbf{W} = \text{softmax}(\mathbf{G}, \; \text{dim}=-1) \in \mathbb{R}^{L \times K}, \quad \sum_{k=1}^{K} W_{f,k} = 1, \; \forall f \in [1, L]
$$

**Step 3: 频带分离与逆变换**
$$
\widehat{\mathbf{X}}^{(k)} = \widehat{\mathbf{X}} \odot \mathbf{W}_{:,k}, \qquad k=1,2,3
$$
$$
\mathbf{X}^{(k)} = \text{IRFFT}\bigl(\widehat{\mathbf{X}}^{(k)}, \; n=T, \; \text{dim}=-1\bigr) \in \mathbb{R}^{B \times N \times T}
$$
其中 $\odot$ 为广播乘法（频率维度 $L$ 广播到复数谱实部与虚部）。

> **物理可解释性**：训练完成后可视化 $\mathbf{W}$，预期低频段权重集中在 $k=1$ (low band)，高频激波段集中在 $k=3$ (high band)。

---

### 4.2 频带级 Trend-Aware PatchEmbed

**Step 4: Patch 切分**
将每个频带子序列 $\mathbf{X}^{(k)}$ 切分为 $M = T/P$ 个 patch。

**Step 5: 频带内统计量**
对第 $k$ 个频带的第 $i$ 个 patch ($i=1,\dots,M$)：
$$
\mu_i^{(k)} = \frac{1}{P} \sum_{j=(i-1)P+1}^{iP} \mathbf{X}^{(k)}_{:,:,j} \in \mathbb{R}^{B \times N}
$$
$$
\tau_i^{(k)} = \frac{\mathbf{X}^{(k)}_{:,:,iP} - \mathbf{X}^{(k)}_{:,:,(i-1)P+1}}{P-1} \in \mathbb{R}^{B \times N}
$$

**Step 6: 频带特异性线性投影**
每个频带独立学习投影矩阵，维度 $d_k = d_{\text{model}} / K$：
$$
\mathbf{z}_i^{(k)} = \mathbf{W}_\mu^{(k)} \mu_i^{(k)} + \mathbf{W}_\tau^{(k)} \tau_i^{(k)} \in \mathbb{R}^{B \times N \times d_k}
$$
其中：
- $\mathbf{W}_\mu^{(k)} \in \mathbb{R}^{d_k}$：均值投影（继承原始 PatchEmbed 初始化逻辑）
- $\mathbf{W}_\tau^{(k)} \in \mathbb{R}^{d_k}$：斜率投影（零初始化或随机初始化）

**Step 7: 频带级趋势调制**
引入频带级可学习标量 $\eta_k \in \mathbb{R}$，通过 softplus 保证非负：
$$
\tilde{\mathbf{z}}_i^{(k)} = \text{softplus}(\eta_k) \cdot \mathbf{z}_i^{(k)} \in \mathbb{R}^{B \times N \times d_k}
$$

**初始化建议**（交通物理引导）：
- $\eta_1$ (low) = $\log(e^{0.2} - 1) \approx -1.33$ → softplus 后 ≈ 0.2（保守）
- $\eta_2$ (mid) = $\log(e^{0.5} - 1) \approx -0.69$ → softplus 后 ≈ 0.5（moderate）
- $\eta_3$ (high) = $\log(e^{1.0} - 1) \approx 0.54$ → softplus 后 ≈ 1.0（激进）

---

### 4.3 频带融合与扩散输入

**Step 8: 拼接 + 融合**
$$
\mathbf{Z} = \text{Concat}\bigl[\tilde{\mathbf{z}}^{(1)}, \tilde{\mathbf{z}}^{(2)}, \tilde{\mathbf{z}}^{(3)}\bigr] \in \mathbb{R}^{B \times N \times M \times d_{\text{model}}}
$$

可选轻量融合（也可省略，直接送 Transformer）：
$$
\mathbf{Z}_0 = \mathbf{Z} \cdot \mathbf{W}_{\text{fuse}} \in \mathbb{R}^{B \times N \times M \times d_{\text{model}}}
$$
其中 $\mathbf{W}_{\text{fuse}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$。

**Step 9: 接入 SimDiff Denoiser**
$\mathbf{Z}_0$ 直接替代原始 PatchEmbed 输出，进入 SimDiff Transformer。扩散前向/反向过程数学形式不变：
$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}), \quad p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{Z}_0)
$$

---

### 4.4 推理时频带级趋势校正（后处理，兼容现有 eta）

**Step 10: 各频带历史末端趋势斜率**
对分离后的频带子序列：
$$
\mathbf{s}^{(k)} = \frac{\mathbf{X}^{(k)}_{:,:,-1} - \mathbf{X}^{(k)}_{:,:,0}}{T-1} \in \mathbb{R}^{B \times N}
$$

**Step 11: 频带级残差校正**
$$
\Delta^{(k)} = \text{softplus}(\eta_k) \cdot \mathbf{s}^{(k)} \cdot \text{pred\_len}
$$

**Step 12: 加权聚合**
$$
\mathbf{Y}_{\text{final}} = \mathbf{Y}_{\text{pred}} + \sum_{k=1}^{K} \beta_k \cdot \Delta^{(k)}
$$
其中权重满足 $\sum_{k=1}^{K} \beta_k = 1$。

**初始化建议**：$\boldsymbol{\beta} = [0.2, \; 0.3, \; 0.5]$，高频校正占主导。

---

## 五、实现步骤 (Step-by-Step)

### Step 0: 环境准备
- 确认 PyTorch 版本支持 `torch.fft.rfft` / `irfft`
- 下载参考代码（可选，用于对照）：
  - FreDN: `git clone https://github.com/An-z-d/FreDN`
  - DualCast: `git clone https://github.com/suzy0223/DualCast`

### Step 1: 实现 `LearnableSpectralGate`
- 新建模块文件，定义参数 `self.G = nn.Parameter(torch.randn(L, K))`
- 前向：输入 `(B, N, T)` → RFFT → softmax 门控 → 分离 $K$ 条频带 → IRFFT → 输出 list of `(B, N, T)`
- **关键注意**：IRFFT 必须指定 `n=T`，否则长度不对
- **初始化**：`G` 可用小随机值初始化，或按预期频率分布做 warm-start（低频位置给 band-1 更大初值）

### Step 2: 改造 `TrendAwarePatchEmbed` → `BandSpecificTrendAwarePatchEmbed`
- 原模块：单输入 → 单输出
- 新模块：接收 $K$ 个频带子序列，每个子序列独立做 Trend-Aware Embed
- 每个 band 的投影维度 $d_k = d_{\text{model}} // K$
- 每个 band 独立拥有 `W_mu` 和 `W_tau`
- 输出 list of `(B, N, M, d_k)`

### Step 3: 实现 `BandTrendModulator`
- 定义 `self.eta = nn.Parameter(torch.tensor([-1.33, -0.69, 0.54]))`（对应 softplus 后 0.2/0.5/1.0）
- 前向：对每个 band 的嵌入 `z` 应用 `softplus(eta[k]) * z`

### Step 4: 实现 `LSTDE` 组装模块
- 组合 Step 1→2→3→Concat→(可选 Fuse Linear)
- 输入：`(B, N, T)` 原始历史序列
- 输出：`(B, N, M, d_model)` 替代原 PatchEmbed 输出
- 在 `HoliDiff` 模型初始化中，用 `LSTDE` 替换原 `PatchEmbed`

### Step 5: 修改后处理（推理阶段）
- 在 `simple_mean` 聚合后、反 RevIN 前（或后，需实验）插入频带级趋势校正
- 复用 Step 1 的 `LearnableSpectralGate` 分离历史序列（**注意**：推理时 gate 参数已固定，无需重新学习）
- 计算各频带斜率 $\mathbf{s}^{(k)}$
- 应用 $\Delta^{(k)}$ 和加权聚合
- **超参**：先固定 $\beta_k = [0.2, 0.3, 0.5]$，后续可扩展为可学习

### Step 6: 训练配置
- 继承当前最优配置：
  - `d_model=128, n_heads=8, e_layers=1, d_layers=1`
  - `train_epochs=30, batch_size=32, lr=0.0001`
  - `loss=MSE`
- **新增可学习参数**：`G` (L×K), 3×`W_mu`, 3×`W_tau`, `eta` (3,), `W_fuse` (可选)
- **冻结策略**：可先 freeze SimDiff Transformer 主体，只训练 LSTDE 模块 10 epoch，再联合训练 20 epoch

### Step 7: 快速验证 (10 epoch)
- 对照组：当前最优（Trend-Aware concat + fixed eta=0.5）
- 实验组 A：LSTDE + 统一后处理 eta=0.5（不加频带级校正）
- 实验组 B：LSTDE + 频带级趋势校正（完整版）
- **判断标准**：
  - CG_MAE < 30.56 → 有效
  - FF_MAE < 13.80 → 未劣化
  - HOL_CG 显著下降 → 节假日漂移被缓解

### Step 8: 可视化与可解释性验证
- **图 1**: 训练后频带权重 $\mathbf{W}$ 热力图（频率 $f$ vs 频带 $k$），验证低频→low、高频→high
- **图 2**: 各频带能量分布箱线图（自由流 / 拥堵 / 节假日 / 常规日）
- **图 3**: 频带级 eta 学习曲线，验证 high band 收敛到更大值
- **图 4**: 24h 频带激活模式热力图

### Step 9: 消融实验（如主实验有效）
- Abl-1: 固定 FFT 掩码（10%/50% 硬阈值）vs 可学习门控
- Abl-2: 去除 $\eta_k$ 调制（所有 band 同等对待）
- Abl-3: 去除后处理频带级校正
- Abl-4: $K=2$ (low/high) vs $K=3$ (low/mid/high)
- Abl-5: 后处理 $\beta_k$ 固定 vs 可学习

### Step 10: 风险评估与备选
- **风险 1**: 频带分离后信息损失（CG_MAE > 31.0）→ 减少频带数 $K=3 \to 2$，或改用 DCT
- **风险 2**: 频带级 eta 过拟合（FF_MAE > 15）→ 固定 eta 比例，或加 L2 正则
- **风险 3**: FFT 计算开销（训练时间增加 > 20%）→ 改用 DCT（实数运算，更快）
- **风险 4**: 频带能量假设不成立（分布重叠严重）→ 放弃频率解耦，转向 DualCast 时域解耦

---

## 六、诊断与验证 Checklist

| 检查项 | 通过标准 | 诊断方法 |
|--------|---------|---------|
| 频带分离可视化 | 低频能量集中在 low band，激波在 high band | 训练后 plot $\mathbf{W}$ 和 band 能量分布 |
| 参数初始化有效性 | 前 3 epoch CG_MAE 不暴涨 | 对比随机初始化 vs 物理引导初始化 |
| eta 学习方向 | high band eta 最终 > low band eta | 打印训练后 `softplus(eta)` 值 |
| 梯度健康度 | gate/eta 梯度不消失/不爆炸 | 每 100 batch 打印梯度范数 |
| 推理一致性 | 同一样本多次推理结果方差可控 | 固定 seed 跑 5 次采样 |
| 后处理兼容性 | 加/不加频带校正总 MAE 变化方向一致 | 分别评估 A/B 组 |

---

## 七、论文叙事定位

### 7.1 章节建议位置
建议放在 **3.2 PatchEmbed 改进** 或新增 **3.3 频域解耦嵌入** 章节。

### 7.2 叙事逻辑
```
3.1 问题背景：Trend Annihilation（patch 平均池化抹平趋势）
    ↓
3.2 基础方案：Trend-Aware PatchEmbed（concat 均值+斜率）
    ↓ 触及天花板 CG_MAE≈30.56
3.3 深化方案：LSTDE
    - 3.3.1 多尺度频带分离（可学习门控）
    - 3.3.2 频带级趋势编码（差异化 PatchEmbed）
    - 3.3.3 频带级趋势调制（可学习 eta_k）
    - 3.3.4 频带融合与扩散输入
    - 3.3.5 推理时频带级校正
```

### 7.3 与参考文献的引用策略
- 分解合理性：FEDformer (ICML 2022), Autoformer (NeurIPS 2021)
- 固定分解缺陷：FreDN (AAAI 2026) — "spectral entanglement"
- 交通解耦必要性：DualCast (IJCAI 2025) — "aperiodic events in traffic"
- 时频双域工程：TFDNet (PR 2025)
- 本文方法：首次将可学习频域解耦引入扩散交通预测，且与 Trend-Aware 编码联合。

---

## 八、一句话总结

> LSTDE 通过**可学习频谱门控**将交通流历史序列软分解为低/中/高三个频带，每个频带独立执行 Trend-Aware PatchEmbed 并施加**频带级可学习趋势调制**（低频保守、高频激进），最终通过拼接融合输入 SimDiff Denoiser；推理阶段进一步施加**频带级趋势残差校正**，使拥堵相在中高频获得更强外推，自由流在低频保持保守，从而突破当前 CG_MAE≈30.56 的天花板。

---

*文档版本: v1.0*  
*创建时间: 2026-05-19*  
*对应实验: HoliDiff Step 2 — 频率解耦模块*
