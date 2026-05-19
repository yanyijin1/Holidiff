# HoliDiff: Three-Stage Holiday Conditioning Scheme

> **方案定位**：节假日结构干预的三阶段渐进注入  
> **基座**：SFCN (Spatial Field Coupled Normalization, Phase-E4)  
> **核心原则**：先验证假设，再堆叠模块；每阶段仅引入最少参数  
> **文档版本**：v2.0 (Stage-wise)  
> **日期**：2026-05-19

---

## 术语声明

本文档使用 **Regime-Specific Field Bank (RSFB)** 替代先前临时使用的 "Climate Bank"。
- **Regime**：交通流理论标准用语，指 free-flow / congested / holiday-synchronized 等动力学态型；
- **RSFB**：按节假日干预状态预计算的场统计量仓库，非 EventTSF 或任何现有工作的专有名词。

## Holiday 标签来源约定

本文档中所有节假日条件变量 $h_b$、$h$、`is_holiday` 都**直接来自数据集原始标签列 `is_holiday`**。实现时应从样本对应预测窗口读取数据集中的 `is_holiday` 值并聚合为二值 holiday 标记；**不要**根据 `time_slot`、时间戳、节日日期表或任何手工规则自行推断节假日状态。

---

## Stage 1: Holiday-Adaptive Coupling (HAC)

> **目标**：验证核心假设——节假日需要更强的空间场耦合  
> **新增参数**：1 个标量 $\Delta\alpha$  
> **改动范围**：仅 SFCN 的 $\alpha$ 计算  
> **预期通过标准**：$\Delta\alpha > 0$ 且 Hol_Cong_MAE 下降

### 1.1 物理假设

节假日是对路网的结构干预：平日自由流态型（free-flow regime）下节点近似独立；节假日同步拥堵态型（holiday-synchronized regime）下拥堵波全场传播，空间一致性显著增强。因此，场耦合强度应随干预状态自适应：

$$\alpha(h_b) = \alpha_{\text{base}} + \Delta\alpha \cdot h_b, \quad h_b \in \{0, 1\}$$

其中 $\alpha_{\text{base}} = 0.05$（Phase-E4 定版值），$\Delta\alpha$ 为唯一新增可学习标量，且 $h_b$ 由该样本预测窗口对应的**数据集 `is_holiday` 标签**生成（只要窗口内存在 holiday 标签即可置为 1）。

### 1.2 场编码（训练时）

保留 Phase-E4 的单库统计量 $\mathbf{F}_\mu, \mathbf{F}_\sigma$（全历史估计）：

$$\tilde{Y}^{(0)}_{b,i,t} = \frac{Y^{\text{fut}}_{b,i,t} - \mu_i}{\sigma_i + \epsilon}$$

$$\tilde{Y}^{(1)}_{b,i,t} = \sum_{j \in \mathcal{N}(i)} \frac{(Y^{\text{fut}}_{b,j,t} - Y^{\text{fut}}_{b,i,t}) - \delta_{ij}}{\nu_{ij} + \epsilon}$$

联合目标：
$$\boxed{\tilde{Y}_{b,i,t} = \tilde{Y}^{(0)}_{b,i,t} + \alpha(h_b) \cdot \tilde{Y}^{(1)}_{b,i,t}}$$

### 1.3 场解码（推理时）

$$\hat{Y}^{(0)}_{b,i,t} = \sigma^{\text{hist}}_i \cdot z_{b,i,t} + \mu^{\text{hist}}_i$$

$$\hat{Y}^{(1)}_{b,i,t} = \sum_{j \in \mathcal{N}(i)} \nu^{\text{hist}}_{ij} \cdot (z_{b,j,t} - z_{b,i,t})$$

$$\boxed{\hat{Y}_{b,i,t} = \hat{Y}^{(0)}_{b,i,t} + \alpha(h_b) \cdot \hat{Y}^{(1)}_{b,i,t}}$$

### 1.4 退化验证

- $h_b = 0$ 时：$\alpha(0) = \alpha_{\text{base}}$，严格退化为 Phase-E4 SFCN；
- $\Delta\alpha = 0$ 时：无论 $h_b$ 为何值，均退化为 Phase-E4。

### 1.5 实验配置

| 组 | 配置 | 新增参数 | 预期结果 |
|---|---|---|---|
| **E4-Baseline** | Phase-E4 定版（$\alpha=0.05$ 固定） | 0 | MAE 0.2267 |
| **HAC-learn** | $\alpha(h) = 0.05 + \Delta\alpha \cdot h$ | 1 标量 | $\Delta\alpha \to 0.03 \sim 0.08$，Hol_Cong_MAE 下降 |
| **HAC-hard** | $\alpha(0)=0.05, \alpha(1)=0.10$ 硬开关 | 0 | 验证方向正确性 |

### 1.6 通过标准（进入 Stage 2 的门槛）

必须同时满足：
1. $\Delta\alpha$ 收敛到 **正值**（$> 0.01$）；
2. **Hol_Cong_MAE** 相对 E4-Baseline 下降（$< 35.79$）；
3. **Reg_MAE** 不显著恶化（$\Delta < +0.005$）。

若 $\Delta\alpha \approx 0$：说明场耦合增强假设不成立，终止节假日条件化路线，转向频带能量偏移（Plan B）。

---

## Stage 2: HAC + Conditional Dropout (CD)

> **目标**：防止 $\Delta\alpha$ 过拟合节假日小样本，增强平日泛化  
> **新增参数**：0  
> **改动范围**：训练策略（数据层面）  
> **预期通过标准**：Hol_MAE 进一步下降，Reg_MAE 持平或改善

### 2.1 动机

Stage 1 中 $\Delta\alpha$ 仅由节假日样本驱动更新。若节假日样本占比小（如 < 10%），标量可能过拟合到特定节日模式。受 Classifier-Free Guidance 启发，在训练时以概率 $p_{\text{drop}}$ 将节假日标签随机置零，迫使模型学习从 $h=0$ 到 $h=1$ 的平滑插值谱。

### 2.2 条件 Dropout

训练时：
$$h_b^{\text{train}} = \begin{cases} 0 & \text{with prob. } p_{\text{drop}} = 0.1 \\ h_b & \text{with prob. } 0.9 \end{cases}$$

**注意**：
- Dropout **仅作用于 $\alpha(\cdot)$ 的计算**，即 $\alpha(h_b^{\text{train}})$；
- 单库统计量 $\mathbf{F}_\mu, \mathbf{F}_\sigma$ 仍使用原始标签对应的历史窗口计算（不 dropout），保证统计纯净性。

### 2.3 物理语义

以 10% 概率让模型"假装今天是平日"去预测节假日样本，它必须学会：即使场统计量被强制使用平日库，也能通过 $\alpha$ 的增量补偿部分节假日效应。这增强了 $\Delta\alpha$ 的鲁棒性。

### 2.4 实验配置

| 组 | 配置 | 新增参数 | 预期结果 |
|---|---|---|---|
| **HAC** | Stage 1 最优 | 1 标量 | 基线 |
| **HAC-CD** | HAC + 10% 条件 dropout | 0 | Hol_MAE 下降，Reg_MAE 不恶化 |

### 2.5 通过标准（进入 Stage 3 的门槛）

1. **HAC-CD** 的 Hol_Cong_MAE **优于** HAC；
2. **Reg_MAE** 相对 HAC 变化 $< +0.003$；
3. $\Delta\alpha$ 收敛值与 HAC 同量级（证明 dropout 未抹除信号）。

若 HAC-CD 劣于 HAC：说明 dropout 破坏了节假日信号，或节假日样本过少。改用更小的 $p_{\text{drop}} = 0.05$ 重试，或直接跳过 Stage 2 进入 Stage 3。

---

## Stage 3: HAC-CD + Dual-Regime Field Bank (DRFB)

> **目标**：验证节假日不仅改变耦合强度，更改变场本身的基准统计量  
> **新增参数**：0（预计算，不可学习）  
> **改动范围**：SFCN 统计量预计算 + 检索逻辑  
> **预期通过标准**：Hol_MAE 显著下降，平日性能不劣化

### 3.1 物理假设

Stage 1-2 使用全历史统计量 $\mathbf{F}^{(\text{all})}$，是对两种态型的粗暴平均：
- 平日场均值结构：通勤双峰（早高峰 + 晚高峰）；
- 节假日场均值结构：午后单峰（出游/返程集中）。

使用混合统计量导致：
- 平日推理：场基准被节假日样本"污染"，耦合基准偏高；
- 节假日推理：场基准被平日样本"稀释"，耦合基准偏低。

### 3.2 Dual-Regime Field Bank (DRFB) 定义

将训练集按**数据集标签 `is_holiday`** 划分为两个子集：
- $\mathcal{D}_{\text{reg}} = \{(\mathbf{X}, \mathbf{Y}) \mid \texttt{is\_holiday}=0\}$
- $\mathcal{D}_{\text{hol}} = \{(\mathbf{X}, \mathbf{Y}) \mid \texttt{is\_holiday}=1\}$

分别估计两套场统计量矩阵。

#### 3.2.1 平日态型场库 $\mathbf{F}^{(\text{reg})}$

$$\mathbf{F}_\mu^{(\text{reg})}[i,j] = \begin{cases} \mu_i^{(\text{reg})} & i=j \\ \delta_{ij}^{(\text{reg})} = \mu_j^{(\text{reg})} - \mu_i^{(\text{reg})} & (i,j) \in \mathcal{E} \\ 0 & \text{otherwise} \end{cases}$$

$$\mathbf{F}_\sigma^{(\text{reg})}[i,j] = \begin{cases} \sigma_i^{(\text{reg})} & i=j \\ \nu_{ij}^{(\text{reg})} = \mathrm{Std}_{\mathcal{D}_{\text{reg}}}[X_j - X_i] & (i,j) \in \mathcal{E} \\ 0 & \text{otherwise} \end{cases}$$

#### 3.2.2 节假日态型场库 $\mathbf{F}^{(\text{hol})}$

$$\mathbf{F}_\mu^{(\text{hol})}[i,j] = \begin{cases} \mu_i^{(\text{hol})} & i=j \\ \delta_{ij}^{(\text{hol})} = \mu_j^{(\text{hol})} - \mu_i^{(\text{hol})} & (i,j) \in \mathcal{E} \\ 0 & \text{otherwise} \end{cases}$$

$$\mathbf{F}_\sigma^{(\text{hol})}[i,j] = \begin{cases} \sigma_i^{(\text{hol})} & i=j \\ \nu_{ij}^{(\text{hol})} = \mathrm{Std}_{\mathcal{D}_{\text{hol}}}[X_j - X_i] & (i,j) \in \mathcal{E} \\ 0 & \text{otherwise} \end{cases}$$

#### 3.2.3 态型检索算子

硬选择（推理时零计算开销）：
$$\mathbf{F}^{(h)} = (1 - h) \cdot \mathbf{F}^{(\text{reg})} + h \cdot \mathbf{F}^{(\text{hol})}$$

等价于：
$$\mathbf{F}^{(h)} = \begin{cases} \mathbf{F}^{(\text{reg})} & h=0 \\ \mathbf{F}^{(\text{hol})} & h=1 \end{cases}$$

### 3.3 条件化场编码（Stage 3 完整版）

输入：
- $\mathbf{Y}^{\text{fut}} \in \mathbb{R}^{B \times N \times H}$
- $\mathbf{h} \in \{0,1\}^B$
- 双库：$\mathbf{F}_\mu^{(\text{reg})}, \mathbf{F}_\sigma^{(\text{reg})}, \mathbf{F}_\mu^{(\text{hol})}, \mathbf{F}_\sigma^{(\text{hol})}$

样本级检索：
$$\mathbf{F}_{\mu,b} = (1 - h_b) \cdot \mathbf{F}_\mu^{(\text{reg})} + h_b \cdot \mathbf{F}_\mu^{(\text{hol})}$$
$$\mathbf{F}_{\sigma,b} = (1 - h_b) \cdot \mathbf{F}_\sigma^{(\text{reg})} + h_b \cdot \mathbf{F}_\sigma^{(\text{hol})}$$

节点残差（态型感知）：
$$\tilde{Y}^{(0)}_{b,i,t} = \frac{Y^{\text{fut}}_{b,i,t} - \mu^{(h_b)}_i}{\sigma^{(h_b)}_i + \epsilon}$$

场耦合残差（态型感知）：
$$\tilde{Y}^{(1)}_{b,i,t} = \sum_{j \in \mathcal{N}(i)} \frac{(Y^{\text{fut}}_{b,j,t} - Y^{\text{fut}}_{b,i,t}) - \delta^{(h_b)}_{ij}}{\nu^{(h_b)}_{ij} + \epsilon}$$

联合目标（HAC + DRFB）：
$$\boxed{\tilde{Y}_{b,i,t} = \tilde{Y}^{(0)}_{b,i,t} + \alpha(h_b) \cdot \tilde{Y}^{(1)}_{b,i,t}}$$

其中 $\alpha(h_b) = \alpha_{\text{base}} + \Delta\alpha \cdot h_b$。

### 3.4 条件化场解码（推理时）

历史窗口同样维护双库 $\mathbf{F}^{\text{hist},(\text{reg})}, \mathbf{F}^{\text{hist},(\text{hol})}$，检索逻辑同上。

$$\hat{Y}^{(0)}_{b,i,t} = \sigma^{\text{hist},(h_b)}_i \cdot z_{b,i,t} + \mu^{\text{hist},(h_b)}_i$$

$$\hat{Y}^{(1)}_{b,i,t} = \sum_{j \in \mathcal{N}(i)} \nu^{\text{hist},(h_b)}_{ij} \cdot (z_{b,j,t} - z_{b,i,t})$$

$$\boxed{\hat{Y}_{b,i,t} = \hat{Y}^{(0)}_{b,i,t} + \alpha(h_b) \cdot \hat{Y}^{(1)}_{b,i,t}}$$

### 3.5 退化验证

- $h=0$ 且 $\Delta\alpha=0$：严格退化为 vanilla LocalNorm；
- $h=0$ 且 $\alpha_{\text{base}}=0.05$：严格退化为 Phase-E4 SFCN；
- $h=1$ 且使用 $\mathbf{F}^{(\text{hol})}$：完整节假日干预响应。

### 3.6 实验配置

| 组 | 配置 | 新增参数 | 预期结果 |
|---|---|---|---|
| **HAC-CD** | Stage 2 最优 | 1 标量 | 基线 |
| **HAC-CD-DRFB** | + 双库检索 | 0 | Hol_MAE 显著下降，Reg_MAE 不劣化 |
| **DRFB-only** | 仅双库，$\alpha$ 固定 0.05 | 0 | 验证双库本身贡献 |

### 3.7 通过标准（最终定版门槛）

1. **HAC-CD-DRFB** 的 Hol_Cong_MAE **优于** HAC-CD；
2. **Reg_MAE** 相对 HAC-CD 变化 $< +0.003$；
3. **DRFB-only** 优于 E4-Baseline（证明双库本身有价值）。

若 DRFB-only 劣于 E4-Baseline：说明双库分离引入噪声（可能节假日样本过少导致统计不稳定），回退到 HAC-CD 作为最终方案。

---

## 4. 三阶段总览与决策树

```
Stage 1 (HAC)
    |
    |-- 通过？Delta_alpha > 0 且 Hol_Cong_MAE 下降
    |
    |-- YES --> Stage 2 (HAC + CD)
    |           |
    |           |-- 通过？Hol_MAE 进一步下降
    |           |
    |           |-- YES --> Stage 3 (HAC-CD + DRFB)
    |           |           |
    |           |           |-- 通过？Hol_MAE 显著下降
    |           |           |
    |           |           |-- YES --> 最终定版：HAC-CD-DRFB
    |           |           |
    |           |           |-- NO  --> 回退：HAC-CD
    |           |
    |           |-- NO  --> 尝试 p_drop=0.05 或直接 Stage 3
    |
    |-- NO  --> 终止节假日路线，转向 Plan B（频带能量偏移）
```

---

## 5. 关键代码骨架（Stage 1 极简版）

```python
class HolidayAdaptiveSFCN(nn.Module):
    def __init__(self, adj, alpha_base=0.05):
        super().__init__()
        N = adj.shape[0]
        self.N = N
        self.alpha_base = alpha_base

        self.register_buffer('adj', adj.float())
        self.register_buffer('field_mask', adj + torch.eye(N, device=adj.device))

        # === Stage 1 唯一新增 ===
        self.delta_alpha = nn.Parameter(torch.tensor(0.01))
        # =======================

        # Phase-E4 原有统计量（单库）
        self.register_buffer('F_mu', torch.zeros(N, N))
        self.register_buffer('F_sigma', torch.zeros(N, N))
        self.register_buffer('F_mu_hist', torch.zeros(N, N))
        self.register_buffer('F_sigma_hist', torch.zeros(N, N))

    def get_alpha(self, is_holiday):
        """
        is_holiday: (B,) LongTensor, 0 or 1, directly loaded from dataset label `is_holiday`
        """
        alpha = self.alpha_base + self.delta_alpha * is_holiday.float()
        return alpha.view(-1, 1, 1)  # (B,1,1) broadcast to (B,N,H)

    def encode(self, Y_fut, is_holiday):
        B, N, H = Y_fut.shape
        alpha = self.get_alpha(is_holiday)  # (B,1,1)

        # 节点残差（Phase-E4 单库）
        mu_node = self.F_mu.diagonal()       # (N,)
        sigma_node = self.F_sigma.diagonal() # (N,)
        node_res = (Y_fut - mu_node.view(1,N,1)) / (sigma_node.view(1,N,1) + 1e-6)

        # 场耦合残差（Phase-E4 单库）
        Y_i = Y_fut.unsqueeze(2)   # (B,N,1,H)
        Y_j = Y_fut.unsqueeze(1)   # (B,1,N,H)
        edge_flow = Y_j - Y_i       # (B,N,N,H)
        edge_res = (edge_flow - self.F_mu.view(1,N,N,1)) /                    (self.F_sigma.view(1,N,N,1) + 1e-6)
        field_res = (self.adj.view(1,N,N,1) * edge_res).sum(dim=2)  # (B,N,H)

        return node_res + alpha * field_res

    def decode(self, z, is_holiday):
        B, N, H = z.shape
        alpha = self.get_alpha(is_holiday)

        mu_node = self.F_mu_hist.diagonal()
        sigma_node = self.F_sigma_hist.diagonal()
        node_pred = z * sigma_node.view(1,N,1) + mu_node.view(1,N,1)

        z_i = z.unsqueeze(2)
        z_j = z.unsqueeze(1)
        z_diff = z_j - z_i
        field_pred = (self.adj.view(1,N,N,1) * 
                      self.F_sigma_hist.view(1,N,N,1) * z_diff).sum(dim=2)

        return node_pred + alpha * field_pred
```

---

## 6. Stage 2 代码修改点（条件 Dropout）

仅修改训练循环，不改动模型结构：

```python
# 在 train_step 中，传给 SFCN 之前
if self.training and random.random() < 0.1:
    is_holiday_for_alpha = torch.zeros_like(is_holiday)
else:
    is_holiday_for_alpha = is_holiday

# 传给 encode/decode 的 alpha 计算使用 is_holiday_for_alpha
# 但统计量 F_mu, F_sigma 仍使用原始 is_holiday（如果有按标签分窗口统计的话）
```

---

## 7. Stage 3 代码修改点（DRFB）

将 `F_mu` / `F_sigma` 从 `(N, N)` 扩展为 `(2, N, N)`：

```python
class DualRegimeSFCN(nn.Module):
    def __init__(self, adj, alpha_base=0.05, dropout_prob=0.1):
        super().__init__()
        N = adj.shape[0]
        self.N = N
        self.alpha_base = alpha_base
        self.dropout_prob = dropout_prob

        self.register_buffer('adj', adj.float())

        # Stage 1 保留
        self.delta_alpha = nn.Parameter(torch.tensor(0.01))

        # Stage 3 双库：索引 0=reg, 1=hol
        self.register_buffer('F_mu_bank', torch.zeros(2, N, N))
        self.register_buffer('F_sigma_bank', torch.zeros(2, N, N))
        self.register_buffer('F_mu_hist_bank', torch.zeros(2, N, N))
        self.register_buffer('F_sigma_hist_bank', torch.zeros(2, N, N))

    def retrieve(self, h, bank):
        """h: (B,) LongTensor -> return (B, N, N)"""
        return bank[h.long()]  # (B, N, N)

    def get_alpha(self, h, training=True):
        if training and self.dropout_prob > 0:
            mask = torch.rand_like(h.float()) > self.dropout_prob
            h = h * mask.long()
        alpha = self.alpha_base + self.delta_alpha * h.float()
        return alpha.view(-1, 1, 1)

    def encode(self, Y_fut, h):
        B, N, H = Y_fut.shape
        alpha = self.get_alpha(h, self.training)

        # Stage 3：检索对应态型库
        F_mu = self.retrieve(h, self.F_mu_bank)       # (B, N, N)
        F_sigma = self.retrieve(h, self.F_sigma_bank) # (B, N, N)

        mu_node = F_mu.diagonal(dim1=1, dim2=2)       # (B, N)
        sigma_node = F_sigma.diagonal(dim1=1, dim2=2)   # (B, N)
        node_res = (Y_fut - mu_node.unsqueeze(-1)) / (sigma_node.unsqueeze(-1) + 1e-6)

        Y_i = Y_fut.unsqueeze(2)
        Y_j = Y_fut.unsqueeze(1)
        edge_flow = Y_j - Y_i
        edge_res = (edge_flow - F_mu.unsqueeze(-1)) / (F_sigma.unsqueeze(-1) + 1e-6)
        field_res = (self.adj.view(1, N, N, 1) * edge_res).sum(dim=2)

        return node_res + alpha * field_res
```

---

## 8. 论文叙事段落（按阶段撰写）

### 8.1 Stage 1 方法段落

> **Holiday-Adaptive Field Coupling.** We hypothesize that holidays constitute a structural intervention that intensifies spatial propagation of congestion waves. Under this intervention, the field coupling strength $\alpha$ should no longer remain static. We parameterize the holiday-adaptive coupling as:
> $$\alpha(h) = \alpha_{\text{base}} + \Delta\alpha \cdot h, \quad h \in \{0, 1\},$$
> where $\alpha_{\text{base}} = 0.05$ is the regular-day coupling calibrated in Phase-E4, and $\Delta\alpha$ is the sole learnable scalar introduced in this stage. When $h=0$, the module strictly degenerates to the vanilla SFCN; when $h=1$, the model learns to strengthen neighborhood consistency to reflect the synchronized congested regime.

### 8.2 Stage 2 方法段落

> **Conditional Dropout for Robust Intervention Learning.** To prevent $\Delta\alpha$ from overfitting to the limited holiday samples, we adopt a classifier-free-guidance-style training strategy: with 10% probability the holiday label is randomly masked to $h=0$ during training. This forces the model to learn a smooth interpolation spectrum between regular-day and holiday regimes, enhancing the robustness of the coupling gate without adding parameters.

### 8.3 Stage 3 方法段落

> **Dual-Regime Field Bank.** Empirical analysis reveals that holiday traffic operates under a distinct baseline distribution (shifted peak timing, widened IQR). A single field statistics bank averaged over both regimes dilutes the holiday signal. We therefore pre-compute separate field statistics matrices $\mathbf{F}^{(\text{reg})}$ and $\mathbf{F}^{(\text{hol})}$ from the corresponding subsets. During encoding and decoding, the model retrieves the appropriate regime-specific bank via the hard selector $\mathbf{F}^{(h)} = (1-h)\mathbf{F}^{(\text{reg})} + h\mathbf{F}^{(\text{hol})}$. This design introduces **zero learnable parameters** yet achieves structural climate adaptation.

### 8.4 因果 DAG 段落

> Figure X depicts the causal mechanism. The holiday intervention $\mathbf{H}$ influences the prediction $\hat{\mathbf{Y}}$ through two mediators: (i) the regime-specific field bank $\mathbf{F}^{(h)}$, which switches the baseline spatial statistics, and (ii) the adaptive coupling gate $\alpha(h)$, which modulates the strength of neighborhood consistency. Historical observation $\mathbf{X}$ affects $\hat{\mathbf{Y}}$ both directly (via the Transformer denoiser) and indirectly (via the retrieved field statistics). Importantly, $\mathbf{H}$ does not enter the input layer; it intervenes on the internal physical parameters.

---

## 9. 与现有工作的边界（Related Work 用）

| 维度 | 现有工作 | 本文（Stage 3 定版） |
|---|---|---|
| **节假日角色** | 输入特征 / 混杂噪声 | **结构干预变量** |
| **因果语言** | Back-door adjustment（去混杂，如 CaST） | **干预响应（Intervention Response）** |
| **技术实现** | Embedding 拼接 / 环境码本 | **参数级标量门控 + 态型库检索** |
| **参数增量** | 大量（Embedding 表、MLP、码本） | **1 个标量（Stage 1-3 均如此）** |
| **调制对象** | 注意力 / 扩散时间步 / 特征图 | **物理场耦合强度 + 场统计量基准** |
| **训练策略** | 无特殊处理 | **条件 Dropout（CFG 风格）** |

---

*文档版本: v2.0 (Stage-wise)*  
*创建时间: 2026-05-19*  
*下次更新: Stage 1 实验验证完成后*
