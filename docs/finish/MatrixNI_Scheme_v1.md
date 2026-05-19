# HoliDiff Phase E: Matrix-NI (场化归一化独立性) 方案文档

> 目标：将 NI 从单点标量扩展为矩阵结构，让 1-hop 空间分布在训练目标中内生，不修改 loss，不增加物理约束，RevIN 分两步验证。
> 
> 基座：SimDiff (AAAI 2026) + HoliDiff Phase D (LSTDE K=4, patch=12)
> 
> 核心创新：Normalization Independence 的矩阵化——对角线存节点未来分布，邻接边存差值未来分布。

---

## 一、核心思想

传统 NI 用 $(\mu_i, \sigma_i)$ 描述节点 $i$ 的未来分布，假设各节点独立。交通流中，节点 $i$ 的未来不仅取决于自身历史，还取决于**1-hop 场内邻居的未来状态**。

**矩阵化 NI** 构造稀疏矩阵 $\mathbf{M}_\mu, \mathbf{M}_\sigma \in \mathbb{R}^{N \times N}$，非零结构 = 有向邻接 $\mathbf{A} + \mathbf{I}$：

| 位置 | 存储内容 | 语义 |
|:---|:---|:---|
| 对角线 $(i,i)$ | $\mu_i, \sigma_i$ | 节点 $i$ 的**自身未来分布** |
| 非零邻接 $(i,j)$ | $\delta_{ij} = \mu_j - \mu_i$, $\sigma^\Delta_{ij} = \mathrm{Std}[X_j - X_i]$ | 边 $(i,j)$ 的**未来差值分布** |

训练时，扩散去噪目标不再是单点偏移，而是**"在场中的相对位置"**；推理时，历史场统计量代理未来场，实现空间一致的反归一化。

---

## 二、数学公式

### 2.1 矩阵化统计量构造（离线，训练/推理前）

**输入**：历史流量矩阵 $\mathbf{X} \in \mathbb{R}^{T \times N}$（全训练集时间序列拼接）

**节点级统计量**：
$$\mu_i = \frac{1}{T} \sum_{t=1}^T X_{t,i}, \quad \sigma_i = \sqrt{\frac{1}{T} \sum_{t=1}^T (X_{t,i} - \mu_i)^2}$$

**边级差值统计量**（仅对邻接边计算）：
$$\delta_{ij} = \frac{1}{T} \sum_{t=1}^T (X_{t,j} - X_{t,i}) = \mu_j - \mu_i$$
$$\sigma^\Delta_{ij} = \sqrt{\frac{1}{T} \sum_{t=1}^T \left[(X_{t,j} - X_{t,i}) - \delta_{ij}\right]^2}$$

**稀疏矩阵组装**：
$$\mathbf{M}_\mu[i,j] = \begin{cases} \mu_i & i=j \\ \delta_{ij} & (i,j) \in \mathcal{E} \\ 0 & \text{otherwise} \end{cases}, \quad
\mathbf{M}_\sigma[i,j] = \begin{cases} \sigma_i & i=j \\ \sigma^\Delta_{ij} & (i,j) \in \mathcal{E} \\ 0 & \text{otherwise} \end{cases}$$

### 2.2 训练阶段：联合归一化目标

对批次内未来真值 $\mathbf{X}^{\mathrm{fut}} \in \mathbb{R}^{B \times N \times H}$：

**节点级归一化**（保留 NI 骨架）：
$$\tilde{X}^{(0)}_{b,i,h} = \frac{X^{\mathrm{fut}}_{b,i,h} - \mu^{\mathrm{fut}}_i}{\sigma^{\mathrm{fut}}_i}$$

**边级归一化**（空间结构注入）：
$$\tilde{X}^{(1)}_{b,i,h} = \sum_{j \in \mathcal{N}(i)} \frac{(X^{\mathrm{fut}}_{b,j,h} - X^{\mathrm{fut}}_{b,i,h}) - \delta^{\mathrm{fut}}_{ij}}{\sigma^{\Delta,\mathrm{fut}}_{ij}}$$

**联合目标**（唯一超参/可学习标量 $\zeta$）：
$$\boxed{\tilde{X}_{b,i,h} = \tilde{X}^{(0)}_{b,i,h} + \zeta \cdot \tilde{X}^{(1)}_{b,i,h}}$$

**扩散训练目标**（loss 形式不变，目标空间改变）：
$$\mathcal{L} = \mathbb{E}_{t,\epsilon} \left[ \left\| \epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_t} \tilde{\mathbf{X}} + \sqrt{1-\bar{\alpha}_t} \epsilon, \; t, \; \tilde{\mathbf{X}}^{\mathrm{hist}}\right) \right\|^2 \right]$$

### 2.3 推理阶段：场化反归一化

模型输出 $\mathbf{z} \in \mathbb{R}^{B \times N \times H}$（标准化空间）。

**节点项**（NI 严格保留）：
$$\hat{X}^{(0)}_{b,i,h} = \sigma^{\mathrm{hist}}_i \cdot z_{b,i,h} + \mu^{\mathrm{hist}}_i$$

**边项**（场渗透）：
$$\hat{X}^{(1)}_{b,i,h} = \sum_{j \in \mathcal{N}(i)} \sigma^{\Delta,\mathrm{hist}}_{ij} \cdot (z_{b,j,h} - z_{b,i,h})$$

**联合反归一化**：
$$\boxed{\hat{X}_{b,i,h} = \hat{X}^{(0)}_{b,i,h} + \zeta \cdot \hat{X}^{(1)}_{b,i,h}}$$

**验证**：$\zeta=0$ 时严格退化 $\hat{X}_{b,i,h} = \sigma^{\mathrm{hist}}_i z_{b,i,h} + \mu^{\mathrm{hist}}_i$（原始 SimDiff NI）。

### 2.4 RevIN 两版

#### 版 A：原版 RevIN（节点孤岛，Phase E1 先用）
$$\tilde{X}^{\mathrm{revin}}_{b,i,:} = \gamma_i \cdot \frac{X_{b,i,:} - \mu^{\mathrm{inst}}_{b,i}}{\sigma^{\mathrm{inst}}_{b,i}} + \beta_i$$

#### 版 B：Field-RevIN（邻域协同，Phase E3 验证）
$$\tilde{X}^{\mathrm{revin}}_{b,i,:} = \gamma_i \cdot \frac{X_{b,i,:} - \mu^{\mathrm{field}}_{b,i}}{\sigma^{\mathrm{field}}_{b,i}} + \beta_i$$

其中场化统计量：
$$\mu^{\mathrm{field}}_{b,i} = \mu^{\mathrm{inst}}_{b,i} + \eta \sum_{j \in \mathcal{N}(i)} (\mu^{\mathrm{inst}}_{b,j} - \mu^{\mathrm{inst}}_{b,i})$$
$$\sigma^{\mathrm{field}}_{b,i} = \sigma^{\mathrm{inst}}_{b,i} + \eta \sum_{j \in \mathcal{N}(i)} (\sigma^{\mathrm{inst}}_{b,j} - \sigma^{\mathrm{inst}}_{b,i})$$

---

## 三、代码骨架

```python
import torch
import torch.nn as nn
from typing import Optional

class MatrixNI(nn.Module):
    def __init__(self, adj: torch.Tensor, zeta_init: float = 0.1):
        super().__init__()
        N = adj.shape[0]
        self.register_buffer('A', adj.float())
        self.register_buffer('mask', adj + torch.eye(N, device=adj.device))
        self.N = N
        self.zeta = nn.Parameter(torch.tensor(zeta_init))

    def build_field_matrices(self, mu: torch.Tensor, sigma: torch.Tensor,
                             X_hist: Optional[torch.Tensor] = None):
        device = mu.device
        M_mu = torch.zeros(self.N, self.N, device=device)
        M_mu.diagonal().copy_(mu)
        diff_mu = mu.unsqueeze(0) - mu.unsqueeze(1)
        M_mu = M_mu + self.A * diff_mu

        M_sigma = torch.zeros(self.N, self.N, device=device)
        M_sigma.diagonal().copy_(sigma)

        if X_hist is not None:
            X_i = X_hist.unsqueeze(1)
            X_j = X_hist.unsqueeze(2)
            delta = X_j - X_i
            sigma_delta = delta.std(dim=0)
            M_sigma = M_sigma + self.A * sigma_delta
        else:
            sigma_sum = sigma.unsqueeze(0) + sigma.unsqueeze(1)
            M_sigma = M_sigma + self.A * sigma_sum * 0.5

        return M_mu, M_sigma

    def normalize(self, X_fut: torch.Tensor, M_mu: torch.Tensor,
                  M_sigma: torch.Tensor) -> torch.Tensor:
        B, N, H = X_fut.shape
        mu_node = M_mu.diagonal()
        sigma_node = M_sigma.diagonal()
        node_term = (X_fut - mu_node.view(1, N, 1)) / (sigma_node.view(1, N, 1) + 1e-6)

        X_i = X_fut.unsqueeze(2)
        X_j = X_fut.unsqueeze(1)
        X_diff = X_j - X_i
        edge_norm = (X_diff - M_mu.unsqueeze(0).unsqueeze(-1)) /                     (M_sigma.unsqueeze(0).unsqueeze(-1) + 1e-6)
        edge_term = (self.A.unsqueeze(0).unsqueeze(-1) * edge_norm).sum(dim=2)

        target = node_term + self.zeta * edge_term
        return target

    def denormalize(self, z: torch.Tensor, M_mu_hist: torch.Tensor,
                    M_sigma_hist: torch.Tensor) -> torch.Tensor:
        B, N, H = z.shape
        mu_node = M_mu_hist.diagonal()
        sigma_node = M_sigma_hist.diagonal()
        node_part = z * sigma_node.view(1, N, 1) + mu_node.view(1, N, 1)

        z_i = z.unsqueeze(2)
        z_j = z.unsqueeze(1)
        z_diff = z_j - z_i
        edge_part = (self.A.unsqueeze(0).unsqueeze(-1) * 
                     M_sigma_hist.unsqueeze(0).unsqueeze(-1) * z_diff).sum(dim=2)

        return node_part + self.zeta * edge_part


class FieldRevIN(nn.Module):
    def __init__(self, adj: torch.Tensor, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer('A', adj.float())
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, mode: str = 'norm'):
        if mode == 'norm':
            mu_inst = x.mean(dim=-1, keepdim=True)
            sigma_inst = x.std(dim=-1, keepdim=True) + self.eps

            mu_diff = mu_inst.unsqueeze(2) - mu_inst.unsqueeze(1)
            mu_field = mu_inst + self.eta * (self.A.unsqueeze(0).unsqueeze(-1) * mu_diff).sum(dim=2)

            sigma_diff = sigma_inst.unsqueeze(2) - sigma_inst.unsqueeze(1)
            sigma_field = sigma_inst + self.eta * (self.A.unsqueeze(0).unsqueeze(-1) * sigma_diff).sum(dim=2)

            x_norm = (x - mu_field) / sigma_field
            x_out = self.gamma.view(1, 1, -1) * x_norm + self.beta.view(1, 1, -1)
            self._cache = (mu_field, sigma_field)
            return x_out
        else:
            mu_field, sigma_field = self._cache
            x = (x - self.beta.view(1, 1, -1)) / (self.gamma.view(1, 1, -1) + self.eps)
            x = x * sigma_field + mu_field
            return x
```

---

## 四、分 Step 注入计划

### Phase E0：基线复现（Day 1）
- **目标**：确认当前最优基线可复现
- **配置**：Phase D (K=4, patch=12, fixed_fft) + 原版 RevIN + 原版 NI
- **指标**：Test MAE 0.2274, Cong MAE 30.53, Hol_Cong MAE 35.91, TPR_CG 0.464
- **通过标准**：指标误差 < 1%

### Phase E1：矩阵化 NI 注入（Day 2-3）
- **目标**：验证矩阵化 NI 在训练目标中内嵌空间结构的有效性
- **配置**：
  - RevIN = **版 A（原版）**
  - NI -> **Matrix-NI**
  - zeta 固定 = 0.1（不训练，grid search 验证集最优）
  - A 使用现有有向邻接矩阵（若原始无向，按流量梯度方向化）
- **修改点**：
  1. 离线预计算 M_mu^hist, M_sigma^hist（全训练集统计量）
  2. 训练时：用未来统计量构造 M_mu^fut, M_sigma^fut，normalize 输出联合目标
  3. 推理时：用历史统计量矩阵 denormalize
- **判断标准**：
  - Hol_Cong MAE < 35.9 -> 空间感知有效
  - Cong MAE < 30.5 -> 激波传播改善
  - TPR_CG > 0.47 -> 拥堵命中提升
  - SGFE（空间梯度误差）显著低于基线（核心证明指标）

### Phase E2：zeta 可学习化（Day 4，若 E1 有效）
- **目标**：让模型自适应调节场耦合强度
- **配置**：zeta 设为 nn.Parameter，softplus 约束，初始化 0.1
- **训练策略**：前 15 epoch freeze zeta，后 5 epoch 联合微调
- **判断标准**：val loss 不劣化且 test 全面优于 E1

### Phase E3：Field-RevIN 注入（Day 5-6，若 E1/E2 有效）
- **目标**：前端归一化也感知空间结构
- **配置**：RevIN -> **版 B（Field-RevIN）**，eta 固定 0.1
- **修改点**：在 RevIN 的实例归一化中引入场化统计量
- **判断标准**：与 E2 对比，若 Cong MAE 进一步下降 > 1% 则保留

### Phase E4：联合验证与论文指标（Day 7）
- **目标**：生成论文可用的对比证据
- **必跑实验**：
  1. 基线：原版 NI + 原版 RevIN
  2. Matrix-NI：矩阵化 NI + 原版 RevIN
  3. Matrix-NI + zeta-learnable：矩阵化 NI（可学习 zeta）+ 原版 RevIN
  4. Matrix-NI + Field-RevIN：矩阵化 NI + 场化 RevIN
- **必报指标**：
  - MAE / MSE / RMSE（总指标）
  - Cong MAE / Hol_Cong MAE（拥堵指标）
  - TPR_CG（真阳性率）
  - SGFE（空间梯度保真度）——新指标，证明核心贡献
  - 计算耗时对比（证明不增加推理负担）

---

## 五、关键设计决策

| 决策项 | 选择 | 理由 |
|:---|:---|:---|
| 邻接方向性 | 有向（车流方向） | 拥堵反压沿上游传播，信息应逆车流方向渗透 |
| zeta 初始化 | 0.1 | 弱耦合起步，避免破坏主干收敛 |
| 边差值标准差 sigma^Delta_ij 计算 | 全历史窗口 | 节假日样本少，滑动窗口估计不稳 |
| RevIN 顺序 | 先 A 后 B | 版 A 验证 Matrix-NI 核心；版 B 验证前端协同 |
| Loss 修改 | 不修改 | 只改变目标空间坐标系，不增加物理正则项 |
| 稀疏格式 | 稠密 mask（N=30） | 30 节点极小，稠密计算可忽略，代码简洁 |

---

## 六、预期结果与叙事

| 阶段 | Test MAE | Cong MAE | Hol_Cong | TPR_CG | SGFE | 叙事定位 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 基线 (Phase D) | 0.2274 | 30.53 | 35.91 | 0.464 | 高 | — |
| E1: Matrix-NI | 持平 | **↓** | **↓** | **↑** | **显著↓** | "空间结构内生" |
| E2: zeta-learnable | 持平 | **↓↓** | **↓↓** | **↑↑** | **↓↓** | "自适应场耦合" |
| E3: +Field-RevIN | 持平或微↓ | **↓↓↓** | **↓↓↓** | **↑↑↑** | **↓↓↓** | "全链路场化" |

**论文核心句**：

> Unlike prior work that treats each sensor as an isolated time series, we propose Matrix-NI, which encodes the future distribution of the local traffic field (1-hop neighborhood) into the normalization space of diffusion training. The denoiser learns not 'how much will I deviate from my own mean', but 'where will I sit in the spatial field of deviations'. This endows the model with intrinsic spatial consistency without modifying the loss or adding physical constraints.

---

*文档版本: v1.0*  
*创建时间: 2026-05-19*  
*下次更新: Phase E1 验证完成后*
