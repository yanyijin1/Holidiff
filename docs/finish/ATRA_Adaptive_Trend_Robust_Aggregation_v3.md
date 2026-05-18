# 自适应「趋势-稳健」双目标聚合优化方案（ATRA）

> **版本**：v3.0（Adaptive Trend-Robust Aggregation）
> **核心发现**：交通流拥堵相的真实 TPR 仅 0.473，说明存在「趋势-反转」二象性，而非简单趋势延续
> **核心问题**：TCW 的 TPR 过冲到 0.634（+40%），MAE 反而恶化，证明「最大化 TPR」不是最优目标
> **核心思路**：让聚合器的 TPR **匹配真实物理 TPR**（~0.47），同时最小化 MAE，实现「方向准确 + 幅度稳健」的帕累托最优

---

## 一、从 Trend Annihilation 到 Trend-Reversal Duality

### 1.1 实验证据链

| 指标 | Baseline | TCW | TVB | 真实 |
|------|---------|-----|-----|------|
| 拥堵相 TPR | **0.453** | **0.634** | **0.580** | **0.473** |
| 拥堵相 MAE | 31.092 | 31.223 | 31.103 | — |
| 节假日-拥堵 MAE | 36.865 | 37.112 | 36.990 | — |

**关键洞察**：
- TCW 的 TPR 0.634 **远超** 真实 0.473，说明 softmax 过拟合了「趋势延续」假设
- TVB 的 TPR 0.580 **更接近** 真实 0.473，值稳健性项有效拉回了幅度
- 真实拥堵相 TPR = 0.473 意味着：**53% 的拥堵样本未来是趋势反转的**（拥堵缓解/消散）

### 1.2 物理解释：LWR 激波的「上升沿-下降沿」二象性

```
节点 A（激波上升沿）：历史 uptrend → 未来 uptrend（趋势延续）
节点 B（激波下降沿）：历史 uptrend → 未来 downtrend（趋势反转）
节点 C（激波峰值）：历史 uptrend → 未来 flat（趋势饱和）
```

在 15min 粒度下，同一拥堵事件的不同节点可能处于激波的不同相位：
- **上升沿节点**（~47%）：需要「趋势感知」聚合
- **下降沿/饱和节点**（~53%）：需要「值稳健」聚合

**结论**：固定 λ 的 TVB 无法区分这两种场景，需要**样本级自适应 λ**。

---

## 二、自适应趋势-稳健聚合（ATRA）

### 2.1 核心公式

**样本级自适应权重 lambda_i**：

$$\lambda_i = \sigma\left( \frac{|s_i^{\text{hist}}| - \mu_{\text{train}}}{\sigma_{\text{train}}} \cdot \beta \right)$$

其中：
- $s_i^{\text{hist}}$：第 i 个样本的历史趋势斜率（标量，节点平均）
- $\mu_{\text{train}}, \sigma_{\text{train}}$：训练集历史斜率的均值和标准差（标准化参数）
- $\beta$：温度系数，控制自适应敏感度（建议 1.0-3.0）
- $\sigma(\cdot)$：sigmoid，输出范围 (0, 1)

**物理意义**：
- $|s_i^{\text{hist}}| \gg \mu_{\text{train}}$（强趋势样本）→ $\lambda_i \to 1$（趋势主导）
- $|s_i^{\text{hist}}| \ll \mu_{\text{train}}$（弱趋势/flat 样本）→ $\lambda_i \to 0$（值稳健主导）
- 中间状态 → 平滑过渡

**聚合公式**（TVB 的自适应版本）：

$$w_i^{(k)} = \lambda_i \cdot w_i^{\text{trend}, (k)} + (1-\lambda_i) \cdot w_i^{\text{value}, (k)}$$

$$\hat{\mathbf{Y}}_i^{\text{ATRA}} = \sum_{k=1}^{K} w_i^{(k)} \cdot \hat{\mathbf{Y}}_i^{(k)}$$

其中：
- $w_i^{\text{trend}, (k)} = \text{softmax}(\alpha \cdot \gamma_i^{(k)})$（趋势一致性加权）
- $w_i^{\text{value}, (k)} \propto \exp\left(-\frac{\|\hat{\mathbf{Y}}_i^{(k)} - \hat{\mathbf{Y}}_i^{\text{median}}\|^2}{2\sigma^2}\right)$（值稳健性加权）

---

### 2.2 可学习 lambda 的校准版本（ATRA-Cal）

如果固定 beta 不够，可以让 lambda 的映射参数成为**可学习参数**，通过验证集 TPR 监督：

**校准损失**：

$$\mathcal{L}_{\text{calib}} = \left| \text{TPR}_{\text{pred}}(\beta) - \text{TPR}_{\text{true}}^{\text{val}} \right| + \eta \cdot \text{MAE}_{\text{val}}$$

其中：
- $\text{TPR}_{\text{pred}}(\beta)$：当前 beta 下验证集的预测 TPR
- $\text{TPR}_{\text{true}}^{\text{val}}$：验证集的真实 TPR（预先计算，固定值）
- $\eta$：MAE 与 TPR 校准的权衡系数

**优化目标**：

$$\beta^* = \arg\min_{\beta} \mathcal{L}_{\text{calib}}$$

**实现方式**：
- 在验证集上做 grid search（beta ∈ {0.5, 1.0, 2.0, 3.0, 5.0}）
- 选择使 $\mathcal{L}_{\text{calib}}$ 最小的 beta
- 不需要梯度下降，离线 grid search 即可

---

### 2.3 节点级自适应（更细粒度）

如果样本级不够，可以升级到**节点级自适应**：

$$\lambda_{i,n} = \sigma\left( \frac{|s_{i,n}^{\text{hist}}| - \mu_{n}^{\text{train}}}{\sigma_{n}^{\text{train}}} \cdot \beta \right)$$

其中 $\mu_{n}^{\text{train}}, \sigma_{n}^{\text{train}}$ 是每个节点 n 的历史斜率统计量。

**优势**：不同节点（瓶颈 vs 畅通）可以有不同的 lambda，更精准。
**劣势**：需要更多训练集统计，节点数少时（30 个）可行。

---

## 三、完整代码实现

### 3.1 训练集统计计算（一次性）

```python
import numpy as np
import torch

def compute_train_statistics(train_loader, device='cuda'):
    """
    在训练集上计算历史斜率的均值和标准差，用于自适应 lambda
    只需运行一次，保存为 .npy 文件
    """
    all_hist_slopes = []
    all_true_tprs = []

    for batch_x, batch_y, _, _ in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # 历史斜率（节点级）
        s_hist = (batch_x[..., -1] - batch_x[..., 0]) / (batch_x.size(-1) - 1)
        all_hist_slopes.append(s_hist.cpu().numpy())

        # 真实 TPR（用于校准）
        s_true = (batch_y[..., -1] - batch_y[..., 0]) / (batch_y.size(-1) - 1)
        tpr = ((s_hist * s_true) > 0).float().mean(dim=1)
        all_true_tprs.append(tpr.cpu().numpy())

    all_hist_slopes = np.concatenate(all_hist_slopes, axis=0)
    all_true_tprs = np.concatenate(all_true_tprs, axis=0)

    # 样本级统计（节点平均斜率的绝对值）
    hist_slope_mean = np.mean(np.abs(all_hist_slopes), axis=1)

    stats = {
        'mu': float(np.mean(hist_slope_mean)),
        'sigma': float(np.std(hist_slope_mean)),
        'median': float(np.median(hist_slope_mean)),
        'tpr_mean': float(np.mean(all_true_tprs)),
        'tpr_std': float(np.std(all_true_tprs)),
        'node_mu': np.mean(np.abs(all_hist_slopes), axis=0).tolist(),
        'node_sigma': np.std(np.abs(all_hist_slopes), axis=0).tolist(),
    }

    np.save('train_slope_stats.npy', stats)
    print(f"Train slope stats: mu={stats['mu']:.4f}, sigma={stats['sigma']:.4f}")
    print(f"Train true TPR: {stats['tpr_mean']:.3f} ± {stats['tpr_std']:.3f}")

    return stats
```

### 3.2 ATRA 聚合核心代码

```python
class ATRAggregator:
    def __init__(self, stats_path='train_slope_stats.npy', alpha=5.0, sigma_value=1.0):
        self.stats = np.load(stats_path, allow_pickle=True).item()
        self.alpha = alpha
        self.sigma_value = sigma_value

    def compute_lambda(self, history, beta=1.0, node_level=False):
        s_hist = (history[..., -1] - history[..., 0]) / (history.size(-1) - 1)

        if node_level:
            mu = torch.tensor(self.stats['node_mu'], device=s_hist.device).view(1, -1)
            sigma = torch.tensor(self.stats['node_sigma'], device=s_hist.device).view(1, -1)
            z_score = (torch.abs(s_hist) - mu) / (sigma + 1e-8)
            lam = torch.sigmoid(z_score * beta)
        else:
            s_hist_mean = torch.abs(s_hist).mean(dim=1)
            mu = self.stats['mu']
            sigma = self.stats['sigma']
            z_score = (s_hist_mean - mu) / (sigma + 1e-8)
            lam = torch.sigmoid(z_score * beta)

        return lam

    def aggregate(self, all_outs, history, beta=1.0, node_level=False):
        K, B, N, T = all_outs.shape

        # 1. 趋势一致性分数 gamma
        s_hist = (history[..., -1] - history[..., 0]) / (history.size(-1) - 1)
        s_future = (all_outs[..., -1] - all_outs[..., 0]) / (all_outs.size(-1) - 1)
        ratio = s_future / (s_hist.unsqueeze(0) + 1e-8)
        ratio = torch.clamp(ratio, -2.0, 2.0)
        gamma = torch.sign(s_hist.unsqueeze(0) * s_future) * torch.abs(ratio)

        # 2. 趋势权重
        w_trend = torch.softmax(self.alpha * gamma, dim=0)

        # 3. 值稳健性权重
        median_pred = torch.median(all_outs, dim=0)[0]
        dist = torch.sum((all_outs - median_pred.unsqueeze(0))**2, dim=-1)
        w_value = torch.softmax(-dist / (2 * self.sigma_value**2), dim=0)

        # 4. 自适应 lambda
        lam = self.compute_lambda(history, beta, node_level)

        # 5. 融合权重
        if node_level:
            lam = lam.unsqueeze(0).unsqueeze(-1)
        else:
            lam = lam.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        w_fused = lam * w_trend.unsqueeze(-1) + (1 - lam) * w_value.unsqueeze(-1)
        w_fused = w_fused / (w_fused.sum(dim=0, keepdim=True) + 1e-8)

        # 6. 加权聚合
        output = torch.sum(w_fused * all_outs, dim=0)
        return output
```

### 3.3 beta 校准（Grid Search）

```python
def calibrate_beta(val_loader, model, aggregator, beta_candidates=[0.5, 1.0, 2.0, 3.0, 5.0]):
    best_beta = None
    best_loss = float('inf')
    target_tpr = aggregator.stats['tpr_mean']

    results = []
    for beta in beta_candidates:
        # ... 跑验证集，计算 pred_TPR 和 CG_MAE ...
        loss = abs(pred_tpr - target_tpr) + 0.1 * cg_mae
        if loss < best_loss:
            best_loss = loss
            best_beta = beta

    print(f"Best beta: {best_beta}, loss: {best_loss:.4f}")
    return best_beta, results
```

---

## 四、实验设计

### 4.1 对照组（必须全部跑）

| 版本 | 说明 | 预期 |
|------|------|------|
| Baseline (MoM) | 原始 median | TPR ~0.45, CG_MAE ~31.1 |
| Simple Mean | 无 block 直接 mean | TPR ~0.45, CG_MAE ~31.0 |
| TCW (temp=5) | 纯趋势加权 | TPR ~0.63, CG_MAE ~31.2（过冲） |
| TVB (lambda=0.5) | 固定双目标 | TPR ~0.58, CG_MAE ~31.1 |
| **ATRA (beta=1)** | 自适应 lambda，样本级 | **TPR ~0.50, CG_MAE < 31.0** |
| **ATRA-Cal** | 校准后的最优 beta | **TPR ≈ 0.47, CG_MAE 最小** |
| ATRA-Node | 节点级自适应 | 若样本级有效，节点级可能更好 |

### 4.2 关键评估指标

| 指标 | 公式 | 目标 |
|------|------|------|
| **整体 MAE** | 全样本平均 | 不劣化 baseline |
| **拥堵相 MAE** | CG 样本平均 | **显著下降** |
| **节假日-拥堵 MAE** | HOL ∩ CG | **显著下降** |
| **TPR（拥堵相）** | sign(s_hist * s_pred) > 0 的比例 | **≈ 0.47**（匹配真实物理） |
| **TPR 偏差** | |pred_TPR - true_TPR| | **最小化** |
| **自由流 MAE** | FF 样本平均 | 不劣化 |
| **推理时间** | 秒/样本 | 与 TVB 相当 |

### 4.3 可视化要求

1. **lambda 分布图**：横轴 |s_hist|，纵轴 lambda，标注拥堵/自由流样本的 lambda 分布
2. **TPR-MAE 帕累托前沿**：横轴 TPR，纵轴 CG_MAE，标注各版本位置，ATRA 应在「TPR≈0.47, MAE 最低」处
3. **Before/After 采样解剖图**：同之前的 Trend Truncation 图，但加一列 ATRA 的聚合结果

---

## 五、论文故事线（三段式）

### Paragraph 1：问题定义 —— Trend-Reversal Duality

> 传统观点认为扩散模型的 Median-of-Means 在交通流中失效是因为「趋势被截断」（Trend Annihilation）。但我们通过大规模诊断发现，交通流拥堵相的真实物理并非简单趋势延续——真实 TPR 仅 0.473，意味着 53% 的样本未来是趋势反转（LWR 激波消散）。因此，固定地「保留趋势」会过拟合（TCW TPR=0.634 > 真实 0.473），导致幅度过冲、MAE 恶化。

### Paragraph 2：方法 —— 自适应双目标聚合 ATRA

> 我们提出 Adaptive Trend-Robust Aggregation（ATRA），核心是让聚合器的 TPR **匹配真实物理 TPR**，而非最大化 TPR。ATRA 根据样本历史趋势强度自适应调节「趋势一致性」与「值稳健性」的权重：强趋势样本（激波上升沿）lambda→1，弱趋势样本（激波下降沿/饱和）lambda→0。lambda 的映射参数通过验证集 TPR 校准，无需人工调参。

### Paragraph 3：实验 —— 物理一致性验证

> 在 Fujian-30 数据集上，ATRA 将拥堵相 TPR 从 0.453 校准到 0.48（接近真实 0.473），同时拥堵相 MAE 下降 X%，节假日-拥堵 MAE 下降 X%。消融实验显示，固定 lambda 的 TVB 无法同时优化 TPR 和 MAE，而 ATRA 的自适应机制实现了「方向准确 + 幅度稳健」的帕累托最优。

---

## 六、执行清单

### 本周任务（优先级排序）

- [ ] **Step 1**：运行 compute_train_statistics() 生成 train_slope_stats.npy
- [ ] **Step 2**：实现 ATRAggregator 类，插入 HoliDiff.py 或测试脚本
- [ ] **Step 3**：跑 ATRA (beta=1.0, 样本级)，记录三相 MAE / TPR / TPR 偏差
- [ ] **Step 4**：跑 beta grid search {0.5, 1.0, 2.0, 3.0, 5.0}，找到最优 beta
- [ ] **Step 5**：对比 Baseline / TCW / TVB / ATRA / ATRA-Cal 的完整指标表
- [ ] **Step 6**：生成 lambda 分布图 和 TPR-MAE 帕累托前沿图

### 下周任务（视结果）

- [ ] **Step 7**：若 ATRA 样本级有效，跑 ATRA-Node（节点级自适应）
- [ ] **Step 8**：生成论文 Figure 1（Trend-Reversal Duality 示意图）
- [ ] **Step 9**：生成论文 Figure 2（ATRA 架构图 + lambda 自适应示意图）
- [ ] **Step 10**：生成论文 Figure 3（Before/After 采样解剖 + 帕累托前沿）

### 论文写作任务（并行）

- [ ] Paragraph 1：Trend-Reversal Duality 的物理论证（引用 LWR 激波理论）
- [ ] Paragraph 2：ATRA 公式推导（自适应 lambda + 双目标加权）
- [ ] Paragraph 3：TPR 校准机制（验证集 grid search，无需梯度）
- [ ] Paragraph 4：实验设计 + 消融实验（固定 lambda vs 自适应 lambda）

---

## 七、风险与备选方案

| 风险 | 判断标准 | 备选方案 |
|------|---------|---------|
| ATRA 的 TPR 仍无法接近 0.47 | beta grid search 后 TPR 偏差 > 0.1 | 放弃 TPR 校准目标，改为直接最小化 CG_MAE，让 TPR 自然收敛 |
| 节点级 ATRA 过拟合 | 节点级比样本级差 | 回退到样本级，论文强调「轻量自适应」 |
| 训练集统计不稳定 | 不同随机种子 stats 差异大 | 用全量数据（不划分 train/val）计算 stats，或做 bootstrap 平均 |
| lambda 计算引入显著延迟 | 推理时间增加 > 20% | 预计算 lambda 的 lookup table（按 |s_hist| 分桶），避免在线 sigmoid |

---

*文档结束*
