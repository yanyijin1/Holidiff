# HOliHOliDiff-MoM 宏观聚合改进方案：趋势感知自适应聚合

> **版本**：v2.0（Trend-Aware Edition）
> **核心问题**：固定 MoM 的「值中心主义」聚合在交通流相态跃迁场景下系统性截断趋势（Trend Truncation）
> **根因**：`sample_times=10, n_blocks=5` 导致每组仅 2 个采样，随机 shuffle 将稀缺的「趋势延续采样」与大量「保守采样」配对，组内 mean 直接湮灭趋势信号
> **目标**：设计以「趋势一致性」为核心的自适应聚合策略，在保持自由流/常规日性能不劣化的前提下，恢复拥堵相/节假日的趋势延续能力

---

## 一、问题重定义：从 Median Bias 到 Trend Annihilation

### 1.1 图证据（Sample 3817/2474）

**左图现象**：
- History（蓝实线）：从深夜 ~100 一路爬升到分割点 ~500，**趋势明确**
- Future True（蓝虚线）：趋势延续并加速，冲向 ~800
- Future Pred（红线）：**flat 在 ~480**，趋势完全丢失
- 10 条灰色采样中**确有上升趋势样本**（冲到 500+），证明生成器**有能力**预测对

**右图现象**：
- 5 个 block mean（彩色线）**全部 flat 在 400-500**
- Ground Truth（蓝虚线）从 0 冲到 600+
- **Median 只能选 flat，因为所有 block mean 已经 flat**

### 1.2 根因：「每组 2 样本」的选举暴政

假设 10 条采样中只有 **2 条**预测正确趋势（↑600, ↑580），其余 8 条保守（→400 左右）：

随机分 5 组每组 2 个：
- Block 1: [↑600, →400] → mean = **500**（还有点趋势）
- Block 2: [↑580, →410] → mean = **495**（还有点趋势）
- Block 3: [→390, ↓350] → mean = **370**
- Block 4: [→420, →400] → mean = **410**
- Block 5: [↓380, ↓360] → mean = **370**

Median 从 [500, 495, 370, 410, 370] 中选第 3 小 → **410**，flat。

**关键洞察**：Median 不是凶手，**「每组 2 样本的随机 mean」才是**。正确趋势信号在组内平均阶段就被保守样本稀释殆尽，导致 5 个 block mean 全部 flat，median 无论怎么选都只能选 flat。

### 1.3 与传统 Median Bias 的区别

| 问题类型 | 传统 Median Bias | 我们的 Trend Annihilation |
|---------|-----------------|------------------------|
| 分布假设 | 对称分布，median ≈ mean | 两极分化（趋势派 vs 保守派） |
| 失效位置 | 组间 median 选择 | 组内 mean 稀释 |
| 修复策略 | 换 weighted median / trimmed mean | 必须先让 block mean 保留趋势 |
| 物理对应 | 一般时序的随机波动 | 交通流的相态跃迁惯性 |

---

## 二、文献支撑

### 2.1 气候模型：Ensemble Median 的 Trend Underestimation

> *"The ensemble median of climate models consistently underestimates historical trends compared to observations."*
> — CMIP6 / Nature / JGR 多篇论文

**迁移**：气候模型 ensemble 成员对未来趋势分歧时，median 取中间值系统性保守。交通流拥堵相的采样轨迹 = 气候模型 ensemble 成员，median 把「强上升趋势」当成 outlier 过滤。

### 2.2 MAPA：Mean vs Median on Trended Data

> *"For trended data, the good long-term performance of MAPA can be explained by temporal aggregation functioning as a low-pass filter that enhances estimation of trend. MAPA(Mean) is more accurate, while MAPA(Median) is less biased."*
> — Kourentzes et al.

**迁移**：对于**有趋势的数据**，简单 mean 比 median 更准确。我们的问题不是「预测不准」，而是「聚合器把对的预测扔了」。

### 2.3 扩散模型趋势分解（2024-2025 新方向）

> *"Existing diffusion models lack mechanisms to preserve structured components (trend, seasonality, residual), often leading to inaccurate long-term predictions."*
> — AAAI/NeurIPS/ICLR 2024-2025

**迁移**：现有工作解决「生成器端」的趋势保留（Fourier decomposition + diffusion），我们解决「聚合端」的趋势截断，两者互补。

### 2.4 交通流物理：LWR 相态跃迁惯性

> *"Under the LWR framework, traffic flow exhibits phase-transition inertia: once a congestion front forms, the density wave propagates with deterministic velocity."*
> — Lighthill-Whitham-Richards

**迁移**：历史 uptrend 是物理惯性的强预测信号，「趋势一致性」是比「值居中」更物理的聚合准则。

---

## 三、改进方案总览

### 3.1 基础改进（不改趋势逻辑，改结构/数量）

| 版本 | 名称 | 核心改动 | 预期效果 |
|------|------|---------|---------|
| A | More-Sample MoM (MS-MoM) | `K=20, M=4`（每组 4 采样） | 正确趋势样本更可能在同 block 聚集，mean 保留趋势 |
| B | Trend-Aware Grouping (TAG-MoM) | 按趋势强度排序后相邻分组，替代随机 shuffle | 趋势派 block vs 保守派 block，block mean 双峰化 |
| C | Trimmed Mean of Means (TMM) | 对 5 个 block mean 做截尾 mean（α=0.2） | 比 median 更灵活，可能保留部分趋势 |
| D | Variance-Adaptive WM (VAWM) | 按组内方差 softmax 加权 | 方差代理弱（ρ=0.14），预期效果有限，作为对照 |
| E | Trend-Aware Trimming (TAT) | 结构化分组 + 截掉趋势分数最低的 block | 直接剔除保守 block，最激进 |
| F | Top-K Trend Selection (TKS) | 跳过 block，直接从 10 条采样选趋势最强的 K' 条取 mean | 完全绕过 block mean 湮灭，但丧失稳健性 |
| G | Phase-Adaptive Hybrid (PAH) | 根据流相态动态选 mean/median | 自由流 mean，拥堵 MoM，但拥堵时 MoM 本身失效，只能作对照 |

### 3.2 趋势感知改进（以趋势一致性为核心聚合依据）

| 版本 | 名称 | 核心思想 |
|------|------|---------|
| 1 | Trend-Consistency WM (TCW) | 对 K 条采样按趋势一致性指数加权平均 |
| 2 | Trend-Guided Block Selection (TGBS) | 保留 block 结构，截掉趋势方向与历史反向的 block，剩余取 mean |
| 3 | Trend Trimming Mean (TTM) | 在原始 K 条采样层面按趋势一致性截尾（保留 top-60%），再取 mean |
| 4 | Trend-Value Bi-Objective (TVB) | 双目标加权：趋势一致性 + 值稳健性，λ 控制偏好 |

---

## 四、详细公式

### 4.0 公共定义

**历史趋势斜率**（节点级，标量）：
$$s_n^{\text{hist}} = \frac{X_{n, T_{\text{hist}}} - X_{n, 1}}{T_{\text{hist}} - 1}$$

**第 k 条采样轨迹的未来趋势斜率**：
$$s_n^{(k)} = \frac{\hat{Y}_{n, T_{\text{pred}}}^{(k)} - \hat{Y}_{n, 1}^{(k)}}{T_{\text{pred}} - 1}$$

**趋势一致性分数**（方向 + 幅度匹配，范围 [-τ, τ]）：
$$\gamma_n^{(k)} = \text{sign}\left(s_n^{\text{hist}} \cdot s_n^{(k)}\right) \cdot \min\left( \left| \frac{s_n^{(k)}}{s_n^{\text{hist}} + \epsilon} \right|, \tau \right)$$
其中 τ 为截断上限（如 τ=2，防止历史斜率接近 0 时爆炸）。

---

### 4.1 版本 A：More-Sample MoM (MS-MoM)

仅改超参：
$$K = 20, \quad B = 5, \quad M = 4, \quad R = 5$$

其余与基线相同：
$$\bar{\mathbf{Y}}_b^{(r)} = \frac{1}{4}\sum_{m=1}^{4} \hat{\mathbf{Y}}^{(\pi_r(4b-4+m))}, \quad \hat{\mathbf{Y}}_{\text{MoM}}^{(r)} = \text{median}\left(\{\bar{\mathbf{Y}}_b^{(r)}\}_{b=1}^{5}\right)$$

---

### 4.2 版本 B：Trend-Aware Grouping (TAG-MoM)

**Step 1：计算每条采样轨迹的趋势分数**
$$\tau_k = s_n^{(k)} = \frac{\hat{Y}_{n, T}^{(k)} - \hat{Y}_{n, 1}^{(k)}}{T - 1}$$

**Step 2：排序索引**
$$\sigma = \text{argsort}(\{\tau_k\}_{k=1}^{K}), \quad \tau_{\sigma(1)} \le \dots \le \tau_{\sigma(K)}$$

**Step 3：相邻聚类分组**（替代随机 shuffle）
$$\mathcal{G}_b = \left\{ \hat{\mathbf{Y}}^{(\sigma(bM-M+1))}, \dots, \hat{\mathbf{Y}}^{(\sigma(bM))} \right\}$$

**Step 4-6**：组内 mean、组间 median、重复随机化同基线。

---

### 4.3 版本 C：Trimmed Mean of Means (TMM)

**排序 block means**（按标量范数或末值）：
$$\|\bar{\mathbf{Y}}_{(1)}\| \le \|\bar{\mathbf{Y}}_{(2)}\| \le \dots \le \|\bar{\mathbf{Y}}_{(B)}\|$$

**截尾比例 α**（建议 α=0.2，即截掉 1 最低 + 1 最高）：
$$\hat{\mathbf{Y}}_{\text{TMM}}^{(\alpha)} = \frac{1}{B - 2\lfloor \alpha B \rfloor} \sum_{b=\lfloor \alpha B \rfloor + 1}^{B - \lfloor \alpha B \rfloor} \bar{\mathbf{Y}}_{(b)}$$

- α=0.2：对中间 3 个 block mean 取 mean
- α=0.4：退化为 median（只留中间 1 个）
- α=0：退化为 simple mean（所有 block mean 平均）

---

### 4.4 版本 D：Variance-Adaptive WM (VAWM)

**组内方差**：
$$v_b = \frac{1}{M}\sum_{m=1}^{M} \left\| \hat{\mathbf{Y}}^{(k_{b,m})} - \bar{\mathbf{Y}}_b \right\|^2$$

**Softmax 权重**（温度 τ，建议 τ ∈ {0.001, 0.01, 0.1}）：
$$w_b = \frac{\exp(-v_b / \tau)}{\sum_{j=1}^{B} \exp(-v_j / \tau)}, \quad \hat{\mathbf{Y}}_{\text{VAWM}} = \sum_{b=1}^{B} w_b \cdot \bar{\mathbf{Y}}_b$$

---

### 4.5 版本 E：Trend-Aware Trimming (TAT)

**Block 级趋势一致性**：
$$\bar{\gamma}_n^{(b)} = \frac{1}{M}\sum_{\hat{\mathbf{Y}} \in \mathcal{G}_b} \gamma_n(\hat{\mathbf{Y}})$$

**截掉趋势分数最低的 α 比例 block**：
$$\mathcal{B}_{\text{keep}} = \left\{ b \mid \bar{\gamma}_n^{(b)} \ge \text{quantile}_{\alpha}(\{\bar{\gamma}_n^{(j)}\}) \right\}$$

**聚合**（对保留 block 取 mean，设最小保留数 B_min=2）：
$$\hat{\mathbf{Y}}_{\text{TAT}} = \begin{cases}\displaystyle \frac{1}{|\mathcal{B}_{\text{keep}}|} \sum_{b \in \mathcal{B}_{\text{keep}}} \bar{\mathbf{Y}}_b & \text{if } |\mathcal{B}_{\text{keep}}| \ge B_{\min} \\ \displaystyle \text{median}(\{\bar{\mathbf{Y}}_b\}) & \text{otherwise (fallback)}\end{cases}$$

---

### 4.6 版本 F：Top-K Trend Selection (TKS)

**趋势分数**：τ_k = s_n^{(k)}（末值减初值除以长度）

**选择 top-K' 条**（建议 K'=3 或 5）：
$$\mathcal{K} = \text{top-}K' \text{ of } \{\tau_k\}, \quad \hat{\mathbf{Y}}_{\text{TKS}} = \frac{1}{K'}\sum_{k \in \mathcal{K}} \hat{\mathbf{Y}}^{(k)}$$

---

### 4.7 版本 G：Phase-Adaptive Hybrid (PAH)

**相态判断**（基于历史均值）：
$$\phi(\mathbf{X}) = \frac{1}{NT}\sum_{n,t} X_{n,t}, \quad \theta_c = \text{quantile}_{66\%}(\phi(\mathbf{X}_{\text{train}}))$$

**动态选择**：
$$\hat{\mathbf{Y}}_{\text{PAH}} = \begin{cases}\displaystyle \frac{1}{K}\sum_{k=1}^{K} \hat{\mathbf{Y}}^{(k)} & \text{if } \phi(\mathbf{X}) < \theta_c \quad (\text{自由流，高效 mean}) \\ \displaystyle \text{MoM}(\{\hat{\mathbf{Y}}^{(k)}\}) & \text{if } \phi(\mathbf{X}) \ge \theta_c \quad (\text{拥堵，稳健 MoM})\end{cases}$$

---

### 4.8 版本 1：Trend-Consistency WM (TCW)

**权重**（α 控制敏感度，建议 α=5 或 10）：
$$w_n^{(k)} = \frac{\exp(\alpha \cdot \gamma_n^{(k)})}{\sum_{j=1}^{K} \exp(\alpha \cdot \gamma_n^{(j)})}$$

**聚合**：
$$\hat{\mathbf{Y}}_n^{\text{TCW}} = \sum_{k=1}^{K} w_n^{(k)} \cdot \hat{\mathbf{Y}}_n^{(k)}$$

**特性**：
- α → ∞：退化为只选最一致的 1 条
- α → 0：退化为 simple mean

---

### 4.9 版本 2：Trend-Guided Block Selection (TGBS)

**Block 级趋势一致性**：
$$\bar{\gamma}_n^{(b)} = \frac{1}{M}\sum_{m=1}^{M} \gamma_n^{(k_{b,m})}$$

**筛选**（只保留与历史趋势同向的 block）：
$$\mathcal{B}_{\text{valid}} = \left\{ b \mid \bar{\gamma}_n^{(b)} > 0 \right\}$$

**聚合**（对 valid block 取 mean，设 B_min=2）：
$$\hat{\mathbf{Y}}_n^{\text{TGBS}} = \begin{cases}\displaystyle \frac{1}{|\mathcal{B}_{\text{valid}}|} \sum_{b \in \mathcal{B}_{\text{valid}}} \bar{\mathbf{Y}}_n^{(b)} & \text{if } |\mathcal{B}_{\text{valid}}| \ge 2 \\ \displaystyle \text{median}(\{\bar{\mathbf{Y}}_b\}) & \text{otherwise}\end{cases}$$

---

### 4.10 版本 3：Trend Trimming Mean (TTM)

**截尾阈值**（保留 top-(1-α) 趋势一致性，建议 α=0.4 即保留 60%）：
$$\eta = \text{quantile}_{\alpha}(\{\gamma_n^{(k)}\}), \quad \mathcal{K}_{\text{keep}} = \left\{ k \mid \gamma_n^{(k)} \ge \eta \right\}$$

**聚合**：
$$\hat{\mathbf{Y}}_n^{\text{TTM}} = \frac{1}{|\mathcal{K}_{\text{keep}}|} \sum_{k \in \mathcal{K}_{\text{keep}}} \hat{\mathbf{Y}}_n^{(k)}$$

---

### 4.11 版本 4：Trend-Value Bi-Objective (TVB)

**趋势权重**：
$$w_n^{\text{trend}, (k)} = \text{softmax}(\alpha \cdot \gamma_n^{(k)})$$

**值权重**（基于与 median 的距离）：
$$w_n^{\text{value}, (k)} \propto \exp\left( -\frac{\|\hat{\mathbf{Y}}_n^{(k)} - \hat{\mathbf{Y}}_n^{\text{median}}\|^2}{2\sigma^2} \right)$$

**融合**（λ ∈ [0,1]）：
$$w_n^{(k)} = \lambda \cdot w_n^{\text{trend}, (k)} + (1-\lambda) \cdot w_n^{\text{value}, (k)}$$

$$\hat{\mathbf{Y}}_n^{\text{TVB}} = \sum_{k=1}^{K} w_n^{(k)} \cdot \hat{\mathbf{Y}}_n^{(k)}$$

---

## 五、测试优先级与实验设计

### 5.1 推荐测试顺序

| 轮次 | 版本 | 目的 | 代码量 | 推理成本 |
|------|------|------|--------|---------|
| **Round 1** | **版本 3 (TTM)** | 验证「直接截掉反向趋势采样」是否有效 | 10 行 | 不变 |
| **Round 1** | **版本 F (TKS)** | 验证「绕过 block mean」是否直接保留趋势 | 10 行 | 不变 |
| **Round 1** | **版本 A (MS-MoM)** | 验证「增加采样数」是否解决趋势湮灭 | 改 1 参数 | ×2 |
| **Round 2** | **版本 2 (TGBS)** | 验证「block 内趋势筛选」是否比 random shuffle 有效 | 15 行 | 略增 |
| **Round 2** | **版本 B (TAG-MoM)** | 验证「结构化分组」是否让 block mean 双峰化 | 15 行 | 略增 |
| **Round 2** | **版本 C (TMM)** | 验证「median 换截尾 mean」是否更灵活 | 改 1 行 | 不变 |
| **Round 3** | **版本 1 (TCW)** | 若 TTM 有效，提供连续加权版本 | 10 行 | 不变 |
| **Round 3** | **版本 4 (TVB)** | 若多版本有效，融合为最终方案 | 20 行 | 略增 |
| **对照** | **版本 D (VAWM)** | 验证方差代理是否有效（预期 negative result） | 10 行 | 不变 |
| **对照** | **版本 G (PAH)** | 验证简单相态切换不够 | 10 行 | 不变 |

### 5.2 必须保留的对照组

| 对照 | 说明 |
|------|------|
| 原始 MoM（median） | 基线 |
| Simple Mean（无 block） | 验证「去掉 median 直接用 mean」的效果上限 |
| Oracle（直接用 GT 趋势选采样） | 验证「如果趋势判断完美，能提升多少」的上界 |

---

## 六、代码插入指南

### 6.1 公共：趋势一致性计算函数

在 `HoliDiff.py` 或测试脚本中定义：

```python
import torch
import numpy as np

def compute_trend_consistency(history, future_samples, epsilon=1e-8, tau=2.0):
    """
    计算趋势一致性分数
    Args:
        history: (B, N, T_hist) 历史序列
        future_samples: (K, B, N, T_pred) K条采样轨迹
    Returns:
        gamma: (K, B, N) 趋势一致性分数
    """
    # 历史斜率
    s_hist = (history[..., -1] - history[..., 0]) / (history.size(-1) - 1)  # (B, N)

    # 每条采样的未来斜率
    s_future = (future_samples[..., -1] - future_samples[..., 0]) / (future_samples.size(-1) - 1)  # (K, B, N)

    # 方向一致性 + 幅度比截断
    ratio = s_future / (s_hist.unsqueeze(0) + epsilon)
    ratio = torch.clamp(ratio, -tau, tau)
    gamma = torch.sign(s_hist.unsqueeze(0) * s_future) * torch.abs(ratio)

    return gamma  # (K, B, N)
```

### 6.2 版本 3 (TTM) 插入代码

在 `HoliDiff.py` 的 `extract_consensus` 或测试脚本中：

```python
# 假设已有 all_outs: (K=10, B, N, T_pred) 和 history: (B, N, T_hist)

# 1. 计算趋势一致性
gamma = compute_trend_consistency(history, all_outs)  # (10, B, N)

# 2. 按 gamma 排序，保留 top-60%（即截掉 bottom 40%）
K_keep = int(K * 0.6)  # = 6
# 对每个 (B, N) 位置独立排序
gamma_sorted, indices = torch.sort(gamma, dim=0, descending=True)  # (10, B, N)
keep_indices = indices[:K_keep]  # (6, B, N)

# 3. gather 保留的采样
# all_outs: (10, B, N, T)
# keep_indices: (6, B, N)
# 需要逐位置 gather
B, N, T = all_outs.shape[1], all_outs.shape[2], all_outs.shape[3]
kept_samples = []
for b in range(B):
    for n in range(N):
        idx = keep_indices[:, b, n]  # (6,)
        kept = all_outs[idx, b, n, :]  # (6, T)
        kept_samples.append(kept.mean(dim=0))  # (T,)

# 堆叠回 (B, N, T)
output_ttm = torch.stack(kept_samples, dim=0).view(B, N, T)
```

**注意**：上面的逐位置 gather 在 PyTorch 中可用 `torch.gather` 或 `advanced indexing` 向量化，避免 for 循环。

### 6.3 版本 F (TKS) 插入代码

```python
# 1. 计算趋势分数（末值减初值）
trend_scores = all_outs[..., -1] - all_outs[..., 0]  # (K, B, N)

# 2. 选 top-K'（如 K'=3）
K_prime = 3
topk_values, topk_indices = torch.topk(trend_scores, k=K_prime, dim=0)  # (3, B, N)

# 3. gather 并 mean
# 向量化 gather（需确认 PyTorch 版本支持）
B, N, T = all_outs.shape[1], all_outs.shape[2], all_outs.shape[3]
output_tks = torch.zeros(B, N, T, device=all_outs.device)
for b in range(B):
    for n in range(N):
        idx = topk_indices[:, b, n]  # (3,)
        output_tks[b, n] = all_outs[idx, b, n].mean(dim=0)
```

### 6.4 版本 2 (TGBS) 插入代码

```python
# 在 _consensus_reduce 中，替换随机 shuffle 为趋势排序分组

def _consensus_reduce_trend_aware(self, tensor, history):
    """
    tensor: (K, B, N, T) = all_outs
    history: (B, N, T_hist)
    """
    K, B, N, T = tensor.shape
    n_blocks = self.n_blocks  # 5
    block_size = K // n_blocks  # 2

    # 1. 计算每条采样的趋势分数
    trend_scores = tensor[..., -1] - tensor[..., 0]  # (K, B, N)

    means = []
    for b in range(n_blocks):
        # 2. 对每个 (B, N) 位置，按趋势排序后取当前 block 的采样
        # 注意：这里需要对每个 (b, n) 独立排序，工程上较复杂
        # 简化方案：对每个 batch 样本整体排序（假设节点间趋势一致）

        # 更实用的方案：按 batch 内平均趋势排序
        avg_trend = trend_scores.mean(dim=(1, 2))  # (K,)
        _, sort_idx = torch.sort(avg_trend, descending=True)

        start = b * block_size
        end = start + block_size
        block_idx = sort_idx[start:end]  # (2,)

        block = tensor[block_idx]  # (2, B, N, T)
        block_mean = block.mean(dim=0)  # (B, N, T)
        means.append(block_mean)

    # 3. 筛选趋势同向的 block（简化：用 block_mean 末值 vs 初值判断）
    # 若 block_mean 末值 > 初值，认为同向
    valid_means = []
    for bm in means:
        if (bm[..., -1] > bm[..., 0]).float().mean() > 0.5:  # 多数节点同向
            valid_means.append(bm)

    if len(valid_means) >= 2:
        return torch.stack(valid_means).mean(dim=0)
    else:
        return torch.median(torch.stack(means), dim=0)[0]
```

**注意**：TGBS 的逐节点/逐 batch 独立排序工程较复杂，上面的 `avg_trend` 简化方案是实用折中。若需要严格逐节点，建议用 `torch_scatter` 或 `einops` 做向量化。

---

## 七、评估指标（必须记录）

### 7.1 核心指标

| 指标 | 公式 | 目的 |
|------|------|------|
| **三相 MAE** | 自由流 / 过渡 / 拥堵 各自 MAE | 验证拥堵相提升，自由流不劣化 |
| **节假日/常规日 MAE** | 分时段 MAE | 验证漂移尖峰被削平 |
| **趋势保留率 (TPR)** | 拥堵相样本中聚合结果趋势方向与历史一致的比例 | 验证趋势方向被保留 |
| **趋势幅度误差 (TME)** | 聚合结果斜率与真实未来斜率的差距 | 验证趋势幅度准确 |
| **Block Mean 双峰度** | 5 个 block mean 的标准差 | 若 TAG-MoM 有效，此值应显著增大 |
| **推理时间** | 秒/样本 | 版本 A 翻倍，其余不变或略增 |

### 7.2 诊断性指标（用于分析）

| 指标 | 说明 |
|------|------|
| 被截掉采样占比 | TTM/TGBS 中，被判定为「反向趋势」而剔除的采样比例 |
| 保留采样趋势分布 | 保留采样的 gamma 分布 vs 被截掉采样的 gamma 分布 |
| Top-K 采样与 GT 相关性 | TKS 中 top-3 采样与真实值的 Pearson 相关 |

---

## 八、决策树

```
Round 1 结果
    |
    |- 版本 3 (TTM) 拥堵相 MAE 下降 > 10%？
    |   |- 是 -> 趋势截尾有效，继续 Round 2 优化截尾阈值 alpha
    |   |- 否 -> 趋势一致性分数 gamma 可能不准确，或生成器正确采样太少
    |
    |- 版本 F (TKS) 拥堵相 MAE 下降 > 15%？
    |   |- 是 -> block 结构本身是问题，最终方案应减少/绕过 block
    |   |- 否 -> 生成器采样质量是瓶颈，需先改进微观生成器
    |
    |- 版本 A (MS-MoM) 拥堵相 MAE 下降 > 8%？
    |   |- 是 -> 采样数不足是主因，可结合 TTM 做「增采样 + 趋势截尾」
    |   |- 否 -> 不是采样数问题，是分组/聚合逻辑问题
    |
Round 2 结果
    |
    |- 版本 2 (TGBS) 优于版本 B (TAG-MoM)？
    |   |- 是 -> 「筛选」比「重分组」更有效，最终走 TTM/TGBS 路线
    |   |- 否 -> 「结构化分组」更有潜力，继续优化分组度量
    |
    |- 版本 C (TMM, alpha=0.2) 优于 Median？
    |   |- 是 -> 截尾 mean 比 median 更灵活，可融入 TGBS
    |   |- 否 -> 问题不是 median 本身，是 block mean 已 flat
    |
Round 3 结果
    |
    |- 版本 1 (TCW) 与版本 3 (TTM) 接近？
    |   |- 是 -> 连续加权与硬截尾等效，选更简单的 TTM
    |   |- 否 -> 加权版本更平滑，选 TCW
    |
    |- 版本 4 (TVB) 在 lambda=0.7 时最优？
        |- 是 -> 趋势主导 + 值辅助是最佳平衡，定稿 TVB
        |- 否 -> 纯趋势（lambda=1）或纯值（lambda=0）更优，简化方案
```

---

## 九、执行清单

### 本周任务

- [ ] Step 1：在 HoliDiff.py 或测试脚本中实现 compute_trend_consistency() 函数
- [ ] Step 2：跑 版本 3 (TTM) + 版本 F (TKS) + 版本 A (MS-MoM)，记录三相 MAE / TPR / TME
- [ ] Step 3：对比原始 MoM / Simple Mean / TTM / TKS / MS-MoM 的 24h MAE 曲线
- [ ] Step 4：若 TTM 或 TKS 有效，生成论文 Figure 1 的 Before/After 趋势截断对比图

### 下周任务（视 Round 1 结果）

- [ ] Step 5：若 TTM 有效，grid search alpha in {0.2, 0.3, 0.4, 0.5}
- [ ] Step 6：若 TKS 有效，grid search K' in {2, 3, 4, 5}
- [ ] Step 7：若 MS-MoM 有效，测试 K=30, M=6 的上界
- [ ] Step 8：跑 版本 2 (TGBS) 和 版本 B (TAG-MoM)，对比结构化分组效果
- [ ] Step 9：若多版本有效，跑 版本 4 (TVB)，grid search lambda in {0.5, 0.7, 0.9}

### 论文写作任务（并行）

- [ ] Paragraph 1：Motivation —— 引用气候模型 median bias + MAPA mean vs median，定义 Trend Annihilation
- [ ] Paragraph 2：Diagnosis —— 引用 Sample 3817/2474 图证据，证明问题在组内 mean 而非 median
- [ ] Paragraph 3：Method —— 提出 Trend-Aware Aggregation 框架，给出 TTM/TGBS/TCW 公式
- [ ] Paragraph 4：Experiment —— 三相误差 + TPR + 24h 曲线 Before/After

---

*文档结束*
