# HoliDiff: 从中心趋势到物理稳态检索 —— 双峰交通流聚合方案集

> **文档定位**：交通流扩散预测中聚合器的范式转换  
> **核心问题**：Mean/Median/MoM 在双峰分布下的系统性失效  
> **解决思路**：以"密度质心"替代"几何质心"，以"物理稳态检索"替代"中心趋势估计"  
> **版本**：v1.0  
> **日期**：2026-05-20

---

## 1. 问题背景：交通流的双峰宿命

### 1.1 基本图的非单调性

交通流基本图 $Q(\rho)$ 存在物理必然的双分支结构：
- **自由流分支**：$Q = v_f \cdot \rho$，$\rho \in [0, \rho_c)$，流量随密度线性增长
- **拥堵分支**：$Q = w \cdot (\rho_{\max} - \rho)$，$\rho \in (\rho_c, \rho_{\max}]$，流量随密度线性下降
- **临界区**：$\rho \approx \rho_c$，$Q \approx Q_{\max}$，物理不稳定

**统计后果**：传感器测得的流量值天然集中在两个稳态区，临界区是"真空带"。

### 1.2 扩散模型采样的双峰继承

扩散模型从噪声生成未来流量时，由于：
1. 训练数据本身双峰（自由流/拥堵样本）
2. MSE loss 在双峰间做"折中"
3. 保守偏差使右峰（拥堵）整体偏低

生成的采样云团 $\{y^{(s)}\}_{s=1}^S$ 呈现：
- **左峰**（自由流/保守估计）：大量样本，较分散，中心 ~150-180
- **右峰**（拥堵真值）：少量样本，较密集，中心 ~220-280
- **真空带**（160-210）：样本稀少

### 1.3 传统聚合器的失效

| 聚合器 | 数学定义 | 在双峰分布下的位置 | 物理含义 | 问题 |
|--------|---------|------------------|---------|------|
| **Mean** | $\frac{1}{S}\sum_s y_s$ | 真空带中央 (~170) | 物理上最不常出现的流量 | ❌ 严重低估拥堵 |
| **Median** | $F^{-1}(0.5)$ | 左峰内部 (~120) | 只代表自由流多数派 | ❌ 完全忽略拥堵相 |
| **MoM** | $\text{Med}(\text{Mean}_b)$ | 真空带偏左 (~160) | 中间组的平均 | ❌ 比 mean 更保守 |

**节假日加剧**（motivation.pdf 图 d）：
- Regular：左峰高、右峰低，mean 被左峰拉低
- Holiday：右峰抬升且展宽，但 mean 仍落在展宽后的中间某处，不代表任何稳态

---

## 2. 思考范式：从几何质心到密度质心

### 2.1 核心转换

**传统范式**：把 $S$ 次采样看作 $S$ 个**等质量质点**，求几何质心。
$$\hat{y} = \frac{\sum_s m_s \cdot y_s}{\sum_s m_s}, \quad m_s = 1$$

**新范式**：把 $S$ 次采样看作 $S$ 个**具有不同密度的粒子**，求密度质心。
$$\hat{y} = \frac{\sum_s \rho_s \cdot y_s}{\sum_s \rho_s}, \quad \rho_s = \text{局部核密度} \times \text{历史兼容性}$$

**物理直觉**：
- 位于 180 的孤立粒子（保守估计）：周围空旷，密度低，质量小
- 位于 250 的紧密核粒子（物理真值）：周围邻居密集，密度高，质量大
- 密度质心被"硬核"吸引，跳出真空带

### 2.2 带宽 $h$ 的杠杆作用

$h$ 控制"邻居半径"，是峰选择的关键旋钮：
- **$h \to \infty$**：所有粒子互为邻居，$\rho_s \to \text{const}$，退化为 **Mean**
- **$h \to 0$**：只认自身，$\rho_s \to 1$，退化为 **Mode**（最近邻聚合）
- **$h \approx \sigma_{\text{peak}}$**：恰好识别单个峰的密集核，**最优工作区**

### 2.3 历史兼容性的引力作用

仅靠内部几何密度（KDWM）可能无法克服左峰的数量优势（15 vs 5）。引入历史真值分布作为外部引力场：
- 历史拥堵真值集中在 $[230, 270]$，形成"拥堵势阱"
- 历史自由流真值集中在 $[80, 120]$，形成"自由流势阱"
- 采样粒子落入哪个势阱深，就获得额外质量

---

## 3. 方案一：DCA (Density-Centroid Aggregation) —— 主方案

### 3.1 数学推导

#### 3.1.1 核密度权重（内部结构）

对每个采样点 $y_s$，计算其 $K$ 近邻核密度：
$$\rho_s^{\text{kde}} = \sum_{t=1}^S K\left(\frac{y_s - y_t}{h}\right), \quad K(u) = \exp\left(-\frac{u^2}{2}\right)$$

**物理意义**：$y_s$ 周围 $h$ 半径内有多少"同伴"。拥堵相的 5 个点若集中在 $\pm 10$ 内，其核密度可超过自由流 15 个分散点的密度。

#### 3.1.2 历史似然权重（外部兼容）

设历史真值库按相态划分为 $\mathcal{H}_{\text{free}}$ 和 $\mathcal{H}_{\text{cong}}$。对每个 $y_s$ 计算两势阱的兼容性：

$$L_s^{\text{free}} = \sum_{y_k \in \mathcal{H}_{\text{free}}} \exp\left(-\frac{(y_s - y_k)^2}{2\sigma^2}\right)$$

$$L_s^{\text{cong}} = \sum_{y_k \in \mathcal{H}_{\text{cong}}} \exp\left(-\frac{(y_s - y_k)^2}{2\sigma^2}\right)$$

**历史主导权重**：
$$\rho_s^{\text{hist}} = \max\left(L_s^{\text{free}}, L_s^{\text{cong}}\right)$$

或软版本（保留两峰信息）：
$$\rho_s^{\text{hist}} = \log\left(\exp(L_s^{\text{free}}) + \exp(L_s^{\text{cong}})\right)$$

#### 3.1.3 联合密度质心

$$w_s = \rho_s^{\text{kde}} \times \rho_s^{\text{hist}}$$

$$\boxed{\hat{y}_{\text{DCA}} = \frac{\sum_{s=1}^S w_s \cdot y_s}{\sum_{s=1}^S w_s}}$$

**退化验证**：
- $h \to \infty$ 且 $\sigma \to \infty$：$w_s \to \text{const}$，退化为 **Mean**
- $h \to 0$ 且仅用 $\rho^{\text{kde}}$：退化为 **Mode**
- $\sigma \to 0$ 且历史库为空：退化为 **KDWM**

### 3.2 物理语义

> "我们不寻找云团的几何中心，而是寻找与历史物理稳态最共振的密集核。"

### 3.3 代码实现

```python
import torch
import numpy as np
from typing import Dict, List, Optional

class DensityCentroidAggregator:
    def __init__(self, 
                 bandwidth_kde: float = 15.0,
                 bandwidth_hist: float = 20.0,
                 mode: str = 'joint'):
        """
        mode: 'kde_only', 'hist_only', 'joint', 'competitive'
        """
        self.h_kde = bandwidth_kde
        self.h_hist = bandwidth_hist
        self.mode = mode

        # 历史真值库，离线构建
        self.hist_free: Optional[np.ndarray] = None
        self.hist_cong: Optional[np.ndarray] = None

    def fit_history(self, y_history: np.ndarray, phase_labels: np.ndarray):
        """
        y_history: (N_hist,) 历史真值
        phase_labels: (N_hist,) 0=free, 1=cong
        """
        self.hist_free = y_history[phase_labels == 0]
        self.hist_cong = y_history[phase_labels == 1]
        print(f"History fitted: free={len(self.hist_free)}, cong={len(self.hist_cong)}")

    def _kde_weights(self, samples: np.ndarray) -> np.ndarray:
        """内部核密度权重"""
        S = len(samples)
        if S == 0:
            return np.array([])

        # 向量化计算: (S, S) 距离矩阵
        diff = samples[:, None] - samples[None, :]  # (S, S)
        w_kde = np.sum(np.exp(-0.5 * (diff / self.h_kde) ** 2), axis=1)  # (S,)
        return w_kde

    def _hist_weights(self, samples: np.ndarray) -> np.ndarray:
        """历史兼容性权重"""
        S = len(samples)
        w_hist = np.ones(S)

        if self.hist_free is not None and len(self.hist_free) > 0:
            # 与自由流历史的兼容性: (S, N_free) -> (S,)
            diff_free = samples[:, None] - self.hist_free[None, :]  # (S, N_free)
            like_free = np.sum(np.exp(-0.5 * (diff_free / self.h_hist) ** 2), axis=1)
        else:
            like_free = np.zeros(S)

        if self.hist_cong is not None and len(self.hist_cong) > 0:
            diff_cong = samples[:, None] - self.hist_cong[None, :]
            like_cong = np.sum(np.exp(-0.5 * (diff_cong / self.h_hist) ** 2), axis=1)
        else:
            like_cong = np.zeros(S)

        if self.mode == 'competitive':
            # 硬选择：哪个历史库兼容性强，就归哪个阵营
            w_hist = np.maximum(like_free, like_cong)
        elif self.mode == 'softmax':
            # 软选择：保留两峰信息，但拥堵相可放大
            w_hist = np.log(np.exp(like_free) + np.exp(like_cong) + 1e-8)
        else:
            # joint 默认：简单相加
            w_hist = like_free + like_cong

        return w_hist

    def aggregate(self, samples_1d: np.ndarray) -> float:
        """
        samples_1d: (S,) numpy array, 单个节点单个时间步的 S 次采样
        return: 标量预测值
        """
        S = len(samples_1d)
        if S == 0:
            return 0.0

        # 1. 内部核密度权重
        w_kde = self._kde_weights(samples_1d)

        # 2. 历史兼容性权重
        w_hist = self._hist_weights(samples_1d)

        # 3. 联合权重
        if self.mode == 'kde_only':
            w = w_kde
        elif self.mode == 'hist_only':
            w = w_hist
        else:
            w = w_kde * w_hist

        # 4. 密度质心
        if w.sum() < 1e-8:
            return float(samples_1d.mean())  # 退化保护

        pred = np.sum(w * samples_1d) / np.sum(w)
        return float(pred)

    def aggregate_batch(self, samples: torch.Tensor, 
                        y_hist_free: Optional[torch.Tensor] = None,
                        y_hist_cong: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        samples: (S, B, N, H) torch.Tensor
        return: (B, N, H) torch.Tensor
        """
        S, B, N, H = samples.shape
        preds = np.zeros((B, N, H))

        # 可选：动态更新历史库
        if y_hist_free is not None:
            self.hist_free = y_hist_free.cpu().numpy()
        if y_hist_cong is not None:
            self.hist_cong = y_hist_cong.cpu().numpy()

        for b in range(B):
            for n in range(N):
                for h_step in range(H):
                    s = samples[:, b, n, h_step].cpu().numpy()
                    preds[b, n, h_step] = self.aggregate(s)

        return torch.from_numpy(preds).to(samples.device)
```

---

## 4. 方案二：ABMS (Adaptive Bandwidth Mean Shift) —— 迭代峰漂移

### 4.1 数学推导

Mean Shift 是迭代版本的密度质心：
$$y^{(k+1)} = \frac{\sum_{s=1}^S K\left(\frac{y^{(k)} - y_s}{h}\right) y_s}{\sum_{s=1}^S K\left(\frac{y^{(k)} - y_s}{h}\right)}$$

**自适应带宽**：每个采样点有自己的带宽，由其到第 $m$ 近邻的距离决定：
$$h_s = d_{s}^{(m)} = \text{第 } m \text{ 小的 } |y_s - y_t|$$

**自适应 Mean Shift**：
$$y^{(k+1)} = \frac{\sum_s K\left(\frac{y^{(k)} - y_s}{h_s}\right) y_s}{\sum_s K\left(\frac{y^{(k)} - y_s}{h_s}\right)}$$

### 4.2 与 DCA 的区别

| 特性 | DCA | ABMS |
|------|-----|------|
| **计算** | 单步加权 | 迭代漂移 |
| **收敛** | 到密度质心 | 到局部密度极大值（mode） |
| **带宽** | 全局固定 $h$ | 每个点自适应 $h_s$ |
| **多峰** | 可能落在两峰之间 | 收敛到最近峰的顶点 |
| **计算量** | $O(S^2)$ | $O(K \cdot S^2)$，$K$ 为迭代次数 |

### 4.3 适用场景

- 云团有**清晰分离的多峰**（自由流峰和拥堵峰相距 > 50 veh/15min）
- 需要**硬选择**一个物理稳态，而非两峰之间的折中
- 迭代起点建议用 **DCA 的输出**而非 mean，避免漂向左峰

### 4.4 代码骨架

```python
def adaptive_mean_shift(samples_1d: np.ndarray, 
                        m_neighbor: int = 3,
                        max_iter: int = 10,
                        tol: float = 1e-3) -> float:
    """
    从 DCA 输出出发，迭代漂移至局部密度极大值
    """
    S = len(samples_1d)

    # 计算自适应带宽: 每个点到第 m 近邻的距离
    dists = np.abs(samples_1d[:, None] - samples_1d[None, :])  # (S, S)
    sorted_dists = np.sort(dists, axis=1)
    h_adaptive = sorted_dists[:, min(m_neighbor, S-1)] + 1e-6  # (S,)

    # 从 DCA 密度质心出发（比 mean 更可能靠近真值峰）
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(samples_1d, bw_method=0.3)
    grid = np.linspace(samples_1d.min(), samples_1d.max(), 200)
    density = kde(grid)
    y = grid[np.argmax(density)]  # 初始化为全局密度最大点

    for _ in range(max_iter):
        diff = y - samples_1d  # (S,)
        k = np.exp(-0.5 * (diff / h_adaptive) ** 2)  # (S,)
        y_new = np.sum(k * samples_1d) / np.sum(k)

        if abs(y_new - y) < tol:
            break
        y = y_new

    return float(y)
```

---

## 5. 方案三：RCL (Regime-Competitive Likelihood) —— 双阵营竞争

### 5.1 数学推导

DCA 和 ABMS 都是"找最密的点"。RCL 换一个思路：**不找点，找阵营**。

对每个采样点 $y_s$，计算它属于自由流阵营和拥堵阵营的后验概率：

$$P(\text{free} | y_s) = \frac{L_s^{\text{free}}}{L_s^{\text{free}} + L_s^{\text{cong}}}, \quad P(\text{cong} | y_s) = \frac{L_s^{\text{cong}}}{L_s^{\text{free}} + L_s^{\text{cong}}}$$

**硬选择版本（Hard RCL）**：
$$\mathcal{S}_{\text{cong}} = \{s \mid P(\text{cong}|y_s) > 0.5\}$$
$$\hat{y}_{\text{hard}} = \frac{1}{|\mathcal{S}_{\text{cong}}|} \sum_{s \in \mathcal{S}_{\text{cong}}} y_s$$

**软选择版本（Soft RCL）**：
$$\hat{y}_{\text{soft}} = \frac{\sum_s P(\text{cong}|y_s) \cdot y_s}{\sum_s P(\text{cong}|y_s)}$$

### 5.2 与 DCA 的本质区别

| 特性 | DCA | RCL |
|------|-----|-----|
| **权重来源** | 局域密度 × 历史兼容 | 历史后验概率 |
| **物理图像** | "粒子有质量" | "粒子投阵营" |
| **中间点处理** | 真空带点密度低，质量小 | 真空带点 $P(\text{cong}) \approx 0.5$，被两阵营抵消 |
| **输出位置** | 密度质心（可能在峰之间） | 拥堵阵营的 mean（在右峰内部） |

### 5.3 适用场景

- 历史库足够大，能可靠估计 $L^{\text{free}}$ 和 $L^{\text{cong}}$
- 需要**明确的相态归属**（如后续要做 TPR 判断，需要知道"预测的是拥堵相"）
- 云团两峰**分离不清晰**（KDWM 可能落在中间，RCL 强制选择右阵营）

### 5.4 代码骨架

```python
def rcl_aggregate(samples_1d: np.ndarray,
                  hist_free: np.ndarray,
                  hist_cong: np.ndarray,
                  h: float = 20.0,
                  mode: str = 'soft') -> float:
    """
    Regime-Competitive Likelihood 聚合
    """
    S = len(samples_1d)

    # 计算每个采样点的两阵营似然
    like_free = np.zeros(S)
    like_cong = np.zeros(S)

    for s in range(S):
        diff_free = hist_free - samples_1d[s]
        like_free[s] = np.sum(np.exp(-0.5 * (diff_free / h) ** 2))

        diff_cong = hist_cong - samples_1d[s]
        like_cong[s] = np.sum(np.exp(-0.5 * (diff_cong / h) ** 2))

    # 后验概率
    total = like_free + like_cong + 1e-8
    p_cong = like_cong / total

    if mode == 'hard':
        mask = p_cong > 0.5
        if mask.sum() == 0:
            # 无点归属拥堵，退化为最大后验点
            return float(samples_1d[np.argmax(p_cong)])
        return float(samples_1d[mask].mean())
    else:
        # soft: 加权平均
        w = p_cong  # 拥堵后验概率作为权重
        return float(np.sum(w * samples_1d) / np.sum(w))
```

---

## 6. 方案四：WBP (Wasserstein Barycenter Projection) —— 分布级匹配

### 6.1 数学推导

以上方案都是**点估计**：从 $S$ 个采样中选一个代表值。

WBP 是**分布估计**：把采样云团 $\hat{P}_S = \frac{1}{S}\sum_s \delta_{y_s}$ 投影到历史真值分布 $P_{\text{hist}}$ 的重心。

**一维 Wasserstein-2 距离**：
$$W_2^2(\hat{P}_S, P_{\text{hist}}) = \int_0^1 (F_S^{-1}(u) - F_{\text{hist}}^{-1}(u))^2 \mathrm{d}u$$

**Barycentric projection**（对 $P_{\text{hist}}$ 的最优传输映射）：
$$T(y_s) = F_{\text{hist}}^{-1} \circ F_S(y_s)$$

即：把采样云团的每个分位数映射到历史分布的对应分位数。

**聚合**：
$$\hat{y}_{\text{WBP}} = \frac{1}{S} \sum_{s=1}^S T(y_s)$$

### 6.2 物理语义

> "我们不问'哪个采样点最好'，而是问'整个云团如何变形才能与历史真值分布对齐'。"

如果历史拥堵分布的 0.7 分位数是 250，而采样云团的 0.7 分位数是 210，WBP 会把所有 210 附近的点"拉"到 250 附近。

### 6.3 与 DCA 的区别

| 特性 | DCA | WBP |
|------|-----|-----|
| **粒度** | 点级加权 | 分布级变形 |
| **利用历史** | 局部核兼容 | 全局分位数匹配 |
| **输出** | 单个密度质心 | 变形后的云团均值 |
| **计算** | $O(S^2 + S \cdot N_{\text{hist}})$ | $O(S \log S + N_{\text{hist}} \log N_{\text{hist}})$ |

### 6.4 适用场景

- 历史库大，分位数估计稳定
- 需要**保持云团形状**的同时整体右移（适合后续不确定性量化）
- 采样次数 $S \geq 20$，分位数估计可靠

### 6.5 代码骨架

```python
def wbp_aggregate(samples_1d: np.ndarray,
                  hist_values: np.ndarray) -> float:
    """
    Wasserstein Barycenter Projection (1D)
    """
    S = len(samples_1d)
    Nh = len(hist_values)

    # 排序
    s_sorted = np.sort(samples_1d)
    h_sorted = np.sort(hist_values)

    # 计算分位数函数（线性插值）
    # F_S(y) 在 s_sorted 处的值: (0.5/S, 1.5/S, ..., 1-0.5/S)
    q_s = (np.arange(S) + 0.5) / S
    q_h = (np.arange(Nh) + 0.5) / Nh

    # 对每个采样点，找到其在历史分布中的对应分位数值
    # 即 T(y_s) = F_hist^{-1}(F_S(y_s))
    # 简化：用历史分布的对应分位数替换

    # 建立从 q_s 到 q_h 的映射
    # 如果 S < Nh：每个采样点映射到历史的一个分位
    # 如果 S > Nh：历史分位数插值

    if S <= Nh:
        # 采样点少：每个采样点直接取历史的对应分位
        idx = np.round(q_s * (Nh - 1)).astype(int)
        mapped = h_sorted[idx]
    else:
        # 采样点多：历史分位数插值到 S 个位置
        h_quantiles = np.interp(q_s, q_h, h_sorted)
        mapped = h_quantiles

    # 输出变形后的均值
    return float(mapped.mean())
```

---

## 7. 方案五：SCP (Skewness-Corrected Peak) —— 偏度解析偏移

### 7.1 数学推导

如果云团是**单峰左偏**（非双峰），以上方案可能过度分裂。SCP 用**偏度**作为偏移强度的解析函数。

**样本偏度**：
$$\hat{\gamma}_1 = \frac{\frac{1}{S}\sum_s (y_s - \bar{y})^3}{\left(\frac{1}{S}\sum_s (y_s - \bar{y})^2\right)^{3/2}}$$

**偏移聚合器**：
$$\hat{y}_{\text{SCP}} = \bar{y} + \lambda \cdot \hat{\gamma}_1 \cdot \sigma$$

其中：
- $\bar{y}$：样本均值
- $\sigma$：样本标准差
- $\lambda > 0$：偏移强度系数（从历史数据校准）
- $\hat{\gamma}_1 < 0$（左偏）时，$\hat{y}_{\text{SCP}} > \bar{y}$，向右偏移

**物理直觉**：
- 左偏分布：尾巴在左，主峰在右
- 偏度越负，主峰离 mean 越远，需要向右偏移越多
- 偏移量与标准差成正比（云团越宽，偏移潜力越大）

### 7.2 与 DCA 的关系

SCP 是 DCA 的**单峰近似**。如果云团实际上是单峰左偏（而非双峰），DCA 可能错误地寻找不存在的右峰，而 SCP 用解析偏移更稳健。

### 7.3 适用场景

- 云团**单峰**但**严重左偏**（拥堵恢复时常见：大量保守估计，少量正确估计形成右尾）
- 历史库**不足**（无法可靠估计 $L^{\text{cong}}$）
- 需要**零历史依赖**的聚合器

### 7.4 代码骨架

```python
def scp_aggregate(samples_1d: np.ndarray, 
                  lambda_skew: float = 0.5) -> float:
    """
    Skewness-Corrected Peak
    lambda_skew: 从历史数据校准，通常 0.3 ~ 0.8
    """
    y_bar = np.mean(samples_1d)
    sigma = np.std(samples_1d)

    if sigma < 1e-6:
        return float(y_bar)

    # 样本偏度
    gamma1 = np.mean((samples_1d - y_bar) ** 3) / (sigma ** 3)

    # 偏移：左偏(gamma1 < 0)时向右移
    pred = y_bar + lambda_skew * gamma1 * sigma
    return float(pred)
```

---

## 8. 五方案对比与实验建议

### 8.1 总表

| 方案 | 核心算子 | 历史依赖 | 计算成本 | 假设 | 最佳场景 |
|------|---------|---------|---------|------|---------|
| **DCA** | 密度质心 $\frac{\sum w_s y_s}{\sum w_s}$ | 需要 | $O(S^2 + S\cdot N_h)$ | 双峰，有密集核 | **通用首选** |
| **ABMS** | 迭代漂移至 mode | 可选 | $O(K S^2)$ | 多峰分离清晰 | 峰间距大 |
| **RCL** | 后验概率加权 | **必须** | $O(S \cdot N_h)$ | 历史库大 | 需明确相态归属 |
| **WBP** | 分位数传输映射 | **必须** | $O(S \log S)$ | 分布形状重要 | 历史库极大 |
| **SCP** | 偏度解析偏移 | 无 | $O(S)$ | 单峰左偏 | 历史库不足/快速验证 |

### 8.2 推荐测试顺序

```
Step 1 (今晚): SCP
    └── 零历史依赖，5分钟实现
    └── 如果 lambda=0.5 时 Cong_MAE 下降 → 确认左偏假设成立

Step 2: DCA (kde_only)
    └── 只用内部密度，不看历史
    └── 如果优于 SCP → 确认双峰/多峰结构

Step 3: DCA (joint)
    └── 加入历史兼容性
    └── 如果显著优于 kde_only → 历史引力有效

Step 4: RCL / WBP
    └── 需要构建历史库（从训练集提取拥堵/自由流真值）
    └── 如果优于 DCA → 阵营竞争/分布匹配更精准

Step 5: ABMS
    └── 以 DCA 输出为起点迭代
    └── 如果优于 DCA → 需要硬收敛到单峰顶点
```

### 8.3 历史库构建

从训练集提取：
```python
# 假设训练集有 (X, Y, phase_label)
hist_free = Y_train[phase_label == 0].flatten()  # 所有自由流真值
hist_cong = Y_train[phase_label == 1].flatten()  # 所有拥堵真值

# 如果无 phase_label，用简单阈值划分
phase_label = (Y_train.mean(axis=-1) > 150).astype(int)  # 阈值需调
```

---

## 9. 论文叙事段落

> **The Bimodal Trap of Central Tendency.** Traffic flow fundamentally operates in two stable regimes: free-flow and congested, separated by a physically unstable critical region. Under this bimodality, the sampling distribution of diffusion-based predictors inherits a two-peak structure: a dominant left peak of conservative estimates and a sparser right peak near the physical ground truth. The mean collapses into the vacuum band between peaks; the median, governed by the majority free-flow samples, ignores the congested regime entirely. Neither can serve as a physically faithful macro-flow estimator.
>
> **Density-Centroid Aggregation.** We propose to replace the geometric centroid with a density centroid. Each microscopic realization is weighted by its local kernel density and its compatibility with historical regime-specific distributions. The aggregator thus migrates toward the densest physical equilibrium—typically the congested peak during congestion phases—rather than being trapped in the statistical vacuum. This is not a post-hoc bias correction; it is a fundamental change in the statistical functional from first-moment estimation to physical-steady-state retrieval.

---

*文档版本: v1.0*  
*创建时间: 2026-05-20*  
*下次更新: SCP/DCA 离线验证完成后*
