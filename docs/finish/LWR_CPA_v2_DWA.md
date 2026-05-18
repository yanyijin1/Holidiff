# LWR-CPA 方案 v2.0：Directional Weighted Aggregation (DWA)

> **定位**：LWR-CPA v1 失败后的修正方案  
> **核心修正**：从「硬选择 min/max」转向「连续权重偏移」  
> **原则**：零训练，基于物理相位 phi 软调制采样权重  
> **目标**：在保留方差控制的前提下，注入与激波方向对齐的有益偏差

---

## 一、v1 失败诊断：min/max 硬选择的方差爆炸

### 1.1 实验结果

| 版本 | MAE | FF | CG | HOL_CG | TPR | TPR_CG |
|------|-----|----|----|--------|-----|--------|
| simple_mean | 0.2288 | 13.799 | 30.981 | 36.849 | 0.446 | 0.453 |
| lwr_cpa_b2.0 | 0.2391 | 15.287 | 32.383 | 38.267 | 0.446 | 0.453 |
| lwr_cpa_b5.0 | 0.2399 | 15.400 | 32.482 | 38.361 | 0.446 | 0.453 |

**关键发现**：
- CG_MAE 从 30.981 恶化到 32.38+，恶化约 4.5%
- FF_MAE 从 13.80 恶化到 15.29+，恶化约 10.8%
- TPR 完全不变（0.453），说明方向没改善，幅度严重过冲

### 1.2 数学证明：为什么 min/max 必然失败

设 K=10 条采样 i.i.d. N(mu, sigma^2)，真值 Y* = mu - delta（拥堵形成期，真值低于均值）。

**simple_mean 的 MSE**：
MSE_mean = delta^2 + sigma^2 / K = delta^2 + 0.1 sigma^2

**min 选择的 MSE**（K=10 标准正态顺序统计量）：
- E[Y_(1)] approx mu - 1.54 sigma
- Var(Y_(1)) approx 0.34 sigma^2

MSE_min = (1.54 sigma - delta)^2 + 0.34 sigma^2
        = 2.37 sigma^2 - 3.08 sigma delta + delta^2 + 0.34 sigma^2
        = delta^2 + 2.71 sigma^2 - 3.08 sigma delta

**min 优于 mean 的临界条件**：
MSE_min < MSE_mean
=> delta^2 + 2.71 sigma^2 - 3.08 sigma delta < delta^2 + 0.1 sigma^2
=> 2.61 sigma^2 < 3.08 sigma delta
=> delta > 0.85 sigma

**致命结论**：只有当真值低于均值超过 0.85 个采样标准差时，选 min 才比 mean 好。

在扩散模型中，采样间标准差 sigma（数十到上百辆车）远大于激波导致的真值偏移 delta（几个到十几个）。因此 **delta << 0.85 sigma 几乎总是成立**，选 min/max 必然劣化。

### 1.3 为什么 TPR 完全不变？

min 的偏差太大（1.54 sigma），即使方向正确（min 确实更低），它也**过度预测了下降幅度**。真值只比均值低 delta，但 min 比均值低 1.54 sigma。这导致：
- 方向可能对（TPR 不变）
- 幅度严重过冲（MAE 恶化）
- 自由流被误判时，min/max 引入巨大噪声（FF_MAE 崩到 15.2+）

### 1.4 本质缺失：不是「相位识别」错了，是「聚合方式」错了

v1 的方向（用物理相位指导聚合）仍然可能是对的，但实现犯了「硬选择」的错误：

| 维度 | v1（硬选择） | 应该是（软加权） |
|------|-------------|----------------|
| 采样使用 | 只用 1 条（min/max） | 用全部 K 条，但权重不同 |
| 方差控制 | Var approx 0.34 sigma^2 | Var approx sigma^2/K = 0.1 sigma^2 |
| 偏差幅度 | 1.54 sigma（过大，不可调） | 可调（通过 alpha） |
| 连续性 | |phi| 阈值处跳变 | phi 连续调制权重 |

**核心修正**：phi 不应该决定「选哪条采样」，而应该决定「**向哪个方向偏移权重分布**」。

---

## 二、v2 方案：Directional Weighted Aggregation (DWA)

### 2.1 核心公式

Y_n^DWA = sum_{k=1}^K w_n^(k) * Y_n^(k)

w_n^(k) = exp(-alpha * phi_n * (Y_n^(k) - m_n)) / sum_{j=1}^K exp(-alpha * phi_n * (Y_n^(j) - m_n))

其中：
- m_n = median({Y_n^(k)}_{k=1}^K)：采样中位数（基准点）
- alpha > 0：温度系数（控制偏移强度）
- phi_n in [-1, 1]：相位（沿用 v1 定义，或改进版）

### 2.2 物理行为

| phi_n | 预期状态 | 权重行为 | 聚合结果 |
|-------|---------|---------|---------|
| > 0 | 拥堵形成（未来更低） | 给低于 median 的采样更高权重 | 均值向**下**偏移 |
| < 0 | 拥堵消散（未来更高） | 给高于 median 的采样更高权重 | 均值向**上**偏移 |
| approx 0 | 稳态 | 权重近似均匀 | 退化为 simple mean |

### 2.3 与 v1 的本质区别

v1：Y = (1-|phi|) * mean + |phi| * extreme（二值混合，方差爆炸）
v2：Y = sum_k w_k(phi) * Y^(k)（连续加权，方差可控）

### 2.4 数学优势：偏差-方差可控

当 alpha 较小时，对权重做一阶泰勒展开：

w_k approx 1/K + (alpha phi / K) * (m - Y^(k)) + O(alpha^2)

（注意：低于 median 的采样 Y^(k) < m，所以 (m - Y^(k)) > 0，权重增大）

聚合期望：
E[Y^DWA] approx mu + alpha phi * c * sigma

其中 c = E[eps_k (m - eps_k)] / sigma^2 > 0 是正的相关系数。

**偏差**：bias approx (mu - Y*) + alpha phi c sigma

如果 phi 与真值偏移方向一致（phi > 0 时 Y* < mu），通过选择 alpha 可以使：
alpha |phi| c sigma approx delta = mu - Y*

从而使 **bias approx 0**。

**方差**：Var(Y^DWA) approx sigma^2 / K + O(alpha^2)

因为仍然使用全部 K 条采样，方差不会爆炸。

**MSE 对比**：
- simple_mean：MSE = delta^2 + sigma^2 / K
- DWA：MSE approx 0 + sigma^2 / K + O(alpha^2)（当 alpha 校准到抵消 delta 时）

只要 delta^2 > O(alpha^2)，DWA 严格优于 simple_mean。

---

## 三、相位 phi 的修正定义

v1 使用二阶差分 a_{n,T}，但实验表明其信号可能被 RevIN 或噪声掩盖。v2 提供两个 phi 候选：

### 候选 A：原始值域二阶差分（v1 延续）

v_{n,t} = x_{n,t} - x_{n,t-1}
a_{n,t} = v_{n,t} - v_{n,t-1}

phi_n = sign(a_{n,T}) * sigmoid((|a_{n,T}| - mu_a) / sigma_a * beta)

### 候选 B：相对位置相位（更鲁棒）

phi_n = 2 * (x_{n,T} - min(X_n)) / (max(X_n) - min(X_n) + eps) - 1

物理意义：
- phi -> +1：当前处于历史区间最高位（即将下降/拥堵形成）
- phi -> -1：当前处于历史区间最低位（即将回升/拥堵消散）
- phi approx 0：历史区间中间位置（稳态）

**优势**：不受 RevIN 影响，只依赖历史区间内的相对位置，对噪声更鲁棒。

### 候选 C：联合相位（推荐）

phi_n = 0.5 * phi_A + 0.5 * phi_B

结合曲率信息和相对位置信息。

---

## 四、与现有方法的本质对比

| 方法 | 选择依据 | 方差 | 偏差可控性 | 物理意义 |
|------|---------|------|-----------|---------|
| MoM | 统计排序（median） | 高（隐式非均匀） | 不可控 | 无 |
| simple_mean | 均匀 | 最低（sigma^2/K） | 无（delta^2 固定） | 无 |
| LWR-CPA v1 | 物理相位 + min/max | 极高（0.34 sigma^2） | 过大（1.54 sigma） | 有 |
| **LWR-CPA v2 (DWA)** | **物理相位 + 连续加权** | **低（sigma^2/K）** | **可调（alpha）** | **有** |

---

## 五、验证实验设计

### 实验 1：DWA 基础验证（候选 A phi）

**配置**：
- phi：候选 A（二阶差分，beta=2.0）
- alpha in {0.1, 0.5, 1.0, 2.0, 5.0}
- 对比：simple_mean, MoM, v1(beta=2.0)

**成功标准**：
- CG_MAE < 30.981（击败 simple_mean）
- FF_MAE 不劣化（DWA 在 phi approx 0 时自动退化为 uniform）
- TPR_CG 提升（证明偏移方向正确）

### 实验 2：phi 候选对比

**配置**：
- 固定 alpha=1.0（或实验 1 最优 alpha）
- 测试 phi_A / phi_B / phi_C

**成功标准**：找到最优 phi 定义。

### 实验 3：alpha 敏感性 + 最优校准

**配置**：
- 固定最优 phi
- alpha grid search：{0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0}

**预期**：
- alpha 过小（<0.1）：接近 simple_mean
- alpha 过大（>5.0）：权重过于尖锐，可能过拟合
- 最优 alpha 应在 0.2-2.0 之间

### 实验 4：分桶 TPR 验证（验证 phi 信号）

**配置**：
- 按 phi 分 5 桶
- 计算每桶真实 TPR

**预期**：
- 桶 [-1, -0.6)：TPR 高（消散延续）
- 桶 [0.6, 1]：TPR 高（形成延续）
- 桶 [-0.2, 0.2]：TPR 约 0.5（稳态随机）

若此预期不成立，说明 phi 定义需要重新设计（转向候选 B/C）。

---

## 六、论文叙事线（v1 失败 → v2 修正）

### Paragraph 1：v1 的尝试与失败

> 基于 LWR 激波相位，我们初步尝试了极端选择策略（拥堵形成期选 min，消散期选 max）。然而实验表明该策略导致 CG_MAE 从 30.981 恶化至 32.38+，且 FF_MAE 恶化 10.8%。数学分析证明：min/max 作为顺序统计量，其方差为 0.34 sigma^2（K=10 时），远高于 uniform mean 的 0.1 sigma^2；且偏差幅度 1.54 sigma 远大于激波真值偏移 delta，导致系统性过冲。

### Paragraph 2：核心洞察——软偏移优于硬选择

> 问题的本质不是「相位识别」方向错误，而是「聚合方式」的硬选择引入了不可控的方差。正确的策略应该是：用物理相位连续调制权重分布，而非二值切换采样选择。这保留了全部 K 条采样的方差控制优势（sigma^2/K），同时通过温度系数 alpha 精确校准偏差幅度。

### Paragraph 3：DWA 方案

> 我们提出 Directional Weighted Aggregation (DWA)：以采样中位数 m 为基准，根据相位 phi 的符号和强度，指数偏移权重——phi > 0（拥堵形成）时给低于 m 的采样更高权重，phi < 0（拥堵消散）时给高于 m 的采样更高权重。该策略在 phi approx 0 时自动退化为 uniform mean，在激波区域通过 alpha 注入与真值对齐的有益偏差。

### Paragraph 4：数学保证

> 一阶分析表明，DWA 的偏差为 alpha phi c sigma（c 为采样与中位数的相关系数），方差为 sigma^2/K + O(alpha^2)。当 alpha 校准至 alpha phi c sigma approx delta 时，偏差趋近于零，而方差仍保持接近 uniform mean 的水平，从而实现 MSE 的帕累托改进。

---

## 七、执行清单

### 本周（验证核心假设）

- [ ] **Step 1**：实现 DWA 聚合器（约 15 行代码）
- [ ] **Step 2**：跑实验 1（alpha grid search，phi=候选 A）
- [ ] **Step 3**：若实验 1 有改善趋势，跑实验 3（alpha 精细搜索）
- [ ] **Step 4**：若实验 1 无改善，切换 phi 候选 B/C，跑实验 2
- [ ] **Step 5**：跑实验 4（分桶 TPR，验证 phi 信号）

### 下周（若核心假设成立）

- [ ] **Step 6**：引入上下游空间信息，升级 phi 为图传播相位
- [ ] **Step 7**：论文 Figure 1：DWA 权重分布可视化（不同 phi 下的权重曲线）
- [ ] **Step 8**：论文 Figure 2：CG_MAE 随 alpha 变化曲线 + 帕累托前沿

---

## 八、风险与备选

| 风险 | 判断标准 | 备选方案 |
|------|---------|---------|
| DWA 仍无法降低 CG_MAE | 实验 1 最优 alpha 下 CG_MAE > 30.981 | 问题回到 phi 定义，需引入空间传播信息或更高阶微分 |
| alpha 敏感度过高 | 最优 alpha 窗口 < 0.5 | 改用自适应 alpha（基于 |phi| 动态调整） |
| phi 分桶 TPR 不显著 | 实验 4 各桶 TPR 差异 < 0.1 | 放弃二阶差分，改用候选 B（相对位置）或候选 C（联合） |
| FF_MAE 劣化 | 最优 alpha 下 FF_MAE 上升 > 2% | 添加自由流保护：当历史方差 < threshold 时强制 phi=0 |

---

*文档结束*
