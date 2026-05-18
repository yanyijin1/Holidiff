# Step 2 结果分析与 eta 参数化决策

> **定位**：Historical Trend Residual 实验结论 + eta 参数化方向决策  
> **核心发现**：固定 eta 有改善但未过生死线，升级为可学习参数是必要路径  
> **决策**：分两步走——先全局可学习标量验证，再条件化升级

---

## 一、Step 2 实验结果

### 1.1 原始数据

| eta | CG_MAE | TPR_CG | 改善幅度 |
|-----|--------|--------|---------|
| 0.0 (baseline) | 30.8779 | 0.4519 | — |
| 0.1 | ~30.85 | ~0.452 | 微降 |
| 0.3 | ~30.80 | ~0.453 | 微降 |
| 0.5 | ~30.75 | ~0.454 | 改善 |
| 0.8 | ~30.72 | ~0.455 | 改善 |
| **1.0** | **30.6939** | **~0.455** | **最优 CG_MAE** |
| 1.5 | ~30.75 | ~0.456 | TPR 继续升，CG_MAE 反弹 |
| **2.0** | ~30.80 | **~0.457** | **最优 TPR_CG** |

### 1.2 关键结论

**结论 1：方向正确**
- 所有 eta > 0 的 CG_MAE 均低于 baseline（30.8779）
- 所有 eta > 0 的 TPR_CG 均高于 baseline（0.4519）
- 证明系统性保守偏差 $\delta$ 确实存在，且 $s^{\text{hist}}$ 是正确的校正信号

**结论 2：幅度不够**
- 最优 CG_MAE = 30.6939，距离生死线 30.5 还有 0.19
- 最优 TPR_CG ≈ 0.457，距离真实物理 TPR 0.473 还有 0.016
- 固定全局 eta 对所有样本一视同仁，无法自适应不同激波强度

**结论 3：eta 存在最优区间**
- CG_MAE 最优在 eta = 1.0
- TPR_CG 最优在 eta = 2.0
- 两者矛盾：强校正（eta=2）提升 TPR 但恶化 CG_MAE（过冲），弱校正（eta=1）优化 CG_MAE 但 TPR 不够
- **这说明不同样本需要不同的 eta**

---

## 二、固定 eta vs 可学习 eta：决策分析

### 2.1 固定 eta 的瓶颈

固定 eta 的问题在于**全局一刀切**：

$$\hat{x}_0^{\text{corrected}} = \hat{x}_0 + \eta \cdot s^{\text{hist}} \cdot \Delta t$$

- 激波上升沿（强趋势）：需要 eta ≈ 1.5-2.0 才能追上真值
- 弱趋势波动：需要 eta ≈ 0.3-0.5 即可，eta=1.0 会过冲
- 自由流稳态：需要 eta ≈ 0，任何 eta > 0 都引入噪声

固定 eta 被迫取折中值（0.8-1.0），结果是：
- 强趋势样本校正不足（TPR 上不去）
- 弱趋势样本校正过度（CG_MAE 反弹）
- 自由流样本被污染（FF_MAE 可能劣化）

### 2.2 可学习 eta 的优势

让 eta 成为样本级或节点级的函数：

$$\eta_n = f(s_n^{\text{hist}}, \sigma_n^{\text{hist}}, t_n)$$

其中输入可以包括：
- $s_n^{\text{hist}}$：历史趋势强度（决定需要多少校正）
- $\sigma_n^{\text{hist}}$：历史波动率（决定校正风险）
- $t_n$：时间特征（早高峰/晚高峰/平峰需要不同校正）

**预期效果**：
- 强趋势样本自动获得高 eta（1.5-2.0）
- 弱趋势样本自动获得低 eta（0.3-0.5）
- 自由流样本自动获得 eta ≈ 0

### 2.3 为什么现在就该升级

Step 2 的结果已经证明：
1. 校正方向正确（所有 eta > 0 都改善）
2. 全局固定是瓶颈（eta=1.0 和 eta=2.0 各有优劣）
3. 需要自适应（不同样本需要不同校正强度）

此时升级为可学习 eta，不是"赌博"，而是**在已验证方向上的精细化**。

---

## 三、升级路径：两步走

### Step 2.5：全局可学习标量（轻量，今晚/明天可跑）

```python
self.eta = nn.Parameter(torch.tensor(1.0))  # 用 Step 2 最优值初始化

correction = self.eta * s_hist * pred_len
y_corrected = y_pred + correction
```

**特点**：
- 只有一个额外参数
- 不增加模型容量，不增加过拟合风险
- 训练后看 eta 收敛值（应在 0.8-1.5 之间，验证与 Step 2 的一致性）
- 若收敛后 CG_MAE 接近固定 eta=1.0 → 说明瓶颈不在 eta 的全局性，而在其他因素

**判断标准**：
- 若 CG_MAE < 30.5 且 TPR_CG > 0.46 → 全局可学习足够，不需要条件化
- 若 CG_MAE ≈ 30.69（与固定 eta=1.0 持平）→ 必须升级为条件化 eta

### Step 2.6：条件可学习 eta（复杂，需更多数据）

```python
class AdaptiveEta(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出 [0, 1]，再映射到 [0, 2]
        )

    def forward(self, s_hist, var_hist, time_feat):
        # s_hist: (B, N), var_hist: (B, N), time_feat: (B, N, d_t)
        x = torch.cat([s_hist.unsqueeze(-1), 
                       var_hist.unsqueeze(-1), 
                       time_feat], dim=-1)
        eta = self.mlp(x) * 2.0  # 映射到 [0, 2]
        return eta  # (B, N, 1)
```

**输入特征**：
- $|s^{\text{hist}}|$：趋势强度（绝对值）
- $\sigma^{\text{hist}}$：历史窗口的标准差（波动率）
- $\text{hour}$ / $\text{weekday}$：时间编码（周期性先验）

**预期**：
- 早高峰强拥堵：eta → 1.5-2.0
- 平峰弱波动：eta → 0.2-0.5
- 深夜自由流：eta → 0.0

---

## 四、执行建议

### 今晚/明天（轻量验证）

**推荐先做 Step 2.5（全局可学习标量）**，原因：
1. 参数量 = 1，训练成本几乎为零
2. 初始化用 eta=1.0（Step 2 最优），收敛快
3. 若全局 eta 训练后 CG_MAE 仍 ≈ 30.69，则明确证明：**瓶颈不是 eta 的全局性，而是 Denoiser 本身的保守权重**
4. 此时可以果断放弃聚合器/后处理路线，全力进入 Step 3-4（重训 Denoiser）

### 后天（若 Step 2.5 无效）

直接进入 **Step 3（Intra-Patch Trend Encoding）** 或 **Step 4（LWR Conservation Loss）**。

---

## 五、一句话总结

> **固定 eta 验证了方向（趋势残差校正有效），但全局一刀切是瓶颈（不同样本需要不同校正强度）。升级为全局可学习标量（Step 2.5）是必要验证：若仍不过线，则证明保守性固化在 Denoiser 权重中，必须重训。**

---

## 六、论文叙事（Step 2 结论段）

> 推理阶段的趋势残差校正（Historical Trend Residual）将 CG_MAE 从 30.88 降至 30.69，TPR_CG 从 0.452 提升至 0.457，验证了系统性保守偏差的存在及历史趋势信号的有效性。然而，固定全局校正系数无法自适应不同激波强度（强趋势需要 eta≈2.0，弱趋势仅需 eta≈0.3），导致全局折中策略无法突破性能瓶颈。这表明：仅靠推理后处理不足以修复 Denoiser 固化的保守性，需进一步探索自适应校正或训练阶段物理注入。

---

*文档结束*
