# Step 3：Trend-Conditional Attention（保效果 Trick）

> **定位**：在保留当前最优配置（concat + eta=0.5）的基础上，向核心 Attention 注入趋势偏置  
> **性质**：轻量 Trick——不改模型容量、不改训练目标、只改 Attention Score  
> **目标**：CG_MAE 从 30.56 突破到 30.0 以下，同时保持总 MAE 不劣化

---

## 一、当前最优基线（Step 1+2 结果）

| 配置 | 总 MAE | CG_MAE | TPR_CG | 说明 |
|------|--------|--------|--------|------|
| SimDiff 原始 | 0.2296 | 30.8779 | 0.4519 | 基线 |
| **trend_aware_concat + eta=0.5** | **0.2276** | **30.5635** | **0.4742** | **当前最优** |

**关键事实**：
- 输入层（PatchEmbed）已优化完毕，天花板 30.56
- 后处理（eta=0.5）已校准完毕，TPR 匹配物理真值
- **剩余保守性大概率在 Attention 的凸组合平滑中**

---

## 二、Step 3 修改：Trend-Conditional Attention Bias

### 2.1 修改内容（轻量 Trick）

原始 Attention Score：
$$S = \frac{QK^T}{\sqrt{d_k}}$$

修改后（加趋势因果偏置）：
$$S = \frac{QK^T}{\sqrt{d_k}} + \underbrace{\lambda \cdot \phi \cdot M_{\text{causal}}}_{\text{Trend Bias}}$$

其中：
- $\lambda \in \{0.1, 0.3, 0.5, 1.0\}$：偏置强度（超参，可 grid search）
- $\phi = \text{sign}(s^{\text{hist}}) \cdot \sigma\left(\frac{|s^{\text{hist}}| - \mu}{\sigma_{\text{train}}} \cdot \beta\right)$：历史趋势相位
- $M_{\text{causal}}[i,j] = \frac{j-i}{L}$（当 $j > i$），否则 $0$：因果趋势掩码

### 2.2 为什么是"保效果的 Trick"

| 维度 | 传统重训 | 本 Trick |
|------|---------|---------|
| 参数量 | 增加（新层/新模块） | **零增加**（只改 score 计算） |
| 训练成本 | 从头训练 30 epoch | **继承权重**，微调 10-20 epoch |
| 模型容量 | 改变 | **不变** |
| 回退风险 | 高（新架构可能崩） | **低**（$\lambda=0$ 即原始模型） |
| 物理意义 | 可能模糊 | **明确**（趋势强时看远端未来） |

**本质**：不给模型更多容量，而是**告诉模型"什么时候该看哪里"**。

### 2.3 与 Step 1（推理注入）的本质区别

Step 1 在**推理时**注入偏置，对已固化权重无效：
- 权重已学会"保守模式"，推理偏置是外部贴标签
- 效果：所有采样仍保守，只是整体偏移

Step 3 在**训练时**注入偏置，让权重**从头学会**利用趋势：
- 权重学会"激波延续时主动看远端"
- 效果：采样产生结构性分化（部分追高、部分保守）

---

## 三、实验配置

### 3.1 初始化策略（保效果关键）

```
加载当前最优权重：trend_aware_concat + eta=0.5 的 checkpoint
├── 保留：PatchEmbed (W_mu, W_tau) —— 已优化
├── 保留：Transformer Layer 3+ —— 深层语义
├── 保留：输出投影 —— 已校准
└── 重置/微调：Transformer Layer 1-2 的 Attention 权重 —— 学习趋势偏置
```

**学习率策略**：
- Layer 1-2 Attention：lr = 0.0001（正常训练）
- 其他层：lr = 0.00001（极小，防止破坏已有优化）

### 3.2 Lambda Grid Search

| 实验组 | lambda | 预期行为 | 训练 epoch |
|--------|--------|---------|-----------|
| A | 0.0 | 对照（原始 Attention） | 10（快速） |
| B | 0.1 | 弱偏置 | 10（快速） |
| C | 0.3 | 中等偏置 | 10（快速） |
| D | 0.5 | 强偏置 | 10（快速） |
| E | 1.0 | 极强偏置 | 10（快速） |

**10 epoch 判断**：
- 若某 lambda 的 CG_MAE < 30.0 → 跑满 30 epoch
- 若所有 lambda 的 CG_MAE > 31.0 → 终止，进入 Step 4

---

## 四、判断标准

### 4.1 成功标准（10 epoch 快筛）

| CG_MAE | TPR_CG | 总 MAE | 判断 |
|--------|--------|--------|------|
| < 30.0 | > 0.47 | < 0.2280 | **成功**，跑满 30 epoch |
| 30.0-30.5 | > 0.47 | < 0.2280 | **边际有效**，跑满 30 epoch |
| > 31.0 | 任意 | 任意 | **无效**，终止 |

### 4.2 与基线的严格对比

**必须同时满足**：
1. CG_MAE < 30.5635（击败 concat + eta=0.5）
2. 总 MAE < 0.2280（不劣化全局精度）
3. TPR_CG > 0.4742（不劣化方向准确性）

**满足 1+2+3 → Step 3 成功，论文核心模块确立**

---

## 五、为什么这是最后轻量机会

当前已验证的层次：

```
输入层（PatchEmbed）    ✅ 已优化，天花板 30.56
后处理（eta）           ✅ 已校准，TPR 匹配物理
核心层（Attention）     ⬜ Step 3 正在验证
训练目标（Loss）        ⬜ Step 4 备选
```

如果 Step 3 无效，说明保守性根植于**损失函数**（MSE 条件均值陷阱），必须进入 Step 4（LWR Conservation Loss）——那是**根本性修改**，成本更高。

**Step 3 是"保效果的 Trick"的最后窗口**。

---

## 六、执行清单

### 今晚/明天
- [ ] 修改 Attention Score 计算（加 lambda * phi * M_causal）
- [ ] 加载 concat + eta=0.5 最优权重作为初始化
- [ ] 跑 lambda grid（0.0, 0.1, 0.3, 0.5, 1.0），各 10 epoch
- [ ] 记录 CG_MAE / TPR_CG / 总 MAE

### 后天
- [ ] 若某 lambda 有效：跑满 30 epoch，与基线严格对比
- [ ] 若无效：准备 Step 4（LWR Conservation Loss）

---

## 七、一句话

> **concat + eta=0.5 把 CG_MAE 磨到了 30.56，输入层和后处理已榨干。现在用 Trend-Conditional Attention Bias 这个零参数量 Trick，让模型在训练时学会"激波延续时主动看远端"——如果 10 epoch 内 CG_MAE 能降到 30.0 以下，这就是突破瓶颈的最后轻量机会；如果无效，说明保守性根植于 MSE 损失，必须改训练目标。**

---

*文档结束*
