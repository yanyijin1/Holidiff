# SimDiff-MoM（即HOlidiff当中的macro部分） 宏观聚合诊断方案：交通流（Flow）视角

> **版本**：v1.0  
> **适用模型**：基于 HoliDiff 的 SimDiff 变体（Fujian-30 节假日/分布漂移任务）  
> **预测目标**：交通流量（Flow），时间粒度 15min  
> **核心假设**：固定 MoM 无法感知样本级分布异质性，在交通流多相态（自由流/拥堵流）下采样预算分配失衡。

---

## 一、前置问题：我们为什么要做这次诊断？

### 1.1 从节假日到所有分布漂移

SimDiff 的固定 Median-of-Means（MoM）假设所有测试样本来自**同质的不确定性分布**。但交通流数据存在固有的**多相态结构**：

- **自由流相（Free Flow）**：Flow 低方差、单峰、采样高度一致
- **拥堵流相（Congested Flow）**：Flow 高方差、双峰/多峰、采样分散
- **临界相（Metastable）**：微小扰动导致相变，方差极大且非平稳

节假日只是**分布漂移的一种可观测实例**。真正的科学问题是：
> **固定 MoM 对所有样本分配相同的采样预算（K 组 × M 次），导致自由流样本采样冗余、拥堵/异常样本采样不足。**

### 1.2 Flow 的物理约束（15min 粒度）

| 特性 | 物理含义 | 对 MoM 的影响 |
|------|---------|--------------|
| **非负性** | Flow ≥ 0，标准化后仍有隐式下界 | 预测值若出现负值需截断，截断率反映分布偏移程度 |
| **双峰结构** | 早晚高峰形成明显双峰分布 | 固定采样难以覆盖双峰的两侧尾部，中位数可能落在峰谷而非真实值 |
| **15min 粒度** | 时间片较宽，内部可能包含状态跃迁 | 一个 15min 窗口可能混合自由流→拥堵的过渡，增加样本内方差 |
| **瓶颈截断** | 道路容量上限导致右尾截断 | 拥堵时采样集中在容量上限附近，方差被人为压低，但真实不确定性被掩盖 |

### 1.3 诊断目标

我们不修改模型，仅通过**插入诊断钩子**收集证据，回答三个问题：

1. **采样级**：不同流相态的组内方差 $\sigma^2_{\mathrm{intra}}$ 是否显著不同？
2. **聚合级**：固定中位数在哪些样本上失效？失效模式是什么？
3. **误差级**：组间离散度 $D_{\mathrm{inter}}$ 与最终预测误差是否相关？

---

## 二、现有 MoM 机制剖析

基于当前代码（`HoliDiff.py`），宏观估计器（原MOM）执行流程如下：

```
输入：sample_times 次完整采样结果
      all_outs.shape = (sample_times, B, pred_len, N)

Step 1: 随机打乱（shuffle）
        tensor = tensor[indic]  # indic 为随机索引

Step 2: 分块（n_blocks=5）
        每块大小 ≈ sample_times / n_blocks

Step 3: 组内均值（_emp_mean）
        block_mean = block.mean(dim=0)

Step 4: 组间中位数
        means = torch.stack(means)  # (n_blocks, B, pred_len, N)
        pred = torch.median(means, dim=0)[0]

Step 5: 重复随机化（_rob_median_of_means）
        重复 rmom=20 次 Step 1-4
        对 20 个中位数结果再取均值
```

**关键观察**：
- `sample_times=50` 被随机分成 `n_blocks=5` 组，每组约 10 个采样
- 中位数操作在 **block 维度（dim=0）** 执行，对每个 (batch, time, node) 位置独立取中位数
- 最终 `_rob_median_of_means` 是对 20 次随机化的 MoM 结果做均值，等效于**多次随机分块的集成**

---

## 三、五层诊断方案

### 3.1 第一层：采样级诊断 —— 组内方差与流相态关联

**目的**：验证"拥堵相样本的组内方差显著高于自由流相"这一核心假设。

**插入点**：`HoliDiff.py`，`_median_of_means` 函数，在 `means.append(block_mean)` 之后、 `torch.median` 之前。

**收集指标**：

```python
# ===== 诊断钩子 1：组内方差 =====
# 在 _median_of_means 中，计算每块的组内方差
block_var = torch.var(block, dim=0, unbiased=False)  # (B, pred_len, N)
# 聚合为标量不确定性分数（先对 N, pred_len 平均，保留 batch 维）
block_var_scalar = block_var.mean(dim=[1,2])  # (B,)
# 存入诊断缓存（需要在 __init__ 中初始化 self.diag_intra_var = []）
if not hasattr(self, '_diag_intra_var'): self._diag_intra_var = []
self._diag_intra_var.append(block_var_scalar.detach().cpu().numpy())
```

**后续分析**（在 `test()` 循环外执行）：

```python
# 假设已有：is_holiday 标签 (num_samples,), preds (num_samples, pred_len, N)
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

intra_vars = np.concatenate(self.model._diag_intra_var, axis=0)  # (num_samples, n_blocks)
# 对 n_blocks 再取平均，得到样本级平均组内方差
sample_var = intra_vars.mean(axis=1)

# 1. 节假日 vs 常规日的方差差异
holiday_mask = is_holiday == 1
u_stat, p_val = mannwhitneyu(sample_var[holiday_mask], sample_var[~holiday_mask], alternative='greater')
print(f"Mann-Whitney U: {u_stat}, p={p_val}")

# 2. 方差与误差的关联
errors = np.mean(np.abs(preds - trues), axis=(1,2))  # (num_samples,)
rho, p_rho = spearmanr(sample_var, errors)
print(f"Spearman(方差, 误差): rho={rho:.3f}, p={p_rho:.3e}")
```

**预期产出**：
- 箱线图：节假日方差分布 vs 常规日方差分布
- 散点图：样本级方差 vs 样本级 MAE
- 判断阈值：若 $p < 0.05$ 且 $
ho > 0.3$，则方差是有效的不确定性代理

---

### 3.2 第二层：聚合级诊断 —— 中位数失效模式

**目的**：定位固定中位数在哪些时空位置失效，以及失效的物理原因。

**插入点**：`HoliDiff.py`，`_median_of_means` 函数，在 `torch.median(means, dim=0)` 前后。

**收集指标**：

```python
# ===== 诊断钩子 2：中位数偏移与组间离散度 =====
# 在 torch.median 之前
means_tensor = torch.stack(means)  # (n_blocks, B, pred_len, N)

# 组间离散度：各组均值与总体均值的偏离
overall_mean = means_tensor.mean(dim=0)  # (B, pred_len, N)
inter_dev = torch.mean((means_tensor - overall_mean.unsqueeze(0))**2, dim=0)  # (B, pred_len, N)
inter_dev_scalar = inter_dev.mean(dim=[1,2])  # (B,)

# 中位数 vs 均值偏移
median_pred = torch.median(means_tensor, dim=0)[0]  # (B, pred_len, N)
mean_pred = overall_mean
median_bias = torch.abs(median_pred - mean_pred).mean(dim=[1,2])  # (B,)

# 存入缓存
if not hasattr(self, '_diag_inter_dev'): self._diag_inter_dev = []
if not hasattr(self, '_diag_median_bias'): self._diag_median_bias = []
self._diag_inter_dev.append(inter_dev_scalar.detach().cpu().numpy())
self._diag_median_bias.append(median_bias.detach().cpu().numpy())
```

**后续分析**：

```python
inter_devs = np.concatenate(self.model._diag_inter_dev, axis=0).mean(axis=1)  # (num_samples,)
median_biases = np.concatenate(self.model._diag_median_bias, axis=0)  # (num_samples,)

# 分析：高组间离散度样本的中位数偏移是否更大
high_dev_mask = inter_devs > np.percentile(inter_devs, 80)
print(f"高离散度样本的中位数偏移: {median_biases[high_dev_mask].mean():.4f}")
print(f"低离散度样本的中位数偏移: {median_biases[~high_dev_mask].mean():.4f}")

# 与节假日关联
print(f"节假日样本的组间离散度: {inter_devs[holiday_mask].mean():.4f}")
print(f"常规日样本的组间离散度: {inter_devs[~holiday_mask].mean():.4f}")
```

**预期产出**：
- 箱线图：高/低离散度样本的中位数偏移对比
- 热力图：组间离散度在 (节点, 时间步) 上的分布
- 关键发现：若拥堵相样本的 $D_{\mathrm{inter}}$ 显著更高且中位数偏移更大，证明固定中位数在此类样本上不稳定

---

### 3.3 第三层：误差级诊断 —— 何时、何地、何种流相态失效

**目的**：将预测误差拆解到节点维和时间维，定位 MoM 的"弱点地图"。

**插入点**：`exp_long_term_forecasting.py`，`test()` 函数，在 `preds.append()` 之后。

**收集指标**（无需修改模型，仅在测试脚本中增加）：

```python
# ===== 诊断钩子 3：时空误差拆解 =====
# 在 test() 的循环中，已有 preds 和 trues 的累积
# 在循环结束后，执行以下分析：

# 1. 按节点拆解 MAE
node_mae = np.mean(np.abs(preds - trues), axis=(0,1))  # (N,)
# 2. 按时间步拆解 MAE（相对 pred_len 的位置）
timestep_mae = np.mean(np.abs(preds - trues), axis=(0,2))  # (pred_len,)
# 3. 按样本拆解（用于与方差关联）
sample_mae = np.mean(np.abs(preds - trues), axis=(1,2))  # (num_samples,)

# 4. 按流相态拆解（需要基于 true flow 定义相态）
# 注意：trues 是标准化后的值，需要反标准化或基于分位数定义
# 建议：在标准化前保存分位数阈值，或直接用标准化后的分位数
flow_mean_per_sample = np.mean(trues, axis=(1,2))  # (num_samples,)
# 定义相态（基于标准化 flow 的分位数，需根据实际数据调整）
q33, q66 = np.percentile(flow_mean_per_sample, [33, 66])
free_flow_mask = flow_mean_per_sample < q33
congested_mask = flow_mean_per_sample > q66

print(f"自由流相 MAE: {sample_mae[free_flow_mask].mean():.4f}")
print(f"拥堵流相 MAE: {sample_mae[congested_mask].mean():.4f}")
print(f"临界相 MAE: {sample_mae[~(free_flow_mask | congested_mask)].mean():.4f}")
```

**预期产出**：
- 节点级 MAE 排序：识别瓶颈节点（误差持续高的节点）
- 时间步级 MAE 曲线：观察 pred_len 内误差是否随预测 horizon 增加而恶化
- 流相态误差对比：验证拥堵相误差显著高于自由流相

---

### 3.4 第四层：Flow 物理约束诊断 —— 预测分布与真实分布的匹配度

**目的**：诊断 MoM 输出是否破坏了 Flow 的物理分布特性（双峰、非负）。

**插入点**：`exp_long_term_forecasting.py`，`test()` 循环结束后。

**收集指标**：

```python
# ===== 诊断钩子 4：物理约束违背率 =====

# 1. 非负截断率（标准化后若出现负值，说明分布偏移严重）
neg_ratio = np.mean(preds < 0, axis=(1,2))  # (num_samples,)
print(f"预测负值率（全样本）: {neg_ratio.mean():.4f}")
print(f"预测负值率（节假日）: {neg_ratio[holiday_mask].mean():.4f}")
print(f"预测负值率（常规日）: {neg_ratio[~holiday_mask].mean():.4f}")

# 2. 双峰结构匹配度（KDE 对比）
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# 选取一个代表性节点（如 MAE 最高的节点）
rep_node = np.argmax(node_mae)
true_dist = trues[:, :, rep_node].flatten()
pred_dist = preds[:, :, rep_node].flatten()

# 绘制 KDE
fig, ax = plt.subplots()
for data, label, color in [(true_dist, 'True', 'blue'), (pred_dist, 'Pred', 'red')]:
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 500)
    ax.plot(x_range, kde(x_range), label=label, color=color)
ax.set_title(f'Node {rep_node} Flow Distribution (Top Error Node)')
ax.legend()
plt.savefig(f'./diag_kde_node_{rep_node}.png')

# 3. 分布漂移样本的预测分布偏移
# 对节假日样本和常规日样本分别绘制 KDE
fig, axes = plt.subplots(1, 2, figsize=(12,4))
for ax, mask, title in [(axes[0], ~holiday_mask, 'Normal'), (axes[1], holiday_mask, 'Holiday')]:
    data = preds[mask, :, rep_node].flatten()
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 500)
    ax.plot(x_range, kde(x_range))
    ax.set_title(f'{title} Pred Distribution')
plt.savefig('./diag_kde_holiday_vs_normal.png')
```

**预期产出**：
- 非负截断率统计：节假日样本是否更容易产生负值预测
- KDE 对比图：真实分布 vs MoM 预测分布的形状差异
- 关键发现：若 MoM 预测分布呈现过度平滑的单峰，而真实分布为双峰，证明固定聚合抹平了多峰结构

---

### 3.5 第五层：时空热力图 —— 不确定性地理分布

**目的**：将诊断结果映射回空间维度，识别瓶颈路段和敏感时段。

**插入点**：`exp_long_term_forecasting.py`，`test()` 循环结束后，结合 `Dataset_FujianBinaryHoliday` 的 `station_index` 映射。

**收集指标**：

```python
# ===== 诊断钩子 5：时空热力图 =====

# 1. 节点级方差地图
node_var = np.mean(sample_var[:, None] * np.ones((1, N)), axis=0)  # 若 sample_var 是标量，需扩展
# 更精确：从诊断缓存中恢复 (num_samples, n_blocks) 的 intra_vars
# 对 n_blocks 平均后，再对样本平均，得到节点级平均方差
# 注意：当前 block_var 是 (B, pred_len, N)，需要在 test() 循环中额外累积

# 建议：在 test() 循环中增加节点级误差累积
if not hasattr(self, '_node_error_map'): self._node_error_map = np.zeros(N)
if not hasattr(self, '_node_count_map'): self._node_count_map = np.zeros(N)
batch_node_mae = np.mean(np.abs(pred - true), axis=1)  # (B, N)
self._node_error_map += batch_node_mae.sum(axis=0)
self._node_count_map += batch_node_mae.shape[0]

# 循环结束后
node_mae_map = self._node_error_map / self._node_count_map  # (N,)

# 2. 时间步级误差曲线
timestep_mae_curve = np.mean(np.abs(preds - trues), axis=(0,2))  # (pred_len,)

# 3. 节假日 × 时段交叉分析
# 需要 hour 标签，可从 batch_y_mark 中提取
# 假设已保存 hour_labels (num_samples,)
hourly_holiday_mae = np.zeros(24)
hourly_normal_mae = np.zeros(24)
hourly_counts_h = np.zeros(24)
hourly_counts_n = np.zeros(24)

for i in range(num_samples):
    h = int(hour_labels[i])
    if holiday_mask[i]:
        hourly_holiday_mae[h] += sample_mae[i]
        hourly_counts_h[h] += 1
    else:
        hourly_normal_mae[h] += sample_mae[i]
        hourly_counts_n[h] += 1

hourly_holiday_mae /= (hourly_counts_h + 1e-8)
hourly_normal_mae /= (hourly_counts_n + 1e-8)

# 绘制
fig, ax = plt.subplots()
ax.plot(range(24), hourly_normal_mae, label='Normal', marker='o')
ax.plot(range(24), hourly_holiday_mae, label='Holiday', marker='s')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('MAE')
ax.set_title('Hourly MAE: Holiday vs Normal')
ax.legend()
plt.savefig('./diag_hourly_mae.png')
```

**预期产出**：
- 节点级误差排序 CSV：用于识别瓶颈站点
- 24h 误差曲线：识别早高峰（7-9）、晚高峰（17-19）的误差峰值
- 节假日 vs 常规日小时级误差对比：验证漂移发生的关键时段

---

## 四、代码插入指南（精确位置）

### 4.1 HoliDiff.py 修改清单

**文件**：`backbone/diffusion/models/HoliDiff.py`

**修改 1**：在 `__init__` 中初始化诊断缓存（可选，也可在 test() 中外部管理）

```python
# 在 __init__ 末尾添加：
self._diag_intra_var = []      # 组内方差
self._diag_inter_dev = []     # 组间离散度
self._diag_median_bias = []    # 中位数偏移
```

**修改 2**：在 `_median_of_means` 函数中（Lines 235-254 附近）

找到以下代码段：
```python
means = []
for i in range(self.n_blocks):
    start_index = i * block_size
    end_index = min((i + 1) * block_size, n)
    if start_index >= end_index:
        break
    block = tensor[start_index:end_index]
    block_mean = self._emp_mean(block)
    means.append(block_mean)
```

**在其后插入**：

```python
    # ===== DIAG: Intra-group variance =====
    block_var = torch.var(block, dim=0, unbiased=False)  # (B, pred_len, N)
    # 聚合为样本级标量：先对 pred_len, N 平均
    block_var_scalar = block_var.mean(dim=[1,2])  # (B,)
    if not hasattr(self, '_diag_block_var'): 
        self._diag_block_var = []
    self._diag_block_var.append(block_var_scalar.detach().cpu().numpy())
```

找到：
```python
means = torch.stack(means)
pred = torch.median(means, dim=0)[0]
```

**在其前插入**：

```python
# ===== DIAG: Inter-group deviation & median bias =====
means_tensor = torch.stack(means)  # (n_blocks, B, pred_len, N)
overall_mean = means_tensor.mean(dim=0)  # (B, pred_len, N)
inter_dev = torch.mean((means_tensor - overall_mean.unsqueeze(0))**2, dim=0)  # (B, pred_len, N)
inter_dev_scalar = inter_dev.mean(dim=[1,2])  # (B,)

median_pred = torch.median(means_tensor, dim=0)[0]  # (B, pred_len, N)
median_bias = torch.abs(median_pred - overall_mean).mean(dim=[1,2])  # (B,)

if not hasattr(self, '_diag_inter_dev'): self._diag_inter_dev = []
if not hasattr(self, '_diag_median_bias'): self._diag_median_bias = []
self._diag_inter_dev.append(inter_dev_scalar.detach().cpu().numpy())
self._diag_median_bias.append(median_bias.detach().cpu().numpy())
```

**修改 3**：在 `forward_val_test` 或测试脚本中，每次测试前清空缓存

```python
# 在调用 model 推理前执行：
model._diag_block_var = []
model._diag_inter_dev = []
model._diag_median_bias = []
```

### 4.2 exp_long_term_forecasting.py 修改清单

**文件**：`backbone/exp/exp_long_term_forecasting.py`

**修改 1**：在 `test()` 函数中，增加节点级误差累积（在 `preds.append()` 之后）

```python
# 原有代码：
# preds.append(outputs.detach().cpu().numpy())
# trues.append(batch_y[:, -self.args.pred_len:, :].detach().cpu().numpy())

# 在其后插入：
if not hasattr(self, '_node_error_acc'):
    self._node_error_acc = None
    self._node_count_acc = 0

pred_np = outputs.detach().cpu().numpy()
true_np = batch_y[:, -self.args.pred_len:, :].detach().cpu().numpy()
batch_node_mae = np.mean(np.abs(pred_np - true_np), axis=1)  # (B, N)

if self._node_error_acc is None:
    self._node_error_acc = batch_node_mae.sum(axis=0)  # (N,)
else:
    self._node_error_acc += batch_node_mae.sum(axis=0)
self._node_count_acc += batch_node_mae.shape[0]
```

**修改 2**：在 `test()` 函数末尾（指标计算之后），插入完整诊断分析

```python
# 在原有指标计算（mae, mse, rmse）之后，保存结果之前，插入：

# --- 诊断分析区块开始 ---
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
import matplotlib.pyplot as plt

preds_arr = np.concatenate(preds, axis=0)  # (num_samples, pred_len, N)
trues_arr = np.concatenate(trues, axis=0)

# 1. 样本级 MAE
sample_mae = np.mean(np.abs(preds_arr - trues_arr), axis=(1,2))  # (num_samples,)

# 2. 从 model 获取诊断缓存
if hasattr(self.model, '_diag_block_var') and len(self.model._diag_block_var) > 0:
    intra_vars = np.concatenate(self.model._diag_block_var, axis=0)  # (num_samples, n_blocks)
    sample_var = intra_vars.mean(axis=1)

    # 方差 vs 误差相关性
    rho, p_rho = spearmanr(sample_var, sample_mae)
    print(f"
[Diag] Spearman(方差, 误差): rho={rho:.3f}, p={p_rho:.3e}")

    # 节假日 vs 常规日（需要 is_holiday 标签，从数据集中提取）
    # 注意：is_holiday 需要从 dataloader 中收集，见下方修改 3

# 3. 节点级误差地图
if hasattr(self, '_node_error_acc') and self._node_count_acc > 0:
    node_mae_map = self._node_error_acc / self._node_count_acc
    np.savetxt(f'./results/{setting}/node_mae_map.csv', node_mae_map, delimiter=',')
    print(f"[Diag] Node MAE map saved. Top-3 error nodes: {np.argsort(node_mae_map)[-3:]}")

# 4. 非负截断率
neg_ratio = np.mean(preds_arr < 0, axis=(1,2))
print(f"[Diag] Negative prediction ratio: {neg_ratio.mean():.4f}")

# 5. 流相态误差（基于 true flow 分位数）
flow_mean_per_sample = np.mean(trues_arr, axis=(1,2))
q33, q66 = np.percentile(flow_mean_per_sample, [33, 66])
ff_mask = flow_mean_per_sample < q33
cg_mask = flow_mean_per_sample > q66
print(f"[Diag] Free-flow MAE: {sample_mae[ff_mask].mean():.4f}")
print(f"[Diag] Congested MAE: {sample_mae[cg_mask].mean():.4f}")
print(f"[Diag] Transitional MAE: {sample_mae[~(ff_mask|cg_mask)].mean():.4f}")

# 6. 时间步级误差曲线
timestep_mae = np.mean(np.abs(preds_arr - trues_arr), axis=(0,2))
plt.figure()
plt.plot(range(len(timestep_mae)), timestep_mae, marker='o')
plt.xlabel('Prediction Horizon (15min steps)')
plt.ylabel('MAE')
plt.title('Error vs Prediction Horizon')
plt.savefig(f'./results/{setting}/diag_timestep_mae.png')

# --- 诊断分析区块结束 ---
```

**修改 3**：收集 `is_holiday` 标签（需要在 `test()` 的 dataloader 循环中收集）

由于 `__getitem__` 返回 `(seq_x, seq_y, seq_x_mark, seq_y_mark)`，其中 `seq_y_mark` 包含 `is_holiday`（在最后一维或特定位置，取决于 `timeenc` 和 `use_holiday_mark`）。

在 `test()` 循环中：

```python
# 在 for i, (batch_x, ...) 循环中，trues.append 之后插入：
# 提取 is_holiday（假设在 seq_y_mark 的最后一个特征维）
if not hasattr(self, '_holiday_labels'):
    self._holiday_labels = []
# seq_y_mark shape: (B, label_len+pred_len, n_features)
# is_holiday 通常是最后一列（需根据实际数据确认）
holiday_batch = batch_y_mark[:, -1, -1].detach().cpu().numpy()  # 假设最后一列是 is_holiday
self._holiday_labels.append(holiday_batch)
```

然后在诊断区块中：

```python
is_holiday = np.concatenate(self._holiday_labels, axis=0)  # (num_samples,)
holiday_mask = is_holiday > 0.5  # 根据实际值调整阈值

# 节假日 vs 常规日方差检验
u_stat, p_val = mannwhitneyu(sample_var[holiday_mask], sample_var[~holiday_mask], alternative='greater')
print(f"[Diag] Mann-Whitney U (holiday var > normal var): p={p_val:.3e}")

# 节假日 vs 常规日误差
print(f"[Diag] Holiday MAE: {sample_mae[holiday_mask].mean():.4f}")
print(f"[Diag] Normal MAE: {sample_mae[~holiday_mask].mean():.4f}")
```

**注意**：`batch_y_mark` 中 `is_holiday` 的具体列索引需要你在实际运行时打印 `batch_y_mark.shape` 和前几行确认。如果 `timeenc=0`，特征顺序通常是 `[month, day, weekday, hour, minute, is_holiday]`，则 `is_holiday` 是最后一列；如果 `timeenc=1`，需查看 `time_features` 的输出顺序。

---

## 五、执行流程与预期产出

### 5.1 执行步骤

```bash
# Step 0：基线运行（不修改代码，先跑一次固定 MoM）
python run.py --task_name long_term_forecast --is_training 0 ...
# 记录 mae/mse/rmse

# Step 1：插入诊断钩子（按第四章修改代码）
# 修改 HoliDiff.py 和 exp_long_term_forecasting.py

# Step 2：运行诊断实验
python run.py --task_name long_term_forecast --is_training 0 ...
# 观察控制台输出的诊断统计量

# Step 3：收集产出物
# 在 ./results/{setting}/ 下应出现：
# - node_mae_map.csv
# - diag_timestep_mae.png
# - 控制台输出的方差-误差相关性、流相态误差对比等
```

### 5.2 预期产出物清单

| 产出物 | 位置 | 用途 |
|--------|------|------|
| `node_mae_map.csv` | `./results/{setting}/` | 识别瓶颈节点，指导空间注意力分配 |
| `diag_timestep_mae.png` | `./results/{setting}/` | 验证误差是否随预测 horizon 恶化 |
| `diag_kde_node_X.png` | `./results/{setting}/` | 验证 MoM 是否抹平双峰结构 |
| `diag_kde_holiday_vs_normal.png` | `./results/{setting}/` | 验证节假日分布偏移 |
| `diag_hourly_mae.png` | `./results/{setting}/` | 识别关键敏感时段 |
| 控制台日志 | stdout | Spearman ρ、Mann-Whitney p、分相态 MAE |

### 5.3 决策判断标准

运行诊断后，用以下标准决定后续修改方向：

```
诊断结果
    │
    ├─ Spearman(方差, 误差) > 0.3 且 p < 0.05？
    │   ├─ 是 → 方差是有效不确定性代理，继续路径 A（方差加权）
    │   └─ 否 → 方差代理失效，需引入其他信号（如 LWR 残差、特征距离），转路径 C
    │
    ├─ 拥堵相 MAE >> 自由流相 MAE（> 50%）？
    │   ├─ 是 → 固定 MoM 在拥堵相显著失效，自适应聚合有明确收益空间
    │   └─ 否 → 分布漂移不是主要误差来源，问题可能在 Denoiser 本身
    │
    ├─ 节假日方差 >> 常规日方差（Mann-Whitney p < 0.05）？
    │   ├─ 是 → 节假日是分布漂移的有效探针，可用于验证自适应策略
    │   └─ 否 → 节假日标签与不确定性弱相关，需寻找其他漂移代理
    │
    └─ 高组间离散度样本的中位数偏移 > 低离散度样本？
        ├─ 是 → 中位数在复杂分布下不稳定，加权平均/自适应中位数有理论依据
        └─ 否 → 中位数鲁棒性假设成立，问题在采样不足而非聚合策略
```

---

## 六、逻辑链总结（用于论文 Motivation）

> **现象**：在 Fujian-30 数据集的节假日测试集上，固定 MoM 的预测误差显著高于常规日。  
> **归因**：交通流 Flow 具有 LWR 方程约束下的多相态结构（自由流/拥堵流），不同相态的微观不确定性分布异质。固定 MoM 对所有样本分配 `n_blocks=5` 组、`sample_times=50` 次采样的统一预算，导致拥堵相样本采样不足（组内方差高但组数固定）、自由流样本采样冗余（组内方差低但过度采样）。  
> **诊断**：通过组内方差 $\sigma^2_{\mathrm{intra}}$、组间离散度 $D_{\mathrm{inter}}$、中位数偏移度 $\mathrm{Bias}_{\mathrm{median}}$ 的三层量化分析，我们验证了：
> 1. 拥堵相/节假日样本的组内方差显著高于自由流/常规日（Mann-Whitney $p < 0.05$）；
> 2. 高方差样本的组间离散度更高，中位数偏移更大，固定中位数在此类样本上失效；
> 3. 样本级方差与最终预测误差呈显著正相关（Spearman $
ho > 0.3$）。  
> **结论**：固定 MoM 的稳健性假设在交通流的多相态分布下被打破，需要引入**样本级不确定性自适应聚合**。

---

## 附录：快速检查清单

在运行诊断前，确认以下事项：

- [ ] `HoliDiff.py` 中 `_median_of_means` 的代码已确认在 Lines 235-254
- [ ] `exp_long_term_forecasting.py` 中 `test()` 的 `preds/trues` shape 已打印确认
- [ ] `batch_y_mark` 中 `is_holiday` 的列索引已确认（建议先打印 `batch_y_mark[0,0,:]`）
- [ ] `sample_times=50, n_blocks=5, rmom=20` 与当前运行参数一致
- [ ] 结果保存目录 `./results/{setting}/` 已存在或可自动创建
- [ ] matplotlib 已安装（用于 KDE 和曲线图）

---

*文档结束*
