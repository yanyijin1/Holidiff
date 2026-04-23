# LWRdiff 模型详细流程文档

> 日期: 2026-04-08
> 版本: v0.2
> 状态: 已完成 D 矩阵修缮

---

## 一、整体架构概览

### 1.1 输入输出格式

| 阶段 | 变量 | 形状 | 说明 |
|------|------|------|------|
| **原始数据** | `msst_speed.csv` | `(时间步, 30节点)` | 30个龙门架的速度观测 |
| **训练输入** | `batch_x` | `(B, L, N)` | 历史序列 |
| **训练目标** | `batch_y` | `(B, L+pred_len, N)` | label_len + pred_len |
| **模型输出** | `outputs` | `(B, pred_len, N)` | 预测的未来序列 |

其中:
- `B`: batch size
- `L`: seq_len (历史长度, 默认96)
- `N`: enc_in = 30 (节点数)
- `pred_len`: 预测长度 (默认12)

### 1.2 数据划分

使用 `Dataset_Custom`，按 70%/20%/10% 划分 train/val/test:

```
原始数据:  [========70%========][===20%===][==10%==]
           |← train+val border1 →|← test border →|
           
train: border1s[0]=0, border2s[0]=num_train*0.7
val:   border1s[1]=num_train-96, border2s[1]=num_train+num_vali
test:  border1s[2]=len-num_test-96, border2s[2]=len
```

### 1.3 标准化

使用 `StandardScaler`，仅用训练集拟合，对所有数据统一变换：

```python
scaler.fit(train_data)  # 仅用训练集
data = scaler.transform(all_data)  # 训练/验证/测试都用同一scaler
```

---

## 二、训练阶段前向流程

### 2.1 数据准备

```
原始 batch_x: (B, L, N)          batch_y: (B, L+label_len+pred_len, N)
     时间步 L                      时间步 L+label_len+pred_len
     节点数 N                      节点数 N
```

**关键代码** (trainer.py:175-176):
```python
dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :])
dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1)
```

### 2.2 RevIN 归一化 (条件编码)

**输入**: `x_enc` = `(B, L, N)`
**输出**: `cond_ts` = `(B, N, L)` (转置后)

**公式**:
```python
# new_norm=True 时使用 RevIN
cond_ts = RevIN_norm(x_enc)  # per-node z-score
cond_ts = cond_ts.permute(0, 2, 1)  # (B, N, L)

# new_norm=False 时使用全局统计
mean = torch.mean(x_enc, dim=-1, keepdims=True)
std = torch.std(x_enc, dim=-1, keepdims=True)
cond_ts = (x_enc - mean) / (std + 1e-5)  # (B, L, N)
cond_ts = cond_ts.permute(0, 2, 1)  # (B, N, L)
```

### 2.3 目标提取与归一化

**目标**: `x` = `batch_y[:, -pred_len:, :]` = `(B, pred_len, N)`

**公式** (ptld_model.py:88-90):
```python
mean_ = torch.mean(x, dim=1).unsqueeze(1)
std_ = torch.ones_like(torch.std(x, dim=1).unsqueeze(1))
x_norm = (x - mean_.repeat(1, pred_len, 1)) / (std_.repeat(1, pred_len, 1) + 1e-5)
```

**重塑**: `(B, pred_len, N)` → `(B*N, pred_len)`

### 2.4 扩散加噪

**输入**: 
- `x_norm` = `(B*N, L2)` 目标序列
- `t` = 随机时间步

**公式** (ptld_model.py:101):
```python
x_k = sqrt(ᾱ_t) * x_norm + sqrt(1-ᾱ_t) * noise
```

其中:
- `ᾱ_t` = `alphas_cumprod[t]`
- `noise` ~ `N(0, 1)`

**时间步采样**:
```python
t = torch.randint(0, num_timesteps, size=[B*N//2])
t = torch.cat([t, num_timesteps-1-t], dim=0)  # 对称采样
```

### 2.5 FormerBone 前向

#### 2.5.1 Patch Embed + Time Embedding

**输入**: 
- `x` = `x_k` = `(B, N, L2)`
- `cond_ts` = `(B, N, L1)`
- `timesteps` = `(B, N)`

**公式** (ptld_model.py:551-557):
```python
# 1. 拼接噪声和条件
x_stack = torch.stack([x, cond_ts[:, :, -L2:]], dim=-1)  # (B, N, L2, 2)

# 2. Patch Embed (2→d_model)
h = PatchEmbed(x_stack)  # (B, N, P, d_model)

# 3. 时间步嵌入
t_flat = timesteps.reshape(B * N)
t_emb = SinusoidalStepEmbedding(d_model)(t_flat)  # (B*N, d_model)
t_emb = step_proj(t_emb).reshape(B, N, 1, d_model)
h = h + t_emb
```

其中 `P` (patch 数) 由下式决定:
```python
P = (L2 + patch_len - 1) // stride  # 近似
```

#### 2.5.2 条件投影

**公式** (ptld_model.py:559):
```python
c = cond_proj(cond_ts).unsqueeze(2)  # (B, N, 1, d_model)
```

将条件序列从 `(B, N, L1)` 投影到 `(B, N, d_model)`。

#### 2.5.3 Denoiser Blocks

每个 `PatchDenoiserBlock` 包含:

**公式** (ptld_model.py:512-520):
```python
# 1. Self-Attention (ALiBi)
xb = xb + self_attn(norm1(xb))

# 2. Cross-Attention (条件)
x2, _ = cross_attn(norm2(xb), hb, hb)
xb = xb + x2

# 3. FFN
xb = xb + ffn(norm3(xb))

# 4. LWR Operator (空间传播)
xb = lwr_op(xb.reshape(B, N, P, d), d_op)
```

### 2.6 LWR 算子详解

**输入**: `x` = `(B, N, P, d)`, `d_op` = `D` = `(N, N)`

**Step 1: 解码**
```python
[dv, dq] = decode(x)  # Linear(d_model → 2)
# dv ∈ R^(B,N,P), dq ∈ R^(B,N,P)
```

**Step 2: 局部梯度计算**
```python
drho = dq / (|dv| + ε)  # (B, N, P)
```

**Step 3: 空间传播**
```python
drho[:, :, 0] = drho[:, :, 0]  # 初始patch保持不变
for i in range(P-1):
    drho[:, :, i+1] = drho[:, :, i] + D^T @ dq[:, :, i]
```
即:
```python
drho[:, :, i+1] = drho[:, :, i] + Σ_j (D[j,i] * dq[:, j, i])
```

**Step 4: 门控残差**
```python
Δx = project(norm(drho))  # Linear(1 → d_model)
x_out = x + sigmoid(gate) * Δx
```

**D 矩阵定义** (已修缮):
```python
# 从 adjacent_gantry.csv 构建
D[j, i] = +1  if j is downstream of i
D[j, i] = -1  if j is upstream of i
D[j, i] =  0  otherwise
```

### 2.7 Patch Unfold (重建)

**输入**: `h` = `(B, N, P, d_model)`
**输出**: `z_out` = `(B, N, pred_len)`

**公式** (ptld_model.py:430):
```python
out = zeros(B, N, pred_len)
for i in range(P):
    for j in range(patch_len):
        ti = i * stride + j - pad
        if 0 <= ti < pred_len:
            out[:, :, ti] += proj(x)[:, :, i, j]
            cnt[ti] += 1
return out / cnt
```

### 2.8 反归一化与输出

**公式** (ptld_model.py:104-106):
```python
model_out = model_out.permute(0, 2, 1)  # (B, pred_len, N)
model_out = RevIN_denorm(model_out)    # new_norm=True
model_out = model_out * std + mean      # new_norm=False
```

---

## 三、测试阶段前向流程

### 3.1 采样初始化

```python
x_past = x_enc.permute(0, 2, 1)  # (B, N, L)
x_past = RevIN_norm(x_past)      # (B, N, L)
x_past = x_past.reshape(B*N, -1) # (B*N, L)

# 采样 sample_times 次
for i in range(sample_times):
    start_code = torch.randn(B*N, pred_len)  # 随机噪声
    diff_samples = sampler.sample(start_code)
```

### 3.2 DPM-Solver 采样

**输入**: `x_T` = 随机噪声
**输出**: `x_0` = 去噪预测

使用 DPM-Solver 进行多步去噪:
```python
x = dpm_solver.sample(
    img=x_T, 
    steps=s_steps,  # 默认2步
    skip_type='time_uniform',
    method='multistep',
    order=2
)
```

### 3.3 MoM 聚合

**输入**: `all_outs` = `(M, B, pred_len, N)` (M次采样)
**输出**: `outs` = `(B, pred_len, N)`

**公式**:
```python
if mom_mode == 'mom':
    # Robust Median of Means
    outs = rob_median_of_means(all_outs)
elif mom_mode == 'cmom':
    # Clustered Mean of Means
    outs = cmom(all_outs, tau=mom_tau)
elif mom_mode == 'mmmom':
    # Mini-batch MoM
    outs = mmmom(all_outs, clusters=mom_clusters)
```

---

## 四、数据流详细图

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              训练阶段前向流程                                     │
└────────────────────────────────────────────────────────────────────────────────┘

【输入层】
│
├── batch_x: (B, L=96, N=30)        # 历史速度序列 [标准化后]
├── batch_y: (B, L+48+12, N=30)     # 完整标签序列
└── x_mark: (B, L, 4)               # 时间特征 (month, day, weekday, hour)

▼ RevIN 归一化 (per-node)
│
├── cond_ts = RevIN_norm(batch_x)   # (B, N, L)
│   └─ 每个节点独立标准化
│
▼ 目标提取
│
├── x = batch_y[:, -pred_len:, :]   # (B, 12, 30)
├── mean = mean(x, dim=1)            # (B, 1, 30) per-node mean
├── std = std(x, dim=1)              # (B, 1, 30) per-node std
└── x_norm = (x - mean) / (std+ε)   # (B, 12, 30) 标准化目标

▼ 重塑
│
└── x_norm: (B*N, pred_len)         # 展平节点维度

▼ 扩散加噪
│
├── t ~ Uniform(0, T)                # 时间步
└── x_k = sqrt(ᾱ_t) * x_norm + sqrt(1-ᾱ_t) * noise

▼ FormerBone 前向
│
├── x: (B, N, pred_len)             # 展平后
├── cond_ts: (B, N, seq_len)        # 条件序列
├── t: (B, N)                      # 时间步
│
├── ┌─────────────────────────────────────────────────────────────┐
│ │ 1. Patch Embed + Time Emb                                    │
│ │    x_stack = [x, cond_ts[:,:,-pred_len:]] → (B,N,pred_len,2) │
│ │    h = PatchEmbed(x_stack) → (B, N, P, d_model)             │
│ │    h = h + TimeEmb(t) → (B, N, P, d_model)                  │
│ └─────────────────────────────────────────────────────────────┘
│ │
├── ┌─────────────────────────────────────────────────────────────┐
│ │ 2. 条件投影                                                    │
│ │    c = cond_proj(cond_ts) → (B, N, d_model)                  │
│ └─────────────────────────────────────────────────────────────┘
│ │
├── ┌─────────────────────────────────────────────────────────────┐
│ │ 3. Denoiser Blocks (×e_layers)                              │
│ │    for block in blocks:                                      │
│ │      h = block(h, c, d_op)  ←── D矩阵驱动LWR                 │
│ │                                                           │
│ │    Block内部:                                              │
│ │      h = SelfAttn(LayerNorm(h))      # ALiBi自注意           │
│ │      h = h + CrossAttn(LayerNorm(h), c)  # 交叉注意          │
│ │      h = h + FFN(LayerNorm(h))       # 前馈网络               │
│ │      h = LWR_Operator(h, d_op)       # 空间传播 ← 核心       │
│ └─────────────────────────────────────────────────────────────┘
│ │
├── ┌─────────────────────────────────────────────────────────────┐
│ │ 4. Patch Unfold                                              │
│ │    z = PatchUnfold(h, pred_len) → (B, N, pred_len)          │
│ └─────────────────────────────────────────────────────────────┘
│
└── z_out: (B, N, pred_len)

▼ 输出层
│
├── z_out = z_out.permute(0,2,1)    # (B, pred_len, N)
├── z_out = RevIN_denorm(z_out)     # 反归一化
└── output: (B, pred_len, N)

▼ 损失计算
│
├── pred = output
├── true = batch_y[:, -pred_len:, :]
└── loss = MAE(pred, true)


┌────────────────────────────────────────────────────────────────────────────────┐
│                              测试阶段采样流程                                     │
└────────────────────────────────────────────────────────────────────────────────┘

【输入】
│
└── x_enc: (B, L, N)                # 测试集历史数据

▼ 条件编码
│
├── x_past = x_enc.permute(0,2,1)  # (B, N, L)
└── x_past = RevIN_norm(x_past)    # (B, N, L)

▼ 多次采样
│
for sample_i in range(sample_times):
│   │
│   ├── start_code = randn(B*N, pred_len)  # 随机噪声
│   │
│   ├── ┌─────────────────────────────────────────┐
│   │  │ DPM-Solver 采样 (多步去噪)               │
│   │  │ x_T → x_{T-s} → ... → x_0              │
│   │  │                                         │
│   │  │ x_{t-1} = sampler(x_t, t, cond)         │
│   │  │                                         │
│   │  │ 调用 FormerBone:                         │
│   │  │   nn(x_k, t, cond_ts, x_mark)            │
│   │  │   (前向过程同上)                         │
│   │  └─────────────────────────────────────────┘
│   │
│   └── diff_samples: (B, N, pred_len)
│
▼ 聚合 (MoM)
│
├── all_outs: (M, B, N, pred_len)
│
└── outs = MoM_aggregate(all_outs)
    │
    ├── 'mean': outs.mean(0)
    ├── 'mom': rob_median_of_means(all_outs)

▼ 最终输出
│
└── output: (B, pred_len, N)
```

---

## 五、维度变换总览表

| 位置 | 变量 | 形状 | 说明 |
|------|------|------|------|
| **DataLoader** | batch_x | `(B, 96, 30)` | 原始输入 |
| **DataLoader** | batch_y | `(B, 156, 30)` | 标签 (96+48+12) |
| **Model.forward** | x_enc | `(B, 96, 30)` | 编码器输入 |
| **Model.forward** | x_dec | `(B, 156, 30)` | 解码器输入 |
| **RevIN.norm** | cond_ts | `(B, 30, 96)` | 归一化条件 |
| **目标提取** | x | `(B, 12, 30)` | 预测目标 |
| **重塑** | x_norm | `(B*30, 12)` | 展平后目标 |
| **加噪** | x_k | `(B*30, 12)` | 加噪后的目标 |
| **FormerBone** | h | `(B, 30, P, 128)` | Patch嵌入后 |
| **PatchUnfold** | z_out | `(B, 30, 12)` | 预测输出 |
| **最终输出** | outputs | `(B, 12, 30)` | 模型输出 |

---

## 六、关键公式汇总

### 6.1 扩散过程

| 公式 | 说明 |
|------|------|
| \( x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon \) | 前向加噪 |
| \( \bar{\alpha}_t = \prod_{i=1}^t \alpha_i \) | 累积alpha |
| \( \beta_t = 1 - \alpha_t \) | beta schedule |
| \( \beta_t = \text{clip}(1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, 0.001, 0.999) \) | Cosine schedule |

### 6.2 LWR 算子

| 公式 | 说明 |
|------|------|
| \( dv, dq = \text{decode}(x) \) | 解码 |
| \( \rho_{i,p} = \frac{dq_{i,p}}{\|dv_{i,p}| + \epsilon} \) | 局部梯度 |
| \( \rho_{i,p+1} = \rho_{i,p} + \sum_j D_{j,i} \cdot dq_{j,p} \) | 空间传播 |
| \( x_{out} = x + \sigma(gate) \cdot \text{project}(\rho) \) | 门控残差 |

### 6.3 D 矩阵构造

| 条件 | D[j,i] |
|------|--------|
| j 是 i 的直接下游 | +1 |
| j 是 i 的直接上游 | -1 |
| 其他 | 0 |

---

