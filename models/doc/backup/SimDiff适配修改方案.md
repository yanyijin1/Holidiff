# SimDiff 适配 + T/S 层插入完整方案

> 日期：2026-04-08
> 目标：
> 1. 先让代码能 train（适配 SimDiff）
> 2. 然后逐步插入 T/S 层

---

## 一、当前差异分析

### 核心差异对比

| 组件 | SimDiff | 当前 ptld_model.py |
|------|---------|-------------------|
| **Attention** | `Attenion`（带 RotaryEmbedding） | `AsymmetricALiBiAttention` |
| **输入投影** | `nn.Linear(patch_len, d_model)` | 自定义 `PatchEmbed` 类 |
| **Patch处理** | `unfold` + `Linear` | `PatchEmbed` + `PatchUnfold` |
| **输出映射** | `W_outs = Linear(...)` | `PatchUnfold` |
| **时间步处理** | `.unsqueeze(-1).unsqueeze(-1)` | 无 |
| **条件注入** | 在 patch 级别 concat | 单独 `cond_proj` |

---

## 二、修改步骤（逐步验证）

### Step 1️⃣：保持 `Model` 类不变

`forward_train` / `forward_val_test` 完全保持现状，reshape 位置不变。

---

### Step 2️⃣：修改 `PatchUVIT.forward`（添加 unsqueeze）

**SimDiff 原始代码：**
```python
timesteps = rearrange(timesteps, '(b n) -> b n', n=self.enc_in).unsqueeze(-1).unsqueeze(-1)
```

**当前代码缺少 `unsqueeze`**，需要添加。

---

### Step 3️⃣：替换 `FormerBone` 内部结构

#### 3.1 删除 `PatchEmbed`，改用 `nn.Linear`

**SimDiff 方式：**
```python
self.W_input_projection = nn.Linear(self.patch_len, configs.d_model)
self.input_dropout = nn.Dropout(configs.dropout)
```

#### 3.2 删除 `PatchUnfold`，改用 `W_outs`

**SimDiff 方式：**
```python
self.W_outs = nn.Linear((patch_num + 1 + patch_num_forecast) * configs.d_model, configs.pred_len)
```

#### 3.3 使用 `unfold` 替代 `PatchEmbed`

**SimDiff 方式：**
```python
zcube0 = cond_ts.unfold(dimension=-1, size=self.patch_len, step=self.stride)
zcube1 = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
zcube = torch.cat([zcube0, zcube1], dim=-2)
z_embed = self.input_dropout(self.W_input_projection(zcube))
```

---

### Step 4️⃣：替换 `AsymmetricALiBiAttention` → `Attenion`

使用 SimDiff 的 `Attenion` 类：
- 带 `RotaryEmbedding`
- 使用 `scaled_dot_product_attention`
- FFN 结构：`Linear → GELU → Dropout → Linear`

---

### Step 5️⃣：调整 `FormerBone.forward` 数据流

**SimDiff 结构：**
```
输入: (b, n, l)
  ↓
unfold → (b, n, patch_num, patch_len)
  ↓
W_input_projection → (b, n, patch_num, d)
  ↓
concat [time_token, z_embed] → (b, n, patch_num+1, d)
  ↓
Attentions (e_layers 向下) → skip connections
  ↓
Attentions_over_token_mid
  ↓
Attentions (e_layers 向上) + skip
  ↓
W_outs → (b*n, pred_len)
```

---

### Step 6️⃣：替换 Reshape 位置（可选）

**目标**：将 reshape 移动到 `Model.forward_train` 入口处

**SimDiff 原 reshape（在 `PatchUVIT.forward`）：**
```python
x = rearrange(x, '(b n) h -> b n h', n=self.enc_in)
cond_ts = rearrange(cond_ts, '(b n) h -> b n h', n=self.enc_in)
timesteps = rearrange(timesteps, '(b n) -> b n', n=self.enc_in)
```

**修改方案**：移动到 `Model.forward_train` 入口处，这样 `FormerBone` 内部就不需要 reshape 了。

---

## 三、T/S 层插入计划

> 基础 SimDiff 架构跑通后，按以下顺序扩展

---

### Step 7️⃣：在 `FormerBone` 中插入 T-层

**位置**：在 Attention 层之前

```python
# 在 PatchUVIT.forward 中：
# 原来：x = self.model(x, timesteps, cond_ts)
# 改成：
x = self.model.temporal_layer(x)  # T-层
x = self.model(x, timesteps, cond_ts)
```

**T-层实现**（简单版本）：
```python
class TemporalLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (b, n, p, d)
        return self.norm(x + self.proj(x))
```

---

### Step 8️⃣：插入 S-层（LWR）

**位置**：T-层之后，Attention 之前

```python
# 在 FormerBone.forward 中：
# 在 Attention 循环之前加：
x = self.spatial_lwr(x)  # S-层（LWR）
```

**S-层实现**（简单版本，先不涉及物理方程）：
```python
class SpatialLWRLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (b, n, p, d)
        # 简单实现：节点间传播
        return self.norm(x + self.proj(x))
```

---

### Step 9️⃣：双通道改造

**目标**：支持 `[q, v]` 双通道输入

#### 9.1 修改输入格式
```python
# 原来：x: (b, n, l)
# 改成：x: (b, n, l, 2)  # 2 = [q, v]
```

#### 9.2 修改 S-层
```python
class SpatialLWRDualChannel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (b, n, p, 2, d)
        q = x[..., 0, :]  # (b, n, p, d)
        v = x[..., 1, :]  # (b, n, p, d)
        
        q_out = self.proj_q(q)
        v_out = self.proj_v(v)
        
        return torch.stack([q_out, v_out], dim=-2)
```

---

## 四、最终架构图

```
输入: (b, n, l, 2)  # [q, v]
  ↓
┌─────────────────────────────────────┐
│  T-层 (TemporalLayer)              │
│  - 时序建模                         │
│  - 输出: (b, n, p, d)              │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│  S-层 (SpatialLWRLayer)            │
│  - LWR 空间传播                     │
│  - 跨节点信息聚合                   │
│  - 输出: (b, n, p, d)              │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│  Attention 层 (Attenion)           │
│  - RotaryEmbedding                 │
│  - 带 skip connection               │
│  - 输出: (b*n, pred_len)           │
└─────────────────────────────────────┘
  ↓
输出: (b, n, pred_len)
```

---

## 五、验证点

| 步骤 | 验证内容 |
|------|----------|
| Step 1-5 | 模型能 import，forward_train 能跑通 |
| Step 6 | 损失正常下降 |
| Step 7 | 加入 T-层后训练稳定 |
| Step 8 | 加入 S-层后训练稳定 |
| Step 9 | 双通道推理结果合理 |

---

## 六、参考代码位置

- SimDiff 原始实现：`need/SimDiff-main/models/SimDiff.py`
- 关键类：`PatchUVIT`, `FormerBone`, `Attenion`