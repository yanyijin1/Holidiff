# TST-LWR 时空联合扩散架构设计方案

> 日期：2026-04-08
> 目标：设计基于 SimDiff 时序Attention + LWR空间约束的时空联合扩散架构

---

## 一、问题回顾

### 当前架构问题

```
原 PTLD 架构数据流：

x: (B, N, T=12)
   ↓ reshape
x: (B*N, T=12)  ← 入口展平，节点维度丢失
   ↓ PatchEmbed (patch_len=3)
h: (B*N, P=14, d)  ← 14个patch，每个patch=3步
   ↓ AsymmetricALiBiAttention (P维度)
   ↓ PatchLWROperatorLayer (P维度传播 ← 只在时间步之间)
   ↓ PatchUnfold
out: (B*N, T=12)
```

**核心问题**：
1. LWR只在patch维度（时间步之间）传播，无法跨节点交互
2. 节点间时空关系被展平丢失
3. patch机制限制了时序Attention的全局建模能力

---

## 二、新架构


```
┌─────────────────────────────────────────────────────────────┐
│                      原 PTLD-Patch 架构                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入: x: (B, N, T)  → reshape → (B*N, T)                   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  PatchEmbed: (B*N, T) → (B*N, P, d)                 │   │
│  │  patch_len=3, stride=1                               │   │
│  │  P = ceil((T + patch_len - 1) / stride) = 14        │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  FormerBone (×e_layers)                             │   │
│  │                                                       │   │
│  │  ┌───────────────────────────────────────────────┐   │   │
│  │  │  PatchDenoiserBlock:                          │   │   │
│  │  │                                               │   │   │
│  │  │  x: (B*N, P=14, d)                           │   │   │
│  │  │       ↓                                       │   │   │
│  │  │  1. ALiBi Self-Attn (P维度) ← 局部3步        │   │   │
│  │  │       ↓                                       │   │   │
│  │  │  2. Cross-Attn (conditioning)                 │   │   │
│  │  │       ↓                                       │   │   │
│  │  │  3. FFN                                       │   │   │
│  │  │       ↓                                       │   │   │
│  │  │  4. PatchLWROperatorLayer:                   │   │   │
│  │  │     drho[:,:,i+1] = drho[:,:,i] + D @ dq   │   │   │
│  │  │     ↑ 在P维度传播 ← 时间递归                  │   │   │
│  │  │                                               │   │   │
│  │  └───────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  PatchUnfold: (B*N, P, d) → (B*N, T)                │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  输出: (B*N, T)  → reshape → (B, N, T)                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 新架构（TST-LWR）

```
┌─────────────────────────────────────────────────────────────┐
│                      新 TST-LWR 架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入: x: (B, N, T)  ← 保持格式，不展平！                    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Input Projection: (B, N, T) → (B, N, T, d)        │   │
│  │  简单的 Linear 投影                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  TST Blocks (×e_layers)                             │   │
│  │                                                       │   │
│  │  ┌───────────────────────────────────────────────┐   │   │
│  │  │  TSTBlock:                                     │   │   │
│  │  │                                               │   │   │
│  │  │  h: (B, N, T, d)                              │   │   │
│  │  │       ↓                                       │   │   │
│  │  │  1. TemporalALiBiAttention (T维度)           │   │   │
│  │  │     ↑ 在T=12上建模完整时序依赖                 │   │   │
│  │  │       ↓                                       │   │   │
│  │  │  2. Cross-Attn (conditioning)                 │   │   │
│  │  │       ↓                                       │   │   │
│  │  │  3. SpatialLWROperator (N维度)               │   │   │
│  │  │     drho[:,n,t] = drho[:,n-1,t] + D @ q     │   │   │
│  │  │     ↑ 在N维度传播 ← 跨节点传播！              │   │   │
│  │  │       ↓                                       │   │   │
│  │  │  4. TemporalALiBiAttention (时空融合)       │   │   │
│  │  │       ↓                                       │   │   │
│  │  │  5. FFN                                       │   │   │
│  │  │                                               │   │   │
│  │  └───────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Output Projection: (B, N, T, d) → (B, N, T)        │   │
│  │  + 出口 reshape: (B, N, T) → (B*N, T)              │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  输出: (B*N, T)  ← 兼容 PTLD 接口                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 核心差异对比

| 模块 | 原 PTLD-Patch | 新 TST-LWR |
|------|---------------|------------|
| **入口处理** | reshape → (B*N, T) | 保持 (B, N, T) |
| **投影方式** | PatchEmbed (patch_len=3) | Linear 直接投影 |
| **时序建模** | ALiBi 在 P=14 维度 | ALiBi 在 T=12 维度 |
| **空间传播** | LWR 在 P 维度（时间步间） | LWR 在 N 维度（跨节点） |
| **数据格式** | (B*N, P, d) | (B, N, T, d) |
| **出口处理** | PatchUnfold | 直接输出 + reshape |

---

## 三、计划实现的公式

### 3.1 LWR 物理方程

LWR (Lighthill-Whitham-Richards) 交通流模型：

$$
\frac{\partial \rho}{\partial t} + \frac{\partial q}{\partial x} = 0
$$

其中：
- $\rho$：交通密度（veh/km）
- $q$：交通流量（veh/h）
- $q = \rho \cdot v$：基本关系

### 3.2 空间传播公式（原 vs 新）

#### 原架构（Patch维度传播）

$$
\Delta \rho_{n,i+1} = \frac{q_{n,i}}{|v_{n,i}| + \epsilon}
$$

$$
\rho_{n,i+1} = \rho_{n,i} + \mathbf{D} \cdot q_{n,i}
$$

- 含义：密度变化只在时间步 $i \to i+1$ 之间传播
- 问题：无法跨节点传播

#### 新架构（节点维度传播）

$$
\Delta \rho_{n,t} = \frac{q_{n,t}}{|v_{n,t}| + \epsilon}
$$

$$
\rho_{n,t} = \rho_{n-1,t} + \mathbf{D} \cdot q_{n-1,t}
$$

- 含义：每个时间步 $t$，密度沿节点维度 $n \to n+1$ 传播
- 优势：**时空联合分布建模**

### 3.3 TST Block 计算公式

#### 符号定义

| 符号 | 含义 | 维度 |
|------|------|------|
| $\mathbf{x} = [\mathbf{q}; \mathbf{v}]$ | 原始输入 | $(B, N, T, 2)$ |
| $\mathbf{h}$ | 输入嵌入后的统一隐表示 | $(B, N, T, d)$ |
| $\tilde{\mathbf{q}}, \tilde{\mathbf{v}}$ | S-层解码出的代理变量 | $(B, N, T, d)$ |
| $\mathbf{q}_{\text{spatial}}$ | q 的空间传播 | $(B, N, T, d)$ |
| $\mathbf{D}$ | 空间算子 | $(N, N)$ |
| $\sigma(\gamma)$ | 门控 | 标量 |

---

#### T-层1：时序自注意

$$
\mathbf{h}^{(1)}_{:,n,t} = \text{ALiBi}\left(\mathbf{h}_{:,n,:}\right)
$$

含义：对每个节点 $n$，在时间维度 $T$ 上做自注意。

---

#### S-层：LWR空间传播（核心改进）

**核心思想**：直接递推 $\mathbf{h}$，不显式传播 $\rho$。

**D矩阵定义**（后向差分算子）：

$$
\mathbf{D} = \mathbf{I} - \mathbf{S}^T
$$

其中 $S^T$ 是上游移位矩阵：

$$
\mathbf{S}^T = 
\begin{pmatrix}
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0
\end{pmatrix}
$$

**第一步**：从隐表示解码出 $\tilde{\mathbf{q}}$

$$
\tilde{\mathbf{q}} = \mathbf{W}_{q'} \mathbf{h}^{(1)}
$$

**第二步**：计算空间梯度（后向差分）

$$
\mathbf{q}_{\text{spatial}} = \mathbf{D} \otimes \tilde{\mathbf{q}}
$$

展开：$[\mathbf{D} \otimes \tilde{\mathbf{q}}]_n = \tilde{q}_n - \tilde{q}_{n-1}$

**第三步**：直接递推

$$
\mathbf{h}^{(2)} = \mathbf{h}^{(1)} + \sigma(\gamma) \cdot \mathbf{W}_{\Delta} \left( \mathbf{q}_{\text{spatial}} \right)
$$

---

#### T-层2：时空融合

$$
\mathbf{h}^{(3)}_{:,n,t} = \text{ALiBi}\left(\mathbf{h}^{(2)}_{:,n,:}\right)
$$

含义：LWR空间传播后，再次在时间维度做自注意，实现时空联合建模。

---

#### FFN 层

$$
\mathbf{h}^{(4)} = \text{FFN}\left(\mathbf{h}^{(3)}\right)
$$

其中 FFN 为标准前馈网络：

$$
\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \cdot \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
$$

---

### 3.4 双通道输入嵌入（InputEmbedding）

**核心问题**：q 和 v 是两个**物理意义不同**的变量，需要分别嵌入后再融合。

#### 方案对比

| 方案 | S-层能否准确提取 $q$ 和 $v$？ | 问题 |
| ----- | -------------------- | ---------------------------------------------------- |
| **A. Concat+MLP** | ✅ **可以** | $h$ 是 $h_q$ 和 $h_v$ 的融合体，$W_q$ 可以学习到从融合表示中重建 $q$ 的映射 |
| B. 直接相加 | ⚠️ 困难 | 直接相加导致 $q$ 和 $v$ 的信号混合，难以分离 |
| C. 共享投影 | ❌ **不推荐** | 共享投影层，$q$ 和 $v$ 在同一个子空间中竞争，S-层解码出的 $q$ 会混入 $v$ 的干扰 |

**选择**：方案A（Concat+MLP）

#### 计算公式

$$
\mathbf{h}_q = \mathbf{W}_q \cdot \mathbf{q}, \quad \mathbf{h}_v = \mathbf{W}_v \cdot \mathbf{v}
$$

$$
\mathbf{h} = \text{MLP}\left(\left[\mathbf{h}_q; \mathbf{h}_v\right]\right)
$$

其中 $[\cdot; \cdot]$ 表示沿特征维度拼接。

#### 实现

```python
class InputEmbedding(nn.Module):
    """
    双通道输入嵌入

    分别对 q 和 v 进行嵌入，然后融合
    """
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(1, d_model)
        self.v_proj = nn.Linear(1, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, T, 2)  - [q, v]

        Returns:
            h: (B, N, T, d)  - 融合后的隐表示
        """
        q = x[..., 0:1]  # (B, N, T, 1)
        v = x[..., 1:2]  # (B, N, T, 1)

        h_q = self.q_proj(q)  # (B, N, T, d)
        h_v = self.v_proj(v)  # (B, N, T, d)

        h = torch.cat([h_q, h_v], dim=-1)  # (B, N, T, 2d)
        h = self.fuse(h)  # (B, N, T, d)

        return h
```

### 3.5 完整数据流

**核心点**：输入嵌入后得到 `h`，所有后续操作都在 `h` 上进行。

$$
\mathbf{x} = [\mathbf{q}; \mathbf{v}]
\xrightarrow{\text{InputEmbedding}}
\mathbf{h} \quad \text{← 隐表示}
\xrightarrow{\text{T-层1: ALiBi}}
\mathbf{h}^{(1)}
\xrightarrow{\text{CrossAttn}}
\mathbf{h}^{(1c)}
\xrightarrow{\text{S-层: LWR}}
\mathbf{h}^{(2)}
\xrightarrow{\text{T-层2: ALiBi}}
\mathbf{h}^{(3)}
\xrightarrow{\text{FFN}}
\mathbf{h}^{(4)}
$$

**说明**：
- T-层1/2 的时序 Attention 已经作用在 `h` 上（不是原始输入）
- S-层从 `h` 解码出 `q̃` 做空间传播
- 所有时空建模都基于统一的隐表示 `h`

### 3.6 与原架构的对比

| 步骤 | 原PTLD-Patch | 新TST-LWR |
|------|--------------|-----------|
| 隐表示解码 | decode → (v, q) | $\mathbf{W}_v, \mathbf{W}_q$ 投影 |
| 传播量 | $\rho = q / |v|$ | $\mathbf{q}_{\text{spatial}} = \mathbf{D} \otimes \tilde{\mathbf{q}}$ |
| 传播维度 | Patch维度 (P) | 节点维度 (N) |
| 更新目标 | $\rho$ 传播 | $\mathbf{h}_v$ 直接递推 |
| 物理意义 | 密度传播 | 速度递推（跳过ρ） |

---

## 四、SimDiff 原代码关键模块

### 4.1 核心文件位置

```
need/SimDiff-main/models/SimDiff.py
```

### 4.2 需要借鉴的模块

#### A. PatchUVIT（reshape wrapper）

```python
# 位置：SimDiff.py Line 273-284
class PatchUVIT(nn.Module):
    def __init__(self, configs, **kwargs):    
        super().__init__()    
        self.model = FormerBone(configs)
        self.enc_in = configs.enc_in
        
    def forward(self, x, timesteps, cond_ts, x_mark_enc=None, *configs, **kwargs):    
        x = rearrange(x, '(b n) h -> b n h', n=self.enc_in)
        cond_ts = rearrange(cond_ts, '(b n) h -> b n h', n=self.enc_in)
        timesteps = rearrange(timesteps, '(b n) -> b n', n=self.enc_in).unsqueeze(-1).unsqueeze(-1)     
        x = self.model(x, timesteps, cond_ts)
        return x
```

**借鉴点**：
- reshape 逻辑：在入口处处理 `(B*N, ...) → (B, N, ...)`
- 时间步展开：`timesteps.unsqueeze(-1).unsqueeze(-1)`

#### B. FormerBone（主干网络）

```python
# 位置：SimDiff.py Line 286-349
class FormerBone(nn.Module):
    def __init__(self, configs):    
        super().__init__()
        
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        
        patch_num = int((configs.seq_len - self.patch_len) / self.stride + 1)     
        patch_num_forecast = int((configs.pred_len - self.patch_len) / self.stride + 1)
        
        self.W_input_projection = nn.Linear(self.patch_len, configs.d_model)  
        self.input_dropout = nn.Dropout(configs.dropout) 
        self.cls = nn.Sequential(nn.Linear(1, configs.d_model))
        
        # 三层Attention结构��down -> mid -> up
        self.Attentions_over_token = nn.ModuleList([Attenion(configs) for i in range(configs.e_layers)])
        self.Attentions_over_token_mid = Attenion(configs) 
        self.Attentions_over_token_up = nn.ModuleList([Attenion(configs) for i in range(configs.e_layers)])
        
        # Skip connection MLP
        self.Attentions_mlp = nn.ModuleList([nn.Linear(configs.d_model*2, configs.d_model) for i in range(configs.e_layers)])
        
    def forward(self, x, timesteps, cond_ts, x_mark_enc=None):     
        b, c, s = x.shape
        
        # Patch Unfold：将序列分成patches
        zcube0 = cond_ts.unfold(dimension=-1, size=self.patch_len, step=self.stride)  
        zcube1 = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)           
        zcube = torch.cat([zcube0, zcube1], dim=-2)        
        
        z_embed = self.input_dropout(self.W_input_projection(zcube)) 
        
        # 时间步token
        time_token = self.cls(timesteps.float())
        z_embed = torch.cat((time_token, z_embed), dim=-2) 
        
        inputs = z_embed
        b, c, t, h = inputs.shape 
        skip = []
        
        # Down path：编码
        for a_2, mlp, drop, norm in zip(self.Attentions_over_token, self.Attentions_mlp, 
                                         self.Attentions_dropout, self.Attentions_norm):
            output = a_2(inputs)
            inputs = drop(output)
            skip.append(inputs)
            
        # Mid path：中间融合
        inputs = self.Attentions_over_token_mid(inputs)
        inputs = self.Attentions_dropout_mid(inputs)
        
        # Up path：解码 + skip connection
        for a_2, mlp, drop, norm in zip(self.Attentions_over_token_up, self.Attentions_mlp, 
                                         self.Attentions_dropout, self.Attentions_norm):
            prev = skip.pop()
            outputs = drop(mlp(torch.cat((prev, inputs), dim=-1))) 
            outputs = norm(outputs.reshape(b*c, t, -1)).reshape(b, c, t, -1) 
            output = a_2(inputs)
            inputs = drop(output)
        
        z_out = self.W_outs(output[:, :, :, :].reshape(b, c, -1)).reshape(b*c, -1)  
        return z_out
```

**借鉴点**：
- **不需要**：patch unfold（我们直接用完整时间序列）
- **需要**：时间步 token 的投影方式
- **需要**：三层 Attention 的结构（Down-Mid-Up）
- **改进**：去掉 skip connection（简化版），加入 LWR S层

#### C. Attenion（RoPE Attention）

```python
# 位置：SimDiff.py Line 352-394
class Attenion(nn.Module):
    def __init__(self, config, over_hidden=False, trianable_smooth=False, untoken=False, *configs, **kwargs):
        super().__init__()
        
        self.num_heads = config.num_heads
        self.c_in = config.enc_in
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=True)
        self.head_dim = config.d_model // config.num_heads
        
        self.ff_1 = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model, bias=True)
        )
        
        # RoPE：旋转位置编码
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim // 2)
        
    def forward(self, src, *configs, **kwargs):
        B, nvars, H, C = src.shape
        
        # QKV 投影 + 分头
        qkv = self.qkv(src).reshape(B, nvars, H, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # RoPE 旋转
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        
        # 注意力计算
        if hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = torch.softmax(attn, dim=-1)
            x = torch.matmul(attn, v)
        
        output1 = rearrange(x, '(b n) h e d -> b n e (h d)', b=B)
        src2 = self.ff_1(output1)
        src = src + src2
        src = src.reshape(B*nvars, -1, self.num_heads * self.head_dim)
        src = self.norm_attn(src)
        src = src.reshape(B, nvars, -1, self.num_heads * self.head_dim)
        return src
```

**借鉴点**：
- **RoPE 旋转位置编码**：比 ALiBi 更简洁
- **FFN 位置**：在 attention 后直接做 FFN（与原 PTLD 不同）
- **简化**：去掉 BatchNorm（后续用 LayerNorm）

#### D. RotaryEmbedding（RoPE 实现）

```python
# 位置：layers/rotaryembedding.py（外部依赖）
# 需要从 SimDiff 的 layers/rotaryembedding.py 复制
```

### 4.3 需要保留的原 PTLD 模块

```python
# 1. Model 类（外层 wrapper）
#    - 位置：ptld_model.py Line 31-273
#    - 保留：forward_train, forward_val_test, noise_ts, set_new_noise_schedule

# 2. RevIN 归一化
#    - 位置：layers/RevIN.py
#    - 保留：实例化和调用逻辑

# 3. DPMSolverSampler
#    - 位置：layers/samplers/dpm_sampler.py
#    - 保留：采样逻辑（不需要修改）
```

---

## 五、新架构详细设计

### 5.1 模块列表

| 模块名 | 类型 | 来源 | 备注 |
|--------|------|------|------|
| `Model` | Wrapper | 原 PTLD | 外层接口 |
| `TSTUVIT` | New | 重写 | 替代 PatchUVIT |
| `FormerBone_TST` | New | 重写 | 替代 FormerBone |
| `InputEmbedding` | New | 新设计 | **双通道 q/v 嵌入** |
| `TSTBlock` | New | 新设计 | T-S-T Block |
| `TemporalAttention` | New | 借鉴 SimDiff | 时序 Attention |
| `SpatialLWROperator` | New | 重写 LWR | 空间 LWR |
| `SinusoidalStepEmbedding` | 保留 | 原 PTLD | 时间步嵌入 |
| `RotaryEmbedding` | New | SimDiff | RoPE（可选） |
| `CrossAttention` | 保留 | PyTorch | 条件注入 |

### 5.2 完整数据流

```
原始输入 (B, N, T, 2) = [q, v]
         ↓
┌───────────────────────────────────────┐
│  InputEmbedding                       │
│  - W_q, W_v 分别投影                 │
│  - Concat + MLP 融合                 │
└───────────────────────────────────────┘
         ↓
隐表示 h (B, N, T, d)
         ↓
┌───────────────────────────────────────┐
│  TSTBlock × N_layers                  │
│  ┌─────────────────────────────────┐ │
│  │  T-层1: ALiBi(h)               │ │
│  │  CrossAttn: 注入条件            │ │
│  │  S-层: LWR空间传播 (直接递推)  │ │
│  │  T-层2: ALiBi(h)               │ │
│  │  FFN                           │ │
│  └─────────────────────────────────┘ │
└───────────────────────────────────────┘
         ↓
输出 (B, N, T, d)
```

### 5.3 TSTBlock 详细结构

```
┌─────────────────────────────────────────────────────────────┐
│                      TSTBlock                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入: h: (B, N, T, d)  ← 已经过 InputEmbedding       │
│        c: (B, N, L)  条件                              │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  norm1: LayerNorm                                   │   │
│  │  TemporalAttention (Q=K=V=h)                       │   │
│  │  + RoPE 或 ALiBi                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓ residual                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  norm2: LayerNorm                                   │   │
│  │  CrossAttention (Q=h, K=V=c)                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓ residual                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  norm3: LayerNorm                                   │   │
│  │  SpatialLWROperator (直接递推 h)                    │   │
│  │  - W_q', W_v': 解码出 q̃, ṽ                         │   │
│  │  - q_spatial = D ⊗ q̃                               │   │
│  │  - Δh = W_Δ · q_spatial                            │   │
│  │  - h_new = h + σ(γ) · Δh  ← 直接递推!             │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓ residual                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  norm4: LayerNorm                                   │   │
│  │  TemporalAttention (Q=K=V=h)                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓ residual                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  FFN: Linear(d, d_ff) → GELU → Linear(d_ff, d)     │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓ residual                         │
│  输出: h: (B, N, T, d)                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 配置精简

```yaml
# 需要删除的配置
patch_len: null  # ❌ 删除
stride: null     # ❌ 删除

# 需要保留/新增的配置
seq_len: 12
pred_len: 12
d_model: 64
num_heads: 8
e_layers: 4  # TST Block 数量
ffn_dim: 128
dropout: 0.1
enc_in: 30    # 节点数（用于 D 矩阵）
```

---

## 六、实现计划

### 阶段一：核心模块实现

1. [ ] 复制 RotaryEmbedding 从 SimDiff
2. [ ] 实现 TemporalAttention（RoPE 或 ALiBi）
3. [ ] 实现 SpatialLWROperator
4. [ ] 实现 TSTBlock
5. [ ] 实现 FormerBone_TST

### 阶段二：外层对接

1. [ ] 重写 PatchUVIT → TSTUVIT
2. [ ] 对接 Model.forward_train / forward_val_test
3. [ ] 验证 reshape 逻辑正确

### 阶段三：测试验证

1. [ ] 单步前向传播测试
2. [ ] 梯度回传测试
3. [ ] 小数据集训练测试

---

## 七、关键参考代码片段

### 7.1 RotaryEmbedding 旋转

```python
# SimDiff/models/SimDiff.py Line 372
self.rotary_emb = RotaryEmbedding(dim=self.head_dim // 2)

# SimDiff/models/SimDiff.py Line 378-379
q = self.rotary_emb.rotate_queries_or_keys(q)
k = self.rotary_emb.rotate_queries_or_keys(k)
```

### 7.2 空间 D 矩阵构建

```python
# 原 PTLD utils/graph.py
def get_d_matrix(device):
    N = 30  # 节点数
    D = torch.zeros(N, N, device=device)
    D[1:, :-1] = torch.eye(N - 1)
    D = D - D.T  # 上游-下游关系
    return D
```

### 7.3 LWR 空间传播（直接递推 H_v）

```python
class SpatialLWROperator(nn.Module):
    """
    LWR 空间传播 - 直接递推 h
    
    物理思想：上游流量变化 → 下游隐表示更新
    q̃ 只是空间传播的载体，不需要显式计算 ρ
    """
    def __init__(self, d_model, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 投影层
        self.q_proj = nn.Linear(d_model, d_model)   # q 投影
        self.v_proj = nn.Linear(d_model, d_model)   # v 投影
        self.update = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.gate = nn.Parameter(torch.zeros(1))
        
    def forward(self, h, d_op):
        """
        Args:
            h: (B, N, T, d)  - 隐表示，已经过 InputEmbedding
            d_op: (N, N)    - 空间算子，上游→下游为正
        
        Returns:
            h_new: (B, N, T, d) - 更新后的隐表示
        """
        B, N, T, d = h.shape
        
        # 第一步：从隐表示解码出 q̃（只需要 q 来做空间传播）
        q_tilde = self.q_proj(h)  # (B, N, T, d)
        
        # 第二步：q 的空间传播 = D ⊗ q̃
        # d_op: (N, N), q_tilde: (B, N, T, d)
        q_spatial = torch.einsum('nm,btmd->btnd', d_op, q_tilde)
        
        # 第三步：计算 Δh = W_Δ · q_spatial
        delta_h = self.update(q_spatial)  # (B, N, T, d)
        
        # 第四步：直接递推 h
        # h_new = h + gate * Δh
        h_new = h + torch.sigmoid(self.gate) * delta_h
        
        return h_new
```

### 7.4 空间算子 D 的构建

D矩阵采用后向差分算子：`D = I - S^T`

- `build_d_matrix_from_adjacency`: 从邻接关系构建
- `build_d_matrix_linear`: 线性拓扑（见 `models/utils/graph.py`）

```python
# 线性拓扑 D = I - S^T（N=4）
D = [[1, -1,  0,  0],
     [0,  1, -1,  0],
     [0,  0,  1, -1],
     [0,  0,  0,  1]]
```

### 7.5 TST Block（完整实现）

```python
class TSTBlock(nn.Module):
    """
    时-空-时 Block
    
    数据流: T-层1 → CrossAttn → S-层(LWR) → T-层2 → FFN
    """
    def __init__(self, d_model, num_heads, ffn_dim, dropout, num_nodes, d_op):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        # T-层1：时序自注意
        self.t_attn = TemporalAttention(d_model, num_heads, dropout)
        
        # CrossAttn：条件注入
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # S-层：LWR 空间传播（直接递推）
        self.spatial_lwr = SpatialLWROperator(d_model, num_nodes)
        self.register_buffer('d_op', d_op)
        
        # T-层2：时空融合
        self.t_fusion = TemporalAttention(d_model, num_heads, dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model)
        )
        self.drop = nn.Dropout(dropout)
        
    def forward(self, h, c):
        """
        Args:
            h: (B, N, T, d)  - 输入隐表示
            c: (B, N, L)     - 条件（历史序列）
        
        Returns:
            h: (B, N, T, d)  - 输出隐表示
        """
        B, N, T, d = h.shape
        
        # ===== T-层1：时序建模 =====
        h = h + self.drop(self.t_attn(self.norm1(h)))
        
        # ===== CrossAttn：注入条件 =====
        h_flat = h.reshape(B * N, T, d)
        c_expanded = c.unsqueeze(1).expand(B, N, T, d).reshape(B * N, T, d)
        ca_out, _ = self.cross_attn(
            self.norm2(h_flat), c_expanded, c_expanded
        )
        h = h_flat.reshape(B, N, T, d) + self.drop(ca_out.reshape(B, N, T, d))
        
        # ===== S-层：LWR 空间传播 =====
        h = h + self.spatial_lwr(self.norm3(h), self.d_op)
        
        # ===== T-层2：时空融合 =====
        h = h + self.drop(self.t_fusion(self.norm4(h)))
        
        # ===== FFN =====
        h = h + self.drop(self.ffn(h))
        
        return h
```

---

## 八、总结

### 8.1 核心改进点

| 改进点 | 原架构 | 新架构 |
|--------|--------|--------|
| 数据格式 | (B*N, T) 展平 | (B, N, T, d) 保持 |
| 时序建模 | patch内 ALiBi | 完整时间 ALiBi/RoPE |
| 空间传播 | LWR 在 P 维度（时间步间） | LWR 在 N 维度（跨节点） |
| 传播内容 | ρ 显式传播 | h_v 直接递推（跳过ρ） |
| 物理约束 | 时间递归 | 时空联合分布 |

### 8.2 预期收益

1. ✅ 解决 patch 导致的局部时序建模问题
2. ✅ LWR 真正在空间维度传播
3. ✅ 时空联合分布建模
4. ✅ h_v 直接递推，物理意义更清晰
5. ✅ 与 SimDiff 时序架构融合

### 8.3 风险点

1. ⚠️ 显存占用增加（保持 (B, N, T, d) 格式）
2. ⚠️ Attention 计算量 $O(B \cdot N \cdot T^2)$
3. ⚠️ 需要验证 RoPE 与 ALiBi 的效果差异
