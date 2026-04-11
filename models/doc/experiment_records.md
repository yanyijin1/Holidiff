# STFormerBone 实验结果

## 性能对比

| Phase | 架构 | MSE | MAE | RMSE |
|-------|------|-----|-----|------|
| P1 | Identity（骨架） | 0.889 | 0.687 | 0.943 |
| P2 | TemporalAttn（时序） | 0.770 | 0.619 | 0.878 |
| P3 | ST-Joint（时空联合） | 0.771 | 0.620 | 0.878 |

## 架构改动对比

### P1 → P2 改动：添加时序分支
- 添加 TemporalAttention（RotaryAttn，无因果掩码）
- Pre-LN 结构
- FFN 残差连接
- 全注意力机制（纯扩散）

### P2 → P3 改动：添加空间分支 + 门控
- 添加 SpatialPropagation（D @ Q，LWR拓扑传播）
- 添加向量级门控（gate = σ(W_g · Concat[H_T, H_S])）
- ST-Joint 融合：H_new = H + gate*H_T + (1-gate)*H_S

---

# 架构流程图

## Phase 1: 骨架 (Identity)
```
输入: x_k (B*N, 12), t (B*N,), cond_ts (B*N, 96)
  │
  ▼
Step 1: Rearrange → (B, N, T)
        Patch Embedding → (B, N, P, d)
  │
  ▼
Step 2: + Time Embed (加法注入)
  │
  ▼
Step 3: ST-Layer × 3 (P1: Identity 透传)
        for layer in [ST-Layer₁, ST-Layer₂, ST-Layer₃]:
            h = h + Identity(h)
  │
  ▼
Step 4: Output Projection → pred (B*N, 12)
```

## Phase 2: 时序分支 (TemporalAttention)
```
输入: x_k (B*N, 12), t (B*N,), cond_ts (B*N, 96)
  │
  ▼
Step 1: Rearrange + Patch Embedding + Time Embed → (B, N, P, d)
  │
  ▼
Step 2: ST-Layer × 3 (P2: TemporalAttention)
        for layer in [ST-Layer₁, ST-Layer₂, ST-Layer₃]:
            │
            ▼
        ┌─────────────────────────────────────────┐
        │ Pre-LayerNorm: h_norm = LayerNorm(h)   │
        └─────────────────────────────────────────┘
            │
            ▼
        ┌─────────────────────────────────────────┐
        │ TemporalAttention (时序分支)             │
        │  qkv = W_qkv(h_norm) → (B,N,P,3d)      │
        │  q, k, v = split(qkv, d)               │
        │  q = rotary.rotate(q)  ← 无位置掩码     │
        │  k = rotary.rotate(k)                   │
        │  attn = softmax(q @ k^T / √d)          │
        │ h_t = h + Proj(attn @ v)  ← 残差       │
        │ h_t = h_t + FFN(h_t)       ← FFN残差   │
        └─────────────────────────────────────────┘
            │
            ▼
        h = h + h_t
  │
  ▼
Step 3: Output Projection → pred (B*N, 12)
```

## Phase 3: 时空联合 (ST-Joint)
```
输入: x_k (B*N, 12), t (B*N,), cond_ts (B*N, 96)
  │
  ▼
Step 1: Rearrange + Patch Embedding + Time Embed → (B, N, P, d)
  │
  ▼
Step 2: ST-Layer × 3 (P3: 时空联合)
        for layer in [ST-Layer₁, ST-Layer₂, ST-Layer₃]:
            │
            ▼
        ┌─────────────────────────────────────────┐
        │ Pre-LayerNorm: h_norm = LayerNorm(h)    │
        └─────────────────────────────────────────┘
            │
            ├──────────────────┬─────────────────┐
            ▼                  ▼                  │
┌─────────────────────┐ ┌─────────────────────────┐
│ 时序分支 (Temporal) │ │ 空间分支 (Spatial)       │
│                     │ │                          │
│ RotaryAttn         │ │ q = W_q(h_norm)         │
│ (无因果掩码)        │ │ q_s = D @ q  ← D=拓扑   │
│                     │ │ D = I - S^T             │
│ h_t = ...          │ │ h_s = W_upd(q_s)        │
└─────────────────────┘ └─────────────────────────┘
            │                  │
            └──────────┬───────┘
                       ▼
        ┌─────────────────────────────────────────┐
        │ 向量级门控融合                            │
        │                                          │
        │ gate = σ(W_g · Concat[h_t, h_s])        │
        │ h_new = h + gate*h_t + (1-gate)*h_s     │
        │                                          │
        │ 物理意义:                                 │
        │  - gate → 1: 自由流，微观主导            │
        │  - gate → 0: 拥堵传播，宏观LWR主导       │
        └─────────────────────────────────────────┘
                       │
                       ▼
                   h = h + h_new
  │
  ▼
Step 3: Output Projection → pred (B*N, 12)
```

## D 矩阵 (拓扑 LWR 算子)
```
D = I - S^T  (后向差分算子)

物理意义：
- S^T: 下游传播矩阵（S^T[i,j] = 1 表示 j 是 i 的下游）
- D @ q: 每个节点值 = 自身值 - 下游节点值（上游deficit）
```

## FormerBone vs STFormerBone 对比
```
FormerBone:
T-Down × 3 → T-Mid → T-Up × 3 → [S-LWR] → T-Fusion
       时序编码固化                    ↑
                          空间修正是"事后打补丁"

STFormerBone (P3):
Input → PatchEmbed → +TimeEmbed → [ST-Joint × 3] → Output
                                     ↕
                                时空纠缠（内生耦合）
```
