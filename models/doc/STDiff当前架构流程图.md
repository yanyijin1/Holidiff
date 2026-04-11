# STDiff 当前架构文档

> 基于 SimDiff 风格的时空扩散预测模型

---

## 一、训练阶段数据流 (forward_train)

```
┌────────────────────────────────────────────────────────────────────────┐
│  训练阶段前向流程                                                        │
└────────────────────────────────────────────────────────────────────────┘

【输入】
  batch_x: (B, L=96, N=30)         # 历史序列
  batch_y: (B, L+48+12, N=30)      # 完整标签

▼ permute: (B,L,N) -> (B,N,L)
  x: (B, N, pred_len=12)           # 目标部分
  cond_ts: (B, N, seq_len=96)      # 条件部分

▼ 归一化 (全局标准化)
  mean = cond_ts.mean(dim=(1,2))
  std = cond_ts.std(dim=(1,2))
  x_norm = (x - mean) / std
  cond_norm = (cond_ts - mean) / std

▼ 扩散加噪
  t ~ Uniform(0, T)
  x_k = sqrt(alpha_t) * x_norm + sqrt(1-alpha_t) * noise

▼ FormerBone 前向（保持 B,N 维度）

  输入: x_k (B, N, 12), cond_norm (B, N, 96), t (B, N)

  [1] Patch化 (unfold)
      cond_ts.unfold(dim=-1, patch_len, stride) -> (B, N, P_cond, 6)
      x_k.unfold(dim=-1, patch_len, stride)     -> (B, N, P_x, 6)
      zcube = concat(cond_patches, x_patches)   -> (B, N, P+P', 6)
      z_embed = W_input_projection(zcube)        -> (B, N, P+P', 128)

  [2] 时间步 Token
      t: (B, N) -> time_token: (B, N, 1, 128)
      z_embed = concat(time_token, z_embed)      -> (B, N, P+P'+1, 128)

  [3] T-Down (编码阶段 x e_layers)
      Attention: (B, N, P, d) -> (B, N, P, d)
      每次保存 h 到 skip list

  [4] T-Mid (最深层)
      Attention: (B, N, P, d) -> (B, N, P, d)

  [5] T-Up (解码阶段 x e_layers)
      逐层: MLP(concat(skip.pop(), h)) + Attention

  [6] S-LWR (空间传播) <- 核心
      输入: h (B, N, T, d)  T=P+P'+1

      # 空间维度从 (B,N,T,d) 转为 (B,d,N,T) 做矩阵乘法
      q = Q_proj(h.permute(0,3,1,2))           -> (B, d, N, T)
      q = q.permute(0,2,3,1)                   -> (B, N, T, d)
      q_spatial = D @ q (D: N x N)             -> (B, N, T, d)
      delta_h = Update(q_spatial)
      h = h + sigmoid(gate) * delta_h

  [7] T-Fusion (时间融合)
      Attention: (B, N, T, d) -> (B, N, T, d)

  输出: z (B, N, pred_len)

▼ 输出映射
  z_flat = W_outs(z)                           -> (B*N, pred_len)
  z_denorm = z_flat.reshape(B,N,12) * std + mean
  model_out = z_denorm.permute(0,2,1)          -> (B, 12, N)

  输出: model_out (B, pred_len, N)
```

---

## 二、推理阶段数据流 (forward_val_test)

```
┌────────────────────────────────────────────────────────────────────────┐
│  推理阶段采样流程                                                        │
└────────────────────────────────────────────────────────────────────────┘

【输入】
  x_enc: (B, L, N)                   # 测试集历史数据

▼ 条件编码
  x_past = x_enc.permute(0,2,1)     -> (B, N, L)
  x_past = RevIN_norm(x_past)         或标准化

▼ 多次采样 (for i in range(sample_times))

  [1] start_code = randn(B*N, pred_len)   # 随机初始噪声

  [2] DPM-Solver 采样 (S步)
      for s in reversed(range(S)):
          x_{t-1} = sampler(x_t, t, cond)
          调用 FormerBone(x_t, t, cond)

  [3] diff_samples: (B, N, pred_len)
      反归一化

  收集 all_outs: (M, B, N, pred_len)

▼ 聚合 (DiffusionAggregator)
  if use_mom:
      result = rob_median_of_means(all_outs)   # 鲁棒MoM
  else:
      result = mean(all_outs)

  输出: (B, pred_len, N)
```

---

## 三、FormerBone 架构

```
FormerBone: T-Down -> T-Mid -> T-Up -> S-LWR -> T-Fusion

输入: x (B,N,T), cond (B,N,L), t (B,N)

  [1] Patch化
      zcube = concat(unfold(cond), unfold(x))  -> (B,N,P+P',6)
      z = W_proj(zcube)                         -> (B,N,P+P',d)

  [2] 时间步
      time_token = cls(t)                       -> (B,N,1,d)
      z = concat(time_token, z)                 -> (B,N,P+P'+1,d)

  [3] T-Down (e_layers 次)
      z = Attention(z)  # 自注意 + RotaryEmbedding
      z = Dropout(z)
      skip.append(z)

  [4] T-Mid
      z = Attention(z)
      z = Dropout(z)

  [5] T-Up (e_layers 次)
      z = MLP(concat(skip.pop(), z))
      z = Attention(z)
      z = Dropout(z)

  [6] S-LWR (空间传播)
      # 空间矩阵乘法 D @ Q
      q = Q_proj(z)
      q_spatial = D @ q  # D 是拉普拉斯矩阵
      z = z + gate * Update(q_spatial)

  [7] T-Fusion
      z = Attention(z)

  [8] 输出
      W_outs(z)                                  -> (B,N,pred_len)
```

---

## 四、S-LWR 空间传播算子

```
SpatialLWROperator:

输入: h (B, N, T, d)

  D 矩阵:  (N x N)
    - 图结构启用: 从邻接矩阵构建 D = I - S^T
    - 图结构禁用: D = I (单位矩阵)

  q_tilde = Q_proj(h)                 -> (B,N,T,d)
  q_spatial = D @ q_tilde              -> (B,N,T,d)  # 节点间传播
  delta_h = Update(q_spatial)         -> (B,N,T,d)
  h_new = h + sigmoid(gate) * delta_h  -> (B,N,T,d)
```

---

## 五、Attention 模块

```
Attenion (SimDiff 风格带 RotaryEmbedding):

输入: src (B, nvars, H, C)
  nvars = N (节点数)
  H = P+P'+1 (patch数量+token)
  C = d_model

  QKV = Linear(src)                  -> (B,nvars,H,3,d_model)
  q,k,v = QKV.split(3, dim=-1)

  q = rotary_emb.rotate(q)            # 旋转位置编码
  k = rotary_emb.rotate(k)

  attn = Attention(q,k,v)             -> (B,nvars,H,d_model)
  output = FFN(attn) + src
```

---

## 六、DPM-Solver 采样

```
DPMSolverSampler.sample:

输入: x_T (B*N, pred_len), conditioning (B*N, seq_len), S (步数)

  for s in reversed(range(S)):
      t = 当前时间步
      x_t = 上一步结果

      noise_pred = model.nn(x_t, t, conditioning)
      x_{t-1} = 去噪一步(noise_pred, t)

  输出: x_0 (B*N, pred_len)
```

---

## 七、MoM 聚合

```
DiffusionAggregator.aggregate:

输入: all_outs (M, B, N, pred_len)

  if use_mom:
      for _ in range(rmom_n):
          shuff = randperm(M)
          shuffled = all_outs[shuff]
          result = median_of_means(shuffled)
          results.append(result)
      return mean(results)

  else:
      return mean(all_outs)

median_of_means:
  分成 n_blocks 组
  每组求 mean
  取所有 mean 的 median
```

---

## 八、数据维度总结

| 阶段 | 变量 | 维度 |
|------|------|------|
| 输入 | batch_x | (B, 96, 30) |
| 内部 | x (目标) | (B, 30, 12) |
| 内部 | cond_ts | (B, 30, 96) |
| FormerBone | h | (B, 30, P+P'+1, 128) |
| S-LWR | h | (B, 30, P+P'+1, 128) |
| 输出 | z | (B, 30, 12) |
| 最终 | output | (B, 12, 30) |
| 采样 | all_outs | (M, B, 12, 30) |

---

## 九、模块依赖

```
fourier/
├── model.py          # Model (主入口)
│   ├── RevIN
│   ├── PatchUVIT
│   ├── DPMSolverSampler
│   └── DiffusionAggregator
│
├── unet_bone.py      # FormerBone
│   ├── Attenion (时间注意力)
│   ├── SpatialLWROperator (空间传播)
│   └── PatchUVIT
│
└── diffusion.py      # 调度
    └── DiffusionAggregator

layers/
├── RevIN.py
├── rotaryembedding.py
└── samplers/dpm_sampler.py
```