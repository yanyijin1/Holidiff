# SimDiff vs TSTFormer 数据流对比

## 当前状态：训练通过 ✅ (Epoch 2 Vali Loss: 0.652488)

TST 替代 SimDiff U-Net attention 层，核心数据流已对齐。

---

## forward_train 数据流（已对齐）

```
输入：
  x_dec = (B, label_len+pred_len, N) = (32, 60, 30)
  x_enc = (B, seq_len, N)            = (32, 96, 30)

Step 1: 取 target（对齐 SimDiff，只取 pred_len）
  x = x_dec[:, -pred_len:, :]        → (32, 12, 30)

Step 2: 全局 norm
  mean/std from cond_ts
  x_norm     = (32, 30, 12)
  cond_norm  = (32, 30, 96)

Step 3: reshape → SimDiff 格式
  x_k_flat   = (B*N, 12)             → (960, 12)
  cond_flat  = (B*N, 96)             → (960, 96)

Step 4: FormerBone_TST.forward(x_k_flat, t, cond_flat)

  4a. reshape
      x       = (B=32, N=30, T=12)
      cond_ts = (B=32, N=30, L=96)

  4b. unfold（patch 化）
      zcube0 = cond.unfold(96,6,3)   → (32, 30, 31, 6)   [cond_p=31]
      zcube1 = x.unfold(12,6,3)     → (32, 30, 3,  6)   [x_p=3]
      zcube  = cat(zcube0, zcube1)  → (32, 30, 34, 6)

  4c. 投影
      z_embed = W_input_proj(zcube)  → (32, 30, 34, 128)
      time_token = cls(t)            → (32, 30, 1, 128)
      z_embed = cat(time_token, z_embed) → (32, 30, 35, 128)

  4d. TST Blocks（时空联合建模，替代 U-Net attention）
      → (32, 30, 35, 128)

  4e. W_outs 投影
      total_patches = patch_num_cond + x_p + 1 = 31 + 3 + 1 = 35
      W_outs = Linear(35*128, 12) = Linear(4480, 12)
      output = z_embed.reshape(32, 30, -1) → (32, 30, 4480)
      z_out  = W_outs(output).reshape(960, 12) → (960, 12)

  输出: (B*N, pred_len=12)
```

---

## 遗留问题（后续处理）

### 问题1：weight_tmp 形状不一致
- **SimDiff**: `weight_tmp = sqrt_one_minus[t].reshape(B*N, pred_len, 1)` → `(960, 12, 1)`
- **PTLD**: `weight_tmp = sqrt_one_minus[t].reshape(B*N, 1, 1)` → `(960, 1, 1)`
- trainer.py 里如何使用 weight_tmp 需确认

### 问题2：forward_val_test 差异
- PTLD: `x_future = x_dec[:, -pred_len:, :]` 只取 pred_len
- SimDiff: `x_future = x_dec[:, -configs.pred_len:, :]` 一致
- 但 PTLD 的 agg 逻辑（rob_mom/mean）与 SimDiff 对齐方式可能不同
