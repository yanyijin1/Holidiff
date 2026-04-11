# STFormerBone 实施指导

> 基于现有代码风格，渐进式迁移

---

## 一、代码修改清单

| 阶段 | 文件 | 修改内容 |
|------|------|----------|
| **新建** | `fourier/stformer_bone.py` | STFormerBone 主干网络 |
| **修改** | `fourier/model.py` | 添加模型切换开关 |

---

## 二、Phase 1: 创建 stformer_bone.py

### 2.1 核心类结构

```python
"""
STFormerBone: 时空联合扩散骨干网络
"""
import torch
import torch.nn as nn
from einops import rearrange
from layers.rotaryembedding import RotaryEmbedding
from utils.graph import build_d_matrix_from_adjacency


class STJointLayer(nn.Module):
    """时空联合层"""
    def __init__(self, d_model, num_heads, n_vars, configs):
        super().__init__()
        self.d_model = d_model
        self.n_vars = n_vars

        # Pre-LN
        self.norm = nn.LayerNorm(d_model)

        # 时序分支：RotaryAttn（无因果掩码）
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim // 2)
        self.proj = nn.Linear(d_model, d_model)

        # 空间分支：D @ Q
        self.q_proj = nn.Linear(d_model, d_model)
        self.update = nn.Linear(d_model, d_model)

        # 门控：向量级
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, h):
        """h: (B, N, P, d)"""
        residual = h
        h = self.norm(h)

        # ===== 时序分支 =====
        B, N, P, d = h.shape
        # reshape for attention: (B, N, P, d) -> (B*N, P, d) or (B*P, N, d)
        # 这里取 (B*N*P, d) 做 self-attention
        h_flat = h.reshape(B * N * P, 1, d)  # 相当于每个位置独立处理

        # 标准 attention 实现（保持现有风格）
        qkv = self.qkv(h).reshape(B, N, P, 3, self.num_heads, d // self.num_heads)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)  # (3, B, N, heads, P, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.reshape(B * N, self.num_heads, P, self.head_dim)
        k = k.reshape(B * N, self.num_heads, P, self.head_dim)
        v = v.reshape(B * N, self.num_heads, P, self.head_dim)

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # 无因果掩码（纯扩散）
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            scale = 1.0 / (self.head_dim ** 0.5)
            attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
            attn = torch.matmul(attn, v)

        h_t = self.proj(attn.reshape(B * N * P, d)).reshape(B, N, P, d)

        # ===== 空间分支 =====
        q = self.q_proj(h)
        # D @ Q: (N,N) @ (B,N,P,d) -> 需要 reshape
        q_perm = q.permute(0, 3, 1, 2)  # (B, d, N, P)
        # 使用固定的 D 矩阵
        D = self.D.to(h.device)
        q_spatial = torch.einsum('nm,bmdp->bndp', D, q_perm)  # (B, d, N, P)
        q_spatial = q_spatial.permute(0, 2, 3, 1)  # (B, N, P, d)
        h_s = self.update(q_spatial)

        # ===== 门控融合 =====
        gate = self.gate(torch.cat([h_t, h_s], dim=-1))
        out = residual + gate * h_t + (1 - gate) * h_s

        return out


class STFormerBone(nn.Module):
    """STFormerBone 主干网络（兼容 FormerBone 接口）"""
    def __init__(self, configs):
        super().__init__()
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.n_vars = configs.enc_in
        self.num_heads = configs.num_heads

        # 计算 patch 数量
        patch_num = int((configs.seq_len - self.patch_len) / self.stride + 1)
        patch_num_forecast = int((configs.pred_len - self.patch_len) / self.stride + 1)
        self.patch_num = patch_num
        self.patch_num_forecast = patch_num_forecast

        # 输入投影
        self.W_input_projection = nn.Linear(self.patch_len, configs.d_model)

        # 时间步嵌入（改为加法）
        self.cls = nn.Sequential(nn.Linear(1, configs.d_model))

        # 输出投影
        total_patches = patch_num + 1 + patch_num_forecast
        self.W_outs = nn.Linear(total_patches * configs.d_model, configs.pred_len)

        # ST-Layer
        self.st_layers = nn.ModuleList([
            STJointLayer(configs.d_model, configs.num_heads, configs.enc_in, configs)
            for _ in range(getattr(configs, 'st_layers', 3))
        ])

        # D 矩阵（从 graph.py 加载）
        if getattr(configs, 'graph_enabled', False):
            D = build_d_matrix_from_adjacency(
                configs.graph_adj_path,
                configs.graph_num_nodes
            )
            # 给每个 STJointLayer 注册 D
            for layer in self.st_layers:
                layer.register_buffer('D', D)
        else:
            # 默认单位矩阵
            for layer in self.st_layers:
                layer.register_buffer('D', torch.eye(configs.enc_in))

    def forward(self, x, timesteps, cond_ts, x_mark_enc=None, *configs, **kwargs):
        """
        兼容 FormerBone 接口
        输入: x (B*N, T), timesteps (B*N,), cond_ts (B*N, L)
        输出: z_out (B*N, pred_len)
        """
        # Rearrange: (B*N, T) -> (B, N, T)
        b = x.shape[0] // self.n_vars
        x = rearrange(x, '(b n) h -> b n h', n=self.n_vars)
        cond_ts = rearrange(cond_ts, '(b n) h -> b n h', n=self.n_vars)
        timesteps = rearrange(timesteps, '(b n) -> b n', n=self.n_vars).unsqueeze(-1).unsqueeze(-1)

        # Patch 化
        zcube0 = cond_ts.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        zcube1 = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        zcube = torch.cat([zcube0, zcube1], dim=-2)

        # 投影
        z_embed = self.W_input_projection(zcube)

        # 时间步嵌入（加法而非拼接）
        time_token = self.cls(timesteps.float())
        z_embed = z_embed + time_token.unsqueeze(-2)  # 广播加法

        # ST-Layer
        for layer in self.st_layers:
            z_embed = layer(z_embed)

        # 输出
        z_out = self.W_outs(z_embed.reshape(b, self.n_vars, -1))
        return z_out.reshape(b * self.n_vars, -1)


class PatchUVIT_STFormer(nn.Module):
    """Wrapper 兼容 PatchUVIT 接口"""
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.model = STFormerBone(configs)
        self.enc_in = configs.enc_in

    def forward(self, x, timesteps, cond_ts, x_mark_enc=None, *configs, **kwargs):
        return self.model(x, timesteps, cond_ts, x_mark_enc, *configs, **kwargs)
```

---

## 三、Phase 2: 修改 model.py

在 `model.py` 的 `__init__` 中添加切换开关：

```python
def __init__(self, configs):
    super(Model, self).__init__()
    # ... 其他初始化 ...

    # 骨干网络选择
    if getattr(configs, 'use_stformer', False):
        from .stformer_bone import PatchUVIT_STFormer
        self.nn = PatchUVIT_STFormer(configs)
        print("Using STFormerBone")
    else:
        from .unet_bone import PatchUVIT
        self.nn = PatchUVIT(configs)
        print("Using PatchUVIT (FormerBone)")

    # ... 其余保持不变 ...
```

---

## 四、配置添加

在 yaml 配置文件中添加：

```yaml
# 模型选择
use_stformer: false  # Phase 1 测试开关，false=原 FormerBone

# STFormerBone 专属
st_layers: 3  # 可调，默认3层
```

---

## 五、验证清单

| 阶段 | 检查项 | 通过标准 |
|------|--------|----------|
| P1 | `use_stformer: false` 时与原代码结果一致 | MSE 差异 < 0.01 |
| P1 | `use_stformer: true` 时输出维度正确 | `(B*N, pred_len)` |
| P2 | 训练 10 epoch 不 NaN | 损失下降 |
| P3 | 空间分支启用后 MAE 下降 | 对比 P2 提升 |

---

## 六、git 提交建议

```bash
# 提交信息
git add fourier/stformer_bone.py fourier/model.py
git commit -m "feat: add STFormerBone with use_stformer switch

- Phase 1: Create stformer_bone.py with STJointLayer
- Add model switch in model.py __init__
- Compatible with FormerBone interface"
```
