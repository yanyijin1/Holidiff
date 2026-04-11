"""
STFormerBone - 时空联合扩散架构

Phase 3: 时空联合层（时序分支 + 空间分支 + 门控）
"""
import torch
import torch.nn as nn
from einops import rearrange
from .gate import STJointLayer


class TimeEmbedding(nn.Module):
    """时间步嵌入 - 加法注入版本"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, timesteps):
        """timesteps: (B*N,), 输出: (B*N, d_model)"""
        half_d = self.d_model // 2
        t = timesteps.float().unsqueeze(-1) / 10000.0
        emb = torch.cat([torch.sin(t), torch.cos(t)], dim=-1).repeat(1, half_d)
        return self.mlp(emb)


class STFormerBone(nn.Module):
    """
    STFormerBone 主干网络 - Phase 3: 时空联合
    
    架构：Patch Embed → Time Add → [ST-Joint × n] → Output
    
    ST-Joint = TemporalBranch + SpatialBranch + Gate
    """
    def __init__(self, configs):
        super().__init__()
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.n_layers = getattr(configs, 'st_layers', 3)
        self.num_heads = configs.num_heads
        
        # Patch Embedding
        patch_num = int((self.seq_len - self.patch_len) / self.stride + 1)
        patch_num_forecast = int((self.pred_len - self.patch_len) / self.stride + 1)
        self.patch_num = patch_num
        self.patch_num_forecast = patch_num_forecast
        
        # 输入投影：patch_len -> d_model
        self.input_proj = nn.Linear(self.patch_len, self.d_model)
        self.input_dropout = nn.Dropout(configs.dropout)
        
        # 时间嵌入（加法注入）
        self.time_embed = TimeEmbedding(self.d_model)
        
        # 图结构配置
        self.graph_enabled = getattr(configs, 'graph_enabled', False)
        self.graph_adj_path = getattr(configs, 'graph_adj_path', None)
        self.graph_num_nodes = getattr(configs, 'graph_num_nodes', None)
        
        # ST-Joint Layer × n
        self.st_layers = nn.ModuleList([
            STJointLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                num_nodes=self.enc_in,
                adj_path=self.graph_adj_path if self.graph_enabled else None,
                dropout=configs.dropout
            )
            for _ in range(self.n_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(patch_num_forecast * self.d_model, self.pred_len)
        
        # Debug
        self._print_count = 0
        self._debug_mode = False
    
    def _debug_print(self, msg):
        if self._debug_mode or self._print_count == 0:
            print(msg)
        self._print_count += 1
    
    def forward(self, x, timesteps, cond_ts, x_mark_enc=None, **kwargs):
        """
        STFormerBone Forward - Phase 3: 时空联合
        
        Args:
            x: (B*N, pred_len) 输入噪声序列
            timesteps: (B*N,) 时间步
            cond_ts: (B*N, seq_len) 条件序列
        
        Returns:
            z_out: (B*N, pred_len) 输出预测
        """
        self._debug_print(f"\n[STFormer-P3] batch {self._print_count} | x: {x.shape}")
        
        # ========== Step 1: Rearrange (B*N, T) -> (B, N, T) ==========
        B_N, T = x.shape
        N = self.enc_in
        B = B_N // N
        
        x = rearrange(x, '(b n) t -> b n t', n=N)  # (B, N, pred_len)
        cond_ts = rearrange(cond_ts, '(b n) t -> b n t', n=N)  # (B, N, seq_len)
        timesteps = rearrange(timesteps, '(b n) -> b n', n=N).unsqueeze(-1)  # (B, N, 1)
        
        # ========== Step 2: Patch Embedding (unfold + Linear) ==========
        cond_patches = cond_ts.unfold(-1, size=self.patch_len, step=self.stride)
        x_patches = x.unfold(-1, size=self.patch_len, step=self.stride)
        
        # Concat cond + x: (B, N, P_cond+P_x, patch_len)
        patches = torch.cat([cond_patches, x_patches], dim=-2)
        
        # Linear projection: (B, N, P, d_model)
        h = self.input_proj(patches)
        h = self.input_dropout(h)
        
        # ========== Step 3: Time Embedding (加法注入) ==========
        t_flat = timesteps.reshape(B * N)  # (B*N,)
        time_embed = self.time_embed(t_flat)  # (B*N, d_model)
        time_embed = time_embed.reshape(B, N, 1, self.d_model)  # (B, N, 1, d_model)
        h = h + time_embed  # 广播加法
        
        # ========== Step 4: ST-Joint Layers × n ==========
        self._debug_print(f"[STFormer-P3] Processing {self.n_layers} ST-Joint layers...")
        for i, st_layer in enumerate(self.st_layers):
            h = st_layer(h)
            self._debug_print(f"[STFormer-P3] Layer {i}: {h.shape}")
        
        # ========== Step 5: Output Projection ==========
        P_cond = cond_patches.shape[2]
        P_x = x_patches.shape[2]
        
        # 只取预测部分的 patch
        h_forecast = h[:, :, P_cond:, :]  # (B, N, P_x, d_model)
        
        # reshape: (B, N, P_x*d) -> (B*N, P_x*d)
        h_flat = h_forecast.reshape(B, N, -1)  # (B, N, P_x*d)
        h_flat = h_flat.reshape(B * N, -1)  # (B*N, P_x*d)
        
        z_out = self.output_proj(h_flat)  # (B*N, pred_len)
        
        return z_out


class PatchUVIT_STFormer(nn.Module):
    """
    PatchUVIT Wrapper for STFormerBone - 适配 Model 的前向传播接口
    """
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.model = STFormerBone(configs)
        self.enc_in = configs.enc_in
    
    def forward(self, x, timesteps, cond_ts, x_mark_enc=None, **kwargs):
        return self.model(x, timesteps, cond_ts, x_mark_enc)
