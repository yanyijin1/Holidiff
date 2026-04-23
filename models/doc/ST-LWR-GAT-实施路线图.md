# LWR-ResDiff 实施路线图

> 基于理论推导的模块化实现指南。三层验证，步步为营。

---

## 一、总体架构

```
输入层                          主干网络                         输出层
┌─────────┐                  ┌─────────────────┐              ┌──────────┐
│ v_cond  │──┐               │                 │              │          │
│(B,N,L)  │  │  ┌──────┐     │  LWR-ST-Backbone│              │ v_LWR    │
└─────────┘  │  │Patch │────▶│  ┌───────────┐  │────────────▶│(B,N,pl)  │
┌─────────┐  └──│Embed │     │  │Layer 0    │  │              │          │
│ q_obs   │────▶│      │     │  │ LWR-Spatial│  │              │ delta_pred│
│(B,N,L)  │     └──────┘     │  │ Temporal   │  │              │(B,N,pl)  │
└─────────┘                  │  └───────────┘  │              │          │
┌─────────┐                  │  ┌───────────┐  │              │ pred =   │
│ v_obs   │─────────────────▶│  │Layer 1    │  │              │ v_LWR +  │
│(B,N,L)  │                  │  │ LWR-Spatial│  │              │ delta    │
└─────────┘                  │  │ Temporal   │  │              └──────────┘
                             └─────────────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │ ResidualDenoiser│
                             │ (仅在扩散步骤)  │
                             └─────────────────┘
```

**核心主张**：物理传播是矩阵乘法主导，扩散只在残差空间。

---

## 二、模块层次与职责

### 2.1 顶层：Model（主模型）

**文件**：`LWRGAT/Model.py`

**职责**：
- 训练/推理入口
- RevIN 归一化/反归一化
- 扩散调度（cosine beta schedule）
- 调用 LWR-ST-Backbone 计算物理基线
- 调用 ResidualDenoiser 预测残差
- 组合输出：`pred = v_LWR + delta`

### 2.2 主干网络：LWR-ST-Backbone

**文件**：`LWRGAT/FormerBone.py`（重构）

**职责**：
- Patch Embedding
- 多层交替 LWR-Spatial / Temporal 传播
- 输出物理基线 v_LWR
- 输出残差特征供 ResidualDenoiser 使用

### 2.3 核心层：LWRSTLayer

**文件**：`LWRGAT/stlwr_gat.py`

**职责**：单层交替传播 = LWR-Spatial（矩阵乘法） + Temporal（Patch 间 Attention）

```
LWRSTLayer = PCGK + LWRSpatialConv + TemporalAttention + FFN
```

### 2.4 物理图核：PCGK

**职责**：
- 从真实物理量计算相态
- 计算 LWR 图核 A_LWR
- 与自适应邻接融合

**关键**：`raw_v_critical` 是 `nn.Parameter`，梯度必须非零。

### 2.5 空间传播：LWRSpatialConv

**职责**：
- **核心算子**：`h_phys = A_LWR @ h`
- **不是 Attention！不是 Attention！不是 Attention！**
- 轻量 FFN 修正

### 2.6 残差去噪器：ResidualDenoiser

**职责**：
- 输入：加噪残差 + 物理特征 + 图结构
- 输出：预测残差
- 仅在扩散步骤使用

---

## 三、实施优先级

| 优先级 | 模块 | 说明 |
|--------|------|------|
| **P0** | PCGK.compute_A_LWR | 核心物理图核，必须正确 |
| **P0** | LWRSpatialConv | 矩阵乘法替代 Attention，架构灵魂 |
| **P0** | Model.forward_train 残差逻辑 | 训练流程 |
| **P1** | LWRSTLayer 多层堆叠 | 交替时空 |
| **P1** | ResidualDenoiser | 扩散残差 |
| **P2** | Model.forward_val_test | 推理采样 |
| **P2** | PCGK 缓存优化 | 测试加速 |
| **P3** | 向量化 SpatialConv | 性能优化 |

---

## 四、三阶段实施

### Phase 1: 核心替换（本周）

**目标**：让训练能跑起来，验证 PCGK 物理约束有效。

```
改动清单：
- stlwr_gat.py: 新建 LWRSpatialConv（矩阵乘法）
- FormerBone.py: 修改 STLWRGATLayer 使用 LWRSpatialConv
- Model.py: 修改 forward_train 返回 v_LWR + delta
```

**刻意不做**：
- ❌ 不引入 pred_lwr + pred_res 显式分解输出（已有）
- ❌ 不纠结混合系数具体值
- ❌ 不做推理采样

**通过标准**：
1. 训练 loss 下降
2. `raw_v_critical.grad` 非零（> 1e-8）
3. regime 分布随时间变化

### Phase 2: 显式分解 + 推理

**前提**：Phase 1 通过。

```
改动清单：
- Model.py: forward_val_test 实现推理采样
- trainer.py: 修改损失在残差空间计算
- stlwr_gat.py: 添加 PCGK 缓存
```

**通过标准**：
1. 推理能跑通
2. v_LWR 和 delta_pred 可单独提取
3. 测试时间比原版不增加太多

### Phase 3: 性能与消融

**改动清单**：
- 向量化 LWRSpatialConv
- 路段异质性编码
- 消融实验：固定邻接 vs PCGK vs 纯自适应

---

## 五、关键检查清单

实现完成后逐项验证：

```
[ ] PCGK.raw_v_critical 是 nn.Parameter，不是 float
[ ] LWRSpatialConv 使用 torch.bmm 或 einsum，不是 F.softmax(Q@K^T)
[ ] LWRSTLayer 中 h_phys = A_LWR @ h，不是 attn = softmax(Q@K^T + log_A)
[ ] Model.forward_train 返回的 loss 是 MSE(delta_pred, delta_true)
[ ] v_LWR 和 delta_pred 可以单独提取（用于监控）
[ ] 测试时 pcgk_cache 只计算一次
[ ] Trainer 传入 flow_x 到 Model.forward
[ ] 梯度监控：raw_v_critical.grad 非 None 且非零（>1e-8）
```

---

## 六、数据流

### 训练阶段

```
v_cond (B, L, N) --RevIN.norm--> v_cond_norm (B, L, N)
                                    │
                                    ▼
                              PatchEmbed --> h_0 (B, N, P_cond, d)
                                    │
                                    ├──>[Layer 0]
                                    │      PCGK(q_obs, v_obs) --> A_LWR_0, regime_0
                                    │      LWRSpatialConv(h_0, A_LWR_0) --> h_phys_0
                                    │      TemporalAttn(h_phys_0) --> h_1
                                    │
                                    ├──>[Layer 1]
                                    │      PCGK(...) --> A_LWR_1, regime_1
                                    │      LWRSpatialConv(h_1, A_LWR_1) --> h_phys_1
                                    │      TemporalAttn(h_phys_1) --> h_2
                                    │
                                    ▼
                              OutputProj --> v_LWR (B, N, pred_len)
                                    │
                                    │--> delta_true = v_target_norm - v_LWR
                                    │      │
                                    │      ▼
                                    │   Diffusion.noise(delta_true, t) --> delta_k
                                    │      │
                                    │      ▼
                                    │   ResidualDenoiser(delta_k_embed, h_2, A_LWR_1)
                                    │      │
                                    │      ▼
                                    │   delta_pred (B, N, pred_len)
                                    │
                                    ▼
                              Loss = weighted_MAE(delta_pred, delta_true)
```

### 推理阶段

```
v_cond (B, L, N) --RevIN.norm--> v_cond_norm
                                    │
                                    ▼
                              LWR-ST-Backbone --> v_LWR (只算一次！)
                                    │
                                    │--> 预计算 PCGK 缓存
                                    │
                                    ▼
                              for sample_id in range(n_samples):
                                  delta_K = randn(B, N, pred_len)
                                  for k in reversed(range(K)):
                                      delta_pred = ResidualDenoiser(...)
                                      delta_{k-1} = DPM-Solver.step(...)
                                  all_deltas.append(delta_0)
                                    │
                                    ▼
                              delta_final = MoM(all_deltas)
                                    │
                                    ▼
                              v_pred = RevIN.denorm(v_LWR + delta_final)
```

---

## 七、与 Trainer 的接口

### 修改点 1：数据加载器解包后传入 flow_x

```python
# 在 train() 方法中
for i, batch in enumerate(train_loader):
    if len(batch) == 6:
        batch_x, batch_y, batch_x_mark, batch_y_mark, batch_flow_x, batch_flow_y = batch
    else:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_flow_x, batch_flow_y = None, None

    outputs, weight_tmp = self.model(
        batch_x, batch_x_mark, dec_inp, batch_y_mark,
        flow_x=batch_flow_x  # 新增！
    )
```

### 修改点 2：Model.forward 接收 flow_x 并拆分为 q_obs / v_obs

```python
# 在 Model.forward_train 中
if flow_x is not None:
    q_obs = flow_x.permute(0, 2, 1)  # (B, N, L) 流量
    v_obs = batch_x[..., 0].permute(0, 2, 1)  # (B, N, L) 速度
else:
    q_obs = batch_x.permute(0, 2, 1)
    v_obs = batch_x.permute(0, 2, 1)
```

### 修改点 3：损失函数在残差空间计算

```python
# 原来：loss = criterion(pred, target)
# 现在：
if self.configs.is_diff:
    # Model.forward_train 返回的 loss 已经是残差损失
    loss = outputs
else:
    loss = criterion(outputs, target)
```

---

## 八、代码结构

```
LWRGAT/
├── __init__.py
├── Model.py              # 主模型：训练/推理入口
├── FormerBone.py         # 主干网络：多层 LWRSTLayer
├── stlwr_gat.py          # 核心层：PCGK + LWRSpatialConv + Temporal
├── temporal.py           # 时间注意力（保留）
├── spatial.py            # 空间 GAT（参考，Phase 1 后可删）
└── gate.py               # 门控机制（保留）
```

---

## 九、运行命令

```bash
cd /root/yanyijin/STdiff/models

# 训练 LWR-ResDiff
python train_from_yaml.py --config config/lwrdiff_phase1_pcgk.yaml

# 对比训练 STFormer (原版)
python train_from_yaml.py --config config/lwrdiff_96_12_v2.yaml
```

---

## 十、一句话

> **先实现 PCGK + LWRSpatialConv，验证 v_c 梯度非零；再堆叠多层 LWRSTLayer；最后接入残差扩散。三层验证，步步为营。**
