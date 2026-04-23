# ST-Koopman-FormerBone 分阶段实验方案

## 一、实验背景与目标

### 问题定义
- **短程预测（12步）**：MAE ~ 1.6，已有方案表现良好
- **长程预测（36步）**：MAE > 2.5，纯时序扩散（SimDiff）因误差累积而失效
- **核心假设**：长程预测需要时空联合的物理约束（空间LWR + 时序Koopman）

### 实验目标
验证从"纯时序扩散"到"时空联合物理嵌入"的渐进提升，支撑以下论文故事：

```
SimDiff 12步 → 你的2分支12步 → 你的2分支36步 → 你的3分支36步
    (基线)        (已有)          (Phase1)          (Phase2-3)
```

---

## 二、分阶段实施路线图

### Phase 0: 基线建立 - SimDiff 36步（3天）

**目标**：证明"纯时序扩散无法处理36步长程"

#### 实验配置

```yaml
# configs/simdiff_36step.yaml
pred_len: 36
sample_times: 5
dpm_steps: 20  # 必须增加，短程S=5，长程需S=20
seq_len: 96
d_model: 128
diff_steps: 1000
seed: 42
epochs: 30
```

#### 预期结果

| 指标 | 预期值 | 判定 |
|------|--------|------|
| 训练Loss | 震荡/不下降 | ⚠️ 异常 |
| 12步MAE | ~1.6 | ✅ 正常 |
| 36步MAE | >2.5 | ✅ 证明失效 |

#### 验证清单

- [ ] 训练loss是否收敛（预期震荡或不下降）
- [ ] 36步MAE > 2.0（vs 12步MAE ~1.6）
- [ ] 可视化：预测曲线在长程（>20步）出现明显相位偏移

#### 产出
基线报告，证明"长程需要空间约束"

---

### Phase 1: STFormerBone扩展到36步（2天）

**目标**：验证2分支架构（Local + Space）能处理36步

#### 修改点

```python
# stformer_bone.py - 动态P适配
def forward(self, x, t, cond, pred_len=36):
    # 动态P适配
    P = x.shape[-1] // 6  # 自动适配12或36
    # 其余保持不变（2分支）
    ...
```

#### 预期结果

| 指标 | 预期值 | 判定 |
|------|--------|------|
| 12步MAE | <1.55 | ✅ 优于基线 |
| 36步MAE | <2.2 | ✅ 优于SimDiff的>2.5 |

#### 决策点

| MAE范围 | 判定 | 行动 |
|---------|------|------|
| <2.0 | 2分支已足够 | Koopman增益有限，跳过Phase2 |
| 2.0~2.2 | 有改进空间 | 进入Phase 2 |
| >2.3 | LWR实现有问题 | 先调D矩阵 |

#### 验证清单

- [ ] 36步MAE < 2.2（优于SimDiff的>2.5）
- [ ] 证明：加入LWR空间约束后，长程性能提升

---

### Phase 2: 添加Koopman分支（核心，3天）

**目标**：从2分支升级到3分支，统一K=3（简化版）

#### 新增文件

```python
# koopman_branch.py
class KoopmanBranch(nn.Module):
    def __init__(self, d_model, n_modes=3):
        self.psi_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * n_modes)
        )
        # K矩阵: 初始化EDMD + 微调
        K_init = self.compute_edmd_init(n_modes)
        self.K = nn.Parameter(K_init, requires_grad=True)
        self.W_koop = nn.Linear(d_model * n_modes, d_model)
    
    def forward(self, h):
        psi_h = self.psi_proj(h)
        B, N, P, _ = psi_h.shape
        psi_h = psi_h.view(B, N, P, self.n_modes, -1)
        koop_modes = self.K * psi_h
        h_koop = self.W_koop(koop_modes.reshape(B, N, P, -1))
        return h_koop
```

#### 修改STLayer

```python
class STKoopmanLayer(nn.Module):
    def __init__(self):
        self.local = RotaryAttention(...)
        self.koopman = KoopmanBranch(n_modes=3)  # 新增
        self.space = SpatialLWRBranch(...)
        self.gate = nn.Linear(d*3, 3)  # 3路门控
    
    def forward(self, h):
        h_l = self.local(h)
        h_k = self.koopman(h)  # 新增
        h_s = self.space(h)
        # 3路softmax门控融合
        g = softmax(self.gate(torch.cat([h_l, h_k, h_s], dim=-1)))
        h_out = h + g[..., 0:1] * h_l + g[..., 1:2] * h_k + g[..., 2:3] * h_s
        return h_out
```

#### 预期结果

| 指标 | 预期值 | 判定 |
|------|--------|------|
| 36步MAE | <1.9 | ✅ 优于Phase1的2.0-2.2 |

#### 验证清单

- [ ] 36步MAE < 1.9（优于Phase 1的2.0-2.2）
- [ ] 关键：可视化门控权重，确认早晚高峰时g_koop > 0.5
- [ ] 训练稳定性：无NaN，梯度范数<10

#### 决策点

| MAE下降 | 判定 | 行动 |
|---------|------|------|
| >0.2 | Koopman有效 | 进入Phase 3 |
| 无变化 | K矩阵问题 | 检查初始化 |

---

### Phase 3: 分层K矩阵（精细化，2天）

**目标**：Layer-wise K=[2,2,3]

#### 修改点

```python
self.layers = nn.ModuleList([
    STKoopmanLayer(n_modes=2, lambdas=[0.5, 0.3]),    # Layer 1
    STKoopmanLayer(n_modes=2, lambdas=[0.9, 0.3]),    # Layer 2
    STKoopmanLayer(n_modes=3, lambdas=[0.9, 0.3, 0.99]) # Layer 3
])
```

#### 预期结果

| 指标 | 预期值 | 判定 |
|------|--------|------|
| 36步MAE | <1.85 | ✅ 优于Phase2的<1.9 |

#### 验证清单

- [ ] 36步MAE < 1.85（优于Phase 2的<1.9）
- [ ] 对比Phase 2的增益：分层 > 统一（预期+0.05 MAE提升）
- [ ] 可视化：Layer 3的g_koop在跨日预测时显著激活

#### 决策点

| 提升幅度 | 判定 | 行动 |
|----------|------|------|
| <0.03 | 收益有限 | 简化回统一K=3 |
| >0.05 | 有效 | 保留分层设计 |

---

### Phase 4: 长程优化与MoM验证（2天）

**目标**：验证MoM在长程的必要性，对比Mean vs MoM

#### 实验设计

```python
sample_times_list = [1, 3, 5, 10, 20]

# 1 sample (单样本，无MoM)
# 3 samples + Mean
# 5 samples + MoM  <-- 预期最优
# 10 samples + MoM
```

#### 验证清单

- [ ] 5 samples + MoM 最优（MAE最低，稳定性最高）
- [ ] 证明：长程预测（36步）比短程（12步）更需要MoM鲁棒聚合
- [ ] 可视化：多采样方差随步长增加（前12步方差小，后24步方差大）

---

## 三、每阶段代码修改量

| Phase | 修改文件 | 新增行数 | 风险等级 | 回滚难度 |
|-------|----------|----------|----------|----------|
| 0 | configs/ | 5行 | 低 | 极易 |
| 1 | stformer_bone.py | 10行 | 低 | 容易 |
| 2 | +koopman_branch.py, 修改layer.py | 50行 | 中 | 中等（保留git分支） |
| 3 | 修改layer初始化 | 15行 | 低 | 容易 |
| 4 | test.py聚合逻辑 | 10行 | 低 | 极易 |

---

## 四、关键检查点（Go/No-Go）

### Phase 0→1 Go条件
- **必须**: SimDiff 36步MAE > 2.0（证明问题存在）
- **否则**: 36步预测本身不难，改进无意义

### Phase 1→2 Go条件
- **必须**: STFormerBone 36步MAE在2.0-2.2之间
- **否则**: >2.3需先调D矩阵，<2.0直接结束

### Phase 2→3 Go条件
- **必须**: 统一Koopman带来>0.15 MAE下降
- **否则**: 检查K矩阵梯度是否传递

---

## 五、时间线

| 天数 | Phase | 产出 |
|------|-------|------|
| Day 1-2 | 0 | SimDiff 36步基线（证明失效） |
| Day 3-4 | 1 | STFormerBone 36步（验证空间约束） |
| Day 5-7 | 2 | +Koopman 3分支（核心验证） |
| Day 8-9 | 3 | 分层K矩阵（精细化） |
| Day 10-11 | 4 | MoM长程验证（对比Mean） |
| Day 12 | 整理 | A/B报告：5组对比实验 |

---

## 六、最简验证方案（时间紧时）

### 跳过Phase 0，直接Phase 1+2

假设SimDiff 36步会崩（符合理论），直接验证2分支→3分支的增益。

### 必须保留的实验

- Phase 1的2分支36步（证明LWR空间约束有效）
- Phase 2的3分支36步（证明Koopman时序约束有效）

### 最简产出对比表

| 实验 | MAE | 说明 |
|------|-----|------|
| SimDiff 12步 | ~1.6 | 文献值，无需复现 |
| 2-branch 12步 | <1.55 | 已有 |
| 2-branch 36步 | ~2.1 | Phase 1 |
| 3-branch 36步 | <1.85 | Phase 2（核心成果） |

**核心成果**：从纯时序(SimDiff)到时空联合(2-branch)再到物理模态嵌入(3-branch)的渐进提升。

---

## 七、Koopman设计决策确认

| # | 设计点 | 选择 | 具体实现 |
|---|--------|------|----------|
| 1 | K矩阵来源 | **C (初始化EDMD+微调)** | `nn.Parameter(K_edmd, requires_grad=True)`，lr=0.01×base |
| 2 | ψ升维映射 | **B (两层MLP)** | `Linear→GELU→Linear` |
| 3 | λ稳定性 | **C (固定值)** | 物理常数，不参与梯度更新 |
| 4 | 模态数K | **分层 K=[2,2,3]** | Layer 1-2用2模态，Layer 3用3模态 |
| 5 | Patch实现 | **B (动态P)** | `P = pred_len // 6`，动态切片对齐 |

### 物理意义

| Layer | λ值 | 物理对应 | 负责现象 |
|-------|-----|----------|----------|
| Layer 1 | [0.5, 0.3] | 快速衰减模态 | 拥堵形成/消散的瞬态动力学（5-15分钟） |
| Layer 2 | [0.9, 0.3] | 中速衰减模态 | 同步流亚稳态过渡（15-45分钟） |
| Layer 3 | [0.9, 0.3, 0.99] | 慢速衰减+日周期 | 早晚高峰转换、跨天周期（45分钟-3小时） |
