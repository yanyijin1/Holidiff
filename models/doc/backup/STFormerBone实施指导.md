# STFormerBone 实施指导 v0.6

> 版本: v0.6
> 日期: 2026-04-11
> 目标：新建 stformer_bone/ 文件夹，与 fourier/ 同级，逐步实现时空联合扩散架构

---

## 一、实验结果汇总

### 1.1 指标对比

| Phase | 架构 | MSE | MAE | RMSE | 训练时间/epoch |
|-------|------|-----|------|------|----------------|
| P1 | Identity（骨架） | 0.889 | 0.687 | 0.943 | ~4s |
| P2 | TemporalAttn（时序） | **0.770** | **0.619** | **0.878** | ~11s |
| **P3** | **ST-Joint（时空联合）** | **0.771** | **0.620** | **0.878** | ~15s |

### 1.2 消融分析

| 阶段 | vs 基线 (P1) | 提升 |
|------|-------------|------|
| P2 时序分支 | MSE 0.889 → 0.770 | ↓13.4% |
| P3 时空联合 | MSE 0.770 → 0.771 | 基本持平 |

**分析**：
- 时序分支带来显著提升（RotaryAttn 无掩码注意力）
- 空间分支在当前配置下增益有限，可能原因：
  1. 图结构信息未被充分利用
  2. 门控机制尚未完全发挥（当前 P3 已整合门控）
  3. 需要更多 epoch 或调参

---

## 二、目录结构

```
models/
├── fourier/              # 原文件（保留不动）
│   ├── unet_bone.py     # FormerBone 原始实现
│   ├── model.py         # 原始 Model 类
│   └── ...
│
├── stformer_bone/       # 新建（已实现）
│   ├── FormerBone.py    # STFormerBone 主干（Phase 1-3）
│   ├── Model.py         # Model 类
│   ├── diffusion.py     # 复制自 fourier
│   ├── temporal.py      # Phase 2: 时序分支 ✅
│   ├── spatial.py       # Phase 3: 空间分支 ✅
│   └── gate.py          # Phase 3: 门控机制 ✅
│
└── doc/
    └── STFormerBone实施指导.md
```

---

## 三、Phase 1: 骨架版本 ✅

### 结果
- MSE: 0.889, MAE: 0.687
- 中间层用 `nn.Identity()` 透传

### 关键设计
- 时间注入：`h = h + time_embed`（加法广播）
- Patch Embedding：unfold + Linear
- 输出投影：`nn.Linear(P_x * d_model, pred_len)`

---

## 四、Phase 2: 时序分支 ✅

### 结果
- MSE: **0.770** (↓13.4%), MAE: **0.619** (↓9.9%)
- 训练时间增加 ~3x（11s vs 4s）

### TemporalAttention 关键特性
- Pre-LN 结构
- RotaryEmbedding（无因果掩码）
- 全注意力机制（纯扩散）
- FFN 残差连接

### 代码位置
`stformer_bone/temporal.py`

---

## 五、Phase 3: 时空联合 ✅

### 结果
- MSE: 0.771, MAE: 0.620
- 训练时间 ~15s/epoch

### STJointLayer 架构
```
H^{(l+1)} = H^{(l)} + gate ⊗ H_T + (1-gate) ⊗ H_S

其中：
- H_T = TemporalAttention(H^{(l)})     # 时序分支
- H_S = SpatialPropagation(H^{(l)})    # 空间分支
- gate = σ(W_g · Concat[H_T, H_S])    # 向量级门控
```

### SpatialPropagation 关键特性
- D @ Q 操作（LWR 物理先验）
- D = I - S^T（拓扑后向差分算子）
- 可学习参数 `spatial_alpha` 控制传播强度

### 代码位置
- `stformer_bone/gate.py` - STJointLayer
- `stformer_bone/spatial.py` - SpatialPropagation

---

## 六、后续优化方向

### 6.1 图结构增强
- 检查 D 矩阵是否正确加载
- 尝试不同的图结构（上游/下游/双向）

### 6.2 门控机制优化
- 调整门控网络结构
- 尝试不同的融合方式

### 6.3 层数消融
- 测试 st_layers = 5, 7 的效果
- 当前使用 st_layers = 3

### 6.4 对比 FormerBone
- 在相同配置下对比原 fourier 架构
- 验证 STFormerBone 优势

---

## 七、验收清单

| 阶段 | 验收项 | 结果 |
|------|--------|------|
| P1 | 骨架运行，能跑通一个 batch | ✅ |
| P1 | 维度正确，输出 (B*N, pred_len) | ✅ |
| P2 | 时序收敛，MAE 下降 | ✅ MSE↓13.4% |
| P3 | 时空联合，MAE ≤ P2 | ⚠️ 基本持平 |
| P4 | 最终对比，MAE ≤ FormerBone | 待测试 |

---

## 八、配置项汇总

```yaml
# stformer_bone 专属
st_layers: 3          # 可调 3/5/7
use_stformer: true     # 切换模型

# graph 配置
graph:
  enabled: true
  adj_path: /root/yanyijin/STdiff/models/data/adjacent_gantry.csv
  num_nodes: 30
```

---

## 九、测试命令

```bash
# Phase 2 测试（时序分支）
cd /root/yanyijin/STdiff
conda activate holiday
python models/train_from_yaml.py --config models/config/lwrdiff_96_12.yaml
```

---

## 十、下一步

1. **消融实验**：测试 `D = I`（纯时序）vs `D = 真实`
2. **层数消融**：st_layers = 3 vs 5 vs 7
3. **对比原版**：使用 fourier 架构训练相同 epoch
4. **调参优化**：调整 learning_rate, dropout 等