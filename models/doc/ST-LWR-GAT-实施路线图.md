# ST-LWR-GAT 实施路线图

> 三段式实施，避免 debug 地狱

---

## 实施路线图

```
Phase 1: 核心替换（本周）     → 训练能跑起来
Phase 2: 显式分解 + 时间层（下周）→ 可解释性
Phase 3: 打磨与消融（后续）   → 审稿人质疑
```

---

## Phase 1: 核心替换

### 目标
让训练能跑起来，验证 PCGK 物理约束有效。

### 改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `stlwr_gat.py` | 新建 | STLWRGATLayer + PCGK |
| `FormerBone.py` | 修改 | STJointLayer → STLWRGATLayer |
| `gate.py` | 保留 | 暂不删除（兼容） |
| `temporal.py` | 保留 | 暂不删除（Phase 2 用） |
| `spatial.py` | 保留 | 暂不删除（参考） |

### Phase 1 刻意不做的事

- ❌ 不引入 pred_lwr + pred_res 显式分解输出
- ❌ 不纠结 lambda_mix 具体值（固定 0.8）
- ❌ 不引入时间层（纯空间 GAT）

### 通过标准

1. 训练 loss 下降
2. `v_f` 收敛到 40~120 km/h
3. `k_j` 收敛到 80~200 veh/km
4. `regime` 分布随时间变化（早高峰接近 0，平峰接近 1）

---

## Phase 2: 显式分解 + 时间层

### 前提
Phase 1 通过。

### 改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `stlwr_gat.py` | 修改 | 加入 PhysicsDecoupledOutput |
| `FormerBone.py` | 修改 | 加入交替时间层 |

### 通过标准

1. `pred_lwr` 呈现平滑的 LWR 传播趋势
2. `pred_res` 是微观修正

---

## Phase 3: 打磨与消融

### 改动清单

| 实验 | 说明 |
|------|------|
| 路段异质性编码 | 高速/匝道/隧道类型嵌入 |
| 多尺度周期编码 | 与频率解耦结合 |
| 消融实验 | 固定邻接 vs PCGK vs 纯自适应 |

---

## 当前状态

### ✅ 已确认

1. **数据来源**：`adjacent_gantry.csv` 包含 `src_FID`, `nbr_FID` 邻接关系
2. **A_phys_down**: 从 CSV 构建（src → nbr 是下游）
3. **A_phys_up**: A_phys_down 的转置

### 🔧 Phase 1 实现要点

1. **PCGK 简化版**：不依赖 edge_attr，只用邻接矩阵
2. **自适应邻接备选**：如果物理拓扑数据有问题，可以用纯自适应启动（lambda_mix = 0）
3. **监控指标**：每 epoch 打印 v_f, k_j, regime 分布

---

## 实施日志

### 2026-04-21: Phase 1 代码完成

- [x] 阅读现有代码结构
- [x] 创建 `LWRGAT/stlwr_gat.py` (PCGK + STLWRGATLayer)
- [x] 修改 `LWRGAT/FormerBone.py` (STJointLayer → STLWRGATLayer)
- [x] 修改 `trainer.py` (添加 use_lwrgat 选项)
- [x] 修改 `train_from_yaml.py` (解析 use_lwrgat)
- [x] 创建 `config/lwrdiff_phase1_pcgk.yaml`
- [ ] 验证训练能跑
- [ ] 监控 v_f, k_j, regime

### 运行命令

```bash
# 切换到 models 目录
cd /root/yanyijin/STdiff/models

# 训练 Phase 1 (LWRGAT)
python train_from_yaml.py --config config/lwrdiff_phase1_pcgk.yaml

# 对比训练 stformer (原版)
python train_from_yaml.py --config config/lwrdiff_96_12_v2.yaml
```

### 切换方式

在 YAML 中修改 `model_switch.use_lwrgat`:
- `use_lwrgat: true` → 使用 PCGK-GAT
- `use_stformer: true` → 使用原 STFormerBone
