# ST-LWR-GAT: 物理约束时空图注意力网络

> 基于 LWR 交通流守恒律的时空联合扩散预测模型

---

## 核心特性

| 特性 | 描述 |
|------|------|
| **PCGK** | Physics-Computed Graph Kernel，用速度阈值判断相态，控制信息传播方向 |
| **向量化 GAT** | 消除 Patch 循环，整体推理加速 55% |
| **显式分解** | pred = pred_lwr + pred_res，物理骨架 + 残差修正 |
| **扩散去噪** | DPM-Solver 多步采样 |

---

## 快速开始

### 训练

```bash
cd models
conda activate holiday
python run_train.py
```

### 测试

```bash
cd models
python run_test_only.py
```

---

## 项目结构

```
models/
├── LWRGAT/                    # ST-LWR-GAT 核心模块
│   ├── Model.py              # 主模型（PCGK 缓存优化）
│   ├── FormerBone.py         # 主干网络
│   └── stlwr_gat.py          # STLWRGATLayer + SpatialGAT 向量化
│
├── layers/samplers/          # 采样器
│   └── dpm_sampler.py        # DPM-Solver
│
├── config/
│   └── lwrdiff_phase1_pcgk.yaml  # 配置文件
│
├── data_loader.py            # 数据加载
├── trainer.py                # 训练器
├── run_train.py              # 训练入口
└── run_test_only.py          # 测试入口
```

---

## 物理设计

### PCGK: 物理图核计算

```python
# 相态检测：自由流 vs 拥堵流
regime = (v_obs > v_critical).float()

# 方向性加权：自由流向下游，拥堵流向上游
downstream_weight = regime * A_phys_down
upstream_weight = (1 - regime) * A_phys_up
```

### 显式分解输出

```python
# pred = pred_lwr + pred_res
pred_lwr: 物理骨架预测（平滑）
pred_res: 残差修正（捕捉振荡）
```

---

## 性能

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| PCGK 调用 | 40 次/batch | 2 次/batch |
| SpatialGAT | for 循环 | 向量化 |
| 完整测试 (132 batches) | ~100s | ~45s |

> 物理可解释性不牺牲计算效率

---

## 文档

- [ST-LWR-GAT 设计文档](./models/doc/ST-LWR-GAT-Design.md)

---

## 参考

- LWR (Lighthill-Whitham-Richards) 交通流守恒律
- DPM-Solver: 高效扩散采样
