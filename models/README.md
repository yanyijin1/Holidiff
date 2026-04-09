# PTLD 模型架构

> **⚠️ 核心设计锁定 - 禁止修改以下内容**

以下设计内容已锁定，代码实现必须严格遵循，禁止随意修改：

| 编号 | 锁定内容 | 说明 |
|------|----------|------|
| 1 | **PTLD方案评估.md** | 完整方案设计文档 |
| 2 | **LWR Encoder 聚合方式** | q通道不变，v通道由q的空间梯度驱动 |
| 3 | **Loss 计算方式** | 预测目标为速度v，训练流程图不可更改 |
| 4 | **数据来源** | 见下方数据格式说明 |
| 5 | **预测长度** | horizon = 24 (15分钟采样 = 3小时预测) |

### 训练配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **训练/验证/测试比例** | 6:2:2 | 训练集60%，验证集20%，测试集20% |
| **随机种子** | 42 | 确保实验可复现 |

### 计算空间

| 阶段 | 计算空间 | 说明 |
|------|----------|------|
| **Loss 计算** | 归一化空间 | 在归一化空间中计算 loss 并反向传播梯度 |
| **评估指标** | 原数据尺度 | MAE、RMSE 等指标在反归一化后打印 |

**注意**：训练时模型输出和真值均经过归一化，loss 在此空间计算；推理评估时需反归一化到原始尺度再计算指标。

---

## 运行环境

### 策略七：环境配置

```bash
conda activate holiday
```

---

## 调试策略

### 策略八：效率约束

测试代码流程时，**不需要在意精度**，合理范围即可：

- **训练 loss 上限**: train loss < 10
- **单轮训练时间**: 应在 1-5 分钟内完成（diff_steps 适当调小）
- **快速迭代**: 先验证流程，再优化精度

---

### 策略九：小样本训练

验证代码流程时，先用小样本量快速迭代：

| 阶段 | 样本量 | 说明 |
|------|--------|------|
| **流程验证** | 20% 训练样本 | 快速验证数值范围、loss 收敛、模型输出 |
| **正式训练** | 100% 训练样本 | 流程验证通过后，用全量数据训练 |

**配置示例**：

```python
# run.py
SUBSAMPLE_RATIO = 0.2  # 策略九：先用 20% 数据验证流程

# 数据加载时采样
train_data = train_data[:int(len(train_data) * SUBSAMPLE_RATIO)]
```

**原则**：
- 数值范围合理 + loss 收敛 → 流程正确，可以扩展到全量
- 先快后稳：快速验证流程，稳定后再优化精度

---

### 策略一：问题定位
如果出现不合理的结果，就采用以下方式定位问题：
- 在中间层多处添加 `print` 语句
- 反复修改、运行直到找到问题位置
- 或写诊断文件到 `models/diagnosis/` 文件夹，命名为 `问题1.md`、`问题2.md`、`问题3.md`

### 策略二：参考代码优先
参考文档 `need/LWRdiff/models/LWRDiff.py` 是可以跑通的。因此如果出现训练问题：
- 优先思考照抄代码的**最小改动**原则
- 如果不知道和原代码的差别，可以在两边都插入 `print` 进行诊断对比

### 策略三：最小实现优先
- 优先做**最小实现**，先跑通全流程
- 再在已通的代码基础上**一点点变复杂**
- **合理的输出 `print` 是最好的诊断工具**

### 策略四：代码解耦

代码按功能拆分为独立模块，便于独立调试：

```
models/
├── core/                    # 🔒 LOCKED - 不可修改
│   └── lwr_solver.py        # LWR 物理骨架
│
├── fourier/                 # 可自由修改
│   ├── flow_matching.py     # 扩散调度 + MoM聚合
│   ├── denoiser_net.py     # 去噪网络
│   └── ptld_model.py        # 整合模型
│
├── data_loader.py           # 数据加载
├── trainer.py               # 训练器
└── run.py                   # 入口
```

每个模块职责单一，耦合度低。

### 策略五：测试即诊断

测试之前必须增加功能实现的 `print` 语句，作为自测手段：

```python
# 示例：在关键步骤添加 print
print(f"[Step1] Data loaded: {x.shape}")
print(f"[Step2] LWR output: {x_lwr.shape}")
print(f"[Step3] Diffusion sample: {sample.shape}")
```

**原则**：
- 每个关键步骤至少有一个 print
- print 内容包含步骤编号和关键变量形状/范围
- 测试通过后，可选择性删除或保留（调试用）
- 这是最快的问题定位方式，比写诊断文件更直接

### 策略六：即时存档

通过一部分确定的代码后，需要即时存档到 `models/versions/` 文件夹：

```bash
mkdir -p models/versions
cp models/fourier/flow_matching.py models/versions/flow_matching_v1.py
```

版本记录格式（在文件开头注释）：
```python
# Version: v0.1-flow-matching
# Date: 2026-04-07
# Description: Flow Matching 扩散核心实现
```

**重要**：每完成一个可运行的模块，立即存档，文件名带版本号，方便回退。

---

## 参考文档

| 参考项目 | 文档路径 | 核心内容 |
|----------|----------|----------|
| **LWRdiff** | `need/LWRdiff/models/LWRDiff.py` | LWR物理约束的空间推演，PatchLWROperatorLayer |
| **TSGDiff** | `need/TSGDiff/model.py` | 时间序列→图神经网络的建边方式，傅里叶谱相似度 |

---

## 数据格式

### 数据来源

| 数据文件 | 路径 | 说明 |
|----------|------|------|
| 时间序列数据 | `models/data/train_15min.csv` | 每15分钟采样，96时间步/天 |
| 路网拓扑 | `data/adjacent_gantry.csv` | 龙门架邻接关系 (src_FID, nbr_FID) |

### 张量形状

- 历史输入: `(B, seq_len, N, 2)` - [flow, speed]
- 预测目标: `(B, pred_len, N, 2)`
- 隐表示: `(B, T, N, d)` - d=32~64

---

## 目录结构

```
models/
├── core/                      # 核心模型
│   ├── lwr_solver.py          # LWR 空间推演层（隐空间）
│   ├── dgnn.py                # D-GNN 空间编码/解码
│   ├── denoiser.py             # 去噪网络
│   └── ptld.py                 # 完整 PTLD 模型
│
├── layers/                     # 网络层组件
│   ├── cross_attention.py      # Cross-Attention
│   ├── time_embedding.py        # 时间步嵌入
│   ├── transformer.py           # Transformer 编码器
│   ├── revin.py                 # RevIN 归一化
│   └── samplers/                # 采样器
│
├── utils/                      # 工具函数
│   ├── diffusion_utils.py       # 扩散工具
│   ├── losses.py                # 损失函数
│   ├── metrics.py               # 评估指标
│   └── data_loader.py           # 数据加载器
│
└── scripts/                    # 训练脚本
    └── run.py                   # 入口
```

## 模块说明

| 文件 | 职责 |
|------|------|
| `lwr_solver.py` | LWR 空间推演层，作用于隐空间 H |
| `dgnn.py` | D 算子约束的空间编码/解码 |
| `denoiser.py` | 去噪网络 |
| `ptld.py` | 整合所有组件的完整模型 |
| `data_loader.py` | 数据加载接口 |
| `exp_long_term_forecasting.py` | 训练/验证/测试流程 |

## 核心公式

### 1. 输入表示

给定历史观测，包含**流量 q** 和**速度 v** 两个通道：
$$\mathbf{x} \in \mathbb{R}^{B \times T \times N \times 2}$$

其中：
- $B$: batch 大小
- $T$: 历史时间步 (96 = 24小时 × 4)
- $N$: 节点数（龙门架数量）
- 通道 0: 流量 $q$ (veh/15min)
- 通道 1: 速度 $v$ (km/h)

### 2. 双通道隐空间嵌入

q 和 v 各自独立嵌入到隐空间，得到独立的 q 通道和 v 通道：

$$\mathbf{h}_i^{(0, q)} = \text{MLP}_q(q_{i,:}) \in \mathbb{R}^d$$
$$\mathbf{h}_i^{(0, v)} = \text{MLP}_v(v_{i,:}) \in \mathbb{R}^d$$

合并为双通道隐表示：
$$\mathbf{H}^{(0)} = [\mathbf{H}^{(0,q)}; \mathbf{H}^{(0,v)}] \in \mathbb{R}^{B \times T \times N \times 2d}$$

### 3. LWR 空间推演（核心）

**物理直觉**：交通流在路网上沿上下游传播，遵循 LWR 守恒方程 $\partial_t \rho + \partial_x q = 0$。

其离散形式为：
$$\boldsymbol{\rho}^{t+1} = \boldsymbol{\rho}^{t} + \mathbf{D} \cdot \mathbf{q}^{t}$$

#### v 与 q 的数学关系推导

交通流基本图理论给出三个核心变量的关系：

$$\text{流量} = \text{密度} \times \text{速度} \quad \Rightarrow \quad q = \rho \cdot v$$

假设存在**平衡速度关系**（Fundamental Diagram）：
$$v = V_e(\rho)$$

则流量可表示为密度的函数：
$$q = \rho \cdot V_e(\rho) = Q(\rho)$$

抽象函数关系：

$$q = F(\rho), \quad v = G(q) = V_e \circ F^{-1}(q)$$

**关键性质**：$v$ 与 $q$ 之间存在**确定的函数映射**，即 $v = \tilde{V}_e(q)$。

**在隐空间中的对应**：
$$\mathbf{H}^{(l, v)} = \mathbf{W}_{q \to v} \cdot \mathbf{H}^{(l, q)}$$

但 LWR 守恒方程要求 $\rho$ 的更新来自 q 的空间梯度。因此：

**关键洞察**：速度的更新由流量 q 通过空间差分算子 D 决定。将此物理约束提升到隐空间：

#### q 通道

q 作为物理驱动力，提供空间传播信息，**不做自更新**：

$$\mathbf{H}^{(q)} = \mathbf{H}^{(0, q)} = \text{MLP}_q(q_{i,:})$$

即 q 的隐表示直接来自观测，不经过空间推演层。

#### v 通道的更新（LWR 物理约束）

**按 LWR 方程显式写出**：

$$\mathbf{H}^{(l+1, v)} = \sigma\left( \mathbf{H}^{(l, v)} + \mathbf{D} \cdot \mathbf{H}^{(q)} \cdot \mathbf{W}_{q \to v}^{(l)} \right)$$

**物理语义**：
- $\mathbf{H}^{(l, v)}$: v 通道的当前状态
- $\mathbf{D} \cdot \mathbf{H}^{(q)}$: q 通道的上游-下游差分（空间梯度），**来自观测 q**
- $\cdot \mathbf{W}_{q \to v}$: 投影到 v 通道的更新方向
- 最终 v 的更新 = 自身 + q 的空间梯度贡献

#### 完整第 l+1 层结果

$$\mathbf{H}^{(l+1)} = \text{LayerNorm}\left( \left[ \mathbf{H}^{(q)} ; \mathbf{H}^{(l+1, v)} \right] \right)$$

注：q 通道在多层推演中保持不变，仅用于驱动 v 的更新。

### 4. D 矩阵构建

从邻接关系 `adjacent_gantry.csv` 构建：
$$D_{ij} = \begin{cases} +1 & \text{if } j \text{ is downstream of } i \\ -1 & \text{if } j \text{ is upstream of } i \\ 0 & \text{otherwise} \end{cases}$$

例如链式路网（节点 0→1→2→3）：
$$\mathbf{D} = \begin{pmatrix} 0 & 1 & 0 & 0 \\ -1 & 0 & 1 & 0 \\ 0 & -1 & 0 & 1 \\ 0 & 0 & -1 & 0 \end{pmatrix}$$

### 5. 完整前向流程

```
Input: x ∈ ℝ^{B×T×N×2}     (q, v)
  ↓
Embedding → h⁽⁰⁾ ∈ ℝ^{B×T×N×d}
  ↓
LWR Spatial Conv (×L layers):
  h⁽ˡ⁺¹⁾ = σ(D · h⁽ˡ⁾ · Wˡ)
  ↓
Decoder → ε ∈ ℝ^{B×T×N×2}   (残差/微观光流波动)
  ↓
Output: x̂ = x + ε
```

### 6. 与扩散模型结合

PTLD 的扩散过程作用于残差 $\boldsymbol{\epsilon}$：
$$p_\theta(\boldsymbol{\epsilon}_{0:T} | \mathbf{c}) = \prod_{t=0}^{T} p_\theta(\boldsymbol{\epsilon}_t | \boldsymbol{\epsilon}_{t+1}, \mathbf{c})$$

其中条件 $\mathbf{c}$ 包含：
- LWR 推演后的隐表示 $\mathbf{H}^{(L)}$
- 时间嵌入 $\mathbf{e}_{temp}$ (hour, dayofweek, holiday)

## 数据格式

### 原始数据
- `train_15min.csv`: 时间序列数据
- `adjacent_gantry.csv`: 路网拓扑 (src_FID, nbr_FID)

### 张量形状
- 历史输入: `(B, seq_len, N, 2)` - [flow, speed]
- 预测目标: `(B, pred_len, N, 2)`
- 隐表示: `(B, T, N, d)` - d=32~64
