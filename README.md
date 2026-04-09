# STdiff - 时空扩散交通预测

基于条件扩散模型与物理约束的时空交通预测框架。

## 核心思想

STdiff 将**条件扩散模型**与**时空联合建模**结合，用于交通速度预测任务。模型采用 TST-LWR（Time-Space-Time + LWR空间传播）架构，在捕捉时序依赖的同时注入交通流物理约束。

## 技术亮点

### 1. TST-LWR 时空联合架构

```
输入 → T-Attention → Cross-Attention → S-LWR → T-Attention → FFN → 输出
```

- **T-层**：时序自注意力（RoPE旋转位置编码）
- **S-层**：基于 LWR（Lighthill-Whitham-Richards）的空间传播
- **Cross-Attention**：历史条件注入

### 2. 条件扩散生成

- 使用余弦噪声调度（Cosine Schedule）
- 支持 DPMSolver 多步采样
- 物理一致性投影确保生成数据满足交通流守恒方程

### 3. 物理约束注入

通过 LWR 方程建模空间节点间的上游-下游传播关系：

```
D = I - S^T  （后向差分算子）
q = ρ · v    （交通流守恒）
```

## 项目结构

```
STdiff/
├── models/                      # 主模型代码
│   ├── fourier/
│   │   ├── ptld_model.py       # 扩散模型 + TST-LWR 主干
│   │   ├── denoiser_net.py     # 去噪网络
│   │   └── flow_matching.py    # 流匹配
│   ├── core/
│   │   └── lwr_solver.py       # LWR 物理骨架
│   ├── layers/                 # 基础层（RevIN、RoPE等）
│   ├── utils/                  # 工具函数
│   ├── data_loader.py          # 数据加载
│   ├── trainer.py              # 训练器
│   └── train_from_yaml.py      # 配置驱动的训练入口
├── baseline/                    # 基线模型对比
│   ├── SimDiff/                # SimDiff 官方实现
│   └── TSGDiff/                # TSGDiff 基线
├── paper/                       # 论文草稿
└── configs/
```

## 快速开始

### 训练

```bash
cd models
python train_from_yaml.py --config config/lwrdiff_msst.yaml --gpu 0
```

### 测试

```bash
python train_from_yaml.py --config config/lwrdiff_msst.yaml --test_only --gpu 0
```

## 配置说明

主要超参数（见 `config/lwrdiff_msst.yaml`）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `seq_len` | 输入序列长度 | 96 |
| `pred_len` | 预测序列长度 | 12 |
| `d_model` | 模型维度 | 128 |
| `e_layers` | 编码器层数 | 1 |
| `diff_steps` | 扩散步数 | 100 |
| `s_steps` | 采样步数 | 2 |

## 数据格式

输入 CSV 格式：
- 列：时间戳 + N 个节点的速度/流量/密度值
- 示例：MSST-CL 数据集（30节点，15分钟粒度）

## 应用场景

- 长期交通速度预测
- 稀疏观测场景的数据增强
- 节假日分布偏移问题
- 多视图生成式增强

## 依赖

- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas
- einops

详见 `requirements.txt`
