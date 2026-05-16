# HATEK · simdiff解耦

> 当前版本名：`simdiff解耦`
>
> 本仓库当前实现已经完成从 `SimDiff` 叙事到 `HA-TEK` 叙事的命名清理。现阶段代码主入口为 `HATEK`，并保持与原始扩散预测逻辑一致的训练与推理流程。

---

## 1. 项目定位

`HATEK`（Holiday-Aware Traffic Evolution Kernel）将交通预测表述为：

- 历史宏观交通状态作为条件输入；
- 扩散过程生成未来的随机微观实现；
- 多次采样后通过稳健聚合得到最终宏观交通流预测。

当前这版 `simdiff解耦` 的目标不是重写算法本体，而是：

- 保留原 `SimDiff` 的数学骨架；
- 完成面向交通物理叙事的命名重构；
- 清理旧别名与冗余兼容层；
- 让训练入口、模型主干、微观模块、宏观采样模块职责明确。

---

## 2. 当前方法结构

`HATEK` 的运行链可以概括为：

1. **历史状态编码**  
   输入历史交通状态序列，必要时进行节点级归一化。

2. **微观实现生成**  
   通过扩散噪声调度构造某一步的未来微观实现。

3. **交通演化核估计**  
   `TEK` 使用时空 token 和注意力骨干，对未来状态进行条件估计。

4. **宏观共识提取**  
   推理阶段重复采样，并通过 Median-of-Means 得到稳健宏观预测。

---

## 3. 最终目录结构

```text
STdiff/
├── Holidiff/
│   ├── __init__.py
│   ├── HoliDiff.py
│   ├── train.py
│   ├── configs/
│   │   └── compare_fujian30_standard_1epoch.yaml
│   ├── data_provider/
│   ├── exp/
│   ├── layers/
│   ├── macro/
│   ├── micro/
│   └── utils/
├── docs/
│   ├── method_theory.md
│   ├── HA-TEK_narrative_mapping.md
│   └── SimDiff_MoM_Flow_Diagnosis_Guide.md
└── README.md
```

---

## 4. 每个关键代码文件是干什么的

### 4.1 顶层入口

#### `Holidiff/HoliDiff.py`
- 顶层模型定义文件。
- 主要类：`HATEK`。
- 负责：
  - 定义扩散噪声调度；
  - 组织训练时的微观实现生成；
  - 组织测试时的多次采样与聚合；
  - 调用 `TEK` 完成未来状态估计。

#### `Holidiff/train.py`
- 当前统一训练入口。
- 负责：
  - 读取 YAML 配置；
  - 构建任务实验类；
  - 启动训练、验证和测试流程。

#### `Holidiff/__init__.py`
- 包级导出入口。
- 当前只导出 `HATEK`，避免旧命名混入新代码路径。

---

### 4.2 微观实现模块

#### `Holidiff/micro/tek.py`
- 定义 `TEK`（Traffic Evolution Kernel）。
- 是顶层模型调用的微观估计器包装层。
- 负责：
  - 调整输入张量维度；
  - 将历史状态、当前扩散步和未来微观实现送入骨干网络；
  - 返回单次条件估计结果。

#### `Holidiff/micro/stek_backbone.py`
- 定义 `STEKBackbone` 和 `TCPAttention`。
- 是 `TEK` 的核心时空骨干。
- 负责：
  - patch 化历史与未来状态；
  - 注入时间 token；
  - 使用注意力堆叠建模时空依赖；
  - 输出未来时段预测。

#### `Holidiff/micro/__init__.py`
- 微观模块导出文件。
- 当前只导出：
  - `TEK`
  - `STEKBackbone`
  - `TCPAttention`
  - `TensorTranspose`

---

### 4.3 宏观采样与聚合模块

#### `Holidiff/macro/dpm_sampler.py`
- DPM-Solver 采样器封装。
- 负责在推理时从随机初始化状态出发，执行快速扩散反推。

#### `Holidiff/macro/dpm_solver.py`
- DPM-Solver 的底层数值实现。
- 负责具体采样步推进逻辑。

---

### 4.4 数据模块

#### `Holidiff/data_provider/data_factory.py`
- 数据入口工厂。
- 根据任务参数和数据集名称选择对应 dataset / dataloader。

#### `Holidiff/data_provider/data_loader.py`
- 通用数据集与加载逻辑。
- 负责大部分标准时序任务的数据读取与切片。

#### `Holidiff/data_provider/fujian30_loader.py`
- `fujian30` 数据的专用加载逻辑。
- 负责：
  - 读取福建 30 站点交通数据；
  - 构造训练/验证/测试切片；
  - 保持与当前实验协议一致。

---

### 4.5 实验流程模块

#### `Holidiff/exp/exp_basic.py`
- 所有实验类的基类。
- 负责：
  - 管理设备；
  - 注册模型字典；
  - 提供实验接口骨架。

#### `Holidiff/exp/exp_long_term_forecasting.py`
- 当前最主要的实验实现。
- 负责：
  - 构建 `HATEK` 模型；
  - 管理训练、验证、测试；
  - 处理输出形状；
  - 保存 checkpoint；
  - 计算 `mae / mse / rmse`。

其他 `exp_*.py` 文件：
- 对应其它任务类型的实验入口；
- 当前这次 `fujian30` 对照主要使用的是 `exp_long_term_forecasting.py`。

---

### 4.6 基础层与工具模块

#### `Holidiff/layers/RevIN.py`
- RevIN 归一化层。
- 当前用作节点级分布对齐的实现基础。

#### `Holidiff/layers/rotaryembedding.py`
- Rotary positional embedding 实现。
- 为时空 token 注意力提供相对位置建模能力。

#### `Holidiff/utils/diffusion_utils.py`
- 扩散相关工具函数。
- 例如时间步张量抽取、默认噪声辅助函数等。

#### `Holidiff/utils/metrics.py`
- 常用评价指标函数。

#### `Holidiff/utils/tools.py`
- 训练过程常用工具。
- 例如 early stopping、学习率调整、结果可视化辅助等。

#### `Holidiff/utils/print_args.py`
- 负责格式化打印实验参数。

---

### 4.7 配置文件

#### `Holidiff/configs/compare_fujian30_standard_1epoch.yaml`
- 当前最重要的验证配置。
- 用于：
  - `fujian30` 数据；
  - 1 epoch 快速回归；
  - 检查 `HATEK` 命名清理后完整运行链是否正常。

当前已切换为：

```yaml
model: HATEK
```

---

## 5. 当前训练方式

推荐在 `conda holiday` 环境中运行。

```bash
conda activate holiday
cd /root/yanyijin/STdiff
python Holidiff/train.py --config Holidiff/configs/compare_fujian30_standard_1epoch.yaml
```

该命令会完成：

- YAML 参数读取
- `Exp_Long_Term_Forecast` 构建
- `HATEK` 实例化
- 训练、验证、测试全流程执行

---

## 6. 当前版本说明

当前版本名：

```text
simdiff解耦
```

这版的核心特点是：

- `SimDiff` 旧别名已基本移除；
- 模型统一使用 `HATEK` 命名；
- 微观模块统一使用 `TEK / STEKBackbone / TCPAttention` 命名；
- 训练入口与配置已切到清理后的结构。

---

## 7. 相关文档

- `docs/method_theory.md`：方法理论说明
- `docs/HA-TEK_narrative_mapping.md`：命名映射与叙事迁移说明
- `docs/SimDiff_MoM_Flow_Diagnosis_Guide.md`：先前的诊断与分析记录

如果后续你继续演化版本，建议继续沿用“版本名 + 结构定位”的方式记录，例如：

- `simdiff解耦`
- `simdiff解耦+频率分支`
- `simdiff解耦+holiday条件`
- `simdiff解耦+macro增强`
