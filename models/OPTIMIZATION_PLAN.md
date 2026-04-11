# PTLD 模型优化方案

## 一、问题诊断

### 1.1 当前问题
- **训练 loss 下降，但 test 性能没有提升**
- **训练过程震荡，稳定性差**

### 1.2 问题根源分析

| 问题 | 根本原因 | 影响 |
|------|---------|------|
| **训练震荡** | 模型过大 + 无 EMA + 梯度不稳定 | 优化器在尖锐的loss landscape上振荡 |
| **过拟合倾向** | 模型参数量过大，数据量相对不足 | 记忆训练数据而非学习泛化特征 |
| **泛化能力差** | 缺少正则化和稳定训练技术 | 难以迁移到测试集 |

---

## 二、核心优化方案

### 方案 A：轻量化模型架构（推荐）

**目标**：减少参数量的同时保持时空耦合思想

**改动点**：
| 参数 | 当前值 (msst) | 建议值 | 效果 |
|------|--------------|--------|------|
| `d_model` | 128 | 64 | -50% 参数 |
| `d_ff` | 512 | 256 | -50% FFN |
| `num_heads` | 8 | 4 | -50% 注意力 |
| `e_layers` | 1 | 保持 | 保持层数 |

**预期效果**：
- 减少约 60% 参数量
- 降低过拟合风险
- 训练更稳定

---

### 方案 B：添加 EMA（指数移动平均）

**目标**：平滑训练过程，减少震荡

**实现方式**：
```python
# 新增 EMA 类
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
                
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
```

**配置建议**：
- `ema_decay`: 0.999（训练稳定）
- `ema_freq`: 每 10 step 更新一次

---

### 方案 C：训练稳定性优化

**改动点**：

1. **梯度裁剪**
```python
# trainer.py 中添加
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

2. **学习率预热（warmup）**
- 当前已有 warmup_epochs=5，可保留

3. **损失加权策略调整**
```python
# 当前加权方式（可能导致震荡）
weight_tmp = self.sqrt_one_minus_alphas_cumprod[t].reshape(...)

# 建议改为：固定权重或移除
weight_tmp = torch.ones_like(...)
```

4. **梯度累积（可选）**
- batch_size 减半，用梯度累积模拟大 batch
- 效果：更平滑的梯度更新

---

### 方案 D：归一化策略优化

**问题**：`forward_train` 中的归一化与 `forward_val_test` 不一致

**当前差异**：
| 阶段 | 归一化方式 |
|------|-----------|
| train | mean/std from cond_ts |
| val/test | RevIN 或 mean/std |

**建议**：统一使用 RevIN（已有实现）

---

## 三、实施计划

### 阶段 1：轻量化模型（优先级：高）
- 修改 `d_model`、`d_ff`、`num_heads`
- 预期：减少震荡，立即见效

### 阶段 2：添加 EMA（优先级：高）
- 新增 EMA 类
- 修改 trainer 支持 EMA
- 预期：大幅减少训练震荡

### 阶段 3：训练稳定性（优先级：中）
- 添加梯度裁剪
- 调整损失加权
- 预期：训练更平滑

### 阶段 4：归一化统一（优先级：低）
- 统一 train/test 归一化
- 预期：减少 train-test 不一致

---

## 四、配置文件调整建议

### `config/lwrdiff_msst.yaml`

```yaml
model_args:
  e_layers: 1          # 保持
  d_model: 64          # 128 -> 64
  num_heads: 4         # 8 -> 4
  stride: 3
  patch_len: 6
  dropout: 0.1         # 0.0 -> 0.1 (正则化)
  d_ff: 256           # 512 -> 256
  skip_dropout: 0.1    # 新增，U-Net skip dropout

optimization:
  learning_rate: 0.0005  # 0.0001 -> 0.0005 (轻量模型需要稍高学习率)
  ema_decay: 0.999      # 新增
```

---

## 五、风险评估

| 方案 | 风险 | 缓解措施 |
|------|------|---------|
| 轻量化 | 模型表达能力不足 | 逐步递减，观察 val loss |
| EMA | 增加显存和训练时间 | 使用 0.999 decay，每10步更新 |
| 梯度裁剪 | 可能过度限制学习 | 从 max_norm=1.0 开始 |

---

## 六、决策清单

请选择要实施的方案（可多选）：

- [ ] **方案 A**：轻量化模型（d_model 128→64）
- [ ] **方案 B**：添加 EMA 指数移动平均
- [ ] **方案 C**：训练稳定性（梯度裁剪+损失加权）
- [ ] **方案 D**：归一化策略统一

我会根据你的选择逐步实施。