# STFormerBone 实施指导

> 版本: v0.2
> 日期: 2026-04-11
> 目标：渐进式迁移，每步可验收、可回滚

---

## 一、实施原则

1. **每步可运行**：每个 Phase 完成后代码都能跑通
2. **每步可对比**：有明确的基线对比指标
3. **每步可回滚**：出问题时能回到上一个稳定版本
4. **逐步启用**：功能从简到繁，不跳跃

---

## 二、Phase 0: 基线记录

### 目标
确认现有 FormerBone 正常工作，记录基线指标

### 操作
1. 运行现有训练 10 epoch，记录 loss 曲线
2. 运行测试，记录 MAE/MSE 指标
3. **git commit 当前状态**：`git add . && git commit -m "baseline: FormerBone working"`

### 验收标准
- [ ] 训练 10 epoch 无 NaN/Inf
- [ ] 测试输出合理数值范围
- [ ] 指标记录在笔记中（如 MAE ≈ 1.6）

### 失败处理
若失败 → 检查数据加载、归一化代码，不进入下一阶段

---

## 三、Phase 1: 接口对齐

### 目标
创建 `stformer_bone.py`，保持与 FormerBone **完全相同的输入输出接口**

### 操作
1. 创建 `models/fourier/stformer_bone.py`
2. 实现 **最小版本**：
   - 输入/输出层复制自 `unet_bone.py`
   - 中间层用 `nn.Identity()` 占位（透传）
   - 时间注入：加法（不同于拼接，后续再改）

3. 修改 `model.py`：添加 `use_stformer` 开关
   ```python
   if getattr(configs, 'use_stformer', False):
       from .stformer_bone import PatchUVIT_STFormer
       self.nn = PatchUVIT_STFormer(configs)
   else:
       from .unet_bone import PatchUVIT
       self.nn = PatchUVIT(configs)
   ```

4. 添加配置项（yaml）：
   ```yaml
   use_stformer: false  # 默认关闭
   ```

### 验收标准
- [ ] `use_stformer: false` 时，结果与 Phase 0 **完全一致**（MSE 差异 < 0.001）
- [ ] `use_stformer: true` 时，能跑通一个 batch
- [ ] 输出维度正确：输入 `(B*N, 12)` → 输出 `(B*N, 12)`

### 失败处理
- 维度报错 → 检查 reshape/unfold 逻辑
- 结果不一致 → 检查 `use_stformer: false` 分支是否正确

---

## 四、Phase 2: 时序分支验证

### 目标
实现时序分支（RotaryAttn），空间分支暂时跳过（D=I）

### 操作
1. 在 `STJointLayer` 中实现：
   - Pre-LN: `nn.LayerNorm`
   - 时序分支：`TemporalAttention`（无因果掩码）
   - 空间分支：`h_s = 0`（跳过）
   - 门控：`gate = 0.5`（固定）

2. 配置开关：
   ```yaml
   use_stformer: true
   st_layers: 3
   ```

3. 训练 10 epoch，与 Phase 0 基线对比

### 验收标准
- [ ] 训练收敛，loss 下降曲线正常
- [ ] MAE 与基线差距 < 5%（说明时序分支正确）
- [ ] 梯度 norm 在 0.1~10 范围（不爆炸/不消失）

### 失败处理
- 收敛慢 → 检查学习率
- 梯度异常 → 检查 Pre-LN 实现

---

## 五、Phase 3: 空间分支验证

### 目标
加入真实 D 矩阵，验证空间分支有效

### 操作
1. 从 `utils/graph.py` 加载 D 矩阵
2. 在 `STJointLayer` 中实现：
   - 空间分支：`q_spatial = D @ q_proj(h)`
   - `spatial_alpha` 初始为 0.1（弱权重）

3. 消融实验：
   - 实验 A：D=I（纯时序）
   - 实验 B：D=真实（弱空间，α=0.1）
   - 实验 C：D=真实（强空间，α=1.0）

### 验收标准
- [ ] 实验 C 优于实验 A（空间分支有效）
- [ ] gate 值在 0.3~0.7 之间（不饱和）
- [ ] 门控可视化：自由流区域 gate 高，拥堵区域 gate 低

### 失败处理
- 性能下降 → 减小 `spatial_alpha` 到 0.01
- gate 饱和 → 检查 Sigmoid 初始化

---

## 六、Phase 4: 门控训练

### 目标
启用可学习门控，完成完整 STFormerBone

### 操作
1. 门控从固定 0.5 改为可训练
2. 可调参数实验：
   ```yaml
   st_layers: [3, 5, 7]  # 测试不同深度
   gate_dropout: [0.0, 0.1]  # 防止过拟合
   ```

3. A/B 测试：与 FormerBone 对比所有指标

### 验收标准
- [ ] MAE ≤ FormerBone（或差距 < 5%）
- [ ] 训练时间增加 < 50%
- [ ] 显存增加 < 30%

### 失败处理
- 性能不达标 → 回到 Phase 2 检查基础实现
- 训练不稳定 → 增加 Dropout 或减小学习率

---

## 七、代码修改清单

| 文件 | 修改内容 | 阶段 |
|------|----------|------|
| **新建** `stformer_bone.py` | STJointLayer, STFormerBone | P1 |
| `model.py` | 添加 `use_stformer` 开关 | P1 |
| `unet_bone.py` | 无需修改（保留备选） | - |

---

## 八、配置项汇总

```yaml
# 模型选择
use_stformer: false  # P1: false, P2+: true

# STFormerBone 专属
st_layers: 3          # 可调 3/5/7
d_model: 128          # 与原配置一致
num_heads: 8          # 与原配置一致
dropout: 0.1          # 与原配置一致
gate_dropout: 0.1     # P4 调参
spatial_alpha_init: 0.1  # P3 初始值
```

---

## 九、git 提交规范

| 阶段 | 提交信息 |
|------|----------|
| P0 | `git commit -m "baseline: FormerBone working"` |
| P1 | `git commit -m "feat: add stformer_bone.py interface"` |
| P2 | `git commit -m "feat: add temporal branch to STJointLayer"` |
| P3 | `git commit -m "feat: add spatial branch to STJointLayer"` |
| P4 | `git commit -m "feat: complete STFormerBone with trainable gate"` |

---

## 十、总结：每步做什么、做到什么

| 阶段 | 做什么 | 做到什么（验收） |
|------|--------|------------------|
| **P0** | 记录基线 | 训练正常、指标记录 |
| **P1** | 创建空壳 | 接口对齐、开关可控 |
| **P2** | 加时序分支 | 收敛、性能接近基线 |
| **P3** | 加空间分支 | 空间有效、性能提升 |
| **P4** | 完整训练 | 全面对比、指标达标 |
