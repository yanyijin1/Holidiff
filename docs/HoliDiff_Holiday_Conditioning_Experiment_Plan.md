# HoliDiff Holiday Conditioning 实验计划模板

> 目的：为 `docs/HoliDiff_Three_Stage_Scheme_v2.md` 提供一份可直接执行、可持续填写结果的实验计划文档。  
> 原则：所有实验共享同一 backbone、同一数据集、同一评估指标；只切换 holiday conditioning 相关变量。  
> 标签约束：**所有 holiday 条件一律直接使用数据集中的 `is_holiday` 标签，不根据时间戳或外部节日表自行推断。**

---

## 1. 当前固定基线

当前 holiday conditioning 实验统一基于以下固定配置：

| 组件 | 固定取值 |
|---|---|
| Backbone | HATEK / HoliDiff Phase-D + SFCN |
| Patch Encoder | Trend-Aware concat |
| Frequency Decomposer | fixed_fft |
| Num Bands | 4 |
| Patch Length | 12 |
| Frequency Injection | embed_replace |
| Frequency Patch Embed | band_trend + concat_proj |
| Residual Type | hybrid_residual |
| Physical Residual Eta | 0.5 |
| Spatial Field Target Adapter | matrix_ni |
| Spatial Coupling | raw = 0.05, effective = softplus(0.05) |
| Spatial Edge Variance Window | 720 |
| Spatial Field Stats Source | future |
| Spatial Adjacency | topology |
| Local Scaling Adapter | vanilla_revin |
| Epoch | 20 |
| Seed | 2021 |

---

## 2. 总体实验路径

| Stage | 名称 | 核心问题 | 新增参数 | 进入条件 | 退出标准 |
|---|---|---|---:|---|---|
| Stage 0 | E4 Baseline | 当前 SFCN 定版是否稳定 | 0 | 已完成 | 作为所有对比基线 |
| Stage 1 | HAC | 节假日是否需要更强空间耦合 | 1 | baseline 可复现 | `Hol_Cong_MAE` 改善 |
| Stage 2 | HAC + CD | holiday 耦合是否会过拟合 | 0 | Stage 1 有正向收益 | `Hol_MAE` 进一步改善 |
| Stage 3 | HAC + DRFB / RSFB | 节假日是否需要独立场统计库 | 0 | Stage 1/2 证明 holiday 信号存在 | holiday 子集显著改善 |

---

## 3. 实验前检查清单

| 检查项 | 说明 | 状态 |
|---|---|---|
| 数据集 `is_holiday` 可读 | 确认 dataset 输出或可从样本窗口读取 | [ ] |
| Baseline 可复现 | 复现实验 `exp5_final_future_softplus` 或等价版本 | [ ] |
| 日志目录存在 | `Holidiff/logs/holiday_conditioning/` | [ ] |
| 结果目录存在 | `test_results/.../diagnostics` 正常写出 | [ ] |
| 指标解析脚本可用 | 能读取 `mae / Hol_Cong / SGFE` | [ ] |

---

## 4. Stage 0：Baseline 锁定

### 4.1 目的
- 锁定 holiday conditioning 前的最优 SFCN 版本；
- 确认后续所有对比共享同一底座。

### 4.2 运行命令

```bash
python /root/yanyijin/STdiff/Holidiff/train.py \
  --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml \
  --train_epochs 20 \
  --phase_e_enable true \
  --phase_e_target_adapter matrix_ni \
  --phase_e_revin_adapter vanilla_revin \
  --phase_e_zeta_init 0.05 \
  --phase_e_edge_var_window 720 \
  --phase_e_field_stats_source future \
  --version exp5_final_future_softplus
```

### 4.3 结果记录

| Exp ID | Log Path | Best Vali | Test MAE | Test RMSE | Cong MAE | Hol_Cong MAE | SGFE | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| E4-Baseline |  |  |  |  |  |  |  |  |

---

## 5. Stage 1：Holiday-Adaptive Coupling (HAC)

### 5.1 目标
验证：**节假日样本是否需要更强的空间场耦合。**

### 5.2 实验步骤

| Step | 操作 | 代码改动 | 输出 |
|---|---|---|---|
| 1 | 从数据集读取 `is_holiday` | dataset / dataloader | batch 级 holiday 标记 |
| 2 | 生成样本级 `h_b` | 预测窗口内任意 `is_holiday=1` 即置 1 | `(B,)` holiday tensor |
| 3 | 实现 `alpha(h_b)` | `alpha_base + delta_alpha * h_b` | holiday-aware coupling |
| 4 | 加入训练/推理 | 仅调节 coupling 强度 | 新版 HAC 模型 |
| 5 | 跑硬开关与可学习版 | hard / learn 两组 | 对比表 |

### 5.3 实验矩阵

| Exp ID | 配置 | 新增参数 | 目的 | 状态 |
|---|---|---:|---|---|
| HAC-hard | `alpha(0)=0.05, alpha(1)=0.10` | 0 | 验证方向是否正确 | [ ] |
| HAC-learn | `alpha(h)=0.05 + delta_alpha * h` | 1 | 验证可学习 holiday 耦合 | [ ] |
| HAC-ablation | `delta_alpha=0` | 0 | 验证退化到 baseline | [ ] |

### 5.4 命令预留

```bash
# HAC-hard 1epoch smoke test
nohup bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && CUDA_VISIBLE_DEVICES=0 python /root/yanyijin/STdiff/Holidiff/train.py --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml --train_epochs 1 --holiday_enable true --holiday_mode hard --holiday_hard_alpha 0.10 --holiday_dual_bank_enable false --version hc_stage1_hard_smoke' > "/root/yanyijin/STdiff/Holidiff/logs/holiday_conditioning/hc_stage1_hard_smoke.log" 2>&1 &
```

```bash
# HAC-learn 1epoch smoke test
nohup bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && CUDA_VISIBLE_DEVICES=1 python /root/yanyijin/STdiff/Holidiff/train.py --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml --train_epochs 1 --holiday_enable true --holiday_mode learned_delta --holiday_alpha_delta 0.05 --holiday_dual_bank_enable false --version hc_stage1_learn_smoke' > "/root/yanyijin/STdiff/Holidiff/logs/holiday_conditioning/hc_stage1_learn_smoke.log" 2>&1 &
```

```bash
# HAC-hard full run
nohup bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && CUDA_VISIBLE_DEVICES=0 python /root/yanyijin/STdiff/Holidiff/train.py --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml --train_epochs 20 --holiday_enable true --holiday_mode hard --holiday_hard_alpha 0.10 --holiday_dual_bank_enable false --version hc_stage1_hard_full' > "/root/yanyijin/STdiff/Holidiff/logs/holiday_conditioning/hc_stage1_hard_full.log" 2>&1 &
```

```bash
# HAC-learn full run
nohup bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && CUDA_VISIBLE_DEVICES=1 python /root/yanyijin/STdiff/Holidiff/train.py --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml --train_epochs 20 --holiday_enable true --holiday_mode learned_delta --holiday_alpha_delta 0.05 --holiday_dual_bank_enable false --version hc_stage1_learn_full' > "/root/yanyijin/STdiff/Holidiff/logs/holiday_conditioning/hc_stage1_learn_full.log" 2>&1 &
```

### 5.5 结果表

| Exp ID | Best Vali | Test MAE | Test RMSE | Reg MAE | Hol MAE | Cong MAE | Hol_Cong MAE | SGFE | Learned delta_alpha | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HAC-hard |  |  |  |  |  |  |  |  | - |  |
| HAC-learn |  |  |  |  |  |  |  |  |  |  |
| HAC-ablation |  |  |  |  |  |  |  |  | 0 |  |

### 5.6 阶段结论

| 判断项 | 结果 |
|---|---|
| `delta_alpha > 0` 是否成立 |  |
| `Hol_Cong_MAE` 是否优于 baseline |  |
| 是否进入 Stage 2 |  |
| 备注 |  |

---

## 6. Stage 2：HAC + Conditional Dropout (CD)

### 6.1 目标
验证：**holiday coupling 是否会过拟合 holiday 小样本。**

### 6.2 实验步骤

| Step | 操作 | 代码改动 | 输出 |
|---|---|---|---|
| 1 | 在训练阶段对 holiday 标签做 dropout | 仅作用于 `alpha(h)` | `h_train` |
| 2 | 设定 `p_drop` | 先试 `0.1`，必要时 `0.05` | 两组对比 |
| 3 | 保持统计库不 dropout | 只 dropout holiday gate | 稳定训练 |
| 4 | 对比 Stage 1 最优 | holiday / regular 双子集指标 | 是否更稳 |

### 6.3 实验矩阵

| Exp ID | 配置 | 新增参数 | 目的 | 状态 |
|---|---|---:|---|---|
| HAC-CD-01 | `p_drop=0.1` | 0 | 主实验 | [ ] |
| HAC-CD-005 | `p_drop=0.05` | 0 | 若 0.1 太强则回退 | [ ] |

### 6.4 命令预留

```bash
# HAC-CD-01 1epoch smoke test
nohup bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && CUDA_VISIBLE_DEVICES=2 python /root/yanyijin/STdiff/Holidiff/train.py --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml --train_epochs 1 --holiday_enable true --holiday_mode learned_delta --holiday_alpha_delta 0.05 --holiday_dropout_prob 0.10 --holiday_dual_bank_enable false --version hc_stage2_cd01_smoke' > "/root/yanyijin/STdiff/Holidiff/logs/holiday_conditioning/hc_stage2_cd01_smoke.log" 2>&1 &
```

```bash
# HAC-CD-005 1epoch smoke test
nohup bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && CUDA_VISIBLE_DEVICES=3 python /root/yanyijin/STdiff/Holidiff/train.py --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml --train_epochs 1 --holiday_enable true --holiday_mode learned_delta --holiday_alpha_delta 0.05 --holiday_dropout_prob 0.05 --holiday_dual_bank_enable false --version hc_stage2_cd005_smoke' > "/root/yanyijin/STdiff/Holidiff/logs/holiday_conditioning/hc_stage2_cd005_smoke.log" 2>&1 &
```

```bash
# HAC-CD-01 full run
nohup bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && CUDA_VISIBLE_DEVICES=2 python /root/yanyijin/STdiff/Holidiff/train.py --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml --train_epochs 20 --holiday_enable true --holiday_mode learned_delta --holiday_alpha_delta 0.05 --holiday_dropout_prob 0.10 --holiday_dual_bank_enable false --version hc_stage2_cd01_full' > "/root/yanyijin/STdiff/Holidiff/logs/holiday_conditioning/hc_stage2_cd01_full.log" 2>&1 &
```

```bash
# HAC-CD-005 full run
nohup bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && CUDA_VISIBLE_DEVICES=3 python /root/yanyijin/STdiff/Holidiff/train.py --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml --train_epochs 20 --holiday_enable true --holiday_mode learned_delta --holiday_alpha_delta 0.05 --holiday_dropout_prob 0.05 --holiday_dual_bank_enable false --version hc_stage2_cd005_full' > "/root/yanyijin/STdiff/Holidiff/logs/holiday_conditioning/hc_stage2_cd005_full.log" 2>&1 &
```

### 6.5 结果表

| Exp ID | Best Vali | Test MAE | Reg MAE | Hol MAE | Cong MAE | Hol_Cong MAE | SGFE | Learned delta_alpha | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HAC-CD-01 |  |  |  |  |  |  |  |  |  |
| HAC-CD-005 |  |  |  |  |  |  |  |  |  |

### 6.6 阶段结论

| 判断项 | 结果 |
|---|---|
| dropout 是否改善 Hol 指标 |  |
| Reg 指标是否稳定 |  |
| 是否进入 Stage 3 |  |
| 备注 |  |

---

## 7. Stage 3：DRFB / RSFB（Dual-Regime / Regime-Specific Field Bank）

### 7.1 目标
验证：**holiday 不仅改变耦合强度，还改变场统计基准本身。**

### 7.2 实验步骤

| Step | 操作 | 代码改动 | 输出 |
|---|---|---|---|
| 1 | 按数据集 `is_holiday` 拆训练子集 | 统计预处理 | reg / hol 两套 field bank |
| 2 | 预计算双库 | `F_mu_bank`, `F_sigma_bank` | bank 文件/内存缓存 |
| 3 | 训练时按 `h_b` 检索 | hard selector | regime-specific stats |
| 4 | 推理时按 `h_b` 检索 | history decode 同样检索 | holiday-aware decode |
| 5 | 对比 Stage 2 最优 | 看 holiday 子集是否继续提升 | 定版依据 |

### 7.3 实验矩阵

| Exp ID | 配置 | 新增参数 | 目的 | 状态 |
|---|---|---:|---|---|
| DRFB-only | 双库 + 固定 coupling | 0 | 验证双库本身作用 | [ ] |
| HAC-DRFB | HAC + 双库 | 1 | 验证 holiday gate 与双库叠加 | [ ] |
| HAC-CD-DRFB | HAC + CD + 双库 | 1 | 完整版 | [ ] |

### 7.4 命令预留

```bash
# DRFB-only 1epoch smoke test
nohup bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && CUDA_VISIBLE_DEVICES=0 python /root/yanyijin/STdiff/Holidiff/train.py --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml --train_epochs 1 --holiday_enable true --holiday_mode none --holiday_dual_bank_enable true --version hc_stage3_drfb_only_smoke' > "/root/yanyijin/STdiff/Holidiff/logs/holiday_conditioning/hc_stage3_drfb_only_smoke.log" 2>&1 &
```

```bash
# HAC-DRFB 1epoch smoke test
nohup bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && CUDA_VISIBLE_DEVICES=1 python /root/yanyijin/STdiff/Holidiff/train.py --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml --train_epochs 1 --holiday_enable true --holiday_mode learned_delta --holiday_alpha_delta 0.05 --holiday_dual_bank_enable true --version hc_stage3_hac_drfb_smoke' > "/root/yanyijin/STdiff/Holidiff/logs/holiday_conditioning/hc_stage3_hac_drfb_smoke.log" 2>&1 &
```

```bash
# HAC-CD-DRFB 1epoch smoke test
nohup bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && CUDA_VISIBLE_DEVICES=2 python /root/yanyijin/STdiff/Holidiff/train.py --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml --train_epochs 1 --holiday_enable true --holiday_mode learned_delta --holiday_alpha_delta 0.05 --holiday_dropout_prob 0.10 --holiday_dual_bank_enable true --version hc_stage3_haccd_drfb_smoke' > "/root/yanyijin/STdiff/Holidiff/logs/holiday_conditioning/hc_stage3_haccd_drfb_smoke.log" 2>&1 &
```

```bash
# HAC-CD-DRFB full run
nohup bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && CUDA_VISIBLE_DEVICES=2 python /root/yanyijin/STdiff/Holidiff/train.py --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml --train_epochs 20 --holiday_enable true --holiday_mode learned_delta --holiday_alpha_delta 0.05 --holiday_dropout_prob 0.10 --holiday_dual_bank_enable true --version hc_stage3_haccd_drfb_full' > "/root/yanyijin/STdiff/Holidiff/logs/holiday_conditioning/hc_stage3_haccd_drfb_full.log" 2>&1 &
```

### 7.5 结果表

| Exp ID | Best Vali | Test MAE | Reg MAE | Hol MAE | Cong MAE | Hol_Cong MAE | SGFE | Learned delta_alpha | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DRFB-only |  |  |  |  |  |  |  | - |  |
| HAC-DRFB |  |  |  |  |  |  |  |  |  |
| HAC-CD-DRFB |  |  |  |  |  |  |  |  |  |

### 7.6 阶段结论

| 判断项 | 结果 |
|---|---|
| 双库是否优于单库 |  |
| holiday 子集是否显著改善 |  |
| 最终定版是否采用 DRFB |  |
| 备注 |  |

---

## 8. 日志与文件记录表

| Exp ID | Version Name | Log Path | Result Path | Diagnostic Path | 是否完成 | 备注 |
|---|---|---|---|---|---|---|
| E4-Baseline |  |  |  |  | [ ] |  |
| HAC-hard |  |  |  |  | [ ] |  |
| HAC-learn |  |  |  |  | [ ] |  |
| HAC-CD-01 |  |  |  |  | [ ] |  |
| HAC-CD-005 |  |  |  |  | [ ] |  |
| DRFB-only |  |  |  |  | [ ] |  |
| HAC-DRFB |  |  |  |  | [ ] |  |
| HAC-CD-DRFB |  |  |  |  | [ ] |  |

---

## 9. 阶段性结论总表

| 阶段 | 最优实验 | 是否通过 | 核心收益 | 核心风险 | 下一步 |
|---|---|---|---|---|---|
| Stage 0 |  |  |  |  |  |
| Stage 1 |  |  |  |  |  |
| Stage 2 |  |  |  |  |  |
| Stage 3 |  |  |  |  |  |

---

## 10. 最终定版记录

| 项目 | 最终选择 | 原因 |
|---|---|---|
| Holiday 标签来源 | 数据集 `is_holiday` | 避免时间规则泄漏与歧义 |
| Holiday coupling |  |  |
| Conditional Dropout |  |  |
| Dual-Regime Field Bank |  |  |
| 最终版本号 |  |  |
| 最终写入论文的方案名 |  |  |

---

## 11. 回滚策略

| 情况 | 回滚动作 |
|---|---|
| HAC 无收益 | 回到 E4-Baseline，停止 holiday 路线 |
| HAC-CD 劣化 | 回到 HAC 最优 |
| DRFB 统计不稳 | 回到 HAC-CD 或 HAC |
| holiday 标签实现异常 | 先检查 dataset `is_holiday` 流，再禁止任何时间戳推断逻辑 |

---

## 12. 待补充事项

- [x] Stage 1 真实命令
- [x] Stage 2 真实命令
- [x] Stage 3 真实命令
- [ ] 指标自动汇总脚本路径
- [ ] 日志目录统一命名规范
- [ ] 最终论文图表编号映射

---

*文档用途：实验执行模板 / 结果填写模板 / 定版决策记录表*  
*建议文件状态：持续更新，直到 holiday conditioning 定版为止*
