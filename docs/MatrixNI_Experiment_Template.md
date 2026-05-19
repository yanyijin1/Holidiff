# Matrix-NI 可拔插实验模板

> 目的：为 `MatrixNI_Scheme_v1.md` 提供一份可直接执行的实验模板。  
> 原则：所有对比共享同一 backbone、同一 loss、同一训练轮数；只切换 adapter 相关变量。

---

## 1. 当前固定基线

当前所有 Phase E 对比，统一基于以下 Phase D 最优设置：

| 组件 | 固定取值 |
|---|---|
| Backbone | SimDiff + HoliDiff Phase D |
| Patch Encoder | Trend-Aware concat |
| Frequency Decomposer | fixed_fft |
| Num Bands | 4 |
| Patch Length | 12 |
| Frequency Injection | embed_replace |
| Frequency Patch Embed | band_trend + concat_proj |
| Residual Type | hybrid_residual |
| Physical Residual Eta | 0.5 |
| Epoch | 20 |
| Seed | 2021 |

---

## 2. 可拔插实验矩阵

| Exp ID | Target Adapter | RevIN Adapter | zeta | 目的 |
|---|---|---|---|---|
| E0 | vanilla_ni | vanilla_revin | - | 复现当前基线 |
| E1 | matrix_ni | vanilla_revin | fixed=0.1 | 验证 Matrix-NI 本体 |
| E2 | matrix_ni | vanilla_revin | learnable | 验证自适应场耦合 |
| E3 | matrix_ni | field_revin | fixed=0.1 / learnable | 验证前端场化归一化 |

---

## 3. 最小配置模板

建议在配置中新增以下字段：

```yaml
phase_e_enable: false
phase_e_target_adapter: vanilla_ni
phase_e_revin_adapter: vanilla_revin
phase_e_graph_mode: directed
phase_e_zeta_mode: fixed
phase_e_zeta_init: 0.1
phase_e_field_eta: 0.1
phase_e_use_edge_sigma_fallback: false
```

### E0: 基线

```yaml
phase_e_enable: false
phase_e_target_adapter: vanilla_ni
phase_e_revin_adapter: vanilla_revin
```

### E1: Matrix-NI

```yaml
phase_e_enable: true
phase_e_target_adapter: matrix_ni
phase_e_revin_adapter: vanilla_revin
phase_e_zeta_mode: fixed
phase_e_zeta_init: 0.1
```

### E2: Matrix-NI + learnable zeta

```yaml
phase_e_enable: true
phase_e_target_adapter: matrix_ni
phase_e_revin_adapter: vanilla_revin
phase_e_zeta_mode: learnable
phase_e_zeta_init: 0.1
```

### E3: Matrix-NI + Field-RevIN

```yaml
phase_e_enable: true
phase_e_target_adapter: matrix_ni
phase_e_revin_adapter: field_revin
phase_e_zeta_mode: fixed
phase_e_zeta_init: 0.1
phase_e_field_eta: 0.1
```

---

## 4. 命令模板

> 以下命令为模板；正式落地后只需要保证新增配置字段已接入 parser。

### E0

```bash
python /root/yanyijin/STdiff/Holidiff/train.py \
  --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml \
  --train_epochs 20 \
  --version phaseE_E0_baseline
```

### E1

```bash
python /root/yanyijin/STdiff/Holidiff/train.py \
  --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml \
  --train_epochs 20 \
  --phase_e_enable true \
  --phase_e_target_adapter matrix_ni \
  --phase_e_revin_adapter vanilla_revin \
  --phase_e_zeta_mode fixed \
  --phase_e_zeta_init 0.1 \
  --version phaseE_E1_matrixni
```

### E2

```bash
python /root/yanyijin/STdiff/Holidiff/train.py \
  --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml \
  --train_epochs 20 \
  --phase_e_enable true \
  --phase_e_target_adapter matrix_ni \
  --phase_e_revin_adapter vanilla_revin \
  --phase_e_zeta_mode learnable \
  --phase_e_zeta_init 0.1 \
  --version phaseE_E2_matrixni_learnzeta
```

### E3

```bash
python /root/yanyijin/STdiff/Holidiff/train.py \
  --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml \
  --train_epochs 20 \
  --phase_e_enable true \
  --phase_e_target_adapter matrix_ni \
  --phase_e_revin_adapter field_revin \
  --phase_e_zeta_mode fixed \
  --phase_e_zeta_init 0.1 \
  --phase_e_field_eta 0.1 \
  --version phaseE_E3_matrixni_fieldrevin
```

---

## 5. 结果记录模板

| Exp ID | Best Vali | Test MAE | Test MSE | Test RMSE | Cong MAE | Hol_Cong MAE | TPR_Cong | SGFE | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E0 |  |  |  |  |  |  |  |  |  |
| E1 |  |  |  |  |  |  |  |  |  |
| E2 |  |  |  |  |  |  |  |  |  |
| E3 |  |  |  |  |  |  |  |  |  |

---

## 6. 通过/回滚标准

### E1 通过标准
- Hol_Cong MAE < 35.91
- Cong MAE < 30.53
- SGFE 优于 E0

### E2 通过标准
- 不劣于 E1 的验证损失
- 至少一项核心指标优于 E1

### E3 通过标准
- 相比 E2，Cong MAE 进一步下降
- 若训练不稳或收益不明显，则回滚到 vanilla_revin

---

## 7. 代码改造顺序

1. 先接 `TargetSpaceAdapter` 抽象
2. 再接 `VanillaNIAdapter`
3. 再接 `MatrixNIAdapter (fixed zeta)`
4. 再接 `learnable zeta`
5. 最后接 `FieldRevINAdapter`

即：**先保证可拔插，再追求效果**。
