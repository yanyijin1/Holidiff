# HoliDiff Phase D 精简搜索计划（仅参数可跑版）

> 目标：当前阶段只为“找更优方案”，不追求完整消融证据链。  
> 范围：只使用**现有代码 + 命令行参数**可直接跑的实验。  
> 统一口径：`train_epochs=20`，固定其余基础配置，单次只改一个变量。

---

## 0. 固定基线（D-core）

以下参数默认固定，作为所有实验对照：

- `patch_embed_mode=trend_aware_concat`
- `physical_injection_enable=true`
- `physical_residual_enable=true`
- `physical_residual_eta=0.5`
- `frequency_enable=true`
- `frequency_decomp_type=fixed_fft`
- `frequency_injection_mode=embed_replace`
- `frequency_patch_embed_mode=band_trend`
- `frequency_patch_embed_fusion=concat_proj`
- `frequency_residual_type=hybrid_residual`
- `frequency_hybrid_mix_alpha=0.5`
- `frequency_conditioner_mode=identity`
- `train_epochs=20`

基线命名建议：`D20_base_k3_p16`

---

## 1. 第一层（只保留你最关心的）：频带数 K

> 目的：先锁定频带粒度，后续全部搜索都基于最优 K。

### 1.1 搜索集合

- K=2
- K=3（当前）
- K=4

### 1.2 可直接跑命令

> 注意：`frequency_num_bands` 和 `frequency_patch_embed_num_bands` 必须同步。

```bash
# K=2
mkdir -p /root/yanyijin/STdiff/Holidiff/logs && \
source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && \
nohup python /root/yanyijin/STdiff/Holidiff/train.py \
  --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml \
  --gpu 0 --train_epochs 20 \
  --patch_embed_mode trend_aware_concat \
  --physical_injection_enable true --physical_residual_enable true --physical_residual_eta 0.5 \
  --frequency_enable true --frequency_decomp_type fixed_fft \
  --frequency_injection_mode embed_replace \
  --frequency_patch_embed_mode band_trend --frequency_patch_embed_fusion concat_proj \
  --frequency_num_bands 2 --frequency_patch_embed_num_bands 2 \
  --frequency_residual_type hybrid_residual --frequency_hybrid_mix_alpha 0.5 \
  --frequency_conditioner_mode identity \
  --version D20_k2 \
  > /root/yanyijin/STdiff/Holidiff/logs/D20_k2.log 2>&1 &

# K=3
mkdir -p /root/yanyijin/STdiff/Holidiff/logs && \
source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && \
nohup python /root/yanyijin/STdiff/Holidiff/train.py \
  --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml \
  --gpu 0 --train_epochs 20 \
  --patch_embed_mode trend_aware_concat \
  --physical_injection_enable true --physical_residual_enable true --physical_residual_eta 0.5 \
  --frequency_enable true --frequency_decomp_type fixed_fft \
  --frequency_injection_mode embed_replace \
  --frequency_patch_embed_mode band_trend --frequency_patch_embed_fusion concat_proj \
  --frequency_num_bands 3 --frequency_patch_embed_num_bands 3 \
  --frequency_residual_type hybrid_residual --frequency_hybrid_mix_alpha 0.5 \
  --frequency_conditioner_mode identity \
  --version D20_k3 \
  > /root/yanyijin/STdiff/Holidiff/logs/D20_k3.log 2>&1 &

# K=4
mkdir -p /root/yanyijin/STdiff/Holidiff/logs && \
source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && \
nohup python /root/yanyijin/STdiff/Holidiff/train.py \
  --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml \
  --gpu 0 --train_epochs 20 \
  --patch_embed_mode trend_aware_concat \
  --physical_injection_enable true --physical_residual_enable true --physical_residual_eta 0.5 \
  --frequency_enable true --frequency_decomp_type fixed_fft \
  --frequency_injection_mode embed_replace \
  --frequency_patch_embed_mode band_trend --frequency_patch_embed_fusion concat_proj \
  --frequency_num_bands 4 --frequency_patch_embed_num_bands 4 \
  --frequency_residual_type hybrid_residual --frequency_hybrid_mix_alpha 0.5 \
  --frequency_conditioner_mode identity \
  --version D20_k4 \
  > /root/yanyijin/STdiff/Holidiff/logs/D20_k4.log 2>&1 &
```

---

## 2. 第二层（重点）：当前“仅参数可跑”的子集

> 说明：以下是**不改代码就能做**的第二层搜索。  
> 像 `eta_init/beta_init` 的“列表网格”在当前 argparse 机制下不适合直接 CLI 覆盖，建议放到后续“改代码阶段”处理。

### 2.1 beta 固定 vs 可学习

- `frequency_use_learnable_beta=false`（基线）
- `frequency_use_learnable_beta=true`

```bash
# beta 可学习
mkdir -p /root/yanyijin/STdiff/Holidiff/logs && \
source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && \
nohup python /root/yanyijin/STdiff/Holidiff/train.py \
  --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml \
  --gpu 0 --train_epochs 20 \
  --patch_embed_mode trend_aware_concat \
  --physical_injection_enable true --physical_residual_enable true --physical_residual_eta 0.5 \
  --frequency_enable true --frequency_decomp_type fixed_fft \
  --frequency_injection_mode embed_replace \
  --frequency_patch_embed_mode band_trend --frequency_patch_embed_fusion concat_proj \
  --frequency_num_bands 3 --frequency_patch_embed_num_bands 3 \
  --frequency_residual_type hybrid_residual --frequency_hybrid_mix_alpha 0.5 \
  --frequency_use_learnable_beta true \
  --frequency_conditioner_mode identity \
  --version D20_betaLearn \
  > /root/yanyijin/STdiff/Holidiff/logs/D20_betaLearn.log 2>&1 &
```

### 2.2 patch_size（当前以 `patch_len` 代理）

- `patch_len=12`
- `patch_len=16`（当前）
- `patch_len=24`

```bash
# patch_len=12
mkdir -p /root/yanyijin/STdiff/Holidiff/logs && \
source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && \
nohup python /root/yanyijin/STdiff/Holidiff/train.py \
  --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml \
  --gpu 0 --train_epochs 20 \
  --patch_len 12 \
  --patch_embed_mode trend_aware_concat \
  --physical_injection_enable true --physical_residual_enable true --physical_residual_eta 0.5 \
  --frequency_enable true --frequency_decomp_type fixed_fft \
  --frequency_injection_mode embed_replace \
  --frequency_patch_embed_mode band_trend --frequency_patch_embed_fusion concat_proj \
  --frequency_num_bands 3 --frequency_patch_embed_num_bands 3 \
  --frequency_residual_type hybrid_residual --frequency_hybrid_mix_alpha 0.5 \
  --frequency_conditioner_mode identity \
  --version D20_patch12 \
  > /root/yanyijin/STdiff/Holidiff/logs/D20_patch12.log 2>&1 &

# patch_len=24
mkdir -p /root/yanyijin/STdiff/Holidiff/logs && \
source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && \
nohup python /root/yanyijin/STdiff/Holidiff/train.py \
  --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml \
  --gpu 0 --train_epochs 20 \
  --patch_len 24 \
  --patch_embed_mode trend_aware_concat \
  --physical_injection_enable true --physical_residual_enable true --physical_residual_eta 0.5 \
  --frequency_enable true --frequency_decomp_type fixed_fft \
  --frequency_injection_mode embed_replace \
  --frequency_patch_embed_mode band_trend --frequency_patch_embed_fusion concat_proj \
  --frequency_num_bands 3 --frequency_patch_embed_num_bands 3 \
  --frequency_residual_type hybrid_residual --frequency_hybrid_mix_alpha 0.5 \
  --frequency_conditioner_mode identity \
  --version D20_patch24 \
  > /root/yanyijin/STdiff/Holidiff/logs/D20_patch24.log 2>&1 &
```

---

## 3. 第三层（重点）：当前“仅参数可跑”的子集

> 说明：真正的 3.1/3.2/3.3 结构改造还需要后续改代码。  
> 但现在可以先做“近似上限探索”：频域分解器切换。

### 3.1 fixed_fft vs learnable_fft

```bash
# learnable_fft
mkdir -p /root/yanyijin/STdiff/Holidiff/logs && \
source /root/miniconda3/etc/profile.d/conda.sh && conda activate holiday && \
nohup python /root/yanyijin/STdiff/Holidiff/train.py \
  --config /root/yanyijin/STdiff/Holidiff/configs/compare_fujian30_standard_1epoch.yaml \
  --gpu 0 --train_epochs 20 \
  --patch_embed_mode trend_aware_concat \
  --physical_injection_enable true --physical_residual_enable true --physical_residual_eta 0.5 \
  --frequency_enable true --frequency_decomp_type learnable_fft \
  --frequency_injection_mode embed_replace \
  --frequency_patch_embed_mode band_trend --frequency_patch_embed_fusion concat_proj \
  --frequency_num_bands 3 --frequency_patch_embed_num_bands 3 \
  --frequency_residual_type hybrid_residual --frequency_hybrid_mix_alpha 0.5 \
  --frequency_conditioner_mode identity \
  --version D20_learnableFFT \
  > /root/yanyijin/STdiff/Holidiff/logs/D20_learnableFFT.log 2>&1 &
```

---

## 4. 推荐最小执行集（先跑这些）

按你当前目标（快速找更优）：

1. `D20_k2 / D20_k3 / D20_k4`
2. 在最优 K 上跑：`D20_betaLearn`
3. 在最优 (K,beta) 上跑：`D20_patch12 / D20_patch24`
4. 最后跑：`D20_learnableFFT`

---

## 5. 下一阶段（需要改代码后再做）

以下搜索建议放到“改代码阶段”再做：

- `eta_init` 网格（列表参数）
- `beta_init` 多组固定列表
- `d_k` 分配策略（均分/低频加权/高频加权）
- 第三层结构增强：3.1/3.2/3.3

这部分我会在你这轮参数搜索跑完后再接着改代码与测试。
