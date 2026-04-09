#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扩散模型诊断脚本 - 检验是否为真扩散

使用方法:
    python diagnose_diffusion.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def diagnose_noise_prediction(model, batch_x, batch_y, device='cuda'):
    """检验1: 训练阶段 - 模型是否真在学"去噪"而非"记忆"
    
    成功标志: Loss收敛到≈1（预测高斯噪声的方差）
    失败标志: Loss快速降到≈0（模型记住数据集）
    """
    print("\n" + "="*60)
    print("检验1: 训练阶段 - 去噪学习 vs 记忆")
    print("="*60)
    
    model.eval()
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    
    # 模拟训练阶段的前向过程
    with torch.no_grad():
        # 获取预测的噪声和weight_tmp
        model_out, weight_tmp = model(batch_x, None, batch_y, None)
    
    # 生成一个标准高斯噪声
    noise_std = torch.randn_like(model_out)
    
    # 计算理论随机MSE（应该≈1，对于标准高斯噪声）
    # 但由于weight_tmp的存在，需要重新计算
    # model_out是预测的噪声，我们需要直接比较
    
    # 获取真实的噪声预测（从模型的nn模块）
    # 重新做一遍前向传播获取中间结果
    model.train()
    
    # 重新前向
    x = batch_y[:, -model.configs.pred_len:, :].permute(0, 2, 1)
    f_dim = -1 if model.configs.features in ['MS'] else 0
    x = x[:, f_dim:, :]
    cond_ts = batch_x
    
    if model.configs.new_norm:
        cond_ts_norm = model.revin_layer(cond_ts.clone().permute(0, 2, 1), 'norm')
        cond_ts_norm = cond_ts_norm.permute(0, 2, 1)
        mean_ = torch.mean(x[:, -x.shape[1]:, :], dim=1).unsqueeze(1)
        std_ = torch.ones_like(torch.std(x, dim=1).unsqueeze(1))
        x_norm = (x - mean_.repeat(1, x.shape[1], 1)) / (std_.repeat(1, x.shape[1], 1) + 0.00001)
    else:
        cond_ts_norm = cond_ts.permute(0, 2, 1)
        mean_ = torch.mean(cond_ts_norm, dim=-1, keepdims=True)
        std_ = torch.std(cond_ts_norm, dim=-1, keepdims=True)
        cond_ts_norm = (cond_ts_norm - mean_) / (std_ + 0.00001)
        x_norm = (x - mean_) / (std_ + 0.00001)
    
    B = x_norm.shape[0]
    N = model.configs.enc_in
    L1 = cond_ts_norm.shape[2]
    L2 = x_norm.shape[2]
    
    cond_ts_flat = cond_ts_norm.reshape(B * N, L1)
    x_flat = x_norm.reshape(B * N, L2)
    
    # 采样时间步
    t = torch.randint(0, model.num_timesteps, size=[B * N // 2]).long().to(device)
    t = torch.cat([t, model.num_timesteps - 1 - t], dim=0)
    
    # 生成噪声
    noise = torch.randn_like(x_flat)
    
    # 加噪
    alpha_bar = model.alphas_cumprod.to(device)[t].reshape(-1, 1)
    x_noisy = torch.sqrt(alpha_bar) * x_flat + torch.sqrt(1 - alpha_bar) * noise
    
    # 模型预测
    noise_pred = model.nn(x_noisy, t, cond_ts_flat, None)
    
    # 计算损失
    loss = torch.mean((noise_pred - noise) ** 2)
    
    # 计算理论随机MSE
    random_mse = torch.mean((torch.randn_like(noise) - noise) ** 2).item()
    
    print(f"\n[结果]")
    print(f"  模型噪声预测MSE: {loss.item():.4f}")
    print(f"  随机噪声MSE(理论): {random_mse:.4f}")
    print(f"  模型/随机比值: {loss.item() / random_mse:.4f}")
    
    # 判断标准
    if loss.item() < 0.3:
        print(f"  ❌ 失败: Loss << 1，模型可能在记忆或过拟合")
        return False
    elif loss.item() > 0.9:
        print(f"  ⚠️  警告: Loss ≈ 1，模型可能在瞎猜（没学到结构）")
        return False
    else:
        print(f"  ✓ 成功: Loss在0.3-0.9之间，模型学到了部分结构")
        return True


def diagnose_sample_diversity(model, batch_x, batch_y, device='cuda', num_samples=8):
    """检验2: 推理阶段 - 样本是否有"多样性"
    
    成功标志: 不同样本之间有可见方差
    失败标志: 所有样本几乎一样（确定性模型）
    """
    print("\n" + "="*60)
    print("检验2: 推理阶段 - 样本多样性")
    print("="*60)
    
    model.eval()
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    
    with torch.no_grad():
        # 多次采样
        samples = []
        for i in range(num_samples):
            torch.manual_seed(i * 42)  # 不同随机种子
            output, _ = model(batch_x, None, batch_y, None, sample_times=1)
            samples.append(output.cpu())
        
        samples = torch.stack(samples, dim=0)  # (num_samples, B, T, N)
        
        # 计算样本方差
        sample_mean = samples.mean(dim=0)
        sample_var = samples.var(dim=0)  # 每个时间步、每个变量的方差
        
        # 计算真实目标的标准差作为参考
        y_true_std = batch_y[:, -model.configs.pred_len:, :].std().item()
        
        # 平均方差
        avg_var = sample_var.mean().item()
        avg_std = np.sqrt(avg_var)
        
        # 计算样本间差异（组内差异）
        samples_np = samples.numpy()
        intra_diffs = []
        for i in range(num_samples):
            for j in range(i+1, num_samples):
                diff = np.abs(samples_np[i] - samples_np[j]).mean()
                intra_diffs.append(diff)
        avg_intra_diff = np.mean(intra_diffs)
        
        # 相对方差（相对于真实标准差）
        rel_var = avg_std / (y_true_std + 1e-8)
        
        print(f"\n[结果]")
        print(f"  样本数量: {num_samples}")
        print(f"  样本平均标准差: {avg_std:.4f}")
        print(f"  真实目标标准差: {y_true_std:.4f}")
        print(f"  相对方差(样本std/真实std): {rel_var:.4f}")
        print(f"  样本间平均差异: {avg_intra_diff:.4f}")
        
        # 判断标准
        threshold = 0.05  # 5%的真实标准差
        if rel_var < threshold:
            print(f"  ❌ 失败: 样本方差太小(rel_var={rel_var:.4f} < {threshold})，模型可能是确定性的")
            print(f"     提示: 检查是否每次采样都用了相同的随机噪声")
            return False
        else:
            print(f"  ✓ 成功: 样本有多样性")
            return True


def diagnose_mom_effectiveness(model, batch_x, batch_y, device='cuda', num_samples=8):
    """检验3: MoM是否真正生效（解耦验证）
    
    成功标志: MoM聚合结果比单样本更准确
    """
    print("\n" + "="*60)
    print("检验3: MoM效果验证")
    print("="*60)
    
    model.eval()
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    
    # 获取真实值
    f_dim = -1 if model.configs.features in ['MS'] else 0
    y_true = batch_y[:, -model.configs.pred_len:, :].permute(0, 2, 1)[:, f_dim:, :]
    
    with torch.no_grad():
        # 多次采样（不聚合）
        samples = []
        for i in range(num_samples):
            torch.manual_seed(i * 42)
            output, all_outputs = model(batch_x, None, batch_y, None, sample_times=1)
            samples.append(output.cpu())
        
        samples = torch.stack(samples, dim=0)  # (num_samples, B, T, N)
        
        # 方法1: 单样本
        pred_single = samples[0]
        
        # 方法2: 均值聚合
        pred_mean = samples.mean(dim=0)
        
        # 方法3: MoM聚合 (Median of Means)
        num_groups = 2
        group_size = num_samples // num_groups
        group_means = []
        for i in range(num_groups):
            group_mean = samples[i*group_size:(i+1)*group_size].mean(dim=0)
            group_means.append(group_mean)
        pred_mom = torch.stack(group_means).median(dim=0)[0]
        
        # 计算MAE
        mae_single = torch.abs(pred_single - y_true.cpu()).mean().item()
        mae_mean = torch.abs(pred_mean - y_true.cpu()).mean().item()
        mae_mom = torch.abs(pred_mom - y_true.cpu()).mean().item()
        
        print(f"\n[结果]")
        print(f"  单样本MAE: {mae_single:.4f}")
        print(f"  均值MAE: {mae_mean:.4f}")
        print(f"  MoM MAE: {mae_mom:.4f}")
        
        # 判断标准
        if abs(mae_mom - mae_single) < 0.01 * mae_single:
            print(f"  ❌ 失败: MoM与单样本几乎相同，样本可能没有多样性")
            return False
        else:
            if mae_mom <= mae_mean and mae_mom <= mae_single:
                print(f"  ✓ 成功: MoM效果最好")
            elif mae_mean <= mae_single:
                print(f"  ⚠️  警告: MoM效果不如均值")
            else:
                print(f"  ⚠️  警告: 单样本效果最好，可能过拟合")
            return True


def diagnose_denoising_visual(model, batch_x, batch_y, device='cuda'):
    """检验4: 单步去噪可视化（物理合理性）
    
    验证模型真的在"去噪"，而不是无视噪声输入
    """
    print("\n" + "="*60)
    print("检验4: 单步去噪物理合理性")
    print("="*60)
    
    model.eval()
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    
    # 取第一个样本
    x = batch_y[0:1, -model.configs.pred_len:, :].permute(0, 2, 1)  # (1, N, T)
    x_enc = batch_x[0:1]
    
    f_dim = -1 if model.configs.features in ['MS'] else 0
    x = x[:, f_dim:, :]
    
    # 选择中间时间步
    t_step = model.num_timesteps // 2
    t = torch.full((x.shape[1],), t_step, dtype=torch.long).to(device)
    
    # 生成噪声
    noise = torch.randn_like(x)
    
    # 加噪到中间步
    alpha_bar = model.alphas_cumprod[t_step].sqrt().to(device)
    x_noisy = alpha_bar * x + (1 - alpha_bar ** 2).sqrt() * noise
    
    # 模型预测
    model.train()  # 需要训练模式以获取正确输出
    with torch.set_grad_enabled(True):
        x_noisy.requires_grad_(True)
        
        # 条件编码
        if model.configs.new_norm:
            cond_ts = model.revin_layer(x_enc.permute(0, 2, 1), 'norm').permute(0, 2, 1)
            cond_ts = cond_ts.permute(0, 2, 1)
            mean_ = torch.mean(x, dim=-1, keepdims=True)
            std_ = torch.ones_like(torch.std(x, dim=-1, keepdims=True))
            x_norm = (x - mean_) / (std_ + 0.00001)
            cond_ts_norm = cond_ts.reshape(x.shape[1], -1)
        else:
            cond_ts = x_enc.permute(0, 2, 1)
            mean_ = torch.mean(cond_ts, dim=-1, keepdims=True)
            std_ = torch.std(cond_ts, dim=-1, keepdims=True)
            cond_ts_norm = (cond_ts - mean_) / (std_ + 0.00001)
            x_norm = (x - mean_) / (std_ + 0.00001)
        
        # 展平
        x_flat = x_norm.reshape(x.shape[1], -1)
        cond_flat = cond_ts_norm.reshape(x.shape[1], -1)
        
        # 重复t以匹配
        t_repeat = t[:x_flat.shape[0]]
        
        # 加噪
        noise_flat = torch.randn_like(x_flat)
        alpha_bar_t = model.alphas_cumprod[t_repeat].to(device).reshape(-1, 1)
        x_noisy_flat = alpha_bar_t.sqrt() * x_flat + (1 - alpha_bar_t).sqrt() * noise_flat
        
        # 预测
        noise_pred = model.nn(x_noisy_flat, t_repeat, cond_flat, None)
        
        # 去噪一步（简化的去噪公式）
        x_denoised = (x_noisy_flat - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
    
    # 计算相关性
    with torch.no_grad():
        # x_denoised 与 x_flat 的相关性应该比 x_noisy_flat 与 x_flat 更高
        x_noisy_1d = x_noisy_flat.flatten()
        x_flat_1d = x_flat.flatten()
        x_denoised_1d = x_denoised.flatten()
        
        corr_noisy = torch.corrcoef(torch.stack([x_noisy_1d, x_flat_1d]))[0, 1].item()
        corr_denoised = torch.corrcoef(torch.stack([x_denoised_1d, x_flat_1d]))[0, 1].item()
        
        # MAE比较
        mae_noisy = torch.abs(x_noisy_flat - x_flat).mean().item()
        mae_denoised = torch.abs(x_denoised - x_flat).mean().item()
    
    print(f"\n[结果]")
    print(f"  原始-加噪 相关性: {corr_noisy:.4f}")
    print(f"  去噪-原始 相关性: {corr_denoised:.4f}")
    print(f"  原始-加噪 MAE: {mae_noisy:.4f}")
    print(f"  去噪-原始 MAE: {mae_denoised:.4f}")
    
    # 判断标准
    if mae_denoised > mae_noisy * 0.9:
        print(f"  ❌ 失败: 去噪后MAE几乎没有改善，模型可能在忽略输入")
        return False
    elif corr_denoised < corr_noisy:
        print(f"  ❌ 失败: 去噪后相关性反而降低")
        return False
    else:
        print(f"  ✓ 成功: 去噪有效果")
        return True


def diagnose_condition_sensitivity(model, batch_x, batch_y, device='cuda'):
    """检验5: 条件敏感性（确保在用历史条件）
    
    验证模型真的在用x_hist，而不是生成与输入无关的"平均交通流"
    """
    print("\n" + "="*60)
    print("检验5: 条件敏感性")
    print("="*60)
    
    model.eval()
    
    # 创建两个完全不同的历史条件
    B = 2
    N = model.configs.enc_in
    L_enc = model.configs.seq_len
    L_pred = model.configs.pred_len
    
    # 历史A: 上升趋势
    hist_A = torch.linspace(0, 1, L_enc).unsqueeze(0).unsqueeze(-1).repeat(B//2, 1, N)
    # 历史B: 下降趋势
    hist_B = torch.linspace(1, 0, L_enc).unsqueeze(0).unsqueeze(-1).repeat(B//2, 1, N)
    
    batch_x_diff = torch.cat([hist_A, hist_B], dim=0).to(device)
    batch_y_diff = torch.zeros(B, L_pred, N).to(device)  # 目标不重要
    
    with torch.no_grad():
        # 各生成5个样本
        samples_A_list = []
        samples_B_list = []
        
        for seed in range(5):
            torch.manual_seed(seed * 42)
            output, _ = model(batch_x_diff[:B//2], None, batch_y_diff[:B//2], None, sample_times=1)
            samples_A_list.append(output.cpu())
            
            torch.manual_seed(seed * 42)
            output, _ = model(batch_x_diff[B//2:], None, batch_y_diff[B//2:], None, sample_times=1)
            samples_B_list.append(output.cpu())
        
        samples_A = torch.stack(samples_A_list, dim=0)  # (5, B/2, T, N)
        samples_B = torch.stack(samples_B_list, dim=0)  # (5, B/2, T, N)
        
        # 计算组内相似度
        intra_A = []
        for i in range(5):
            for j in range(i+1, 5):
                diff = torch.abs(samples_A[i] - samples_A[j]).mean().item()
                intra_A.append(diff)
        intra_A_mean = np.mean(intra_A)
        
        intra_B = []
        for i in range(5):
            for j in range(i+1, 5):
                diff = torch.abs(samples_B[i] - samples_B[j]).mean().item()
                intra_B.append(diff)
        intra_B_mean = np.mean(intra_B)
        
        # 计算组间差异
        inter_AB = []
        for i in range(5):
            for j in range(5):
                diff = torch.abs(samples_A[i] - samples_B[j]).mean().item()
                inter_AB.append(diff)
        inter_AB_mean = np.mean(inter_AB)
        
        # 各组的均值
        mean_A = samples_A.mean(dim=0)
        mean_B = samples_B.mean(dim=0)
        mean_AB_diff = torch.abs(mean_A - mean_B).mean().item()
    
    print(f"\n[结果]")
    print(f"  组A内平均差异: {intra_A_mean:.4f}")
    print(f"  组B内平均差异: {intra_B_mean:.4f}")
    print(f"  组AB间平均差异: {inter_AB_mean:.4f}")
    print(f"  均值差异: {mean_AB_diff:.4f}")
    print(f"  inter_AB / intra_A 比率: {inter_AB_mean / (intra_A_mean + 1e-8):.2f}x")
    
    # 判断标准
    if inter_AB_mean < intra_A_mean * 1.5:
        print(f"  ❌ 失败: 不同历史条件产生的样本差异不大，模型可能在忽略历史条件")
        return False
    else:
        print(f"  ✓ 成功: 模型对历史条件敏感")
        return True


def main():
    """主诊断流程"""
    print("\n" + "#"*60)
    print("# 扩散模型诊断工具 v1.0")
    print("#"*60)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 创建模型
    print("\n加载模型...")
    from fourier.ptld_model import Model
    
    # 读取配置
    class Args:
        seq_len = 96
        label_len = 48
        pred_len = 12
        enc_in = 30
        dec_in = 30
        c_out = 30
        d_model = 128
        e_layers = 1
        d_layers = 1
        d_ff = 512
        num_heads = 8
        dropout = 0.0
        stride = 3
        patch_len = 6
        diff_steps = 100
        s_steps = 2
        coss = 0.008
        new_norm = 1
        features = 'M'
        rmom = 20
        n_b = 5
        sample_times = 20
        batch_size = 32
        lower_order_final = 'true'
        method = 'multistep'
        skip_type = 'time_uniform'
        order = 2
    
    args = Args()
    model = Model(args).to(device)
    model.eval()
    
    # 生成测试数据
    print("生成测试数据...")
    B = 8
    batch_x = torch.randn(B, args.seq_len, args.enc_in)
    batch_y = torch.randn(B, args.pred_len + args.label_len, args.enc_in)
    
    # 运行诊断
    results = {}
    
    try:
        results['检验1_去噪学习'] = diagnose_noise_prediction(model, batch_x, batch_y, device)
    except Exception as e:
        print(f"\n❌ 检验1失败: {e}")
        import traceback
        traceback.print_exc()
        results['检验1_去噪学习'] = False
    
    try:
        results['检验2_样本多样性'] = diagnose_sample_diversity(model, batch_x, batch_y, device)
    except Exception as e:
        print(f"\n❌ 检验2失败: {e}")
        import traceback
        traceback.print_exc()
        results['检验2_样本多样性'] = False
    
    try:
        results['检验3_MoM效果'] = diagnose_mom_effectiveness(model, batch_x, batch_y, device)
    except Exception as e:
        print(f"\n❌ 检验3失败: {e}")
        import traceback
        traceback.print_exc()
        results['检验3_MoM效果'] = False
    
    try:
        results['检验4_去噪物理'] = diagnose_denoising_visual(model, batch_x, batch_y, device)
    except Exception as e:
        print(f"\n❌ 检验4失败: {e}")
        import traceback
        traceback.print_exc()
        results['检验4_去噪物理'] = False
    
    try:
        results['检验5_条件敏感'] = diagnose_condition_sensitivity(model, batch_x, batch_y, device)
    except Exception as e:
        print(f"\n❌ 检验5失败: {e}")
        import traceback
        traceback.print_exc()
        results['检验5_条件敏感'] = False
    
    # 汇总报告
    print("\n" + "="*60)
    print("诊断汇总报告")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n通过率: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n🎉 所有检验通过！你的模型是真正的扩散模型！")
    elif passed_count >= total_count // 2:
        print("\n⚠️ 部分检验通过，模型可能需要调优")
    else:
        print("\n❌ 大部分检验失败，请检查模型实现")
    
    return results


if __name__ == "__main__":
    main()
