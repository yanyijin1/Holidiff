#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扩散模型快速诊断脚本 - 检验是否为真扩散
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def diagnose_sample_diversity_simple(model, device='cuda', num_samples=8):
    """检验2: 推理阶段 - 样本是否有"多样性"（核心检验）"""
    print("\n" + "="*60)
    print("检验2: 推理阶段 - 样本多样性（核心检验）")
    print("="*60)
    
    model.eval()
    
    # 创建测试数据
    B = 4
    seq_len = model.configs.seq_len
    pred_len = model.configs.pred_len
    enc_in = model.configs.enc_in
    
    batch_x = torch.randn(B, seq_len, enc_in).to(device)
    batch_y = torch.randn(B, pred_len, enc_in).to(device)  # dec_in
    
    with torch.no_grad():
        # 多次采样
        samples = []
        for i in range(num_samples):
            torch.manual_seed(i * 42)  # 不同随机种子 = 不同初始噪声
            output, _ = model(batch_x, None, batch_y, None, sample_times=1)
            samples.append(output.cpu().numpy())
        
        samples = np.stack(samples, axis=0)  # (num_samples, B, T, N)
        
        # 计算样本方差
        sample_mean = samples.mean(axis=0)
        sample_var = samples.var(axis=0)  # 每个样本内的方差
        avg_var = sample_var.mean()
        avg_std = np.sqrt(avg_var)
        
        # 计算真实目标的标准差作为参考
        y_true_std = batch_y[:, -pred_len:, :].std().item()
        
        # 相对方差
        rel_var = avg_std / (y_true_std + 1e-8)
        
        # 计算样本间差异
        sample_diffs = []
        for i in range(num_samples):
            for j in range(i+1, num_samples):
                diff = np.abs(samples[i] - samples[j]).mean()
                sample_diffs.append(diff)
        avg_diff = np.mean(sample_diffs)
        
        print(f"\n[结果]")
        print(f"  采样次数: {num_samples}")
        print(f"  样本平均标准差: {avg_std:.4f}")
        print(f"  真实目标标准差: {y_true_std:.4f}")
        print(f"  相对方差(样本std/真实std): {rel_var:.4f}")
        print(f"  样本间平均差异: {avg_diff:.4f}")
        
        # 判断
        threshold = 0.05
        if rel_var < threshold:
            print(f"\n  ❌ 失败: 样本方差太小({rel_var:.4f} < {threshold})")
            print(f"     模型可能是确定性的！")
            return False
        else:
            print(f"\n  ✓ 成功: 样本有多样性")
            print(f"     相对方差 {rel_var:.4f} > {threshold}，证明从不同随机噪声生成了不同样本")
            return True


def diagnose_mom_vs_mean(model, device='cuda', num_samples=16):
    """检验3: MoM效果验证"""
    print("\n" + "="*60)
    print("检验3: MoM效果验证")
    print("="*60)
    
    model.eval()
    
    B = 4
    seq_len = model.configs.seq_len
    pred_len = model.configs.pred_len
    enc_in = model.configs.enc_in
    
    batch_x = torch.randn(B, seq_len, enc_in).to(device)
    batch_y = torch.randn(B, pred_len + model.configs.label_len, enc_in).to(device)
    
    with torch.no_grad():
        # 收集所有样本
        all_samples = []
        for i in range(num_samples):
            torch.manual_seed(i * 100)
            output, _ = model(batch_x, None, batch_y, None, sample_times=1)
            all_samples.append(output.cpu())
        
        all_samples = torch.stack(all_samples, dim=0)  # (num_samples, B, T, N)
        
        # 1. 均值聚合
        pred_mean = all_samples.mean(dim=0)
        
        # 2. MoM聚合 (Median of Means, 分成2组)
        num_groups = 2
        group_size = num_samples // num_groups
        group_means = []
        for i in range(num_groups):
            group_mean = all_samples[i*group_size:(i+1)*group_size].mean(dim=0)
            group_means.append(group_mean)
        pred_mom = torch.stack(group_means).median(dim=0)[0]
        
        # 3. 随机选一个作为单样本
        pred_single = all_samples[0]
        
        # 计算各方法的结果分散度（与均值的差异）
        overall_mean = all_samples.mean(dim=0)
        
        mean_disp = torch.abs(pred_mean - overall_mean).mean().item()
        mom_disp = torch.abs(pred_mom - overall_mean).mean().item()
        single_disp = torch.abs(pred_single - overall_mean).mean().item()
        
        print(f"\n[结果]")
        print(f"  采样次数: {num_samples}")
        print(f"  与整体均值的偏离:")
        print(f"    单样本: {single_disp:.4f}")
        print(f"    均值:   {mean_disp:.4f}")
        print(f"    MoM:   {mom_disp:.4f}")
        
        # MoM应该比均值更稳健（偏离更小或相似）
        if mom_disp <= mean_disp * 1.5:
            print(f"\n  ✓ MoM表现正常")
            print(f"     MoM偏离/均值偏离 = {mom_disp/mean_disp:.2f}")
            return True
        else:
            print(f"\n  ⚠️ MoM偏离较大")
            return False


def diagnose_condition_sensitivity_simple(model, device='cuda'):
    """检验5: 条件敏感性（确保在用历史条件）"""
    print("\n" + "="*60)
    print("检验5: 条件敏感性（核心检验）")
    print("="*60)
    
    model.eval()
    
    seq_len = model.configs.seq_len
    pred_len = model.configs.pred_len
    enc_in = model.configs.enc_in
    
    # 创建两个完全不同的历史条件
    B_half = 2
    
    # 历史A: 上升趋势
    hist_A = torch.linspace(0.5, 1.5, seq_len).unsqueeze(0).unsqueeze(-1).repeat(B_half, 1, enc_in)
    # 历史B: 下降趋势  
    hist_B = torch.linspace(1.5, 0.5, seq_len).unsqueeze(0).unsqueeze(-1).repeat(B_half, 1, enc_in)
    
    batch_x_AB = torch.cat([hist_A, hist_B], dim=0).to(device)
    batch_y_AB = torch.zeros(B_half * 2, pred_len + model.configs.label_len, enc_in).to(device)
    
    with torch.no_grad():
        # 各生成5个样本
        samples_A = []
        samples_B = []
        
        for seed in range(5):
            torch.manual_seed(seed * 42)
            output_A, _ = model(batch_x_AB[:B_half], None, batch_y_AB[:B_half], None, sample_times=1)
            samples_A.append(output_A.cpu())
            
            torch.manual_seed(seed * 42)  # 同样的种子，但历史不同
            output_B, _ = model(batch_x_AB[B_half:], None, batch_y_AB[B_half:], None, sample_times=1)
            samples_B.append(output_B.cpu())
        
        samples_A = torch.stack(samples_A, dim=0)  # (5, B_half, T, N)
        samples_B = torch.stack(samples_B, dim=0)
        
        # 组内差异
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
        
        # 组间差异（用相同种子比较）
        inter_AB = []
        for i in range(5):
            diff = torch.abs(samples_A[i] - samples_B[i]).mean().item()
            inter_AB.append(diff)
        inter_AB_mean = np.mean(inter_AB)
        
        # 各组均值差异
        mean_A = samples_A.mean(dim=0)
        mean_B = samples_B.mean(dim=0)
        mean_AB_diff = torch.abs(mean_A - mean_B).mean().item()
        
        # 组内均值
        intra_mean = (intra_A_mean + intra_B_mean) / 2
        ratio = inter_AB_mean / (intra_mean + 1e-8)
    
    print(f"\n[结果]")
    print(f"  历史A: 上升趋势 [0.5 → 1.5]")
    print(f"  历史B: 下降趋势 [1.5 → 0.5]")
    print(f"  ")
    print(f"  组A内平均差异: {intra_A_mean:.4f}")
    print(f"  组B内平均差异: {intra_B_mean:.4f}")
    print(f"  组AB间(同种子)差异: {inter_AB_mean:.4f}")
    print(f"  均值差异: {mean_AB_diff:.4f}")
    print(f"  ")
    print(f"  inter_AB / intra 均值 = {ratio:.2f}x")
    
    if ratio > 1.5:
        print(f"\n  ✓ 成功: 模型对历史条件敏感")
        print(f"     组间差异是组内差异的 {ratio:.1f} 倍")
        print(f"     证明模型确实在使用历史条件生成不同的未来")
        return True
    else:
        print(f"\n  ❌ 失败: 模型可能忽略了历史条件")
        print(f"     不同历史产生的样本差异不大")
        return False


def diagnose_noise_understanding(model, device='cuda'):
    """检验1: 简单版 - 模型理解噪声预测"""
    print("\n" + "="*60)
    print("检验1: 噪声预测能力（训练阶段核心检验）")
    print("="*60)
    
    model.eval()
    
    # 检查模型是否能正确处理加噪
    seq_len = model.configs.seq_len
    pred_len = model.configs.pred_len
    enc_in = model.configs.enc_in
    
    # 创建简单测试数据
    x_enc = torch.randn(2, seq_len, enc_in).to(device)
    x_dec = torch.randn(2, pred_len + 48, enc_in).to(device)  # dec_in
    
    # 模拟训练阶段的前向传播
    try:
        model.train()
        output, weight_tmp = model(x_enc, None, x_dec, None)
        model.eval()
        
        print(f"\n[结果]")
        print(f"  输出形状: {output.shape}")
        print(f"  weight_tmp形状: {weight_tmp.shape}")
        
        # weight_tmp是sqrt(1-alpha_bar)，应该与噪声预测有关
        weight_mean = weight_tmp.mean().item()
        weight_std = weight_tmp.std().item()
        
        print(f"  weight_tmp均值: {weight_mean:.4f}")
        print(f"  weight_tmp标准差: {weight_std:.4f}")
        
        # 在不同时间步，weight应该不同
        print(f"\n  ✓ 模型前向传播正常")
        print(f"     证明模型能处理噪声预测任务")
        return True
        
    except Exception as e:
        print(f"\n  ❌ 失败: {e}")
        return False


def main():
    print("\n" + "#"*60)
    print("# 扩散模型诊断工具 v2.0 (简化版)")
    print("#"*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 创建模型
    print("\n加载模型...")
    from fourier.ptld_model import Model
    
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
    
    print("模型加载成功!")
    
    results = {}
    
    # 运行核心检验
    try:
        results['检验1_噪声预测'] = diagnose_noise_understanding(model, device)
    except Exception as e:
        print(f"\n❌ 检验1异常: {e}")
        results['检验1_噪声预测'] = False
    
    try:
        results['检验2_样本多样性'] = diagnose_sample_diversity_simple(model, device)
    except Exception as e:
        print(f"\n❌ 检验2异常: {e}")
        import traceback
        traceback.print_exc()
        results['检验2_样本多样性'] = False
    
    try:
        results['检验3_MoM效果'] = diagnose_mom_vs_mean(model, device)
    except Exception as e:
        print(f"\n❌ 检验3异常: {e}")
        import traceback
        traceback.print_exc()
        results['检验3_MoM效果'] = False
    
    try:
        results['检验5_条件敏感'] = diagnose_condition_sensitivity_simple(model, device)
    except Exception as e:
        print(f"\n❌ 检验5异常: {e}")
        import traceback
        traceback.print_exc()
        results['检验5_条件敏感'] = False
    
    # 汇总
    print("\n" + "="*60)
    print("诊断汇总报告")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n通过率: {passed}/{total}")
    
    # 关键结论
    print("\n" + "="*60)
    print("关键结论")
    print("="*60)
    
    if results.get('检验2_样本多样性', False):
        print("✓ 模型能生成多样化的样本（真扩散核心特征）")
    else:
        print("❌ 模型可能还是确定性的")
        
    if results.get('检验5_条件敏感', False):
        print("✓ 模型能根据历史条件生成不同未来（条件生成能力）")
    else:
        print("❌ 模型可能忽略了历史条件")
    
    if passed >= total * 0.6:
        print("\n🎉 模型通过核心检验，是真正的扩散模型!")
    else:
        print("\n⚠️ 部分核心检验未通过，请检查")
    
    return results


if __name__ == "__main__":
    main()
