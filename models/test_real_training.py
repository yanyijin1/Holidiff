"""
真实训练测试：验证 T/S 层效果
使用 MSST 数据集 (custom 格式)
"""
import sys
sys.path.insert(0, '/home/yanyijin/research/MSST-CL/models')

import argparse
import torch
import random
import numpy as np

from trainer import Exp_Long_Term_Forecast


def test_training(num_batches=10):
    """测试几个 batch 的训练"""

    parser = argparse.ArgumentParser(description='LWRdiff')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='test_TS_layer')
    parser.add_argument('--model', type=str, default='LWRdiff')

    # data loader - 使用 MSST 数据 (custom 格式)
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='./data/')
    parser.add_argument('--data_path', type=str, default='msst_speed.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='s0')  # MSST 数据的一个列
    parser.add_argument('--freq', type=str, default='t')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', action='store_true', default=False)

    # model define
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--enc_in', type=int, default=30)       # 30个节点
    parser.add_argument('--dec_in', type=int, default=30)
    parser.add_argument('--c_out', type=int, default=30)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--e_layers', type=int, default=1)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true', default=False)

    # patch parameters
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--patch_len', type=int, default=6)
    parser.add_argument('--skip_dropout', type=float, default=0.1)

    # diffusion parameters
    parser.add_argument('--coss', type=float, default=0.008)
    parser.add_argument('--diff_steps', type=int, default=100)
    parser.add_argument('--s_steps', type=int, default=2)
    parser.add_argument('--skip_type', type=str, default='time_uniform')
    parser.add_argument('--method', type=str, default='multistep')
    parser.add_argument('--lower_order_final', type=str, default='true')
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--is_diff', type=int, default=1)  # 开启扩散
    parser.add_argument('--rmom', type=int, default=20)
    parser.add_argument('--n_b', type=int, default=5)
    parser.add_argument('--sample_times', type=int, default=20)
    parser.add_argument('--vs_times', type=int, default=5)
    parser.add_argument('--use_mom', type=int, default=1)
    parser.add_argument('--mom_mode', type=str, default='mom')
    parser.add_argument('--mom_tau', type=float, default=5.0)
    parser.add_argument('--mom_clusters', type=int, default=2)
    parser.add_argument('--new_norm', type=int, default=1)

    # optimization
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='test_TS_layer')
    parser.add_argument('--loss_type', type=str, default='MAE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2021)

    args = parser.parse_args([])

    fix_seed = int(args.seed)
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.use_gpu = True if torch.cuda.is_available() else False
    print('=' * 60)
    print('CUDA available:', torch.cuda.is_available())
    print('=' * 60)

    # 创建实验
    exp = Exp_Long_Term_Forecast(args)

    # 获取数据
    train_data, train_loader = exp._get_data(flag='train')
    print(f'\n数据加载成功!')
    print(f'训练集大小: {len(train_data)}')

    # 获取模型
    model = exp.model
    print(f'\n模型创建成功!')
    print(f'模型参数数量: {sum(p.numel() for p in model.parameters()):,}')
    print(f'设备: {exp.device}')

    # 测试训练
    model.train()
    model_optim = exp._select_optimizer()
    criterion = exp._select_criterion()

    print(f'\n' + '=' * 60)
    print(f'开始测试训练 (测试 {num_batches} 个 batch)')
    print('=' * 60)

    losses = []
    iter_count = 0

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        if i >= num_batches:
            break

        iter_count += 1
        model_optim.zero_grad()

        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float().to(exp.device)
        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)

        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)

        # 前向传播
        outputs, weight_tmp = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        # 调整输出维度
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, f_dim:, -args.pred_len:]
        batch_y_adj = batch_y[:, -args.pred_len:, f_dim:].to(exp.device)

        # 计算损失
        loss = criterion(outputs / weight_tmp, batch_y_adj.permute(0, 2, 1) / weight_tmp)

        # 反向传播
        loss.backward()
        model_optim.step()

        losses.append(loss.item())

        print(f'Batch {iter_count}/{num_batches}: Loss = {loss.item():.6f}')

    # 统计
    print('\n' + '=' * 60)
    print('训练测试完成!')
    print('=' * 60)
    print(f'平均 Loss: {np.mean(losses):.6f}')
    print(f'Loss 标准差: {np.std(losses):.6f}')
    print(f'Loss 范围: [{min(losses):.6f}, {max(losses):.6f}]')

    if len(losses) > 1:
        diff = losses[-1] - losses[0]
        print(f'\nLoss 变化: {diff:+.6f} ({("下降" if diff < 0 else "上升")})')
        # 计算趋势
        if len(losses) > 2:
            early_avg = np.mean(losses[:3])
            late_avg = np.mean(losses[-3:])
            trend = (late_avg - early_avg) / early_avg * 100
            print(f'Loss 趋势 (前3 vs 后3): {trend:+.2f}%')

    return losses


if __name__ == '__main__':
    test_training(num_batches=10)
