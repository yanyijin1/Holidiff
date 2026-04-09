# Version: v0.1-run (Exact match with LWRdiff original)
# Date: 2026-04-07
# Description: Entry Point - Copy from LWRdiff

import argparse
import torch
import random
import numpy as np

from trainer import Exp_Long_Term_Forecast
from utils.print_args import print_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LWRdiff')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, required=True, default='LWRdiff')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1')
    parser.add_argument('--root_path', type=str, default='./data/ETT/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', action='store_true', default=False)

    # model define
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true', default=False)

    # patch parameters
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--patch_len', type=int, default=16)

    # diffusion parameters
    parser.add_argument('--coss', type=float, default=0.008)
    parser.add_argument('--diff_steps', type=int, default=100)
    parser.add_argument('--s_steps', type=int, default=5)
    parser.add_argument('--skip_type', type=str, default='time_uniform')
    parser.add_argument('--method', type=str, default='multistep')
    parser.add_argument('--lower_order_final', type=str, default='false')
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--is_diff', type=int, default=0)
    parser.add_argument('--rmom', type=int, default=20)
    parser.add_argument('--n_b', type=int, default=5)
    parser.add_argument('--sample_times', type=int, default=50)
    parser.add_argument('--vs_times', type=int, default=10)
    parser.add_argument('--use_mom', type=int, default=1)
    parser.add_argument('--mom_mode', type=str, default='mom')
    parser.add_argument('--mom_tau', type=float, default=5.0)
    parser.add_argument('--mom_clusters', type=int, default=2)
    parser.add_argument('--new_norm', type=int, default=1)

    # optimization
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss_type', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)

    args = parser.parse_args()

    fix_seed = int(args.seed)
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.use_gpu = True if torch.cuda.is_available() else False
    print('CUDA available:', torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    else:
        raise ValueError("Only long_term_forecast is supported")

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_eb{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.num_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.embed,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        exp = Exp(args)
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_eb{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.num_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.embed,
            args.des, ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()