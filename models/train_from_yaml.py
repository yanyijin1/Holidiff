"""
从 YAML 配置加载参数并运行训练/测试
用法:
  训练: python train_from_yaml.py --config config/xxx.yaml
  测试: python train_from_yaml.py --config config/xxx.yaml --test_only
"""
import argparse
import os
import sys
import random
import yaml

import numpy as np
import torch

# 确保项目根目录在 path 中
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trainer import Exp_Long_Term_Forecast
from utils.print_args import print_args


def load_config_from_yaml(config_path):
    """从 YAML 文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def yaml_to_args(config):
    """将 YAML 配置转换为 argparse.Namespace"""
    args = argparse.Namespace()

    # Experiment
    args.task_name = config.get('experiment', {}).get('task_name', 'long_term_forecast')
    args.model = config.get('experiment', {}).get('model', 'LWRdiff')
    args.model_id = config.get('experiment', {}).get('model_id', 'test')
    args.des = config.get('experiment', {}).get('des', 'test')
    args.itr = config.get('experiment', {}).get('itr', 1)

    # 如果有 overrides 嵌套，则优先从 overrides 读取
    cfg = config.get('overrides', config)

    # Data
    data_cfg = cfg.get('data', config.get('data', {}))
    args.data = data_cfg.get('data', 'custom')
    args.root_path = data_cfg.get('root_path', './data/')
    args.data_path = data_cfg.get('data_path', 'data.csv')
    args.features = data_cfg.get('features', 'M')
    args.target = data_cfg.get('target', 'OT')
    args.freq = data_cfg.get('freq', 'h')
    args.enc_in = data_cfg.get('enc_in', 7)
    args.dec_in = data_cfg.get('dec_in', 7)
    args.c_out = data_cfg.get('c_out', 7)

    # Window
    win_cfg = cfg.get('window', config.get('window', {}))
    args.seq_len = win_cfg.get('seq_len', 96)
    args.label_len = win_cfg.get('label_len', 48)
    args.pred_len = win_cfg.get('pred_len', 96)

    # Model
    model_cfg = cfg.get('model_args', config.get('model_args', {}))
    args.e_layers = model_cfg.get('e_layers', 2)
    args.d_layers = model_cfg.get('d_layers', 1)
    args.d_model = model_cfg.get('d_model', 512)
    args.num_heads = model_cfg.get('num_heads', 8)
    args.d_ff = model_cfg.get('d_ff', 2048)
    args.dropout = model_cfg.get('dropout', 0.1)
    args.stride = model_cfg.get('stride', 8)
    args.patch_len = model_cfg.get('patch_len', 16)
    args.embed = model_cfg.get('embed', 'timeF')
    args.activation = model_cfg.get('activation', 'gelu')
    args.output_attention = model_cfg.get('output_attention', False)
    args.skip_dropout = model_cfg.get('skip_dropout', 0.1)
    args.use_tst_layer = model_cfg.get('use_tst_layer', False)

    # Optimization
    opt_cfg = cfg.get('optimization', config.get('optimization', {}))
    args.seed = opt_cfg.get('seed', 2021)
    args.batch_size = opt_cfg.get('batch_size', 32)
    args.train_epochs = opt_cfg.get('train_epochs', 10)
    args.patience = opt_cfg.get('patience', 3)
    args.learning_rate = opt_cfg.get('learning_rate', 0.0001)
    args.loss_type = opt_cfg.get('loss_type', 'MSE')
    args.num_workers = opt_cfg.get('num_workers', 10)
    args.lradj = opt_cfg.get('lradj', 'type1')
    args.use_amp = opt_cfg.get('use_amp', False)

    # Diffusion
    diff_cfg = cfg
    args.is_diff = diff_cfg.get('is_diff', 1)
    args.diff_steps = diff_cfg.get('diff_steps', 100)
    args.s_steps = diff_cfg.get('s_steps', 5)
    args.skip_type = diff_cfg.get('skip_type', 'time_uniform')
    args.method = diff_cfg.get('method', 'multistep')
    args.order = diff_cfg.get('order', 2)
    args.lower_order_final = diff_cfg.get('lower_order_final', 'false')
    args.sample_times = diff_cfg.get('sample_times', 50)
    args.vs_times = diff_cfg.get('vs_times', 10)
    args.use_mom = diff_cfg.get('use_mom', 1)
    args.mom_mode = diff_cfg.get('mom_mode', 'mom')
    args.mom_tau = diff_cfg.get('mom_tau', 5.0)
    args.mom_clusters = diff_cfg.get('mom_clusters', 2)
    args.new_norm = diff_cfg.get('new_norm', 1)
    args.rmom = diff_cfg.get('rmom', 20)
    args.n_b = diff_cfg.get('n_b', 5)
    args.coss = diff_cfg.get('coss', 0.008)
    args.sampler = diff_cfg.get('sampler', 'dpmsolver')
    args.ddim_steps = diff_cfg.get('ddim_steps', 20)
    args.solver_order = diff_cfg.get('solver_order', 3)

    # GPU
    args.use_gpu = True
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = '0,1'

    # Misc
    args.seasonal_patterns = 'Monthly'
    args.inverse = False
    args.checkpoints = './checkpoints/'
    args.augmentation_ratio = 0

    return args


def main():
    parser = argparse.ArgumentParser(description='Train from YAML config')
    parser.add_argument('--config', type=str, required=True, help='YAML config file path')
    parser.add_argument('--test_only', action='store_true', help='Only run testing')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    args_cli = parser.parse_args()

    # 加载 YAML 配置
    config = load_config_from_yaml(args_cli.config)

    # 转换为 args
    args = yaml_to_args(config)
    args.gpu = args_cli.gpu

    # 固定随机种子
    fix_seed = args.seed
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.use_gpu = True if torch.cuda.is_available() else False

    if args_cli.test_only:
        exp = Exp_Long_Term_Forecast(args)
        setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.num_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_eb{args.embed}_{args.des}_0"
        exp.test(setting, test=1)
    else:
        # 训练模式
        for ii in range(args.itr):
            exp = Exp_Long_Term_Forecast(args)
            setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.num_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_eb{args.embed}_{args.des}_{ii}"

            print(f'>>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
            torch.cuda.empty_cache()

    print('[Done]')


if __name__ == '__main__':
    main()
