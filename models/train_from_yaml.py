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
from datetime import datetime

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


def save_best_params(model, save_dir, setting):
    """
    保存模型的最佳物理参数到文件
    支持 LWRGAT 模型和其他模型
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'best_params_{setting}_{timestamp}.txt')
    
    params_info = []
    params_info.append(f"=" * 60)
    params_info.append(f"Best Physical Parameters - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    params_info.append(f"Setting: {setting}")
    params_info.append(f"=" * 60)
    
    # 尝试获取 LWRGAT 模型的物理参数
    if hasattr(model, 'nn') and hasattr(model.nn, 'st_layers'):
        for i, layer in enumerate(model.nn.st_layers):
            params_info.append(f"\n[STLWRGATLayer {i}]")
            if hasattr(layer, 'pcgk'):
                pcgk = layer.pcgk
                params_info.append(f"  v_critical: {pcgk.v_critical.item():.2f} km/h")
                if hasattr(pcgk, 'alpha'):
                    params_info.append(f"  alpha (mix): {pcgk.alpha.item():.4f}")
                if hasattr(pcgk, 'A_phys_down'):
                    params_info.append(f"  A_phys_down: shape={pcgk.A_phys_down.shape}")
            if hasattr(layer, 'spatial_gat'):
                gat = layer.spatial_gat
                if hasattr(gat, 'log_A_weight'):
                    params_info.append(f"  log_A_weight: {gat.log_A_weight.item():.4f}")
    
    # 获取总参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_info.append(f"\n[Model Stats]")
    params_info.append(f"  Total params: {total_params:,}")
    params_info.append(f"  Trainable params: {trainable_params:,}")
    params_info.append(f"=" * 60)
    
    # 保存到文件
    with open(save_path, 'w') as f:
        f.write('\n'.join(params_info))
    
    # 同时保存为 JSON 格式方便程序读取
    json_path = save_path.replace('.txt', '.json')
    import json
    json_data = {
        'setting': setting,
        'timestamp': timestamp,
        'layers': []
    }
    
    if hasattr(model, 'nn') and hasattr(model.nn, 'st_layers'):
        for i, layer in enumerate(model.nn.st_layers):
            layer_data = {'layer_idx': i}
            if hasattr(layer, 'pcgk'):
                pcgk = layer.pcgk
                layer_data['v_critical'] = float(pcgk.v_critical.item())
                if hasattr(pcgk, 'alpha'):
                    layer_data['alpha'] = float(pcgk.alpha.item())
            if hasattr(layer, 'spatial_gat'):
                gat = layer.spatial_gat
                if hasattr(gat, 'log_A_weight'):
                    layer_data['log_A_weight'] = float(gat.log_A_weight.item())
            json_data['layers'].append(layer_data)
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"[Best Params] Saved to {save_path}")
    print(f"[Best Params] JSON saved to {json_path}")
    
    return save_path


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
    args.flow_path = data_cfg.get('flow_path', None)  # 流量数据路径（LWRGAT 双流输入）
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

    # Graph - 空间图结构配置
    graph_cfg = cfg.get('graph', {})
    args.graph_enabled = graph_cfg.get('enabled', False)
    args.graph_adj_path = graph_cfg.get('adj_path', None)
    args.graph_num_nodes = graph_cfg.get('num_nodes', None)

    # Optimization
    opt_cfg = cfg.get('optimization', config.get('optimization', {}))
    args.seed = opt_cfg.get('seed', 2021)
    args.batch_size = opt_cfg.get('batch_size', 32)
    args.train_epochs = opt_cfg.get('train_epochs', 10)
    args.patience = opt_cfg.get('patience', 3)
    args.learning_rate = opt_cfg.get('learning_rate', 0.0001)
    args.loss_type = opt_cfg.get('loss_type', 'MSE')
    args.num_workers = 0  # 避免多进程导致的问题
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
    
    # Metrics - DTW
    misc_cfg = cfg.get('metrics', {})
    args.use_dtw = misc_cfg.get('use_dtw', False)
    
    # MoE / Expert
    moe_cfg = cfg.get('moe', {})
    args.num_experts = moe_cfg.get('num_experts', 8)
    args.moe_enabled = moe_cfg.get('enabled', False)
    
    # Graph
    graph_cfg = cfg.get('graph', {})
    args.graph_enabled = graph_cfg.get('enabled', False)
    args.graph_adj_path = graph_cfg.get('adj_path', None)
    args.graph_num_nodes = graph_cfg.get('num_nodes', None)
    
    # 模型切换配置
    switch_cfg = cfg.get('model_switch', config.get('model_switch', {}))
    args.use_stformer = switch_cfg.get('use_stformer', False)
    args.use_lwrgat = switch_cfg.get('use_lwrgat', False)
    args.use_lwrres = switch_cfg.get('use_lwrres', False)
    args.st_layers = switch_cfg.get('st_layers', 3)
    args.use_simple_layer = switch_cfg.get('use_simple_layer', False)  # 简化层（快速验证）
    
    return args


def main():
    parser = argparse.ArgumentParser(description='Train from YAML config')
    parser.add_argument('--config', type=str, required=True, help='YAML config file path')
    parser.add_argument('--test_only', action='store_true', help='Only run testing')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
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

    # 设置日志目录
    log_dir = args_cli.log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = args.model_id or 'default'
    log_file = os.path.join(log_dir, f'train_{model_name}_{timestamp}.log')
    
    # 保存最佳参数的目录
    best_params_dir = os.path.join(log_dir, 'best_params')
    os.makedirs(best_params_dir, exist_ok=True)

    # 打开日志文件，同时输出到屏幕和文件
    log_f = open(log_file, 'a', buffering=1)
    
    # 保存原始 stdout
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    class Logger:
        def __init__(self, file, stdout):
            self.file = file
            self.stdout = stdout
        def write(self, msg):
            self.stdout.write(msg)
            self.file.write(msg)
            self.file.flush()
        def flush(self):
            self.stdout.flush()
            self.file.flush()
    
    sys.stdout = Logger(log_f, old_stdout)
    sys.stderr = Logger(log_f, old_stderr)

    try:
        if args_cli.test_only:
            exp = Exp_Long_Term_Forecast(args)
            setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.num_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_eb{args.embed}_{args.des}_0"
            
            print(f"\n{'='*60}")
            print(f"Test Mode - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}\n")
            
            exp.test(setting, test=1)
        else:
            # 训练模式
            for ii in range(args.itr):
                exp = Exp_Long_Term_Forecast(args)
                setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.num_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_eb{args.embed}_{args.des}_{ii}"

                print(f"\n{'='*60}")
                print(f"Training Iteration {ii+1}/{args.itr} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Setting: {setting}")
                print(f"{'='*60}\n")
                
                model = exp.train(setting)

                # 训练完成后保存最佳参数
                save_best_params(model, best_params_dir, setting)
                
                print(f"\n{'='*60}")
                print(f"Testing - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}\n")
                exp.test(setting)
                torch.cuda.empty_cache()
                
            # 保存最终最佳参数汇总
            print(f"\n{'='*60}")
            print(f"All training iterations completed!")
            print(f"Best params saved to: {best_params_dir}")
            print(f"Log saved to: {log_file}")
            print(f"{'='*60}\n")

        print('[Done]')
        
    finally:
        # 恢复原始 stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_f.close()
        print(f"\nLog saved to: {log_file}")
        print(f"Best params saved to: {best_params_dir}")


if __name__ == '__main__':
    main()
