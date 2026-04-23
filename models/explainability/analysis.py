"""
可解释性分析脚本
基于训练好的模型参数，分析物理可解释性

使用方法:
  python explainability_analysis.py --checkpoint checkpoints/xxx/checkpoint.pth --config config/xxx.yaml

输出:
  - logs/explainability/ 目录下的分析结果和可视化图片
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_config_from_yaml(config_path):
    """从 YAML 文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def yaml_to_args(config):
    """将 YAML 配置转换为 argparse.Namespace"""
    args = argparse.Namespace()
    
    cfg = config.get('overrides', config)
    
    data_cfg = cfg.get('data', config.get('data', {}))
    args.data = data_cfg.get('data', 'custom')
    args.root_path = data_cfg.get('root_path', './data/')
    args.data_path = data_cfg.get('data_path', 'data.csv')
    args.features = data_cfg.get('features', 'M')
    args.enc_in = data_cfg.get('enc_in', 7)
    
    win_cfg = cfg.get('window', config.get('window', {}))
    args.seq_len = win_cfg.get('seq_len', 96)
    args.label_len = win_cfg.get('label_len', 48)
    args.pred_len = win_cfg.get('pred_len', 96)
    
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
    args.n_b = cfg.get('n_b', 5)
    args.rmom = cfg.get('rmom', 20)
    
    opt_cfg = cfg.get('optimization', config.get('optimization', {}))
    args.batch_size = opt_cfg.get('batch_size', 32)
    args.seed = opt_cfg.get('seed', 2021)
    
    args.diff_steps = cfg.get('diff_steps', 100)
    args.s_steps = cfg.get('s_steps', 5)
    args.sample_times = cfg.get('sample_times', 50)
    args.vs_times = cfg.get('vs_times', 10)
    args.new_norm = cfg.get('new_norm', 1)
    args.rmom = cfg.get('rmom', 20)
    args.n_b = cfg.get('n_b', 5)
    args.coss = cfg.get('coss', 0.008)
    args.is_diff = cfg.get('is_diff', 1)
    args.loss_type = 'MSE'
    args.lower_order_final = 'true'
    args.method = cfg.get('method', 'multistep')
    args.skip_type = cfg.get('skip_type', 'time_uniform')
    args.order = cfg.get('order', 2)
    args.sampler = cfg.get('sampler', 'dpmsolver')
    args.ddim_steps = cfg.get('ddim_steps', 20)
    args.solver_order = cfg.get('solver_order', 3)
    
    args.use_gpu = True
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = '0,1'
    args.checkpoints = './checkpoints/'
    args.freq = 'h'
    args.target = data_cfg.get('target', 'OT')
    args.num_workers = 0
    args.itr = 1
    args.train_epochs = 10
    args.patience = 10
    
    switch_cfg = cfg.get('model_switch', config.get('model_switch', {}))
    args.use_stformer = switch_cfg.get('use_stformer', False)
    args.use_lwrgat = switch_cfg.get('use_lwrgat', False)
    args.st_layers = switch_cfg.get('st_layers', 3)
    
    graph_cfg = cfg.get('graph', {})
    args.graph_enabled = graph_cfg.get('enabled', False)
    args.graph_adj_path = graph_cfg.get('adj_path', None)
    
    return args


class ExplainabilityAnalyzer:
    """可解释性分析器"""
    
    def __init__(self, model, args, device='cuda'):
        self.model = model
        self.args = args
        self.device = device
        self.model.eval()
        self.results = {
            'regime_distribution': [],
            'direction_ratio': [],
            'decouple_ratio': [],
            'timestamps': []
        }
    
    def extract_hooks(self):
        """提取中间层输出的 hooks"""
        self.regime_values = []
        self.downstream_mass = []
        self.upstream_mass = []
        self.res_ratio = []
        
        def hook_regime(module, input, output):
            if len(input) > 0 and isinstance(input[0], torch.Tensor):
                v_obs = input[0]
                if hasattr(module, 'v_critical'):
                    regime = (v_obs > module.v_critical).float().mean().item()
                    self.regime_values.append(regime)
        
        def hook_pcgk(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                A_kernel, regime = output[0], output[1]
                self.regime_values.append(regime.mean().item())
                
                if hasattr(self.model, 'nn') and hasattr(self.model.nn, 'A_phys_down'):
                    A_phys_down = self.model.nn.A_phys_down
                    A_phys_up = self.model.nn.A_phys_up
                    B = A_kernel.shape[0]
                    
                    down = (A_kernel * A_phys_down.unsqueeze(0)).sum(dim=-1).mean(dim=0).mean().item()
                    up = (A_kernel * A_phys_up.unsqueeze(0)).sum(dim=-1).mean(dim=0).mean().item()
                    self.downstream_mass.append(down)
                    self.upstream_mass.append(up)
        
        self.hooks = []
        if hasattr(self.model, 'nn') and hasattr(self.model.nn, 'st_layers'):
            for layer in self.model.nn.st_layers:
                if hasattr(layer, 'pcgk'):
                    h = layer.pcgk.register_forward_hook(hook_pcgk)
                    self.hooks.append(h)
    
    def remove_hooks(self):
        """移除 hooks"""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def analyze_single_batch(self, batch_x, batch_y, x_mark_enc, dec_inp, y_mark_dec):
        """分析单个 batch"""
        with torch.no_grad():
            if self.args.is_diff:
                outputs, all_samples = self.model(
                    batch_x, x_mark_enc, dec_inp, y_mark_dec,
                    sample_times=1
                )
            else:
                outputs = self.model(batch_x, x_mark_enc, dec_inp, y_mark_dec)
            
            B, T, N = batch_x.shape
            
            # 计算相态分布 (regime > 0.5 为自由流)
            # 从 velocity 数据中提取 (假设 velocity 是第3个特征或可以从 flow 推导)
            if batch_x.shape[-1] >= 3:
                v_obs = batch_x[:, :, 2].to(self.device)  # velocity 特征
            else:
                # 从 flow/density 推导速度 v = q / k
                v_obs = torch.ones(B, T, N, device=self.device) * 30.0  # 默认值
            
            # 使用模型中的 v_critical
            if hasattr(self.model, 'nn') and hasattr(self.model.nn, 'st_layers'):
                v_criticals = []
                for layer in self.model.nn.st_layers:
                    if hasattr(layer, 'pcgk'):
                        v_criticals.append(layer.pcgk.v_critical.item())
                v_critical = np.mean(v_criticals)
                
                regime = (v_obs > v_critical).float().mean().item()
                self.results['regime_distribution'].append(regime)
                self.results['timestamps'].append(datetime.now().isoformat())
                
                # 图核方向性
                if len(self.downstream_mass) > 0:
                    down = self.downstream_mass[-1] if self.downstream_mass else 0
                    up = self.upstream_mass[-1] if self.upstream_mass else 0
                    self.results['direction_ratio'].append({
                        'downstream': down,
                        'upstream': up,
                        'ratio': up / (down + 1e-6)
                    })
    
    def get_physical_params(self):
        """获取物理参数"""
        params = {}
        
        # 从 state_dict 读取参数（更可靠）
        state_dict = self.model.state_dict()
        
        layers_params = []
        for key in sorted(state_dict.keys()):
            if 'st_layers' in key and 'pcgk' in key:
                layer_idx = int(key.split('.st_layers.')[1].split('.')[0])
                if layer_idx >= len(layers_params):
                    layers_params.append({})
                
                if 'raw_v_critical' in key:
                    raw_val = state_dict[key].item()
                    # v_critical = softplus(raw) + min
                    v_crit = np.log(1 + np.exp(max(raw_val, 0))) + 5.0
                    layers_params[layer_idx]['raw_v_critical'] = raw_val
                    layers_params[layer_idx]['v_critical'] = v_crit
                elif 'alpha' in key:
                    alpha_val = state_dict[key].item()
                    layers_params[layer_idx]['alpha'] = alpha_val
            
            # 提取 GAT 参数
            if 'st_layers' in key and 'spatial_gat' in key:
                layer_idx = int(key.split('.st_layers.')[1].split('.')[0])
                if layer_idx >= len(layers_params):
                    layers_params.append({'layer_idx': layer_idx})
                
                if 'log_A_weight' in key:
                    layers_params[layer_idx]['log_A_weight'] = state_dict[key].item()
                elif 'beta' in key:
                    layers_params[layer_idx]['gat_beta'] = state_dict[key].item()
        
        if layers_params:
            params['layers'] = [{'layer_idx': i, **p} for i, p in enumerate(layers_params)]
        
        if hasattr(self.model, 'nn') and hasattr(self.model.nn, 'st_layers'):
            params['graph_info'] = {
                'num_nodes': self.model.nn.enc_in,
                'num_layers': len(self.model.nn.st_layers),
                'has_physical_adj': hasattr(self.model.nn, 'A_phys_down')
            }
        
        params['model_stats'] = {
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'device': str(next(self.model.parameters()).device)
        }
        
        return params
    
    def print_physical_params(self):
        """打印物理参数"""
        params = self.get_physical_params()
        
        print("\n" + "="*60)
        print("[Physical Parameters]")
        print("="*60)
        
        if 'layers' in params:
            for layer in params['layers']:
                print(f"\n[STLWRGATLayer {layer['layer_idx']}]")
                if 'raw_v_critical' in layer:
                    print(f"  raw_v_critical = {layer['raw_v_critical']:.4f}")
                if 'v_critical' in layer:
                    print(f"  v_critical = {layer['v_critical']:.2f} (after softplus + 5)")
                if 'alpha' in layer:
                    alpha_sig = 1.0 / (1.0 + np.exp(-layer['alpha']))
                    print(f"  alpha (sigmoid) = {alpha_sig:.4f}")
                if 'log_A_weight' in layer:
                    print(f"  log_A_weight = {layer['log_A_weight']:.4f}")
                if 'gat_beta' in layer:
                    print(f"  gat_beta = {layer['gat_beta']:.4f}")
        
        if 'graph_info' in params:
            print(f"\n[Graph Info]")
            print(f"  num_nodes: {params['graph_info']['num_nodes']}")
            print(f"  num_layers: {params['graph_info']['num_layers']}")
            print(f"  has_physical_adj: {params['graph_info']['has_physical_adj']}")
        
        if 'model_stats' in params:
            print(f"\n[Model Stats]")
            print(f"  Total params: {params['model_stats']['total_params']:,}")
            print(f"  Trainable params: {params['model_stats']['trainable_params']:,}")
            print(f"  Device: {params['model_stats']['device']}")
        
        print("="*60)
        
        return params
    
    def analyze_data_patterns(self, data_loader, num_batches=10):
        """分析数据中的物理模式"""
        print("\n" + "="*60)
        print("[Data Pattern Analysis]")
        print("="*60)
        
        self.extract_hooks()
        
        v_obs_all = []
        flow_obs_all = []
        all_timestamps = []
        
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
            
            if len(batch) == 6:
                batch_x, batch_y, batch_x_mark, batch_y_mark, _, _ = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            
            B, T, N = batch_x.shape[:3]
            
            # 数据已经是归一化的，分析第一列 (通常是 flow/q)
            flow_data = batch_x[:, :, 0].cpu().numpy()
            flow_obs_all.extend(flow_data.flatten())
            
            # 如果有多个特征
            if batch_x.shape[-1] >= 2:
                v_data = batch_x[:, :, 1].cpu().numpy()
                v_obs_all.extend(v_data.flatten())
            
            # 从 timestamp 提取小时信息
            if batch_x_mark.shape[-1] >= 4:
                hours = batch_x_mark[:, :, 3].cpu().numpy()
                all_timestamps.extend(hours.flatten())
        
        self.remove_hooks()
        
        flow_obs_all = np.array(flow_obs_all)
        print(f"\n[Flow Distribution (normalized)]")
        print(f"  Mean: {flow_obs_all.mean():.4f}")
        print(f"  Std: {flow_obs_all.std():.4f}")
        print(f"  Min: {flow_obs_all.min():.4f}")
        print(f"  Max: {flow_obs_all.max():.4f}")
        
        if len(v_obs_all) > 0:
            v_obs_all = np.array(v_obs_all)
            print(f"\n[Velocity Distribution (normalized)]")
            print(f"  Mean: {v_obs_all.mean():.4f}")
            print(f"  Std: {v_obs_all.std():.4f}")
            print(f"  Min: {v_obs_all.min():.4f}")
            print(f"  Max: {v_obs_all.max():.4f}")
        
        # 时空模式分析
        if len(all_timestamps) > 0:
            all_timestamps = np.array(all_timestamps)
            unique_hours = np.unique(all_timestamps)
            print(f"\n[Time Pattern]")
            print(f"  Unique hours in data: {len(unique_hours)}")
            print(f"  Hour range: {unique_hours.min():.0f} - {unique_hours.max():.0f}")
            
            # 分析不同时间段的流量分布
            if len(unique_hours) > 0:
                # 早高峰 (6-9), 晚高峰 (17-20), 夜间 (22-5)
                morning_rush = (flow_obs_all[(all_timestamps >= 6) & (all_timestamps < 9)]).mean() if ((all_timestamps >= 6) & (all_timestamps < 9)).sum() > 0 else 0
                evening_rush = (flow_obs_all[(all_timestamps >= 17) & (all_timestamps < 20)]).mean() if ((all_timestamps >= 17) & (all_timestamps < 20)).sum() > 0 else 0
                night = (flow_obs_all[(all_timestamps >= 22) | (all_timestamps < 5)]).mean() if ((all_timestamps >= 22) | (all_timestamps < 5)).sum() > 0 else 0
                
                print(f"\n[Traffic Pattern by Time of Day]")
                print(f"  Morning rush (6-9): {morning_rush:.4f}")
                print(f"  Evening rush (17-20): {evening_rush:.4f}")
                print(f"  Night (22-5): {night:.4f}")
                
                self.results['traffic_pattern'] = {
                    'morning_rush': float(morning_rush),
                    'evening_rush': float(evening_rush),
                    'night': float(night)
                }
        
        print("="*60)
        
        return {
            'flow_distribution': flow_obs_all,
            'v_distribution': v_obs_all if len(v_obs_all) > 0 else None,
            'timestamps': all_timestamps if len(all_timestamps) > 0 else None
        }
    
    def generate_visualizations(self, data_patterns, output_dir):
        """生成可视化图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        params = self.get_physical_params()
        
        # Figure 1: 物理参数收敛图
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        if 'layers' in params and len(params['layers']) > 0:
            layer_ids = [l['layer_idx'] for l in params['layers']]
            
            # v_critical (左上)
            v_crits = [l.get('v_critical', 0) for l in params['layers']]
            axes[0, 0].bar(layer_ids, v_crits, color='steelblue', alpha=0.8, edgecolor='black')
            axes[0, 0].set_xlabel('Layer Index')
            axes[0, 0].set_ylabel('v_critical (km/h)')
            axes[0, 0].set_title('Critical Velocity (Physics)')
            for i, v in enumerate(v_crits):
                axes[0, 0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=10)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            axes[0, 0].set_ylim(0, max(v_crits) * 1.3 if v_crits else 50)
            
            # alpha (右上)
            alphas = [l.get('alpha', 0) for l in params['layers']]
            axes[0, 1].bar(layer_ids, alphas, color='coral', alpha=0.8, edgecolor='black')
            axes[0, 1].set_xlabel('Layer Index')
            axes[0, 1].set_ylabel('Alpha')
            axes[0, 1].set_title('Physics Adaptation Coefficient')
            for i, a in enumerate(alphas):
                axes[0, 1].text(i, a + 0.02, f'{a:.3f}', ha='center', fontsize=10)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # log_A_weight (左下)
            log_A_weights = [l.get('log_A_weight', 0) for l in params['layers']]
            axes[1, 0].bar(layer_ids, log_A_weights, color='seagreen', alpha=0.8, edgecolor='black')
            axes[1, 0].set_xlabel('Layer Index')
            axes[1, 0].set_ylabel('log(A_weight)')
            axes[1, 0].set_title('Graph Attention Weight')
            for i, w in enumerate(log_A_weights):
                offset = 0.02 if w >= 0 else -0.15
                axes[1, 0].text(i, w + offset, f'{w:.2f}', ha='center', fontsize=10)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # gat_beta (右下)
            betas = [l.get('gat_beta', 0) for l in params['layers']]
            axes[1, 1].bar(layer_ids, betas, color='purple', alpha=0.8, edgecolor='black')
            axes[1, 1].set_xlabel('Layer Index')
            axes[1, 1].set_ylabel('Beta')
            axes[1, 1].set_title('GAT Beta (Residual)')
            for i, b in enumerate(betas):
                offset = 0.02 if b >= 0 else -0.15
                axes[1, 1].text(i, b + offset, f'{b:.3f}', ha='center', fontsize=10)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            plt.suptitle('Physical Parameters Analysis', fontsize=14, fontweight='bold')
        else:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No layer parameters found', ha='center', va='center')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'physical_params.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Viz] Saved physical_params.png")
        
        # Figure 2: 数据分布
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        flow_dist = data_patterns.get('flow_distribution', np.array([]))
        if len(flow_dist) > 0:
            axes[0].hist(flow_dist, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            axes[0].set_xlabel('Flow (normalized)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Flow Distribution')
            axes[0].grid(True, alpha=0.3)
        
        v_dist = data_patterns.get('v_distribution', np.array([]))
        if len(v_dist) > 0:
            axes[1].hist(v_dist, bins=50, color='coral', alpha=0.7, edgecolor='black')
            axes[1].set_xlabel('Velocity (normalized)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Velocity Distribution')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'data_distribution.png'), dpi=150)
        plt.close()
        print(f"[Viz] Saved data_distribution.png")
        
        # Figure 3: 时段分析
        traffic_pattern = self.results.get('traffic_pattern', {})
        if traffic_pattern:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            times = ['Morning\n(6-9)', 'Evening\n(17-20)', 'Night\n(22-5)']
            values = [traffic_pattern.get('morning_rush', 0),
                     traffic_pattern.get('evening_rush', 0),
                     traffic_pattern.get('night', 0)]
            
            colors = ['#f39c12', '#e74c3c', '#3498db']
            bars = ax.bar(times, values, color=colors, alpha=0.8, edgecolor='black')
            
            ax.set_ylabel('Average Flow')
            ax.set_title('Traffic Pattern by Time of Day')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'traffic_pattern.png'), dpi=150)
            plt.close()
            print(f"[Viz] Saved traffic_pattern.png")
        
        # Figure 4: 模型架构概览
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        summary_text = f"""
        ╔══════════════════════════════════════════════════════════════════════╗
        ║                    LWRGAT Model Summary                                ║
        ╠══════════════════════════════════════════════════════════════════════╣
        ║  Total Parameters:     {params.get('model_stats', {}).get('total_params', 'N/A'):>10,}                              ║
        ║  Trainable Parameters:  {params.get('model_stats', {}).get('trainable_params', 'N/A'):>10,}                              ║
        ║  Device:                {params.get('model_stats', {}).get('device', 'N/A'):>15}                              ║
        ╠══════════════════════════════════════════════════════════════════════╣
        ║  Graph Structure                                                     ║
        ║    - Num Nodes:        {params.get('graph_info', {}).get('num_nodes', 'N/A'):>10}                              ║
        ║    - Num Layers:       {params.get('graph_info', {}).get('num_layers', 'N/A'):>10}                              ║
        ║    - Has Physical Adj: {params.get('graph_info', {}).get('has_physical_adj', 'N/A'):>10}                              ║
        ╚══════════════════════════════════════════════════════════════════════╝
        """
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', horizontalalignment='center',
               fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.savefig(os.path.join(output_dir, 'model_summary.png'), dpi=150)
        plt.close()
        print(f"[Viz] Saved model_summary.png")
        
        return output_dir
    
    def save_report(self, output_dir):
        """保存分析报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'physical_params': self.get_physical_params(),
            'regime_distribution': self.results['regime_distribution'],
            'direction_ratio': self.results['direction_ratio'],
            'decouple_ratio': self.results['decouple_ratio']
        }
        
        report_path = os.path.join(output_dir, 'explainability_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[Report] Saved to {report_path}")
        return report_path


def main():
    parser = argparse.ArgumentParser(description='Explainability Analysis')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--config', type=str, required=True, help='YAML config file path')
    parser.add_argument('--output_dir', type=str, default='./logs/explainability', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--num_batches', type=int, default=20, help='Number of batches to analyze')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config_from_yaml(args.config)
    args_cfg = yaml_to_args(config)
    args_cfg.gpu = args.gpu
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # 构建模型
    print("Loading model...")
    from trainer import Exp_Long_Term_Forecast
    exp = Exp_Long_Term_Forecast(args_cfg)
    exp.model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    exp.model = exp.model.to(device)
    
    print(f"Model loaded from {args.checkpoint}")
    
    # 创建分析器
    analyzer = ExplainabilityAnalyzer(exp.model, args_cfg, device)
    
    # 打印物理参数
    analyzer.print_physical_params()
    
    # 获取数据
    print("\nLoading data...")
    from data_loader import data_provider
    _, train_loader = data_provider(args_cfg, 'train')
    _, test_loader = data_provider(args_cfg, 'test')
    
    # 分析训练数据模式
    print("\nAnalyzing training data patterns...")
    train_patterns = analyzer.analyze_data_patterns(train_loader, num_batches=args.num_batches)
    
    # 分析测试数据模式
    print("\nAnalyzing test data patterns...")
    test_patterns = analyzer.analyze_data_patterns(test_loader, num_batches=args.num_batches)
    
    # 生成可视化
    print("\nGenerating visualizations...")
    combined_patterns = {
        'flow_distribution': np.concatenate([train_patterns['flow_distribution'], test_patterns['flow_distribution']]),
        'v_distribution': np.concatenate([train_patterns['v_distribution'], test_patterns['v_distribution']]) if train_patterns.get('v_distribution') is not None else np.array([]),
        'timestamps': np.concatenate([train_patterns['timestamps'], test_patterns['timestamps']]) if train_patterns.get('timestamps') is not None else np.array([])
    }
    analyzer.generate_visualizations(combined_patterns, args.output_dir)
    
    # 保存报告
    analyzer.save_report(args.output_dir)
    
    print(f"\n[Done] Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
