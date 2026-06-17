import argparse
import os
import random
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from Holidiff.exp import Exp_Long_Term_Forecast
from Holidiff.utils.print_args import print_args
from Holidiff.utils.run_artifacts import build_run_name, log_path

try:
    import yaml
except Exception as exc:  # pragma: no cover
    yaml = None
    _yaml_import_error = exc


NETWORK_PROFILES = {
    'tiny': {
        'd_model': 64,
        'd_ff': 128,
        'n_heads': 4,
        'num_heads': 4,
        'e_layers': 1,
        'd_layers': 1,
    },
    'small': {
        'd_model': 96,
        'd_ff': 256,
        'n_heads': 4,
        'num_heads': 4,
        'e_layers': 1,
        'd_layers': 1,
    },
    'base': {
        'd_model': 128,
        'd_ff': 512,
        'n_heads': 8,
        'num_heads': 8,
        'e_layers': 1,
        'd_layers': 1,
    },
    'large': {
        'd_model': 192,
        'd_ff': 768,
        'n_heads': 8,
        'num_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
    },
}


CONFIG_ALIAS_PAIRS = [
    ('physical_residual_free_flow_epsilon', 'physical_free_flow_epsilon'),
]


def _apply_config_aliases(cfg):
    normalized = dict(cfg)
    for new_key, old_key in CONFIG_ALIAS_PAIRS:
        if new_key in normalized and old_key not in normalized:
            normalized[old_key] = normalized[new_key]
        elif old_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized[old_key]
    return normalized


def _apply_network_profile(cfg):
    normalized = dict(cfg)
    profile_name = str(normalized.get('network_profile', '') or '').strip().lower()
    if not profile_name:
        return normalized
    if profile_name not in NETWORK_PROFILES:
        valid = ', '.join(sorted(NETWORK_PROFILES.keys()))
        raise ValueError(f"Unknown network_profile='{profile_name}'. Available profiles: {valid}")
    for key, value in NETWORK_PROFILES[profile_name].items():
        normalized.setdefault(key, value)
    return normalized


def _load_yaml(path):
    if yaml is None:
        raise RuntimeError(f'PyYAML is required but could not be imported: {_yaml_import_error}')
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = _apply_config_aliases(cfg)
    cfg = _apply_network_profile(cfg)
    return cfg


def _build_parser():
    parser = argparse.ArgumentParser(description='HoliDiff YAML runner')
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config')
    parser.add_argument('--version', type=str, default='', help='Short experiment version used in checkpoint/result naming')
    parser.add_argument('--load_checkpoint', type=str, default='', help='Optional explicit checkpoint path for test-only runs')
    parser.add_argument(
        '--history_ratio',
        type=float,
        default=0.70,
        help=(
            'Continuous training history ratio over the full time series. '
            'For a 7:1:2 split, this must be in (0, 0.70]. '
            'Train uses the tail segment [0.7T-history_ratio*T, 0.7T). '
            'Validation and test remain fixed.'
        ),
    )
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Chronological train split ratio over the full time series.')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Chronological validation split ratio over the full time series.')
    parser.add_argument('--save_dir', type=str, default='', help='Optional explicit run directory for checkpoints and saved metrics.')
    parser.add_argument('--dataset_name', type=str, default='', help='Optional dataset display name used in saved metrics.')
    return parser


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'true', '1', 'yes', 'y', 'on'}:
        return True
    if value in {'false', '0', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


CLI_OVERRIDE_KEYS = {
    'config',
    'version',
    'load_checkpoint',
    'history_ratio',
    'train_ratio',
    'val_ratio',
    'save_dir',
    'dataset_name',
    'eval_gamma',
    'eval_delta_t',
    'eval_peak_radius',
    'eval_bandwidth_scale',
    'train_val_aggregation_mode',
    'test_aggregation_mode',
    'test_times',
}


def _merge_args_from_yaml(parser, cfg):
    for k, v in cfg.items():
        if k in CLI_OVERRIDE_KEYS:
            continue
        arg_name = f'--{k}'
        if any(a.option_strings and arg_name in a.option_strings for a in parser._actions):
            continue
        if isinstance(v, bool):
            parser.add_argument(arg_name, type=_str2bool, default=v)
        else:
            parser.add_argument(arg_name, type=type(v), default=v)


def _set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self.streams)

    def reconfigure(self, **kwargs):
        for stream in self.streams:
            reconfigure = getattr(stream, 'reconfigure', None)
            if callable(reconfigure):
                reconfigure(**kwargs)


if __name__ == '__main__':
    base_parser = _build_parser()
    base_args, _ = base_parser.parse_known_args()
    cfg = _load_yaml(base_args.config)
    if base_args.version:
        cfg['version'] = base_args.version
    if base_args.load_checkpoint:
        cfg['load_checkpoint'] = base_args.load_checkpoint

    parser = argparse.ArgumentParser(description='HoliDiff YAML runner', add_help=False)
    _merge_args_from_yaml(parser, cfg)
    parser.add_argument('--config', type=str, default=base_args.config)
    parser.add_argument('--version', type=str, default=cfg.get('version', base_args.version))
    parser.add_argument('--load_checkpoint', type=str, default=cfg.get('load_checkpoint', base_args.load_checkpoint))
    parser.add_argument('--history_ratio', type=float, default=cfg.get('history_ratio', getattr(base_args, 'history_ratio', 0.70)))
    parser.add_argument('--train_ratio', type=float, default=cfg.get('train_ratio', getattr(base_args, 'train_ratio', 0.7)))
    parser.add_argument('--val_ratio', type=float, default=cfg.get('val_ratio', getattr(base_args, 'val_ratio', 0.1)))
    parser.add_argument('--save_dir', type=str, default=cfg.get('save_dir', getattr(base_args, 'save_dir', '')))
    parser.add_argument('--dataset_name', type=str, default=cfg.get('dataset_name', getattr(base_args, 'dataset_name', '')))
    parser.add_argument('--train_val_aggregation_mode', type=str, default=cfg.get('train_val_aggregation_mode', 'single'))
    parser.add_argument('--test_aggregation_mode', type=str, default=cfg.get('test_aggregation_mode', 'dca'))
    parser.add_argument('--test_times', type=int, default=cfg.get('test_times', cfg.get('vs_times', cfg.get('sample_times', 1))))
    parser.add_argument('--eval_gamma', type=float, default=cfg.get('eval_gamma', 0.5))
    parser.add_argument('--eval_delta_t', type=float, default=cfg.get('eval_delta_t', 1.0))
    parser.add_argument('--eval_peak_radius', type=int, default=cfg.get('eval_peak_radius', 1))
    parser.add_argument('--eval_bandwidth_scale', type=float, default=cfg.get('eval_bandwidth_scale', 0.5))
    args = parser.parse_args()

    for k, v in cfg.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    _set_seeds(getattr(args, 'seed', 2021))
    args.use_gpu = True if torch.cuda.is_available() else False

    if getattr(args, 'use_gpu', False) and getattr(args, 'use_multi_gpu', False):
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print(torch.cuda.is_available(), flush=True)
    print('Args in experiment:', flush=True)
    print_args(args)

    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            run_name = build_run_name(args, ii)
            current_log_path = log_path(run_name)

            with open(current_log_path, 'a', encoding='utf-8') as log_file:
                original_stdout, original_stderr = sys.stdout, sys.stderr
                sys.stdout = _TeeStream(original_stdout, log_file)
                sys.stderr = _TeeStream(original_stderr, log_file)
                run_started_at = time.time()
                try:
                    print(f'[run] version={run_name}', flush=True)
                    print(f'[run] auto log path: {current_log_path}', flush=True)
                    print(f'[run] started_at={time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_started_at))}', flush=True)
                    exp = Exp(args)
                    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(run_name), flush=True)
                    train_started_at = time.time()
                    exp.train(run_name)
                    print(f'[timing] train_seconds={time.time() - train_started_at:.3f}', flush=True)

                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(run_name), flush=True)
                    test_started_at = time.time()
                    exp.test(run_name)
                    print(f'[timing] test_seconds={time.time() - test_started_at:.3f}', flush=True)
                    print(f'[timing] total_run_seconds={time.time() - run_started_at:.3f}', flush=True)
                    print(f'[run] finished_at={time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}', flush=True)
                finally:
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
            torch.cuda.empty_cache()
    else:
        ii = 0
        run_name = build_run_name(args, ii)
        current_log_path = log_path(run_name)

        with open(current_log_path, 'a', encoding='utf-8') as log_file:
            original_stdout, original_stderr = sys.stdout, sys.stderr
            sys.stdout = _TeeStream(original_stdout, log_file)
            sys.stderr = _TeeStream(original_stderr, log_file)
            run_started_at = time.time()
            try:
                print(f'[run] version={run_name}', flush=True)
                print(f'[run] auto log path: {current_log_path}', flush=True)
                print(f'[run] started_at={time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_started_at))}', flush=True)
                exp = Exp(args)
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(run_name), flush=True)
                test_started_at = time.time()
                exp.test(run_name, test=1)
                print(f'[timing] test_seconds={time.time() - test_started_at:.3f}', flush=True)
                print(f'[timing] total_run_seconds={time.time() - run_started_at:.3f}', flush=True)
                print(f'[run] finished_at={time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}', flush=True)
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
        torch.cuda.empty_cache()
