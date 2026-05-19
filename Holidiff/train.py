import argparse
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from Holidiff.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from Holidiff.exp.exp_imputation import Exp_Imputation
from Holidiff.exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from Holidiff.exp.exp_anomaly_detection import Exp_Anomaly_Detection
from Holidiff.exp.exp_classification import Exp_Classification
from Holidiff.utils.print_args import print_args

try:
    import yaml
except Exception as exc:  # pragma: no cover
    yaml = None
    _yaml_import_error = exc


def _load_yaml(path):
    if yaml is None:
        raise RuntimeError(f'PyYAML is required but could not be imported: {_yaml_import_error}')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _build_parser():
    parser = argparse.ArgumentParser(description='HoliDiff YAML runner')
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config')
    parser.add_argument('--version', type=str, default='', help='Short experiment version used in checkpoint/result naming')
    parser.add_argument('--load_checkpoint', type=str, default='', help='Optional explicit checkpoint path for test-only runs')
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


def _merge_args_from_yaml(parser, cfg):
    for k, v in cfg.items():
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


def _build_setting(args, ii):
    run_tag = args.version if getattr(args, 'version', '') else args.des
    return '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        run_tag, ii)


if __name__ == '__main__':
    base_parser = _build_parser()
    base_args, _ = base_parser.parse_known_args()
    cfg = _load_yaml(base_args.config)

    parser = argparse.ArgumentParser(description='HoliDiff YAML runner', add_help=False)
    _merge_args_from_yaml(parser, cfg)
    parser.add_argument('--config', type=str, default=base_args.config)
    parser.add_argument('--version', type=str, default=base_args.version)
    parser.add_argument('--load_checkpoint', type=str, default=base_args.load_checkpoint)
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

    print(torch.cuda.is_available())
    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = _build_setting(args, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        exp = Exp(args)
        setting = _build_setting(args, ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
