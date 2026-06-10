from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from Holidiff.exp.common.metrics import compute_complex_state_metrics, dumps_pretty
from Holidiff.exp.exp2_forecasting.exp2_12h import Exp2Forecast12H


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'true', '1', 'yes', 'y', 'on'}:
        return True
    if value in {'false', '0', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_args(config_path: str, checkpoint_path: str, gpu: int, test_times: int | None):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['config'] = config_path
    cfg['load_checkpoint'] = checkpoint_path
    cfg['is_training'] = 0
    cfg['use_gpu'] = bool(torch.cuda.is_available())
    cfg['use_multi_gpu'] = False
    cfg['gpu'] = gpu
    if test_times is not None:
        cfg['test_times'] = int(test_times)
    cfg.setdefault('test_times', cfg.get('vs_times', cfg.get('sample_times', 1)))
    cfg.setdefault('train_val_aggregation_mode', cfg.get('aggregation_mode', 'simple'))
    cfg.setdefault('test_aggregation_mode', cfg.get('aggregation_mode', 'simple'))
    return SimpleNamespace(**cfg)


def collect_train_future(loader):
    futures = []
    for batch in loader:
        batch_y = batch[1].float()
        futures.append(batch_y[:, -batch_y.shape[1]:, :].numpy())
    return np.concatenate(futures, axis=0)


def run_inference(exp: Exp2Forecast12H, loader):
    preds, trues, histories, masks = [], [], [], []
    previous_aggregation_mode = exp._set_model_aggregation_mode(
        getattr(exp.args, 'test_aggregation_mode', getattr(exp.args, 'aggregation_mode', None))
    )
    exp.model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch[0], batch[1], batch[2], batch[3]
            batch_y_mask = batch[4] if len(batch) > 4 else None
            batch_holiday = batch[5] if len(batch) > 5 else None

            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float()

            if exp._is_tsdiff_model():
                core_model = exp._core_model()
                pred = core_model._sample_once(
                    batch_x,
                    int(getattr(exp.args, 'tsdiff_sampling_steps', 20)),
                    float(getattr(exp.args, 'tsdiff_guidance_scale', 1.0)),
                    float(getattr(exp.args, 'tsdiff_guidance_clip', 10.0)),
                ).detach().cpu().numpy()
                future_true = batch_y[:, -exp.args.pred_len:, :].numpy()
                preds.append(pred)
                trues.append(future_true)
                histories.append(batch_x.detach().cpu().numpy())
                if batch_y_mask is not None:
                    masks.append(batch_y_mask.detach().cpu().numpy())
                continue

            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)

            model_output = exp._run_model(
                exp.model,
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                sample_times=getattr(exp.args, 'test_times', exp.args.vs_times),
                holiday_flag=batch_holiday,
                future_target=batch_y[:, -exp.args.pred_len:, :].to(exp.device),
            )
            outputs = model_output[0] if exp.args.is_diff else model_output
            outputs = exp._process_model_output(outputs, is_diff=exp.args.is_diff)
            future_true = batch_y[:, -exp.args.pred_len:, :]

            preds.append(outputs.detach().cpu().numpy())
            trues.append(future_true.detach().cpu().numpy())
            histories.append(batch_x.detach().cpu().numpy())
            if batch_y_mask is not None:
                masks.append(batch_y_mask.detach().cpu().numpy())

    exp._restore_model_aggregation_mode(previous_aggregation_mode)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    histories = np.concatenate(histories, axis=0)
    valid_mask = np.concatenate(masks, axis=0) if masks else None
    return preds, trues, histories, valid_mask


def main():
    parser = argparse.ArgumentParser(description='Compute generic complex-state metrics from a checkpoint.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--test_times', type=int, default=None)
    parser.add_argument('--high_state_quantile', type=float, default=0.9)
    parser.add_argument('--peak_radius', type=int, default=1)
    parser.add_argument('--delta_t', type=float, default=15.0)
    parser.add_argument('--output_json', type=str, default='')
    args = parser.parse_args()

    runtime_args = build_args(args.config, args.checkpoint, args.gpu, args.test_times)
    set_seeds(int(getattr(runtime_args, 'seed', 2021)))

    exp = Exp2Forecast12H(runtime_args)
    train_data, train_loader = exp._get_data(flag='train')
    test_data, test_loader = exp._get_data(flag='test')
    exp._load_checkpoint_compat(args.checkpoint)

    train_future = collect_train_future(train_loader)
    preds, trues, histories, valid_mask = run_inference(exp, test_loader)
    metrics = compute_complex_state_metrics(
        y_pred=preds,
        y_true=trues,
        histories=histories,
        adj=test_data.adj.detach().cpu().numpy() if isinstance(test_data.adj, torch.Tensor) else test_data.adj,
        train_future=train_future,
        high_state_quantile=args.high_state_quantile,
        peak_radius=args.peak_radius,
        delta_t=args.delta_t,
        valid_mask=valid_mask,
    )

    payload = {
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'metrics': metrics,
    }
    print(dumps_pretty(payload))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[saved] {out_path}')


if __name__ == '__main__':
    main()
