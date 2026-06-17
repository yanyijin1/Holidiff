from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from Holidiff.exp import Exp_Long_Term_Forecast

try:
    import yaml
except Exception as exc:
    yaml = None
    _yaml_import_error = exc

DEFAULT_CONFIG = ROOT / 'configs' / 'fujian30' / 'holidiff_h12.yaml'
DEFAULT_CHECKPOINT = Path('/root/autodl-tmp/STdiff_runs/checkpoints/fujian30/final_holidiff__h12/checkpoint.pth')
DEFAULT_OUT_DIR = Path('/root/autodl-tmp/STdiff_runs/exp7_sensitivity/fujian30_h12')


class Exp7SensitivityExperiment:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _set_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        if yaml is None:
            raise RuntimeError(f'PyYAML import failed: {_yaml_import_error}')
        with open(path, 'r', encoding='utf-8') as f:
            cfg = dict(yaml.safe_load(f))
        cfg['use_gpu'] = bool(cfg.get('use_gpu', True) and torch.cuda.is_available())
        cfg['train_val_aggregation_mode'] = cfg.get('train_val_aggregation_mode', 'single')
        cfg['test_aggregation_mode'] = cfg.get('test_aggregation_mode', 'dca')
        cfg['test_times'] = int(cfg.get('test_times', cfg.get('vs_times', cfg.get('sample_times', 1))))
        if cfg.get('use_multi_gpu', False):
            devices = str(cfg.get('devices', '0')).replace(' ', '')
            cfg['devices'] = devices
            cfg['device_ids'] = [int(x) for x in devices.split(',') if x]
            cfg['gpu'] = cfg['device_ids'][0]
        return cfg

    def _build_exp_args(self, k: int, seed: int) -> SimpleNamespace:
        cfg = self._load_yaml(Path(self.args.config))
        cfg.update({
            'config': str(self.args.config),
            'version': f'exp7_k{k}_seed{seed}',
            'load_checkpoint': str(self.args.checkpoint),
            'is_training': 0,
            'seed': int(seed),
            'test_times': int(k),
            'sample_times': int(k),
            'vs_times': int(k),
            'train_val_aggregation_mode': str(self.args.train_val_aggregation_mode),
            'test_aggregation_mode': str(self.args.test_aggregation_mode),
        })
        if self.args.batch_size is not None:
            cfg['batch_size'] = int(self.args.batch_size)
            cfg['eval_batch_size'] = int(self.args.batch_size)
        if self.args.num_workers is not None:
            cfg['num_workers'] = int(self.args.num_workers)
        return SimpleNamespace(**cfg)

    def _existing_metric_rows(self) -> list[dict]:
        rows = []
        for path in sorted(self.output_dir.glob('k_*_rep*_metrics.json')):
            try:
                rows.append(json.loads(path.read_text(encoding='utf-8')))
            except Exception as exc:
                print(f'[exp7] skip unreadable metrics file {path}: {exc}', flush=True)
        return rows

    def evaluate_once(self, k: int, seed: int) -> dict:
        self._set_seeds(seed)
        exp_args = self._build_exp_args(k, seed)
        exp = Exp_Long_Term_Forecast(exp_args)
        test_data, test_loader = exp._get_data('test')

        ckpt = Path(self.args.checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt}')

        exp._load_checkpoint_compat(str(ckpt))
        exp.model.eval()
        previous_mode = exp._set_model_aggregation_mode(getattr(exp.args, 'test_aggregation_mode', 'dca'))

        preds = []
        trues = []
        exp._sync_cuda()
        started = time.perf_counter()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader, start=1):
                if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == len(test_loader):
                    print(f'[exp7] K={k} batch {batch_idx}/{len(test_loader)}', flush=True)
                batch_x = batch[0].float().to(exp.device)
                batch_y = batch[1].float()
                batch_x_mark = batch[2].float().to(exp.device)
                batch_y_mark = batch[3].float().to(exp.device)
                batch_holiday = batch[5] if len(batch) > 5 else None

                dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)

                outputs = exp._run_model(
                    exp.model,
                    batch_x,
                    batch_x_mark,
                    dec_inp,
                    batch_y_mark,
                    sample_times=int(k),
                    holiday_flag=batch_holiday,
                    future_target=batch_y[:, -exp.args.pred_len:, :].to(exp.device),
                )
                outputs = outputs[0] if exp.args.is_diff else outputs
                outputs = exp._process_model_output(outputs, is_diff=exp.args.is_diff)
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y[:, -exp.args.pred_len:, :].detach().cpu().numpy())

        exp._sync_cuda()
        elapsed = time.perf_counter() - started
        exp._restore_model_aggregation_mode(previous_mode)

        pred_arr = np.concatenate(preds, axis=0)
        true_arr = np.concatenate(trues, axis=0)
        pred_raw = exp._inverse_transform(test_data, pred_arr)
        true_raw = exp._inverse_transform(test_data, true_arr)

        mae = float(np.mean(np.abs(pred_arr - true_arr)))
        mse = float(np.mean((pred_arr - true_arr) ** 2))
        rmse = float(np.sqrt(mse))

        mae_raw = float(np.mean(np.abs(pred_raw - true_raw)))
        mse_raw = float(np.mean((pred_raw - true_raw) ** 2))
        rmse_raw = float(np.sqrt(mse_raw))

        return {
            'K': int(k),
            'seed': int(seed),
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAE_raw': mae_raw,
            'MSE_raw': mse_raw,
            'RMSE_raw': rmse_raw,
            'metric_space': 'normalized_like_exp2_test',
            'inference_time_seconds': float(elapsed),
            'avg_time_per_batch': float(elapsed / max(len(test_loader), 1)),
            'num_test_batches': int(len(test_loader)),
            'num_test_samples': int(true_arr.shape[0]),
            'checkpoint': str(self.args.checkpoint),
            'config': str(self.args.config),
            'test_aggregation_mode': str(self.args.test_aggregation_mode),
        }

    def _write_tables(self, rows: list[dict]) -> Path:
        raw_df = pd.DataFrame(rows).sort_values(['K', 'repeat', 'seed']).reset_index(drop=True)
        raw_path = self.output_dir / 'k_sensitivity_raw.csv'
        raw_df.to_csv(raw_path, index=False)

        if int(self.args.repeat) > 1:
            summary = raw_df.groupby('K', as_index=False).agg(
                MAE_mean=('MAE', 'mean'),
                MAE_std=('MAE', 'std'),
                MSE_mean=('MSE', 'mean'),
                MSE_std=('MSE', 'std'),
                RMSE_mean=('RMSE', 'mean'),
                RMSE_std=('RMSE', 'std'),
                MAE_raw_mean=('MAE_raw', 'mean'),
                MAE_raw_std=('MAE_raw', 'std'),
                RMSE_raw_mean=('RMSE_raw', 'mean'),
                RMSE_raw_std=('RMSE_raw', 'std'),
                time_mean=('inference_time_seconds', 'mean'),
                time_std=('inference_time_seconds', 'std'),
                avg_time_per_batch_mean=('avg_time_per_batch', 'mean'),
                avg_time_per_batch_std=('avg_time_per_batch', 'std'),
                num_test_batches=('num_test_batches', 'max'),
                num_test_samples=('num_test_samples', 'max'),
            )
        else:
            keep_cols = [
                'K', 'MAE', 'MSE', 'RMSE', 'MAE_raw', 'MSE_raw', 'RMSE_raw',
                'inference_time_seconds', 'avg_time_per_batch', 'num_test_batches', 'num_test_samples'
            ]
            summary = raw_df[keep_cols].copy()

        summary_path = self.output_dir / 'k_sensitivity_summary.csv'
        summary.to_csv(summary_path, index=False)

        meta = {
            'config': str(self.args.config),
            'checkpoint': str(self.args.checkpoint),
            'k_list': [int(k) for k in self.args.k_list],
            'repeat': int(self.args.repeat),
            'seed': int(self.args.seed),
            'train_val_aggregation_mode': str(self.args.train_val_aggregation_mode),
            'test_aggregation_mode': str(self.args.test_aggregation_mode),
            'metric_space': 'normalized_like_exp2_test',
            'also_saved_raw_space_metrics': True,
        }
        with open(self.output_dir / 'run_meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f'[exp7] Saved raw results to: {raw_path}', flush=True)
        print(f'[exp7] Saved summary to: {summary_path}', flush=True)
        return summary_path

    def run(self) -> Path:
        rows = self._existing_metric_rows()
        seen = {(int(r['K']), int(r.get('repeat', 0))) for r in rows if 'K' in r}
        if rows:
            print(f'[exp7] Loaded {len(rows)} existing metric files for resume', flush=True)

        for k in self.args.k_list:
            for rep in range(int(self.args.repeat)):
                key = (int(k), int(rep))
                if key in seen:
                    print(f'[exp7] Skip existing K={k}, repeat={rep}', flush=True)
                    continue
                seed = int(self.args.seed) + rep
                print(f'[exp7] Evaluating K={k}, repeat={rep}, seed={seed}', flush=True)
                result = self.evaluate_once(int(k), seed)
                result['repeat'] = int(rep)
                rows.append(result)
                seen.add(key)
                metric_path = self.output_dir / f'k_{int(k)}_rep{int(rep)}_metrics.json'
                with open(metric_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(json.dumps(result, ensure_ascii=False), flush=True)

        return self._write_tables(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='HoliDiff K-sensitivity evaluation')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument('--checkpoint', type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument('--k_list', type=int, nargs='+', default=[1, 5, 10, 15, 20, 30, 40, 50])
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_val_aggregation_mode', type=str, default='single')
    parser.add_argument('--test_aggregation_mode', type=str, default='dca')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    Exp7SensitivityExperiment(args).run()


if __name__ == '__main__':
    main()