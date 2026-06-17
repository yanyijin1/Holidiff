from Holidiff.data_provider.data_factory import data_provider
from Holidiff.data_provider.traffic_warehouse_loader import load_adj
from Holidiff.exp.exp_basic import Exp_Basic
from Holidiff.utils.tools import EarlyStopping, ReduceLROnPlateauWithWarmup, adjust_learning_rate
from Holidiff.utils.eval_holiday import evaluate_all
from Holidiff.utils.run_artifacts import checkpoint_dir, checkpoint_path, result_json_path, result_text_path, test_result_dir
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu, spearmanr
except Exception:
    mannwhitneyu = None
    spearmanr = None

warnings.filterwarnings('ignore')


class _TSDiffWinDataset(Dataset):
    def __init__(self, pack):
        self.x = pack['x']
        self.y = pack['y']
        self.m = pack['mask']
        self.h = pack['holiday']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i]).float()
        y = torch.from_numpy(self.y[i]).float()
        m = torch.from_numpy(self.m[i]).float()
        h = torch.from_numpy(self.h[i]).float()
        x_mark = torch.zeros(x.shape[0], 1)
        y_mark = torch.zeros(x.shape[0] + y.shape[0], 1)
        return x, y, x_mark, y_mark, m, h


class Exp2Forecast12H(Exp_Basic):

    # ---------------------------------------------------------------------
    # model / data entry
    # ---------------------------------------------------------------------
    def __init__(self, args):
        super(Exp2Forecast12H, self).__init__(args)

    def _is_tsdiff_model(self):
        return str(getattr(self.args, 'model', '')).lower() == 'tsdiff'

    def _is_diffusionts_model(self):
        return str(getattr(self.args, 'model', '')).lower() == 'diffusionts'

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _make_tsdiff_splits(self):
        csv_path = os.path.join(self.args.root_path, self.args.data_path)
        df = pd.read_csv(csv_path)
        time_col = getattr(self.args, 'time_col', 'time_slot')
        node_col = getattr(self.args, 'node_col', 'station_index')
        target_col = getattr(self.args, 'target_col', 'traffic_flow')
        holiday_col = getattr(self.args, 'holiday_col', 'is_holiday')
        df[time_col] = pd.to_datetime(df[time_col])
        if holiday_col not in df.columns:
            df[holiday_col] = 0.0
        df = df.sort_values([time_col, node_col]).reset_index(drop=True)
        piv = df.pivot(index=time_col, columns=node_col, values=target_col).sort_index()
        raw = piv.values.astype(np.float32)
        valid = (np.isfinite(raw) & (raw > 0)).astype(np.float32)
        raw = pd.DataFrame(raw).mask(~np.isfinite(raw) | (raw <= 0)).interpolate(axis=0, limit_direction='both').ffill().bfill().values.astype(np.float32)
        holiday = df.groupby(time_col).first().sort_index()[holiday_col].values.astype(np.float32)
        T, _ = raw.shape
        ctx, pred = int(self.args.seq_len), int(self.args.pred_len)
        tr = int(T * 0.7)
        va = int(T * 0.8)
        mean, std = raw[:tr].mean(0, keepdims=True), raw[:tr].std(0, keepdims=True)
        std[std < 1e-5] = 1.0
        data = ((raw - mean) / std).astype(np.float32) if getattr(self.args, 'scale', True) else raw
        starts = {'train': [], 'val': [], 'test': []}
        for s in range(T - ctx - pred + 1):
            e = s + ctx + pred
            if e <= tr:
                starts['train'].append(s)
            elif tr <= s and e <= va:
                starts['val'].append(s)
            elif va <= s:
                starts['test'].append(s)
        packs = {}
        for k, ss in starts.items():
            packs[k] = {'x': [], 'y': [], 'mask': [], 'holiday': [], 'indices': np.asarray(ss, dtype=np.int64)}
            for s in ss:
                e, p = s + ctx, s + ctx + pred
                packs[k]['x'].append(data[s:e])
                packs[k]['y'].append(data[e:p])
                packs[k]['mask'].append(valid[e:p])
                packs[k]['holiday'].append(holiday[e:p])
            for kk in ('x', 'y', 'mask', 'holiday'):
                packs[k][kk] = np.stack(packs[k][kk]).astype(np.float32) if packs[k][kk] else np.empty((0,), dtype=np.float32)
        return packs, mean, std, piv.columns.to_numpy()

    def _get_data(self, flag):
        if self._is_tsdiff_model():
            if not hasattr(self, '_tsdiff_cache'):
                packs, mean, std, stations = self._make_tsdiff_splits()
                self._tsdiff_cache = {
                    'packs': packs,
                    'mean': mean,
                    'std': std,
                    'stations': stations,
                }
            pack = self._tsdiff_cache['packs'][flag]
            data_set = _TSDiffWinDataset(pack)
            data_set.scale = bool(getattr(self.args, 'scale', True))
            data_set.scaler = type('Scaler', (), {'mean': self._tsdiff_cache['mean'], 'std': self._tsdiff_cache['std']})()
            data_set.indices = pack['indices'].tolist()
            data_set.adj = torch.from_numpy(load_adj(getattr(self.args, 'adj_path', os.path.join(self.args.root_path, 'adjacent_gantry.csv')), default_num_nodes=len(self._tsdiff_cache['stations']))).float()
            shuffle_flag = flag == 'train'
            data_loader = DataLoader(
                data_set,
                batch_size=self.args.batch_size,
                shuffle=shuffle_flag,
                num_workers=self.args.num_workers,
                drop_last=False,
            )
            print(flag, len(data_set))
            return data_set, data_loader
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader


    # ---------------------------------------------------------------------
    # train-time helpers
    # ---------------------------------------------------------------------
    def _select_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self._is_diffusionts_model():
            beta1 = float(getattr(self.args, 'diffusion_ts_beta1', 0.9))
            beta2 = float(getattr(self.args, 'diffusion_ts_beta2', 0.96))
            return optim.Adam(trainable_params, lr=self.args.learning_rate, betas=(beta1, beta2))
        model_optim = optim.Adam(trainable_params, lr=self.args.learning_rate)
        return model_optim

    @staticmethod
    def _cycle_loader(loader):
        while True:
            for batch in loader:
                yield batch

    def _build_diffusionts_scheduler(self, optimizer):
        return ReduceLROnPlateauWithWarmup(
            optimizer=optimizer,
            factor=float(getattr(self.args, 'diffusion_ts_scheduler_factor', 0.5)),
            patience=int(getattr(self.args, 'diffusion_ts_scheduler_patience', 4000)),
            min_lr=float(getattr(self.args, 'diffusion_ts_scheduler_min_lr', 1.0e-5)),
            threshold=float(getattr(self.args, 'diffusion_ts_scheduler_threshold', 1.0e-1)),
            threshold_mode=str(getattr(self.args, 'diffusion_ts_scheduler_threshold_mode', 'rel')),
            warmup_lr=float(getattr(self.args, 'diffusion_ts_warmup_lr', 8.0e-4)),
            warmup=int(getattr(self.args, 'diffusion_ts_warmup', 500)),
            verbose=bool(getattr(self.args, 'diffusion_ts_scheduler_verbose', False)),
        )

    def _diffusionts_train_num_steps(self, train_loader):
        grad_accum = max(1, int(getattr(self.args, 'diffusion_ts_gradient_accumulate_every', 2)))
        batches_per_epoch = max(1, len(train_loader))
        updates_per_epoch = int(np.ceil(batches_per_epoch / float(grad_accum)))
        return max(1, int(getattr(self.args, 'diffusion_ts_train_steps', self.args.train_epochs * updates_per_epoch)))

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAE':
            return nn.L1Loss()
        else:
            return nn.MSELoss()

    def _core_model(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _model_param_stats(self):
        core_model = self._core_model()
        total = sum(param.numel() for param in core_model.parameters())
        trainable = sum(param.numel() for param in core_model.parameters() if param.requires_grad)
        return {
            'total': int(total),
            'trainable': int(trainable),
            'frozen': int(total - trainable),
        }

    def _sync_cuda(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _masked_loss(self, outputs, target, mask, loss_type='MSE'):
        if mask is None:
            if loss_type == 'MSE':
                return F.mse_loss(outputs, target)
            return F.l1_loss(outputs, target)
        if loss_type == 'MSE':
            elem = F.mse_loss(outputs, target, reduction='none')
        else:
            elem = F.l1_loss(outputs, target, reduction='none')
        return (elem * mask).sum() / mask.sum().clamp(min=1)

    def _save_split_info(self, train_data, val_data, test_data):
        split_info = getattr(train_data, 'split_info', None)
        if split_info is None:
            return
        save_dir = str(getattr(self.args, 'save_dir', '') or '').strip()
        if not save_dir:
            return
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'split_info.json'), 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)

    def _normalize_model_name(self):
        model_name = str(getattr(self.args, 'model', '') or '')
        if model_name == 'HATEK':
            return 'RegDiff'
        return model_name

    def _dataset_identity(self):
        dataset_name = str(getattr(self.args, 'dataset_name', '') or getattr(self.args, 'model_id', '') or getattr(self.args, 'data', '')).strip()
        source = ''
        region = ''
        parts = dataset_name.split('-')
        if len(parts) >= 3 and parts[0].upper().startswith('PEMS'):
            source = parts[0]
            region = parts[-1]
        return dataset_name, source, region

    def _append_boundary_metrics_csv(self, payload):
        csv_path = os.path.join('/root/yanyijin/STdiff', 'results', 'pems_history_boundary_metrics.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        fieldnames = [
            'dataset', 'source', 'region', 'model', 'history_ratio', 'seed',
            'mae', 'mse', 'rmse', 'train_timesteps', 'val_timesteps', 'test_timesteps', 'save_dir'
        ]
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({key: payload.get(key, '') for key in fieldnames})

    def _save_run_metrics(self, basic_metrics, split_info):
        save_dir = str(getattr(self.args, 'save_dir', '') or '').strip()
        if not save_dir:
            return
        os.makedirs(save_dir, exist_ok=True)
        dataset_name, source, region = self._dataset_identity()
        metrics_payload = {
            'dataset': dataset_name,
            'source': source,
            'region': region,
            'model': self._normalize_model_name(),
            'history_ratio': float(getattr(self.args, 'history_ratio', getattr(self.args, 'train_ratio', 0.7))),
            'seed': int(getattr(self.args, 'seed', 2021)),
            'mae': float(basic_metrics['mae']),
            'mse': float(basic_metrics['mse']),
            'rmse': float(basic_metrics['rmse']),
            'train_timesteps': int(split_info.get('train_timesteps', 0)) if split_info else 0,
            'val_timesteps': int(split_info.get('val_timesteps', 0)) if split_info else 0,
            'test_timesteps': int(split_info.get('test_timesteps', 0)) if split_info else 0,
            'save_dir': save_dir,
        }
        with open(os.path.join(save_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
        self._append_boundary_metrics_csv(metrics_payload)


    # ---------------------------------------------------------------------
    # diffusion / sampling helpers
    # ---------------------------------------------------------------------
    def _run_model(self, model, batch_x, batch_x_mark, dec_inp, batch_y_mark, sample_times=None, holiday_flag=None, future_target=None):
        if self.args.is_diff:
            try:
                return model(
                    batch_x,
                    batch_x_mark,
                    dec_inp,
                    batch_y_mark,
                    sample_times=sample_times,
                    holiday_flag=holiday_flag,
                    future_target=future_target,
                )
            except TypeError:
                return model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        try:
            return model(batch_x, batch_x_mark, dec_inp, batch_y_mark, holiday_flag=holiday_flag)
        except TypeError:
            return model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def _process_model_output(self, outputs, is_diff=True):
        """处理模型输出，确保形状为 (B, pred_len, N)
        
        SimDiff在训练时返回 (B, N, pred_len)，评估时返回 (B, pred_len, N)
        """
        if is_diff:
            if outputs.shape[1] == self.args.enc_in and outputs.shape[2] == self.args.pred_len:
                outputs = outputs.transpose(1, 2)
        outputs = outputs[:, -self.args.pred_len:, :]
        return outputs


    # ---------------------------------------------------------------------
    # metrics / evaluation helpers
    # ---------------------------------------------------------------------
    def _safe_spearman(self, x, y):
        if spearmanr is None:
            return np.nan, np.nan
        res = spearmanr(x, y)
        return float(res.statistic), float(res.pvalue)

    def _safe_mwu(self, x, y, alternative='two-sided'):
        if mannwhitneyu is None or len(x) == 0 or len(y) == 0:
            return np.nan, np.nan
        res = mannwhitneyu(x, y, alternative=alternative)
        return float(res.statistic), float(res.pvalue)

    def _inverse_transform(self, dataset, arr):
        if not getattr(dataset, 'scale', False) or not hasattr(dataset, 'scaler'):
            return arr.copy()
        flat = arr.reshape(-1, arr.shape[-1])
        return dataset.scaler.inverse_transform(flat).reshape(arr.shape)

    def _load_checkpoint_compat(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location=self.device)
        model = self._core_model()
        current_state = model.state_dict()
        filtered_state = {}
        skipped_missing_shape = []

        for key, value in state.items():
            if key not in current_state:
                continue
            current_value = current_state[key]
            if not isinstance(value, torch.Tensor) or not isinstance(current_value, torch.Tensor):
                filtered_state[key] = value
                continue
            if current_value.shape != value.shape:
                skipped_missing_shape.append((key, tuple(value.shape), tuple(current_value.shape)))
                continue
            filtered_state[key] = value

        missing_keys = [k for k in current_state.keys() if k not in filtered_state]
        unexpected_keys = [k for k in state.keys() if k not in current_state]
        load_msg = model.load_state_dict(filtered_state, strict=False)

        print('[ckpt] loaded params: {}/{}'.format(len(filtered_state), len(current_state)))
        print('[ckpt] missing keys: {}'.format(len(load_msg.missing_keys)))
        print('[ckpt] unexpected keys ignored: {}'.format(len(unexpected_keys)))
        print('[ckpt] shape-mismatch keys ignored: {}'.format(len(skipped_missing_shape)))
        if load_msg.missing_keys:
            print('[ckpt] first missing keys:', load_msg.missing_keys[:10])
        if unexpected_keys:
            print('[ckpt] first unexpected keys:', unexpected_keys[:10])
        if skipped_missing_shape:
            print('[ckpt] first shape mismatch:', skipped_missing_shape[:10])

    def _load_fujian30_meta(self, dataset):
        csv_path = os.path.join(self.args.root_path, self.args.data_path)
        df = pd.read_csv(csv_path, usecols=['time_slot', 'station_index', 'traffic_flow', 'is_holiday'])
        df['time_slot'] = pd.to_datetime(df['time_slot'])
        df = df.sort_values(['time_slot', 'station_index']).reset_index(drop=True)
        pivot = df.pivot(index='time_slot', columns='station_index', values='traffic_flow').sort_index()
        meta = df.groupby('time_slot').first().sort_index()
        starts = np.asarray(dataset.indices, dtype=np.int64)
        pred_starts = starts + self.args.seq_len
        pred_ends = pred_starts + self.args.pred_len - 1
        return {
            'time_index': pivot.index,
            'station_ids': pivot.columns.to_numpy(),
            'holiday_flag': meta['is_holiday'].to_numpy(dtype=np.float32),
            'starts': starts,
            'pred_starts': pred_starts,
            'pred_ends': pred_ends,
        }

    def _is_pems_dataset(self):
        data_name = str(getattr(self.args, 'data', '') or '').lower()
        model_id = str(getattr(self.args, 'model_id', '') or '').lower()
        root_path = str(getattr(self.args, 'root_path', '') or '').lower()
        data_path = str(getattr(self.args, 'data_path', '') or '').lower()
        return any('pems' in value for value in (data_name, model_id, root_path, data_path))

    def _format_metric_value(self, value):
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, (float, np.floating)):
            return '{:.6f}'.format(float(value))
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    def _compute_extra_metrics(self, preds, trues, histories, holidays, masks, dataset):
        is_pems = self._is_pems_dataset()
        if holidays is None:
            if is_pems:
                holidays = np.zeros(preds.shape[:2], dtype=np.float32)
            else:
                return {}

        holiday_sum = np.nansum(holidays)
        if holiday_sum <= 0 and not is_pems:
            return {}

        adj = getattr(dataset, 'adj', None)
        if isinstance(adj, torch.Tensor):
            adj = adj.detach().cpu().numpy()

        extra_metrics = evaluate_all(
            y_pred=preds,
            y_true=trues,
            hist=histories,
            holiday=holidays,
            adj=adj,
            gamma=getattr(self.args, 'eval_gamma', 0.5),
            delta_t=getattr(self.args, 'eval_delta_t', 1.0),
            peak_radius=getattr(self.args, 'eval_peak_radius', 1),
            valid_mask=masks,
            bandwidth_scale=getattr(self.args, 'eval_bandwidth_scale', 0.5),
        )

        if is_pems:
            holiday_only_prefixes = ('Hol-', 'Reg-', 'HDR', 'CG-HDR', 'holiday_')
            holiday_only_keys = {'CG-F1', 'CG-CSI', 'TP', 'FP', 'FN'}
            extra_metrics = {
                k: v for k, v in extra_metrics.items()
                if not k.startswith(holiday_only_prefixes) and k not in holiday_only_keys
            }

        return extra_metrics

    def _print_and_write_metrics(self, handle, basic_metrics, extra_metrics):
        basic_text = 'mae: {mae}, mse: {mse}, rmse: {rmse}'.format(**basic_metrics)
        print(basic_text)
        handle.write(basic_text)
        handle.write('\n')
        for key in sorted(extra_metrics.keys()):
            text = '{}: {}'.format(key, self._format_metric_value(extra_metrics[key]))
            print(text)
            handle.write(text)
            handle.write('\n')

    def _set_model_aggregation_mode(self, mode):
        if mode is None:
            return None
        core_model = self._core_model()
        if not hasattr(core_model, 'aggregation_mode'):
            return None
        previous = core_model.aggregation_mode
        core_model.aggregation_mode = str(mode).lower()
        return previous

    def _restore_model_aggregation_mode(self, previous):
        if previous is None:
            return
        core_model = self._core_model()
        if hasattr(core_model, 'aggregation_mode'):
            core_model.aggregation_mode = previous

    def _effective_sample_times(self, requested_times, aggregation_mode):
        mode = str(aggregation_mode or '').lower()
        if mode == 'single':
            return 1
        return int(requested_times)


    # ---------------------------------------------------------------------
    # validation / training
    # ---------------------------------------------------------------------
    def vali(self, model, vali_loader, criterion):
        total_loss = []
        previous_aggregation_mode = self._set_model_aggregation_mode(
            getattr(self.args, 'train_val_aggregation_mode', 'single')
        )
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch[0], batch[1], batch[2], batch[3]
                batch_y_mask = batch[4] if len(batch) > 4 else None
                batch_holiday = batch[5] if len(batch) > 5 else None
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self._is_tsdiff_model():
                    core_model = self._core_model()
                    loss = core_model.training_loss(batch_x, batch_y.to(self.device))
                    total_loss.append(loss.item())
                    continue

                if self._is_diffusionts_model() and bool(getattr(self.args, 'diffusion_ts_validate_with_sampling', False)):
                    sample_times = self._effective_sample_times(
                        getattr(self.args, 'diffusion_ts_validation_sample_times', getattr(self.args, 'vs_times', self.args.sample_times)),
                        getattr(self.args, 'train_val_aggregation_mode', 'single'),
                    )
                    core_model = self._core_model()
                    outputs, _ = core_model(batch_x, sample_times=sample_times)
                    outputs = self._process_model_output(outputs, is_diff=True)
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()
                    pred_mask = batch_y_mask.detach().cpu() if batch_y_mask is not None else None
                    loss = self._masked_loss(pred, true, pred_mask, 'MAE')
                    total_loss.append(loss.item())
                    continue

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                core_model = self._core_model()
                if hasattr(core_model, 'training_loss'):
                    try:
                        loss = core_model.training_loss(
                            batch_x,
                            batch_y.to(self.device),
                            batch_y_mask=batch_y_mask.float().to(self.device) if batch_y_mask is not None else None,
                            x_mark_enc=batch_x_mark,
                            x_mark_dec=batch_y_mark,
                            holiday_flag=batch_holiday.float().to(self.device) if batch_holiday is not None else None,
                        )
                    except TypeError:
                        loss = core_model.training_loss(
                            batch_x,
                            batch_y.to(self.device),
                            batch_y_mask.float().to(self.device) if batch_y_mask is not None else None,
                        )
                    total_loss.append(loss.item())
                    continue

                model_output = self._run_model(
                    model,
                    batch_x,
                    batch_x_mark,
                    dec_inp,
                    batch_y_mark,
                    sample_times=self._effective_sample_times(getattr(self.args, 'vs_times', self.args.sample_times), getattr(self.args, 'train_val_aggregation_mode', 'single')),
                    holiday_flag=batch_holiday,
                    future_target=batch_y[:, -self.args.pred_len:, :].to(self.device),
                )
                outputs = model_output[0] if self.args.is_diff else model_output

                outputs = self._process_model_output(outputs, is_diff=self.args.is_diff)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                pred_mask = batch_y_mask.detach().cpu() if batch_y_mask is not None else None
                loss = self._masked_loss(pred, true, pred_mask, self.args.loss_type)
                total_loss.append(loss.item())

        model.train()
        self._restore_model_aggregation_mode(previous_aggregation_mode)
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        val_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        self._save_split_info(train_data, val_data, test_data)

        if self._is_diffusionts_model():
            path = str(checkpoint_dir(self.args.checkpoints, setting, save_dir=getattr(self.args, 'save_dir', '')))
            if not os.path.exists(path):
                os.makedirs(path)

            model_optim = self._select_optimizer()
            scheduler = self._build_diffusionts_scheduler(model_optim)
            grad_accum = max(1, int(getattr(self.args, 'diffusion_ts_gradient_accumulate_every', 2)))
            grad_clip = float(getattr(self.args, 'diffusion_ts_grad_clip', 1.0))
            log_interval = max(1, int(getattr(self.args, 'log_interval', 100)))
            train_num_steps = self._diffusionts_train_num_steps(train_loader)
            data_iter = self._cycle_loader(train_loader)
            train_started_at = time.time()
            self.model.train()
            model_optim.zero_grad()

            for step in range(train_num_steps):
                total_loss = 0.0
                core_model = self._core_model()
                for _ in range(grad_accum):
                    batch = next(data_iter)
                    batch_x = batch[0].float().to(self.device)
                    batch_y = batch[1].float().to(self.device)
                    loss = core_model.training_loss(batch_x, batch_y)
                    loss = loss / grad_accum
                    loss.backward()
                    total_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                model_optim.step()
                if hasattr(core_model, 'after_optimizer_step'):
                    core_model.after_optimizer_step()
                scheduler.step(total_loss)
                model_optim.zero_grad()

                if (step + 1) % log_interval == 0 or step == 0:
                    print('\tsteps: {0}/{1} | loss: {2:.7f} | lr: {3:.6e}'.format(
                        step + 1, train_num_steps, total_loss, model_optim.param_groups[0]['lr']), flush=True)

            torch.save(self.model.state_dict(), path + '/checkpoint.pth')
            print('DiffusionTS baseline-style training done, time: {:.2f}'.format(time.time() - train_started_at))
            return self.model

        if self._is_tsdiff_model():
            path = str(checkpoint_dir(self.args.checkpoints, setting, save_dir=getattr(self.args, 'save_dir', '')))
            if not os.path.exists(path):
                os.makedirs(path)

            train_steps = len(train_loader)
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
            model_optim = self._select_optimizer()

            for epoch in range(self.args.train_epochs):
                epoch_time = time.time()
                losses = []
                self.model.train()
                for i, batch in enumerate(train_loader):
                    batch_x, batch_y = batch[0].float().to(self.device), batch[1].float().to(self.device)
                    model_optim.zero_grad()
                    core_model = self._core_model()
                    loss = core_model.training_loss(batch_x, batch_y)
                    loss.backward()
                    model_optim.step()
                    losses.append(float(loss.detach().cpu()))
                    log_interval = max(1, int(getattr(self.args, 'log_interval', 100)))
                    if (i + 1) % log_interval == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, losses[-1]), flush=True)
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = float(np.mean(losses)) if losses else 0.0
                vali_loss = self.vali(self.model, vali_loader, None)
                test_loss = vali_loss
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            return self.model

        path = str(checkpoint_dir(self.args.checkpoints, setting, save_dir=getattr(self.args, 'save_dir', '')))
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss_type)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch[0], batch[1], batch[2], batch[3]
                batch_y_mask = batch[4].float().to(self.device) if len(batch) > 4 else None
                batch_holiday = batch[5].float().to(self.device) if len(batch) > 5 else None
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                core_model = self._core_model()
                if hasattr(core_model, 'training_loss'):
                    try:
                        loss = core_model.training_loss(
                            batch_x,
                            batch_y,
                            batch_y_mask=batch_y_mask,
                            x_mark_enc=batch_x_mark,
                            x_mark_dec=batch_y_mark,
                            holiday_flag=batch_holiday,
                        )
                    except TypeError:
                        loss = core_model.training_loss(batch_x, batch_y, batch_y_mask)
                    outputs = None
                else:
                    model_output = self._run_model(
                        self.model,
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark,
                        sample_times=self.args.sample_times,
                        holiday_flag=batch_holiday,
                        future_target=batch_y[:, -self.args.pred_len:, :],
                    )
                    outputs = model_output[0] if self.args.is_diff else model_output

                    outputs = self._process_model_output(outputs, is_diff=self.args.is_diff)
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                    loss = self._masked_loss(outputs, batch_y, batch_y_mask, self.args.loss_type)
                if epoch == 0 and i == 0 and batch_y_mask is not None:
                    print('\t[mask] valid ratio: {:.4f}, zero elements: {}/{} (first batch)'.format(
                        batch_y_mask.mean().item(), int((batch_y_mask == 0).sum().item()), batch_y_mask.numel()))
                train_loss.append(loss.item())

                log_interval = max(1, int(getattr(self.args, 'log_interval', 100)))
                if (i + 1) % log_interval == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()), flush=True)
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time), flush=True)
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                physical_module = getattr(self._core_model(), 'physical_injection', None)
                if physical_module is not None and hasattr(physical_module, 'record_eta_gradients'):
                    physical_module.record_eta_gradients()
                model_optim.step()
                if hasattr(core_model, 'after_optimizer_step'):
                    core_model.after_optimizer_step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(self.model, vali_loader, criterion)
            test_loss = vali_loss
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model


    # ---------------------------------------------------------------------
    # test / sampling
    # ---------------------------------------------------------------------
    def test(self, setting, test=0):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if self._is_tsdiff_model():
            if test:
                if getattr(self.args, 'load_checkpoint', ''):
                    checkpoint_path_value = self.args.load_checkpoint
                else:
                    checkpoint_path_value = str(checkpoint_path(self.args.checkpoints, setting, save_dir=getattr(self.args, 'save_dir', '')))
                print('loading model from', checkpoint_path_value)
                self._load_checkpoint_compat(checkpoint_path_value)

            self.model.eval()
            preds = []
            trues = []
            all_masks = []
            all_histories = []
            all_holidays = []
            infer_total_seconds = 0.0
            infer_num_batches = 0
            param_stats = self._model_param_stats()
            print('[params] total={total} trainable={trainable} frozen={frozen}'.format(**param_stats))

            with torch.no_grad():
                for batch in test_loader:
                    batch_x = batch[0].float().to(self.device)
                    batch_y = batch[1].float()
                    batch_y_mask = batch[4] if len(batch) > 4 else None
                    batch_holiday = batch[5] if len(batch) > 5 else None
                    core_model = self._core_model()
                    self._sync_cuda()
                    infer_started_at = time.perf_counter()
                    pred = core_model._sample_once(
                        batch_x,
                        int(getattr(self.args, 'tsdiff_sampling_steps', 20)),
                        float(getattr(self.args, 'tsdiff_guidance_scale', 1.0)),
                        float(getattr(self.args, 'tsdiff_guidance_clip', 10.0)),
                    )
                    self._sync_cuda()
                    infer_total_seconds += time.perf_counter() - infer_started_at
                    infer_num_batches += 1
                    pred = pred.detach().cpu().numpy()
                    true = batch_y[:, -self.args.pred_len:, :].numpy()
                    preds.append(pred)
                    trues.append(true)
                    all_histories.append(batch_x.detach().cpu().numpy())
                    if batch_y_mask is not None:
                        all_masks.append(batch_y_mask.detach().cpu().numpy())
                    if batch_holiday is not None:
                        all_holidays.append(batch_holiday.detach().cpu().numpy())

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            masks = np.concatenate(all_masks, axis=0) if all_masks else None
            holidays = np.concatenate(all_holidays, axis=0) if all_holidays else None
            histories = np.concatenate(all_histories, axis=0)
            print('test shape:', preds.shape, trues.shape)

            if masks is not None:
                mae = (np.abs(preds - trues) * masks).sum() / max(masks.sum(), 1)
                mse = ((preds - trues) ** 2 * masks).sum() / max(masks.sum(), 1)
            else:
                mae = np.mean(np.abs(preds - trues))
                mse = np.mean((preds - trues) ** 2)
            rmse = np.sqrt(mse)
            basic_metrics = {'mae': mae, 'mse': mse, 'rmse': rmse}
            extra_metrics = self._compute_extra_metrics(preds, trues, histories, holidays, masks, test_data)
            timing_metrics = {
                'test_infer_total_seconds': float(infer_total_seconds),
                'test_infer_avg_batch_seconds': float(infer_total_seconds / max(infer_num_batches, 1)),
                'test_num_batches': int(infer_num_batches),
            }
            print('[timing] test_infer_total_seconds={test_infer_total_seconds:.6f}'.format(**timing_metrics))
            print('[timing] test_infer_avg_batch_seconds={test_infer_avg_batch_seconds:.6f}'.format(**timing_metrics))

            results_txt_path = result_text_path(setting)
            with open(results_txt_path, 'a', encoding='utf-8') as f:
                f.write(setting + "  \n")
                self._print_and_write_metrics(f, basic_metrics, extra_metrics)
                for key, value in param_stats.items():
                    line = f'param_{key}: {value}'
                    print(line)
                    f.write(line + '\n')
                for key, value in timing_metrics.items():
                    line = f'{key}: {value}'
                    print(line)
                    f.write(line + '\n')
                f.write('\n')

            metrics_path = result_json_path(setting)
            metrics_payload = {
                'setting': setting,
                'basic': {k: float(v) for k, v in basic_metrics.items()},
                'extra': extra_metrics,
                'params': param_stats,
                'timing': timing_metrics,
            }
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

            self._save_run_metrics(basic_metrics, getattr(test_data, 'split_info', None))
            return mae, mse, rmse

        if test:
            if getattr(self.args, 'load_checkpoint', ''):
                checkpoint_path_value = self.args.load_checkpoint
            else:
                checkpoint_path_value = str(checkpoint_path(self.args.checkpoints, setting, save_dir=getattr(self.args, 'save_dir', '')))
            print('loading model from', checkpoint_path_value)
            self._load_checkpoint_compat(checkpoint_path_value)

        core_model = self._core_model()
        if hasattr(core_model, '_mom_kwargs') and hasattr(test_data, 'scaler') and hasattr(test_data.scaler, 'mean'):
            core_model._mom_kwargs['scaler_mean'] = torch.tensor(test_data.scaler.mean, dtype=torch.float32).squeeze(0)
            core_model._mom_kwargs['scaler_std'] = torch.tensor(test_data.scaler.std, dtype=torch.float32).squeeze(0)
        self.model.eval()
        previous_aggregation_mode = self._set_model_aggregation_mode(
            getattr(self.args, 'test_aggregation_mode', 'dca')
        )

        preds = []
        trues = []
        all_marks = []
        all_masks = []
        all_histories = []
        all_holidays = []
        folder_path = str(test_result_dir(setting))
        diag_path = os.path.join(folder_path, 'diagnostics')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(diag_path):
            os.makedirs(diag_path)

        infer_total_seconds = 0.0
        infer_num_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch[0], batch[1], batch[2], batch[3]
                batch_y_mask = batch[4] if len(batch) > 4 else None
                batch_holiday = batch[5] if len(batch) > 5 else None
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                all_marks.append(batch_y_mark.detach().cpu().numpy())

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                self._sync_cuda()
                infer_started_at = time.perf_counter()
                model_output = self._run_model(
                    self.model,
                    batch_x,
                    batch_x_mark,
                    dec_inp,
                    batch_y_mark,
                    sample_times=self._effective_sample_times(getattr(self.args, 'test_times', self.args.vs_times), getattr(self.args, 'test_aggregation_mode', 'dca')),
                    holiday_flag=batch_holiday,
                    future_target=batch_y[:, -self.args.pred_len:, :].to(self.device),
                )
                self._sync_cuda()
                infer_total_seconds += time.perf_counter() - infer_started_at
                infer_num_batches += 1
                outputs = model_output[0] if self.args.is_diff else model_output

                outputs = self._process_model_output(outputs, is_diff=self.args.is_diff)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                all_histories.append(batch_x.detach().cpu().numpy())
                if batch_y_mask is not None:
                    all_masks.append(batch_y_mask.detach().cpu().numpy())
                if batch_holiday is not None:
                    all_holidays.append(batch_holiday.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        masks = np.concatenate(all_masks, axis=0) if all_masks else None
        holidays = np.concatenate(all_holidays, axis=0) if all_holidays else None
        marks = np.concatenate(all_marks, axis=0) if all_marks else np.empty((0, self.args.label_len + self.args.pred_len, 1))
        histories = np.concatenate(all_histories, axis=0)
        print('test shape:', preds.shape, trues.shape)

        if masks is not None:
            mae = (np.abs(preds - trues) * masks).sum() / max(masks.sum(), 1)
            mse = ((preds - trues) ** 2 * masks).sum() / max(masks.sum(), 1)
        else:
            mae = np.mean(np.abs(preds - trues))
            mse = np.mean((preds - trues) ** 2)
        rmse = np.sqrt(mse)
        basic_metrics = {'mae': mae, 'mse': mse, 'rmse': rmse}
        extra_metrics = self._compute_extra_metrics(preds, trues, histories, holidays, masks, test_data)
        timing_metrics = {
            'test_infer_total_seconds': float(infer_total_seconds),
            'test_infer_avg_batch_seconds': float(infer_total_seconds / max(infer_num_batches, 1)),
            'test_num_batches': int(infer_num_batches),
        }
        param_stats = self._model_param_stats()
        print('[params] total={total} trainable={trainable} frozen={frozen}'.format(**param_stats))
        print('[timing] test_infer_total_seconds={test_infer_total_seconds:.6f}'.format(**timing_metrics))
        print('[timing] test_infer_avg_batch_seconds={test_infer_avg_batch_seconds:.6f}'.format(**timing_metrics))

        results_txt_path = result_text_path(setting)

        with open(results_txt_path, 'a', encoding='utf-8') as f:
            f.write(setting + "  \n")
            self._print_and_write_metrics(f, basic_metrics, extra_metrics)
            for key, value in param_stats.items():
                line = f'param_{key}: {value}'
                print(line)
                f.write(line + '\n')
            for key, value in timing_metrics.items():
                line = f'{key}: {value}'
                print(line)
                f.write(line + '\n')
            f.write('\n')

        metrics_path = result_json_path(setting)
        metrics_payload = {
            'setting': setting,
            'basic': {k: float(v) for k, v in basic_metrics.items()},
            'extra': extra_metrics,
            'params': param_stats,
            'timing': timing_metrics,
        }
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

        self._restore_model_aggregation_mode(previous_aggregation_mode)
        self._save_run_metrics(basic_metrics, getattr(test_data, 'split_info', None))
        return mae, mse, rmse


Exp_Long_Term_Forecast = Exp2Forecast12H
