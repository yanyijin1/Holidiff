from Holidiff.data_provider.data_factory import data_provider
from Holidiff.exp.exp_basic import Exp_Basic
from Holidiff.utils.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
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


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        model_optim = optim.Adam(trainable_params, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAE':
            return nn.L1Loss()
        else:
            return nn.MSELoss()

    def _process_model_output(self, outputs, is_diff=True):
        """处理模型输出，确保形状为 (B, pred_len, N)
        
        SimDiff在训练时返回 (B, N, pred_len)，评估时返回 (B, pred_len, N)
        """
        if is_diff:
            if outputs.shape[1] == self.args.enc_in and outputs.shape[2] == self.args.pred_len:
                outputs = outputs.transpose(1, 2)
        outputs = outputs[:, -self.args.pred_len:, :]
        return outputs

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

    def _core_model(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

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
            if current_state[key].shape != value.shape:
                skipped_missing_shape.append((key, tuple(value.shape), tuple(current_state[key].shape)))
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

    def vali(self, model, vali_loader, criterion):
        total_loss = []
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

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.is_diff:
                    outputs, _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, sample_times=self.args.sample_times, holiday_flag=batch_holiday)
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, holiday_flag=batch_holiday)

                outputs = self._process_model_output(outputs, is_diff=self.args.is_diff)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                pred_mask = batch_y_mask.detach().cpu() if batch_y_mask is not None else None
                loss = self._masked_loss(pred, true, pred_mask, self.args.loss_type)
                total_loss.append(loss.item())

        model.train()
        return np.average(total_loss)

    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')
        _, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
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
                    loss = core_model.training_loss(batch_x, batch_y, batch_y_mask)
                elif self.args.is_diff:
                    outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, sample_times=self.args.sample_times, holiday_flag=batch_holiday)
                    outputs = self._process_model_output(outputs, is_diff=self.args.is_diff)
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    loss = self._masked_loss(outputs, batch_y, batch_y_mask, self.args.loss_type)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, holiday_flag=batch_holiday)
                    outputs = self._process_model_output(outputs, is_diff=self.args.is_diff)
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    loss = self._masked_loss(outputs, batch_y, batch_y_mask, self.args.loss_type)
                if epoch == 0 and i == 0 and batch_y_mask is not None:
                    print('\t[mask] valid ratio: {:.4f}, zero elements: {}/{} (first batch)'.format(
                        batch_y_mask.mean().item(), int((batch_y_mask == 0).sum().item()), batch_y_mask.numel()))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
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

    def test(self, setting, test=0):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if test:
            if getattr(self.args, 'load_checkpoint', ''):
                checkpoint_path = self.args.load_checkpoint
            else:
                checkpoint_path = os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')
            print('loading model from', checkpoint_path)
            self._load_checkpoint_compat(checkpoint_path)

        core_model = self._core_model()
        if hasattr(core_model, 'reset_diagnostics'):
            core_model.reset_diagnostics()
        if hasattr(core_model, '_mom_kwargs') and hasattr(test_data, 'scaler') and hasattr(test_data.scaler, 'mean'):
            core_model._mom_kwargs['scaler_mean'] = torch.tensor(test_data.scaler.mean, dtype=torch.float32).squeeze(0)
            core_model._mom_kwargs['scaler_std'] = torch.tensor(test_data.scaler.std, dtype=torch.float32).squeeze(0)
        self.model.eval()

        preds = []
        trues = []
        all_marks = []
        all_masks = []
        all_histories = []
        folder_path = './test_results/' + setting + '/'
        diag_path = os.path.join(folder_path, 'diagnostics')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(diag_path):
            os.makedirs(diag_path)

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

                if self.args.is_diff:
                    outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, sample_times=self.args.vs_times, holiday_flag=batch_holiday)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, holiday_flag=batch_holiday)

                outputs = self._process_model_output(outputs, is_diff=self.args.is_diff)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                all_histories.append(batch_x.detach().cpu().numpy())
                if batch_y_mask is not None:
                    all_masks.append(batch_y_mask.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        masks = np.concatenate(all_masks, axis=0) if all_masks else None
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

        print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}'.format(mae, mse, rmse))

        folder_path = './results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        f = open(os.path.join(folder_path, setting + ".txt"), 'a')
        f.write(setting + "  \n")
        f.write('mae: {}, mse: {}, rmse: {}'.format(mae, mse, rmse))
        f.write('\n')
        f.write('\n')
        f.close()

        if self.args.data == 'fujian30':
            preds_raw = self._inverse_transform(test_data, preds)
            trues_raw = self._inverse_transform(test_data, trues)
            if masks is not None:
                sample_mae = np.divide((np.abs(preds_raw - trues_raw) * masks).sum(axis=(1, 2)),
                                       np.maximum(masks.sum(axis=(1, 2)), 1))
                node_mae = np.divide((np.abs(preds_raw - trues_raw) * masks).sum(axis=(0, 1)),
                                      np.maximum(masks.sum(axis=(0, 1)), 1))
                timestep_mae = np.divide((np.abs(preds_raw - trues_raw) * masks).sum(axis=(0, 2)),
                                          np.maximum(masks.sum(axis=(0, 2)), 1))
            else:
                sample_mae = np.abs(preds_raw - trues_raw).mean(axis=(1, 2))
                node_mae = np.abs(preds_raw - trues_raw).mean(axis=(0, 1))
                timestep_mae = np.abs(preds_raw - trues_raw).mean(axis=(0, 2))
            negative_ratio = (preds_raw < 0).mean(axis=(1, 2))

            meta = self._load_fujian30_meta(test_data)
            time_index = meta['time_index']
            station_ids = meta['station_ids']
            starts = meta['starts'][:len(preds)]
            pred_starts = meta['pred_starts'][:len(preds)]
            pred_ends = meta['pred_ends'][:len(preds)]
            holiday_flag = meta['holiday_flag']
            start_times = pd.to_datetime(time_index[pred_starts])
            future_holiday = np.asarray([float(holiday_flag[s + self.args.seq_len:s + self.args.seq_len + self.args.pred_len].sum() > 0) for s in starts])
            flow_mean_raw = trues_raw.mean(axis=(1, 2))
            q33, q66 = np.percentile(flow_mean_raw, [33, 66])
            phase = np.where(flow_mean_raw < q33, 'free_flow', np.where(flow_mean_raw > q66, 'congested', 'transition'))

            diag_block_var = torch.cat(core_model._diag_block_var, dim=0).numpy() if getattr(core_model, '_diag_block_var', None) else np.zeros(len(preds), dtype=np.float32)
            diag_inter_dev = torch.cat(core_model._diag_inter_dev, dim=0).numpy() if getattr(core_model, '_diag_inter_dev', None) else np.zeros(len(preds), dtype=np.float32)
            diag_median_bias = torch.cat(core_model._diag_median_bias, dim=0).numpy() if getattr(core_model, '_diag_median_bias', None) else np.zeros(len(preds), dtype=np.float32)
            diag_block_var = diag_block_var[:len(preds)] if len(diag_block_var) >= len(preds) else np.pad(diag_block_var, (0, len(preds) - len(diag_block_var)))
            diag_inter_dev = diag_inter_dev[:len(preds)] if len(diag_inter_dev) >= len(preds) else np.pad(diag_inter_dev, (0, len(preds) - len(diag_inter_dev)))
            diag_median_bias = diag_median_bias[:len(preds)] if len(diag_median_bias) >= len(preds) else np.pad(diag_median_bias, (0, len(preds) - len(diag_median_bias)))

            sample_df = pd.DataFrame({
                'sample_id': np.arange(len(preds)),
                'window_start_index': starts,
                'pred_start_index': pred_starts,
                'pred_end_index': pred_ends,
                'pred_start_time': start_times.astype(str),
                'start_hour': start_times.hour.to_numpy(),
                'is_holiday': future_holiday.astype(int),
                'phase': phase,
                'sample_mae': sample_mae,
                'negative_ratio': negative_ratio,
                'intra_var': diag_block_var,
                'inter_dev': diag_inter_dev,
                'median_bias': diag_median_bias,
            })
            sample_df.to_csv(os.path.join(diag_path, 'sample_mae.csv'), index=False)

            pd.DataFrame({'station_index': station_ids, 'node_mae': node_mae}).sort_values('node_mae', ascending=False).to_csv(os.path.join(diag_path, 'node_mae_map.csv'), index=False)
            timestep_df = pd.DataFrame({'timestep': np.arange(1, self.args.pred_len + 1), 'minutes_ahead': np.arange(1, self.args.pred_len + 1) * 15, 'timestep_mae': timestep_mae})
            timestep_df.to_csv(os.path.join(diag_path, 'timestep_mae.csv'), index=False)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(timestep_df['minutes_ahead'], timestep_df['timestep_mae'], marker='o')
            ax.set_xlabel('minutes ahead')
            ax.set_ylabel('timestep_mae')
            ax.set_title('Fujian30 timestep MAE')
            fig.tight_layout()
            fig.savefig(os.path.join(diag_path, 'diag_timestep_mae.png'), dpi=200)
            plt.close(fig)

            holiday_df = sample_df[sample_df['is_holiday'] == 1]
            normal_df = sample_df[sample_df['is_holiday'] == 0]
            free_df = sample_df[sample_df['phase'] == 'free_flow']
            congested_df = sample_df[sample_df['phase'] == 'congested']
            rho_var_mae, p_var_mae = self._safe_spearman(sample_df['intra_var'], sample_df['sample_mae'])
            rho_inter_mae, p_inter_mae = self._safe_spearman(sample_df['inter_dev'], sample_df['sample_mae'])
            rho_bias_mae, p_bias_mae = self._safe_spearman(sample_df['median_bias'], sample_df['sample_mae'])
            _, p_holiday = self._safe_mwu(holiday_df['sample_mae'], normal_df['sample_mae'], alternative='greater')
            _, p_phase = self._safe_mwu(congested_df['sample_mae'], free_df['sample_mae'], alternative='greater')

            print('[diag] batch_y_mark shape:', marks.shape)
            print('[diag] batch_y_mark unique example:', np.unique(marks)[:10])
            print('[diag] batch_y_mark all zero:', bool(np.allclose(marks, 0)))
            print('[diag] negative prediction ratio mean:', float(negative_ratio.mean()))
            print('[diag] Spearman(intra_var, sample_mae): {:.4f}, p={:.4e}'.format(rho_var_mae, p_var_mae))
            print('[diag] Spearman(inter_dev, sample_mae): {:.4f}, p={:.4e}'.format(rho_inter_mae, p_inter_mae))
            print('[diag] Spearman(median_bias, sample_mae): {:.4f}, p={:.4e}'.format(rho_bias_mae, p_bias_mae))
            print('[diag] holiday vs normal MAE: {:.4f} vs {:.4f}, MWU p={:.4e}'.format(float(holiday_df['sample_mae'].mean()) if len(holiday_df) else float('nan'), float(normal_df['sample_mae'].mean()) if len(normal_df) else float('nan'), p_holiday))
            print('[diag] free_flow vs congested MAE: {:.4f} vs {:.4f}, MWU p={:.4e}'.format(float(free_df['sample_mae'].mean()) if len(free_df) else float('nan'), float(congested_df['sample_mae'].mean()) if len(congested_df) else float('nan'), p_phase))

            # ── MoM 拆解指标 ──
            mom_ver = getattr(self.args, 'mom_version', 'baseline')
            ff_mask = (phase == 'free_flow')
            tr_mask = (phase == 'transition')
            cg_mask = (phase == 'congested')
            hol_mask = (future_holiday > 0)

            ff_mae = float(sample_mae[ff_mask].mean()) if ff_mask.sum() > 0 else float('nan')
            tr_mae = float(sample_mae[tr_mask].mean()) if tr_mask.sum() > 0 else float('nan')
            cg_mae = float(sample_mae[cg_mask].mean()) if cg_mask.sum() > 0 else float('nan')
            hol_mae = float(sample_mae[hol_mask].mean()) if hol_mask.sum() > 0 else float('nan')
            nor_mae = float(sample_mae[~hol_mask].mean()) if (~hol_mask).sum() > 0 else float('nan')
            hol_cg_mask = hol_mask & cg_mask
            nor_cg_mask = (~hol_mask) & cg_mask
            hol_cg_mae = float(sample_mae[hol_cg_mask].mean()) if hol_cg_mask.sum() > 0 else float('nan')
            nor_cg_mae = float(sample_mae[nor_cg_mask].mean()) if nor_cg_mask.sum() > 0 else float('nan')

            adj_path = os.path.join(self.args.root_path, 'adjacent_gantry.csv')
            if os.path.exists(adj_path):
                adj_df = pd.read_csv(adj_path)
                if {'src_FID', 'nbr_FID'}.issubset(adj_df.columns):
                    nodes = sorted(set(adj_df['src_FID'].astype(int)).union(set(adj_df['nbr_FID'].astype(int))))
                    node_to_idx = {nid: i for i, nid in enumerate(nodes)}
                    adj = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
                    for _, row in adj_df.iterrows():
                        adj[node_to_idx[int(row['src_FID'])], node_to_idx[int(row['nbr_FID'])]] = 1.0
                else:
                    adj = adj_df.values.astype(np.float32)
            else:
                adj = np.eye(preds_raw.shape[-1], dtype=np.float32)
            edge_pairs = np.argwhere(adj > 0)
            if len(edge_pairs) > 0:
                pred_edge = np.stack([preds_raw[:, :, j] - preds_raw[:, :, i] for i, j in edge_pairs], axis=-1)
                true_edge = np.stack([trues_raw[:, :, j] - trues_raw[:, :, i] for i, j in edge_pairs], axis=-1)
                sgfe = float(np.mean(np.abs(pred_edge - true_edge)))
            else:
                sgfe = float('nan')

            # TPR: 趋势保留率
            hist_raw = self._inverse_transform(test_data, histories) if hasattr(self, '_inverse_transform') else histories
            s_hist = hist_raw[:, -1, :] - hist_raw[:, 0, :]
            s_pred = preds_raw[:, -1, :] - preds_raw[:, 0, :]
            s_true = trues_raw[:, -1, :] - trues_raw[:, 0, :]
            tpr_all = float(np.mean((np.sign(s_hist) == np.sign(s_pred)) & (np.abs(s_hist) > 1e-6)))
            tpr_cg = float(np.mean((np.sign(s_hist[cg_mask]) == np.sign(s_pred[cg_mask])) & (np.abs(s_hist[cg_mask]) > 1e-6))) if cg_mask.sum() > 0 else float('nan')
            tpr_true_cg = float(np.mean((np.sign(s_hist[cg_mask]) == np.sign(s_true[cg_mask])) & (np.abs(s_hist[cg_mask]) > 1e-6))) if cg_mask.sum() > 0 else float('nan')

            eta_diag = None

            print('[mom-metrics] version={}'.format(mom_ver))
            print('[agg] mode={}, sample_times={}'.format(getattr(self.args, 'aggregation_mode', 'mom' if getattr(self.args, 'use_mom', False) else 'simple'), getattr(self.args, 'sample_times', 1)))
            print('[mom-metrics] phase_mae: free={:.4f}, trans={:.4f}, cong={:.4f}'.format(ff_mae, tr_mae, cg_mae))
            print('[mom-metrics] holiday_mae: hol={:.4f}, nor={:.4f}, hol_cong={:.4f}, nor_cong={:.4f}'.format(hol_mae, nor_mae, hol_cg_mae, nor_cg_mae))
            print('[mom-metrics] tpr: all={:.4f}, cong={:.4f}, true_cong={:.4f}'.format(tpr_all, tpr_cg, tpr_true_cg))
            print('[diag] sgfe: {:.4f}'.format(sgfe))

            with open(os.path.join(diag_path, 'diagnostic_summary.json'), 'w', encoding='utf-8') as fp:
                json.dump({
                    'setting': setting,
                    'mask_enabled': masks is not None,
                    'mask_valid_ratio': float(masks.mean()) if masks is not None else 1.0,
                    'mask_zero_count': int((masks == 0).sum()) if masks is not None else 0,
                    'batch_y_mark_shape': list(marks.shape),
                    'batch_y_mark_all_zero': bool(np.allclose(marks, 0)),
                    'mark_column_note': 'Fujian30CsvDataset returns zero placeholder marks; holiday/hour are reconstructed from CSV timeline.',
                    'spearman_intra_mae': rho_var_mae,
                    'spearman_inter_mae': rho_inter_mae,
                    'spearman_bias_mae': rho_bias_mae,
                    'holiday_mae_mean': float(holiday_df['sample_mae'].mean()) if len(holiday_df) else None,
                    'normal_mae_mean': float(normal_df['sample_mae'].mean()) if len(normal_df) else None,
                    'free_flow_mae_mean': float(free_df['sample_mae'].mean()) if len(free_df) else None,
                    'congested_mae_mean': float(congested_df['sample_mae'].mean()) if len(congested_df) else None,
                    'mwu_holiday_pvalue': p_holiday,
                    'mwu_congested_pvalue': p_phase,
                    'sgfe': sgfe,
                    'aggregation_mode': getattr(self.args, 'aggregation_mode', 'mom' if getattr(self.args, 'use_mom', False) else 'simple'),
                    'sample_times': int(getattr(self.args, 'sample_times', 1)),
                }, fp, ensure_ascii=False, indent=2)

        return mae, mse, rmse
