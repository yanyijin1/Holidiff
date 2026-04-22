# Version: v0.2-trainer (Add DTW metric support)
# Date: 2026-04-11
# Description: Trainer - Add DTW metric support

import os
import torch
import torch.nn as nn
from torch import optim
import warnings
import numpy as np
import time
import math
import argparse

from data_loader import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, metric_with_dtw, dtw_metric

warnings.filterwarnings('ignore')


def l1loss(pred, target):
    return (target - pred).abs().mean()


class Args:
    """简单的参数类，用于从yaml转换"""
    def __init__(self):
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1'
        self.device_ids = [0]


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        raise NotImplementedError
        return None

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.args = args
        self.model = self._build_model()

    def _build_model(self):
        # 根据 use_stformer 配置选择不同的模型
        use_stformer = getattr(self.args, 'use_stformer', False)
        use_lwrgat = getattr(self.args, 'use_lwrgat', False)
        
        if use_lwrgat:
            # Phase 1: STLWRGAT 模型
            from LWRGAT.Model import Model as LWRGATModel
            model = LWRGATModel(self.args).float()
        elif use_stformer:
            from stformer_bone.Model import Model as STFormerModel
            model = STFormerModel(self.args).float()
        else:
            from fourier.ptld_model import Model
            model = Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model.to(self.device)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss_type == 'MSE':
            criterion = nn.MSELoss()
        else:
            criterion = l1loss
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        import time
        t0 = time.time()
        debug_count = 0
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                if i >= 2:
                    break
                # 支持双流数据（有6个值）和原始数据（有4个值）
                if len(batch) == 6:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_flow_x, batch_flow_y = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_flow_x, batch_flow_y = None, None
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                t1 = time.time()
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    if self.args.is_diff:
                        outputs, sampless = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                      sample_times=self.args.vs_times)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                t2 = time.time()
                if i == 0:
                    print(f"  [vali] batch 0 model forward: {t2-t1:.1f}s")

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)

        print(f"  [vali] total: {time.time()-t0:.1f}s, {len(total_loss)} batches")
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        self.warmup_epochs = 5
        
        def adjust_learning_rate_new(optimizer, epoch, args):
            min_lr = 0
            if epoch < self.warmup_epochs:
                lr = self.args.learning_rate * epoch / self.warmup_epochs
            else:
                lr = min_lr + (self.args.learning_rate - min_lr) * 0.5 * \
                    (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.args.train_epochs - self.warmup_epochs)))
            for param_group in optimizer.param_groups:
                if "lr_scale" in param_group:
                    param_group["lr"] = lr * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr
            print(f'Updating learning rate to {lr:.7f}')
            return lr

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            if self.args.is_diff:
                adjust_learning_rate_new(model_optim, epoch + 1, self.args)

            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                # 支持双流数据（有6个值）和原始数据（有4个值）
                if len(batch) == 6:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_flow_x, batch_flow_y = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_flow_x, batch_flow_y = None, None
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    if self.args.is_diff:
                        outputs, weight_tmp = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # outputs 现在是 (B, pred_len, N)，weight_tmp 是 (B*N, 1, 1)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs_for_loss = outputs[:, -self.args.pred_len:, :]

                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                if self.args.is_diff and self.args.loss_type == 'MAE':
                    B = batch_x.shape[0]
                    N = batch_x.shape[2]
                    batch_y_2d = batch_y.permute(0, 2, 1).reshape(B * N, self.args.pred_len)
                    outputs_2d = outputs_for_loss.permute(0, 2, 1).reshape(B * N, self.args.pred_len)
                    weight_2d = weight_tmp.squeeze(-1)  # (B*N, 1)
                    loss = criterion(outputs_2d / weight_2d, batch_y_2d / weight_2d)
                else:
                    loss = criterion(outputs_for_loss, batch_y)
                    
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break
                
            if self.args.is_diff == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        print("[TEST] Step 1: Getting test data...", flush=True)
        test_data, test_loader = self._get_data(flag='test')
        print(f"[TEST] Step 1: Done. test_data={len(test_data)}, batches={len(test_loader)}", flush=True)
        
        if test:
            print('[TEST] Step 2: Loading model checkpoint...', flush=True)
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            print('[TEST] Step 2: Done.', flush=True)
            
        predms = []
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        print(f'[TEST] Step 3: Starting test inference loop... (total batches: {len(test_loader)})', flush=True)
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                print(f'[TEST] Batch {i}: Getting batch...', flush=True)
                
                # 支持双流数据（有6个值）和原始数据（有4个值）
                if len(batch) == 6:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_flow_x, batch_flow_y = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
                    batch_flow_x, batch_flow_y = None, None
                    
                print(f'[TEST] Batch {i}: Moving to device...', flush=True)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                print(f'[TEST] Batch {i}: Calling model forward...', flush=True)
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    if self.args.is_diff:
                        outputs, sampless = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                       sample_times=self.args.sample_times)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                print(f'[TEST] Batch {i}: Model forward done, processing outputs...', flush=True)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y
                
                if self.args.is_diff:
                    predm = sampless.detach().cpu().numpy()
                    predms.append(predm)
                preds.append(pred)
                trues.append(true)
                
                print(f'[TEST] Batch {i}: Done. Accumulated: {len(preds)} batches', flush=True)
                
                if (i + 1) % 5 == 0:
                    print(f'  test batch {i+1}/{len(test_loader)}', flush=True)

        predms = np.array(predms) if predms else None
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        print('Test inference completed!')

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        
        if hasattr(self.args, 'use_dtw') and self.args.use_dtw:
            # 使用DTW指标
            metrics = metric_with_dtw(preds, trues, use_dtw=True, use_accelerated_dtw=True)
            f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, dtw:{}'.format(
                metrics['mse'], metrics['mae'], metrics['rmse'], 
                metrics['mape'], metrics['mspe'], metrics['dtw']))
            np.save(folder_path + 'metrics.npy', np.array([
                metrics['mae'], metrics['mse'], metrics['rmse'], 
                metrics['mape'], metrics['mspe'], metrics['dtw']]))
        else:
            # 标准指标
            f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        
        f.write('\n')
        f.write('\n')
        f.close()
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return