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
        # 根据配置选择不同的模型
        use_stformer = getattr(self.args, 'use_stformer', False)
        use_lwrgat = getattr(self.args, 'use_lwrgat', False)
        use_lwrres = getattr(self.args, 'use_lwrres', False)
        
        if use_lwrres:
            # LWRRes 模型（物理残差扩散）
            from LWRRes.model import LWRResModel
            model = LWRResModel(self.args).float()
        elif use_lwrgat:
            # LWRGAT 模型
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
        """
        选择优化器，物理参数（v_critical, alpha）单独设置大学习率
        temperature 是固定 buffer，不参与优化
        
        关键设计：
        - 物理参数 lr=1e-2（基础参数的 100 倍）
        - 原因：物理参数对 loss 的梯度通常较小，需要更大步长
        """
        # 分离物理参数和基础参数
        phys_param_names = ['raw_v_critical', 'raw_alpha', 'v_critical', 'alpha']

        base_params = []
        phys_params = []

        for name, param in self.model.named_parameters():
            if any(pn in name for pn in phys_param_names):
                phys_params.append(param)
            else:
                base_params.append(param)

        base_count = len(base_params)
        phys_count = len(phys_params)
        print(f"[Optimizer Config]")
        print(f"  Base params: {base_count}, lr={self.args.learning_rate}")
        print(f"  Phys params: {phys_count}, lr={self.args.learning_rate * 200} (with grad clip max_norm=1.0)")

        self.phys_params = phys_params
        model_optim = optim.Adam([
            {'params': base_params, 'lr': self.args.learning_rate},
            {'params': phys_params, 'lr': self.args.learning_rate * 200, 'lr_scale': 200},
        ], weight_decay=0.0)
        
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
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                if i >= 2:
                    break
                # 支持双流数据（有6个值）和原始数据（有4个值）
                if len(batch) == 6:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_flow_x, batch_flow_y = batch
                    batch_flow_x = batch_flow_x.float().to(self.device) if batch_flow_x is not None else None
                    batch_flow_y = batch_flow_y.float().to(self.device) if batch_flow_y is not None else None
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
                    # LWRRes: 直接推理（与 train 一致），不用 sample_times
                    if self.args.use_lwrres:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                             flow_x=batch_flow_x, flow_y=batch_flow_y)
                    elif self.args.is_diff:
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
            """
            学习率调度，物理参数保持 100 倍于基础参数的学习率
            """
            base_lr = self.args.learning_rate
            
            if epoch < self.warmup_epochs:
                lr = base_lr * epoch / self.warmup_epochs
            else:
                lr = base_lr * 0.5 * \
                    (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.args.train_epochs - self.warmup_epochs)))
            
            # 关键：物理参数保持 100 倍学习率（通过 lr_scale 标记）
            for param_group in optimizer.param_groups:
                if "lr_scale" in param_group:
                    param_group["lr"] = lr * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr
            
            print(f'Updating learning rate to {lr:.7f} (phys: {lr*100:.5f})')
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
                
                # 调试：检查 flow_x
                if i == 0 and batch_flow_x is not None:
                    print(f"[DEBUG] batch_flow_x shape: {batch_flow_x.shape}, requires_grad: {batch_flow_x.requires_grad}")
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if batch_flow_x is not None:
                    batch_flow_x = batch_flow_x.float().to(self.device)
                    batch_flow_y = batch_flow_y.float().to(self.device)
                    # 调试：检查转到 GPU 后
                    if i == 0:
                        print(f"[DEBUG] batch_flow_x on GPU: {batch_flow_x.device}, requires_grad: {batch_flow_x.requires_grad}")
                
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    if self.args.is_diff:
                        outputs, weight_tmp = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                        flow_x=batch_flow_x, flow_y=batch_flow_y)
                    else:
                        outputs, weight_tmp = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                        flow_x=batch_flow_x, flow_y=batch_flow_y)

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
                
                # === 诊断：物理参数梯度 ===
                if (i + 1) % 100 == 0:
                    target_param = None
                    for name, p in self.model.named_parameters():
                        if 'raw_v_critical' in name and ('st_layers.0' in name or 'layers.0' in name):
                            target_param = p
                            break
                    if target_param is not None:
                        grad = torch.autograd.grad(
                            outputs=loss,
                            inputs=target_param,
                            retain_graph=True,
                            allow_unused=True
                        )[0]
                        if grad is None:
                            print(f"\n[DIAG] grad is None -> 计算图断了!")
                        else:
                            print(f"\n[DIAG] v_c grad = {grad.item():.2e}")
                            if abs(grad.item()) < 1e-8:
                                print("[DIAG] WARNING: grad 极小，数值抵消")
                
                loss.backward()

                # 物理参数梯度裁剪（防止高 lr 导致 v_c 撞边界）
                torch.nn.utils.clip_grad_norm_(self.phys_params, max_norm=1.0)

                # 打印物理参数梯度 + regime + v_critical
                if (i + 1) % 100 == 0:
                    print("\n[Grad Monitor] batch {}".format(i + 1))
                    for name, param in self.model.named_parameters():
                        if 'raw_v_critical' in name or 'raw_alpha' in name:
                            grad_norm = param.grad.norm().item() if param.grad is not None else 0
                            print(f"  {name}: grad_norm={grad_norm:.2e}")
                    
                    # 打印 v_critical 当前值
                    for name, param in self.model.named_parameters():
                        if 'raw_v_critical' in name:
                            print(f"  {name}: value={param.item():.4f}")
                    
                    # 诊断：直接测试 v_critical 梯度
                    v_c = None
                    for name, param in self.model.named_parameters():
                        if 'layers.0.pcgk.raw_v_critical' in name or 'st_layers.0.pcgk.raw_v_critical' in name:
                            v_c = param
                            break
                    if v_c is not None and v_c.grad is None:
                        # 手动反向传播测试
                        test_loss = loss.clone()
                        test_grad = torch.autograd.grad(
                            outputs=test_loss,
                            inputs=v_c,
                            retain_graph=True
                        )[0]
                        print(f"  [DIAG] manual v_c grad: {test_grad.abs().mean().item():.2e}")
                    
                    # regime 统计（用真实速度 batch_x，不是 flow_x）
                    # 复制 forward_train 的归一化逻辑（RevIN with std=1）：每条时序独立归一化
                    try:
                        if batch_flow_x is not None:
                            # batch_x: (B, seq_len, N) -> (B, N, L) = 速度
                            v_obs_raw = batch_x.permute(0, 2, 1)  # (B, N, L)
                            # RevIN 归一化：每条时序独立减均值（除以 std=1 不变）
                            mean_v = v_obs_raw.mean(dim=-1, keepdim=True)  # (B, N, 1)
                            v_obs_normed = v_obs_raw - mean_v          # (B, N, L)
                            # reshape: (B, N, L) -> (B*N, L)
                            v_obs_flat = v_obs_normed.reshape(v_obs_raw.shape[0] * v_obs_raw.shape[1], -1)

                            v_c_val = None
                            for name, module in self.model.named_modules():
                                if 'st_layers.0.pcgk' in name or 'layers.0.pcgk' in name:
                                    v_c_val = module.v_critical.item()
                                    temp_val = getattr(module, 'temperature', torch.tensor(1.0)).item()
                                    break
                            regime = torch.sigmoid((v_obs_flat - v_c_val) * temp_val).mean().item()
                            print(f"  regime (speed, normed): {regime:.3f}")
                    except Exception as e:
                        print(f"  regime calc error: {e}")
                
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
            
            # 监控物理参数变化
            print("\n[Physical Params Monitor]")
            for name, param in self.model.named_parameters():
                if 'raw_v_critical' in name or 'raw_alpha' in name:
                    print(f"  {name}: {param.item():.6f}")
            # temperature 是固定 buffer，打印当前值
            for name, buf in self.model.named_buffers():
                if 'temperature' in name:
                    print(f"  {name}: {buf.item():.4f} (fixed)")
            
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
        test_data, test_loader = self._get_data(flag='test')
        
        # 加载模型
        if test:
            checkpoint_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path))
                print(f"[Test] Loaded checkpoint from {checkpoint_path}")
            else:
                print(f"[Test] WARNING: checkpoint not found at {checkpoint_path}")
        
        # 打印关键参数确认
        for name, param in self.model.named_parameters():
            if 'raw_v_critical' in name or 'raw_alpha' in name:
                print(f"[Test] {name}: {param.item():.6f}")
        for name, buf in self.model.named_buffers():
            if 'temperature' in name:
                print(f"[Test] {name}: {buf.item():.4f} (fixed)")
        
        predms = []
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                # 支持双流数据（有6个值）和原始数据（有4个值）
                if len(batch) == 6:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_flow_x, batch_flow_y = batch
                    batch_flow_x = batch_flow_x.float().to(self.device) if batch_flow_x is not None else None
                    batch_flow_y = batch_flow_y.float().to(self.device) if batch_flow_y is not None else None
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
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
                        outputs, sampless = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                       sample_times=self.args.sample_times)
                    elif self.args.use_lwrres:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                             flow_x=batch_flow_x, flow_y=batch_flow_y)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
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
                
                if (i + 1) % 20 == 0:
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