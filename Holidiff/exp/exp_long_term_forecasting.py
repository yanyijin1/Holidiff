from Holidiff.data_provider.data_factory import data_provider
from Holidiff.exp.exp_basic import Exp_Basic
from Holidiff.utils.tools import EarlyStopping, adjust_learning_rate, visual
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
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
            # 训练时是 (B, N, pred_len)，评估时是 (B, pred_len, N)
            if outputs.shape[1] == self.args.enc_in and outputs.shape[2] == self.args.pred_len:
                outputs = outputs.transpose(1, 2)  # (B, N, pred_len) -> (B, pred_len, N)
            # 如果已经是 (B, pred_len, N) 则不需要处理
        
        outputs = outputs[:, -self.args.pred_len:, :]
        return outputs

    def vali(self, model, vali_loader, criterion):
        total_loss = []
        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.is_diff:
                    outputs, _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, sample_times=self.args.sample_times)
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = self._process_model_output(outputs, is_diff=self.args.is_diff)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss.item())

        model.train()
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.is_diff:
                    outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, sample_times=self.args.sample_times)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = self._process_model_output(outputs, is_diff=self.args.is_diff)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

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
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')
        _, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))

        self.model.eval()

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.is_diff:
                    outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, sample_times=self.args.vs_times)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = self._process_model_output(outputs, is_diff=self.args.is_diff)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

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

        return mae, mse, rmse
