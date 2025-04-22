import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.losses import CustomLoss, asymmetric_mse_loss
from functools import partial

warnings.filterwarnings('ignore')


class Exp_NE_Forecast(Exp_Long_Term_Forecast):
    """
    新能源发电功率预测的实验类
    继承自长期预测实验类，主要改进在于利用气象预报数据优化decoder输入
    """

    def __init__(self, args):
        super(Exp_NE_Forecast, self).__init__(args)
        # 定义气象特征和功率特征的维度
        self.weather_dims = args.weather_dims  # 气象特征的维度列表
        self.power_dims = args.power_dims  # 功率特征的维度列表

    def _process_decoder_input(self, batch_y, batch_y_mark):
        """
        处理解码器输入，将气象预报数据纳入考虑
        Args:
            batch_y: 包含气象和功率特征的输入数据
            batch_y_mark: 时间特征标记
        Returns:
            处理后的解码器输入
        """
        B, L, M = batch_y.shape

        # 1. 分离历史数据中的气象特征和功率特征
        weather_features = batch_y[:, :, self.weather_dims]
        power_features = batch_y[:, :, self.power_dims]

        # 2. 构造解码器输入
        # 2.1 历史部分（label_len）保持不变
        dec_inp_history = batch_y[:, :self.args.label_len, :]

        # 2.2 预测部分（pred_len）
        # - 气象特征使用预报数据
        # - 功率特征使用零值
        dec_inp_future = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
        # 将气象预报数据填入相应位置
        dec_inp_future[:, :,
                       self.weather_dims] = batch_y[:, -self.args.pred_len:,
                                                    self.weather_dims]

        # 2.3 拼接历史部分和预测部分
        dec_inp = torch.cat([dec_inp_history, dec_inp_future], dim=1)

        return dec_inp.float().to(self.device)

    def _select_criterion(self):
        """
        选择损失函数
        使用自定义的非对称MSE损失，对于光伏/风电预测更合适
        """
        criterion = CustomLoss(partial(asymmetric_mse_loss, alpha=2.5))
        return criterion

    def train(self, setting):
        """
        模型训练函数
        Args:
            setting: 实验设置
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience,
                                       verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark,
                    batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 使用新的解码器输入处理方法
                dec_inp = self._process_decoder_input(batch_y, batch_y_mark)

                # 使用amp
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark,
                                                 dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark,
                                                 dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:,
                                          f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp,
                                             batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp,
                                             batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:,
                                      f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                    loss.backward()
                    model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(
                        speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1,
                                                   time.time() - epoch_time))

            # 验证
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss,
                        test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        """
        验证函数
        """
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark,
                    batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = self._process_decoder_input(batch_y, batch_y_mark)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark,
                                                 dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark,
                                                 dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp,
                                             batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp,
                                             batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        """
        测试函数
        """
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(
                    os.path.join('./checkpoints/' + setting,
                                 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark,
                    batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = self._process_decoder_input(batch_y, batch_y_mark)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark,
                                                 dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark,
                                                 dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp,
                                             batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp,
                                             batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # 计算指标
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # 计算新能源预测评价指标CR
        def calculate_cr(pred, true, n=96):
            """计算每日预测准确率CR"""
            diff = pred - true
            cr = (1 - np.sqrt(np.mean(
                (diff / np.maximum(true, 0.2))**2))) * 100
            return cr

        # 按天计算CR
        daily_cr = []
        for i in range(0, len(preds), 96):  # 假设一天96个点
            day_pred = preds[i:i + 96]
            day_true = trues[i:i + 96]
            cr = calculate_cr(day_pred, day_true)
            daily_cr.append(cr)

        avg_cr = np.mean(daily_cr)
        print('Average CR: {:.2f}%'.format(avg_cr))

        return
