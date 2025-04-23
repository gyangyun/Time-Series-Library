import csv
import os
import time
import warnings
from datetime import datetime
from functools import partial
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.augmentation import run_augmentation, run_augmentation_single
from utils.dtw_metric import accelerated_dtw, dtw
from utils.losses import CustomLoss, asymmetric_mse_loss
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings("ignore")


class Exp_Industry_Electricity_Forecast(Exp_Basic):

    def __init__(self, args):
        super(Exp_Industry_Electricity_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(),
                                 lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = CustomLoss(partial(asymmetric_mse_loss, alpha=2.5))
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark,
                    batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = (torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1).float().to(self.device))
                # encoder - decoder
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

                # 每个输出结果的多少列参与损失计算
                # f_dim = -1 if self.args.features == "MS" else 0
                if self.args.features in ["MS", "S"]:
                    f_dim = -1
                else:
                    if self.args.target == "":
                        f_dim = 0
                    else:
                        targets = [t.strip() for t in self.args.target.split()]
                        f_dim = -len(targets)

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

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

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

        best_vali_loss = float("inf")
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

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = (torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1).float().to(self.device))

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark,
                                                 dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark,
                                                 dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:,
                                          f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp,
                                             batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp,
                                             batch_y_mark)

                    # 每个输出结果的多少列参与损失计算
                    # f_dim = -1 if self.args.features == "MS" else 0
                    if self.args.features in ["MS", "S"]:
                        f_dim = -1
                    else:
                        if self.args.target == "":
                            f_dim = 0
                        else:
                            targets = [
                                t.strip() for t in self.args.target.split()
                            ]
                            f_dim = -len(targets)

                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:,
                                      f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                        speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1,
                                                   time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss,
                        test_loss))

            # 每个epoch都更新最佳验证损失，以便使用超参搜索时作为优化目标
            if vali_loss < best_vali_loss:
                best_vali_loss = vali_loss

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, best_vali_loss

    def test(self, setting, load=False):
        test_data, test_loader = self._get_data(flag="test")
        if load:
            print("loading model")
            best_model_path = os.path.join(self.args.checkpoints, setting,
                                           "checkpoint.pth")
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device))

        preds = []
        trues = []
        folder_path = os.path.join(self.args.root_path, setting,
                                   "test_results", "figure")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark,
                    batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = (torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1).float().to(self.device))
                # encoder - decoder
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

                # 每个输出结果的多少列参与损失计算
                # f_dim = -1 if self.args.features == "MS" else 0
                if self.args.features in ["MS", "S"]:
                    f_dim = -1
                else:
                    if self.args.target == "":
                        f_dim = 0
                    else:
                        targets = [t.strip() for t in self.args.target.split()]
                        f_dim = -len(targets)

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(
                        outputs.reshape(shape[0] * shape[1],
                                        -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(
                        batch_y.reshape(shape[0] * shape[1],
                                        -1)).reshape(shape)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                preds.append(pred)
                trues.append(true)
                # 不需要绘制了
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(
                #             input.reshape(shape[0] * shape[1], -1)
                #         ).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        # result save
        folder_path = os.path.join(self.args.root_path, setting,
                                   "test_results", "data")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        # 记录各项评测指标到csv文件
        mae, mse, rmse, mape, mspe, smape, r2 = metric(preds, trues)
        # 打印主要结果
        print("mse:{}, mae:{}, dtw:{}".format(mse, mae, dtw))
        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 文件路径
        file_path = os.path.join(self.args.root_path,
                                 "result_long_term_forecast.csv")
        # 检查文件是否存在，如果不存在则创建并写入表头
        file_exists = os.path.isfile(file_path)
        # 打开CSV文件，准备写入数据
        with open(file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # 如果文件不存在，先写入表头
            if not file_exists:
                writer.writerow([
                    "setting",
                    "mae",
                    "mse",
                    "rmse",
                    "mape",
                    "mspe",
                    "smape",
                    "r2",
                    "dtw",
                    "timestamp",
                ])

            # 写入数据，包含各个评价指标和当前时间
            writer.writerow([
                setting, mae, mse, rmse, mape, mspe, smape, r2, dtw,
                current_time
            ])

        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        # print("mse:{}, mae:{}, dtw:{}".format(mse, mae, dtw))
        # f = open("result_long_term_forecast.txt", "a")
        # f.write(setting + "  \n")
        # f.write("mse:{}, mae:{}, dtw:{}".format(mse, mae, dtw))
        # f.write("\n")
        # f.write("\n")
        # f.close()

        np.save(
            os.path.join(folder_path, "metrics.npy"),
            np.array([mae, mse, rmse, mape, mspe]),
        )
        np.save(os.path.join(folder_path, "pred.npy"), preds)
        np.save(os.path.join(folder_path, "true.npy"), trues)

        # 将评测结果封装到字典中
        evaluation_results = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "mspe": mspe,
            "smape": smape,
            "r2": r2,
            "dtw": dtw,
        }

        # 返回评测结果字典
        return evaluation_results

    def predict(self, setting, load=True):
        pred_data, pred_loader = self._get_data(flag="pred")

        if load:
            print("loading model")
            best_model_path = os.path.join(self.args.checkpoints, setting,
                                           "checkpoint.pth")
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark,
                    batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = (torch.zeros(
                    [batch_y.shape[0], self.args.pred_len,
                     batch_y.shape[2]]).float().to(self.device))
                dec_inp = (torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1).float().to(self.device))

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

                # 每个输出结果的多少列为关注的输出值
                # f_dim = -1 if self.args.features == "MS" else 0
                if self.args.features in ["MS", "S"]:
                    f_dim = -1
                else:
                    if self.args.target == "":
                        f_dim = 0
                    else:
                        targets = [t.strip() for t in self.args.target.split()]
                        f_dim = -len(targets)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(
                        outputs.reshape(shape[0] * shape[1],
                                        -1)).reshape(shape)

                preds.append(outputs)

        preds = np.concatenate(preds, axis=0)

        folder_path = os.path.join(self.args.root_path, setting,
                                   "predict_results", "data")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(os.path.join(folder_path, "real_prediction.npy"), preds)

        return preds
