# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import pdb

import numpy as np
import torch as t
import torch.nn as nn


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = 0.0
    result[result == np.inf] = 0.0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(
        self,
        insample: t.Tensor,
        freq: int,
        forecast: t.Tensor,
        target: t.Tensor,
        mask: t.Tensor,
    ) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(
        self,
        insample: t.Tensor,
        freq: int,
        forecast: t.Tensor,
        target: t.Tensor,
        mask: t.Tensor,
    ) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(
            divide_no_nan(
                t.abs(forecast - target), t.abs(forecast.data) + t.abs(target.data)
            )
            * mask
        )


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(
        self,
        insample: t.Tensor,
        freq: int,
        forecast: t.Tensor,
        target: t.Tensor,
        mask: t.Tensor,
    ) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


def weighted_mse_loss(y_pred, y_true, weights):
    # y_pred, y_true, weights 应该是形状相同的张量
    loss = t.mean(weights * (y_pred - y_true) ** 2)
    return loss


def msle_loss(y_pred, y_true):
    log_true = t.log(y_pred + 1)
    log_pred = t.log(y_true + 1)
    loss = t.mean((log_true - log_pred) ** 2)
    return loss


def huber_loss(y_pred, y_true, delta=1.0):
    error = y_pred - y_true
    abs_error = t.abs(error)
    quadratic = t.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return t.mean(loss)


def quantile_loss(y_pred, y_true, tau=0.5):
    error = y_pred - y_true
    # 根据误差的符号决定损失的权重
    loss = t.mean((tau - (error < 0).float()) * error)
    return loss


def asymmetric_mse_loss(y_pred, y_true, alpha=1.2):
    """
    不对称均方误差损失，其中低估部分的惩罚更大。

    :param y_pred: 预测值张量。
    :param y_true: 真实值张量。
    :param alpha: 低估惩罚的权重（alpha > 1 会鼓励更高的预测）。
    :return: 损失值。
    """
    error = y_pred - y_true
    squared_error = error**2
    underestimation_penalty = t.where(error < 0, squared_error * alpha, squared_error)
    loss = t.mean(underestimation_penalty)
    return loss


def exponential_underestimation_penalty(y_pred, y_true, beta=2.0):
    """
    对低估的预测值施加指数级别的惩罚。

    :param y_pred: 预测值张量。
    :param y_true: 真实值张量。
    :param beta: 低估惩罚的指数因子。
    :return: 损失值。
    """
    error = y_pred - y_true
    underestimation_penalty = t.where(error < 0, t.exp(beta * error), error**2)
    loss = t.mean(underestimation_penalty)
    return loss


def combined_mse_quantile_loss(y_pred, y_true, tau=0.7, mse_weight=0.5):
    """
    组合MSE和分位数损失，强调低估部分。

    :param y_pred: 预测值张量。
    :param y_true: 真实值张量。
    :param tau: 分位数参数（tau > 0.5 强调低估）。
    :param mse_weight: MSE部分的权重。
    :return: 损失值。
    """
    mse_loss = t.mean((y_pred - y_true) ** 2)
    quantile_error = y_pred - y_true
    quantile_loss = t.mean((tau - (quantile_error < 0).float()) * quantile_error)
    loss = mse_weight * mse_loss + (1 - mse_weight) * quantile_loss


class CustomLoss(nn.Module):
    def __init__(self, loss_func):
        super(CustomLoss, self).__init__()
        self.loss_func = loss_func

    def forward(self, y_pred, y_true):
        # 调用传入的损失计算函数
        return self.loss_func(y_pred, y_true)
