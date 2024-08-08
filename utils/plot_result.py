from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

# Import Plotly

pd.options.plotting.backend = "plotly"

import plotly.io as pio

pio.templates.default = "simple_white"

from sklearn.metrics import mean_absolute_percentage_error


def plot_result(tmp_df, x, y, add_info="", dirpath=None):
    start_time = tmp_df["datetime"].min()
    end_time = tmp_df["datetime"].max()
    timerange_info = f"{start_time.strftime('%Y%m%d')}-{end_time.strftime('%Y%m%d')}"
    title = f"{timerange_info} {add_info}" if add_info else timerange_info

    tmp_fig = px.line(
        tmp_df,
        x=x,
        y=y,
        # line_shape=line_shape,
        markers=True,
        # labels=dict(
        #     datetime="数据时间",
        #     variable="",
        #     value="功率（kW）",
        # ),
        # hover_data=dict(
        #     precision=':.2%',
        #     ),
        # labels=dict(
        #     datetime="datetime",
        #     variable="",
        #     value="power",
        # ),
    )
    legend = dict(
        orientation="h",
        y=1,
        x=0.5,
        yanchor="bottom",
        xanchor="center",
    )
    hours_axis = dict(
        type="date",
        # tickformat="%H",
        tickformat="%Y-%m-%d %H",
        # dtick=1e3 * 60 * 60,
        tickangle=-45,
        showgrid=False,
        showticklabels=True,
        matches=None,
        rangeslider_visible=False,
    )
    linear_axis = dict(
        type="linear",
        tickformat=":.2f",
        showgrid=True,
        showticklabels=True,
        matches=None,
    )
    exponent_axis = dict(
        tickformat=".2e",
        showgrid=True,
        showticklabels=True,
        matches=None,
    )
    tmp_fig.update_layout(
        dict(
            title=title,
            title_x=0.5,
            xaxis=hours_axis,
            yaxis=exponent_axis,
            legend=legend,
            #  hovermode='x',
        ),
    )
    if dirpath:
        dirpath = Path(dirpath)
        if not dirpath.is_dir():
            dirpath.mkdir(parents=True)
        # tmp_fig.write_html(dirpath.joinpath(f"{title}.html"))
        tmp_fig.write_image(dirpath.joinpath(f"{title}.png"), width=1600, height=1200)
    return tmp_fig


def plot_test_result(
    result_dp, fig_dp, start_time, end_time, freq, scaler_y=None
):
    result_dp = Path(result_dp)
    fig_dp = Path(fig_dp)

    # mae, mse, rmse, mape, mspe = np.load(result_dp.joinpath("metrics.npy"))
    # [batch_size, pred_len, output_size]
    pred = np.load(result_dp.joinpath("pred.npy"))
    # [batch_size, pred_len, output_size]
    true = np.load(result_dp.joinpath("true.npy"))

    time_range = pd.date_range(
        pd.to_datetime(start_time),
        pd.to_datetime(end_time),
        freq=freq,
    )
    for i in range(len(pred)):
        # 每个时间步长的输出结果只能绘制1维，所以把第1维取出来
        tmp_pred = pred[i, :, 0]
        tmp_true = true[i, :, 0]
        tmp_time_range = time_range[i: i+tmp_pred.shape[0]]

        if scaler_y:
            tmp_pred = scaler_y.inverse_transform(
                np.array(tmp_pred).reshape(-1, 1)
            ).squeeze()
            tmp_true = scaler_y.inverse_transform(
                np.array(tmp_true).reshape(-1, 1)
            ).squeeze()

        tmp_df = pd.DataFrame(
            {"order": i, "datetime": tmp_time_range, "pred": tmp_pred, "true": tmp_true}
        )

        mape = mean_absolute_percentage_error(tmp_df["true"], tmp_df["pred"])
        precision = 1 - mape
        add_info = f"precision: {precision: .2%}"
        plot_result(
            tmp_df, x="datetime", y=["pred", "true"], add_info=add_info, dirpath=fig_dp
        )


def plot_predict_result(
    result_dp, fig_dp, start_time, end_time, freq, scaler_y=None
):
    result_dp = Path(result_dp)
    fig_dp = Path(fig_dp)

    pred = np.load(result_dp.joinpath("real_prediction.npy"))

    time_range = pd.date_range(
        pd.to_datetime(start_time),
        pd.to_datetime(end_time),
        freq=freq,
    )
    for i in range(len(pred)):
        # 每个时间步长的输出结果只能绘制1维，所以把第1维取出来
        tmp_pred = pred[i, :, 0]
        tmp_time_range = time_range[i: i+tmp_pred.shape[0]]

        if scaler_y:
            tmp_pred = scaler_y.inverse_transform(
                np.array(tmp_pred).reshape(-1, 1)
            ).squeeze()

        tmp_df = pd.DataFrame(
            {"order": i, "datetime": tmp_time_range, "pred": tmp_pred}
        )

        plot_result(tmp_df, x="datetime", y="pred", add_info="", dirpath=fig_dp)
