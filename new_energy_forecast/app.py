import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.data_loader import DataLoader

# 初始化数据加载器
data_dir = os.path.join(project_root, 'dataset')
loader = DataLoader(data_dir)

# 变量中英文映射
variable_names = {
    'power': '功率',
    'hour': '小时',
    'date': '日期',
    'dayofweek': '星期',
    'u100': '100米纬向风',
    'v100': '100米经向风',
    't2m': '2米气温',
    'tp': '总降水量',
    'tcc': '总云量',
    'sp': '地面气压',
    'msl': '海平面气压',
    'poai': '光伏面板辐照度',
    'ghi': '水平面总辐照度'
}

# 变量单位映射
variable_units = {
    'power': 'MW',
    'hour': '',
    'date': '',
    'dayofweek': '',
    'u100': 'm/s',
    'v100': 'm/s',
    't2m': 'K',
    'tp': 'm',
    'tcc': '',
    'sp': 'Pa',
    'msl': 'Pa',
    'poai': 'W/m²',
    'ghi': 'W/m²'
}

# 初始化Dash应用
app = dash.Dash(__name__, title='新能源发电数据可视化')

# 定义布局
app.layout = html.Div([
    # 标题
    html.H1('新能源发电数据可视化平台', style={'textAlign': 'center'}),

    # 数据选择区域
    html.Div(
        [
            # 左侧：数据类型和场站选择
            html.Div(
                [
                    # 数据类型选择
                    html.Label('数据类型'),
                    dcc.RadioItems(id='data-type',
                                   options=[{
                                       'label': '功率数据',
                                       'value': 'power'
                                   }, {
                                       'label': '气象数据',
                                       'value': 'nwp'
                                   }, {
                                       'label': '关联分析',
                                       'value': 'correlation'
                                   }],
                                   value='power',
                                   style={'marginBottom': '20px'}),

                    # 场站选择
                    html.Label('场站选择'),
                    dcc.Dropdown(id='station-id',
                                 options=[{
                                     'label':
                                     f'场站{i} ({loader.get_station_type(i)})',
                                     'value': i
                                 } for i in range(1, 11)],
                                 value=1,
                                 style={'marginBottom': '20px'}),

                    # 数据源选择（仅用于气象数据和关联分析）
                    html.Div(
                        id='nwp-source-container',
                        children=[
                            html.Label('气象数据源'),
                            dcc.Dropdown(id='nwp-source',
                                         options=[{
                                             'label': source,
                                             'value': source
                                         } for source in loader.nwp_sources],
                                         value='NWP_1')
                        ],
                        style={
                            'display': 'none',
                            'marginBottom': '20px'
                        }),

                    # 经纬度选择（仅用于气象数据）
                    html.Div(id='location-container',
                             children=[
                                 html.Label('纬度选择'),
                                 dcc.Dropdown(id='lat-select',
                                              options=[{
                                                  'label': f'纬度 {i}',
                                                  'value': i
                                              } for i in range(11)],
                                              value=5,
                                              style={'marginBottom': '10px'}),
                                 html.Label('经度选择'),
                                 dcc.Dropdown(id='lon-select',
                                              options=[{
                                                  'label': f'经度 {i}',
                                                  'value': i
                                              } for i in range(11)],
                                              value=5,
                                              style={'marginBottom': '20px'})
                             ],
                             style={
                                 'display': 'none',
                                 'marginBottom': '20px'
                             }),

                    # 变量选择
                    html.Label('变量选择'),
                    dcc.Dropdown(id='variables',
                                 multi=True,
                                 style={'marginBottom': '20px'}),
                ],
                style={
                    'width': '25%',
                    'float': 'left',
                    'padding': '20px'
                }),

            # 右侧：时间选择和图表
            html.Div(
                [
                    # 时间范围选择
                    html.Label('时间范围'),
                    dcc.DatePickerRange(id='date-range',
                                        start_date='2024-01-01',
                                        end_date='2024-12-31',
                                        style={'marginBottom': '20px'}),

                    # 图表
                    dcc.Graph(id='time-series-plot', style={'height': '800px'})
                ],
                style={
                    'width': '75%',
                    'float': 'right',
                    'padding': '20px'
                })
        ],
        style={
            'display': 'flex',
            'margin': '20px'
        }),

    # 底部信息
    html.Div([html.P('注：图表会根据选择的条件自动更新。', style={'fontStyle': 'italic'})],
             style={
                 'clear': 'both',
                 'padding': '20px'
             })
])


# 回调函数：控制气象数据源和经纬度选择框的显示/隐藏
@app.callback([
    Output('nwp-source-container', 'style'),
    Output('location-container', 'style')
], Input('data-type', 'value'))
def toggle_nwp_controls(data_type):
    nwp_source_style = {'display': 'block', 'marginBottom': '20px'} \
        if data_type in ['nwp', 'correlation'] else {'display': 'none', 'marginBottom': '20px'}
    location_style = {'display': 'block', 'marginBottom': '20px'} \
        if data_type in ['nwp', 'correlation'] else {'display': 'none', 'marginBottom': '20px'}
    return nwp_source_style, location_style


# 回调函数：更新变量选择下拉框
@app.callback(Output('variables', 'options'), Output('variables', 'value'),
              Input('data-type', 'value'), Input('nwp-source', 'value'))
def update_variable_options(data_type, nwp_source):
    if data_type == 'power':
        options = [{
            'label': f'{variable_names[var]}',
            'value': var
        } for var in ['power', 'hour', 'date', 'dayofweek']]
        default_value = ['power']
    elif data_type == 'nwp':
        variables = loader.variables[nwp_source]
        options = [{
            'label': f'{variable_names[var]}',
            'value': var
        } for var in variables]
        default_value = [variables[0]]
    else:  # correlation
        variables = loader.variables[nwp_source]
        options = [{
            'label': f'{variable_names[var]}',
            'value': var
        } for var in variables]
        default_value = []  # 默认不选择任何变量

    return options, default_value


# 回调函数：更新时间序列图
@app.callback(Output('time-series-plot', 'figure'), [
    Input('data-type', 'value'),
    Input('station-id', 'value'),
    Input('nwp-source', 'value'),
    Input('variables', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('lat-select', 'value'),
    Input('lon-select', 'value')
])
def update_time_series(data_type, station_id, nwp_source, variables,
                       start_date, end_date, lat, lon):
    if not variables:
        # 如果没有选择变量，返回空白图表但带有提示文字
        fig = go.Figure()
        fig.update_layout(title_text="请选择要显示的变量",
                          annotations=[
                              dict(text="请在左侧面板选择要显示的变量",
                                   xref="paper",
                                   yref="paper",
                                   x=0.5,
                                   y=0.5,
                                   showarrow=False,
                                   font=dict(size=20))
                          ])
        return fig

    # 转换日期字符串为datetime对象
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if data_type == 'correlation':
        # 加载功率数据和气象数据
        power_df = loader.load_processed_power_data(station_id)
        nwp_df = loader.load_processed_nwp_data(station_id, nwp_source)

        if power_df is None or nwp_df is None:
            return go.Figure()

        # 对于气象数据，选择指定的经纬度点
        nwp_df = nwp_df.xs((lat, lon), level=('lat', 'lon'), drop_level=False)

        # 将nwp_df的索引转换为DataFrame的列
        nwp_df = nwp_df.reset_index()

        # 确保时间列是datetime类型
        power_df['time'] = pd.to_datetime(power_df['time'])
        nwp_df['time'] = pd.to_datetime(nwp_df['time'])

        # 合并功率数据和气象数据
        merged_df = pd.merge(power_df, nwp_df, on='time', how='inner')

        # 过滤时间范围
        mask = (merged_df['time'] >= start_date) & (merged_df['time']
                                                    <= end_date)
        merged_df = merged_df[mask]

        # 创建子图，每个变量一个子图
        subplot_titles = [
            f"功率与{variable_names[var]}的关系 (纬度: {lat}, 经度: {lon})"
            for var in variables
        ]
        fig = make_subplots(
            rows=len(variables),
            cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            row_heights=[1 / len(variables)] * len(variables),
            specs=[[{
                "secondary_y": True
            }] for _ in range(len(variables))]  # 为每个子图启用双Y轴
        )

        # 为每个选择的变量创建一个子图
        for i, var in enumerate(variables, 1):
            if var in merged_df.columns:
                # 计算两个y轴的范围，并调整以留出一定边距
                power_min, power_max = merged_df['power'].min(
                ), merged_df['power'].max()
                power_margin = (power_max - power_min) * 0.1
                power_range = [
                    power_min - power_margin, power_max + power_margin
                ]

                var_min, var_max = merged_df[var].min(), merged_df[var].max()
                var_margin = (var_max - var_min) * 0.1
                var_range = [var_min - var_margin, var_max + var_margin]

                # 添加功率数据（使用左侧y轴）
                fig.add_trace(
                    go.Scatter(
                        x=merged_df['time'],
                        y=merged_df['power'],
                        name='功率',
                        mode='lines',
                        line=dict(color='blue'),
                        showlegend=True if i == 1 else False  # 只在第一个子图显示图例
                    ),
                    row=i,
                    col=1)

                # 添加气象数据（使用右侧y轴）
                fig.add_trace(
                    go.Scatter(x=merged_df['time'],
                               y=merged_df[var],
                               name=variable_names[var],
                               mode='lines',
                               line=dict(color='red'),
                               showlegend=True),
                    row=i,
                    col=1,
                    secondary_y=True  # 使用次坐标轴
                )

                # 设置左侧y轴（功率）
                fig.update_yaxes(title=dict(text='功率 (MW)',
                                            font=dict(color='blue')),
                                 tickfont=dict(color='blue'),
                                 range=power_range,
                                 row=i,
                                 col=1,
                                 secondary_y=False)

                # 设置右侧y轴（气象变量）
                fig.update_yaxes(title=dict(
                    text=f'{variable_names[var]} ({variable_units[var]})',
                    font=dict(color='red')),
                                 tickfont=dict(color='red'),
                                 range=var_range,
                                 row=i,
                                 col=1,
                                 secondary_y=True)

                # 设置x轴标签
                if i == len(variables):  # 只在最后一个子图显示x轴标签
                    fig.update_xaxes(title_text="时间", row=i, col=1)
                else:
                    fig.update_xaxes(showticklabels=True, row=i, col=1)

        # 更新布局
        station_type = '风电' if loader.get_station_type(
            station_id) == 'wind' else '光伏'
        fig.update_layout(
            height=300 * len(variables) + 100,  # 根据子图数量调整总高度
            title=dict(
                text=
                f'场站{station_id}({station_type}) - 功率与{nwp_source}气象数据关联分析',
                y=0.99),
            showlegend=True,
            legend=dict(orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1),
            margin=dict(t=100, b=50)  # 调整上下边距
        )

        return fig
    else:
        # 原有的功率数据和气象数据可视化逻辑
        if data_type == 'power':
            df = loader.load_processed_power_data(station_id)
        else:
            df = loader.load_processed_nwp_data(station_id, nwp_source)
            if df is not None:
                df = df.xs((lat, lon), level=('lat', 'lon'), drop_level=False)

        if df is None:
            return go.Figure()

        # 过滤数据
        if data_type == 'power':
            mask = (df['time'] >= start_date) & (df['time'] <= end_date)
            df = df[mask]
            x = df['time']
        else:
            mask = (df.index.get_level_values('time') >= start_date) & \
                   (df.index.get_level_values('time') <= end_date)
            df = df[mask]
            x = df.index.get_level_values('time')

        # 创建子图
        subplot_titles = [
            f"{variable_names[var]}{f' ({variable_units[var]})' if variable_units[var] else ''}"
            for var in variables
        ]
        fig = make_subplots(rows=len(variables),
                            cols=1,
                            subplot_titles=subplot_titles,
                            shared_xaxes=True,
                            vertical_spacing=0.05)

        # 添加每个变量的时间序列
        for i, var in enumerate(variables, 1):
            if var in df.columns:
                fig.add_trace(go.Scatter(x=x,
                                         y=df[var],
                                         name=variable_names[var],
                                         mode='lines'),
                              row=i,
                              col=1)

        # 更新布局
        station_type = '风电' if loader.get_station_type(
            station_id) == 'wind' else '光伏'
        title = f'场站{station_id}({station_type}) - {"功率" if data_type == "power" else nwp_source}数据'
        if data_type == 'nwp':
            title += f' (纬度: {lat}, 经度: {lon})'

        fig.update_layout(height=400 * len(variables),
                          title_text=title,
                          showlegend=True)

        # 更新y轴标签
        for i, var in enumerate(variables, 1):
            fig.update_yaxes(
                title_text=
                f"{variable_names[var]}{f' ({variable_units[var]})' if variable_units[var] else ''}",
                row=i,
                col=1)

        return fig


if __name__ == '__main__':
    app.run(debug=True, port=8050)
