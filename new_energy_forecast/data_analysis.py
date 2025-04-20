import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import os

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data_loader import DataLoader


class DataAnalyzer:
    """数据分析类，用于分析气象数据和功率数据"""

    def __init__(self, data_dir: str, output_dir: str):
        """
        初始化数据分析器
        
        Args:
            data_dir: 数据根目录
            output_dir: 分析结果输出目录
        """
        self.loader = DataLoader(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_power_data(self, station_id: int, use_processed: bool = True):
        """
        分析单个场站的功率数据
        
        Args:
            station_id: 场站ID
            use_processed: 是否使用处理后的数据，默认为True
        """
        # 根据参数选择加载方法
        if use_processed:
            power_df = self.loader.load_processed_power_data(station_id)
            data_source = "处理后"
        else:
            power_df = self.loader.load_raw_power_data(station_id)
            data_source = "原始"

        if power_df is None:
            print(f"无法加载场站{station_id}的{data_source}数据")
            return

        station_type = self.loader.get_station_type(station_id)
        print(f"\n=== 场站{station_id}({station_type}){data_source}功率数据分析 ===")

        # 基本统计信息
        print("\n基本统计信息：")
        print(power_df['power'].describe())

        # 缺失值信息
        print("\n缺失值统计：")
        missing = power_df.isnull().sum()
        print(missing)

        # 创建子图布局
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=[
                f'场站{station_id}({station_type}){data_source}功率时间序列',
                f'场站{station_id}({station_type}){data_source}功率分布',
                f'场站{station_id}({station_type}){data_source}各小时功率分布'
            ],
            vertical_spacing=0.1,
            row_heights=[0.4, 0.3, 0.3])

        # 时间序列图
        fig.add_trace(go.Scatter(x=power_df['time'],
                                 y=power_df['power'],
                                 mode='lines',
                                 name='功率'),
                      row=1,
                      col=1)

        # 功率分布图
        fig.add_trace(go.Histogram(x=power_df['power'], nbinsx=50,
                                   name='功率分布'),
                      row=2,
                      col=1)

        # 按小时统计平均功率
        hourly_power = power_df.copy()
        hourly_power['hour'] = hourly_power['time'].dt.hour

        # 添加箱线图
        fig.add_trace(go.Box(x=hourly_power['hour'],
                             y=hourly_power['power'],
                             name='小时功率分布'),
                      row=3,
                      col=1)

        # 更新布局
        fig.update_layout(
            height=1200,
            showlegend=False,
            title_text=f"场站{station_id}({station_type}){data_source}功率数据分析")

        # 更新x轴和y轴标签
        fig.update_xaxes(title_text="时间", row=1, col=1)
        fig.update_xaxes(title_text="功率", row=2, col=1)
        fig.update_xaxes(title_text="小时", row=3, col=1)

        fig.update_yaxes(title_text="功率", row=1, col=1)
        fig.update_yaxes(title_text="频次", row=2, col=1)
        fig.update_yaxes(title_text="功率", row=3, col=1)

        # 保存图表
        fig.write_html(
            self.output_dir /
            f'power_analysis_station_{station_id}_{data_source}.html')

    def analyze_nwp_data(self, station_id: int, use_processed: bool = True):
        """
        分析单个场站的所有气象数据
        
        Args:
            station_id: 场站ID
            use_processed: 是否使用处理后的数据，默认为True
        """
        station_type = self.loader.get_station_type(station_id)
        data_source = "处理后" if use_processed else "原始"
        print(f"\n=== 场站{station_id}({station_type}){data_source}气象数据分析 ===")

        for source in self.loader.nwp_sources:
            # 根据参数选择加载方法
            if use_processed:
                df = self.loader.load_processed_nwp_data(station_id, source)
            else:
                df = self.loader.load_raw_nwp_data(station_id, source)

            if df is None:
                print(f"无法加载{source}数据")
                continue

            print(f"\n{source}数据概览：")
            print(df.info())

            # 分析每个变量
            for var in df.columns:
                print(f"\n{var}变量统计信息：")
                print(df[var].describe())

                # 创建子图
                fig = make_subplots(rows=2,
                                    cols=2,
                                    subplot_titles=[
                                        f'{var}时间序列', f'{var}分布统计',
                                        f'{var}日变化特征', f'{var}lead_time特征'
                                    ])

                # 1. 时间序列图（取第一个lead_time的数据）
                time_series = df.xs(0, level='lead_time')[var]
                fig.add_trace(go.Scatter(x=time_series.index,
                                         y=time_series.values,
                                         mode='lines',
                                         name=f'{var}时间变化'),
                              row=1,
                              col=1)

                # 2. 分布统计
                fig.add_trace(go.Histogram(x=df[var].values,
                                           nbinsx=50,
                                           name=f'{var}分布'),
                              row=1,
                              col=2)

                # 3. 日变化特征（按小时统计）
                df['hour'] = pd.to_datetime(
                    df.index.get_level_values('time')).hour
                hourly_data = df.groupby('hour')[var].mean()
                fig.add_trace(go.Box(x=list(range(24)),
                                     y=df.groupby('hour')[var].apply(list),
                                     name=f'{var}日变化'),
                              row=2,
                              col=1)

                # 4. lead_time特征
                lead_time_data = df.groupby('lead_time')[var].mean()
                fig.add_trace(go.Box(
                    x=df.index.get_level_values('lead_time').unique(),
                    y=df.groupby('lead_time')[var].apply(list),
                    name=f'{var}lead_time特征'),
                              row=2,
                              col=2)

                # 更新布局
                fig.update_layout(
                    height=1000,
                    title_text=
                    f"场站{station_id} - {source} - {var}{data_source}数据分析",
                    showlegend=False)

                # 保存图表
                fig.write_html(
                    self.output_dir /
                    f'nwp_analysis_station_{station_id}_{source}_{var}_{data_source}.html'
                )

    def analyze_correlation(self, station_id: int, use_processed: bool = True):
        """
        分析气象数据与功率的相关性
        
        Args:
            station_id: 场站ID
            use_processed: 是否使用处理后的数据，默认为True
        """
        station_type = self.loader.get_station_type(station_id)
        data_source = "处理后" if use_processed else "原始"
        print(f"\n=== 场站{station_id}({station_type}){data_source}数据相关性分析 ===")

        # 加载功率数据
        if use_processed:
            power_df = self.loader.load_processed_power_data(station_id)
        else:
            power_df = self.loader.load_raw_power_data(station_id)

        if power_df is None:
            return

        for source in self.loader.nwp_sources:
            # 加载气象数据
            if use_processed:
                nwp_df = self.loader.load_processed_nwp_data(
                    station_id, source)
            else:
                nwp_df = self.loader.load_raw_nwp_data(station_id, source)

            if nwp_df is None:
                continue

            # 准备相关性分析数据
            # 使用第一个lead_time的数据进行相关性分析
            nwp_data = nwp_df.xs(0, level='lead_time')

            # 将功率数据与气象数据对齐
            power_df_aligned = power_df.set_index('time')
            merged_data = pd.merge(power_df_aligned['power'],
                                   nwp_data,
                                   left_index=True,
                                   right_index=True,
                                   how='inner')

            # 计算相关系数
            corr_matrix = merged_data.corr()

            # 创建热力图
            fig = make_subplots(rows=2,
                                cols=1,
                                subplot_titles=['相关性热力图', '时间序列对比（标准化）'],
                                row_heights=[0.4, 0.6],
                                vertical_spacing=0.15)

            # 1. 相关性热力图
            fig.add_trace(go.Heatmap(z=corr_matrix.values,
                                     x=corr_matrix.columns,
                                     y=corr_matrix.columns,
                                     colorscale='RdBu_r',
                                     zmid=0,
                                     text=np.round(corr_matrix.values, 2),
                                     texttemplate='%{text}',
                                     textfont={"size": 10},
                                     hoverongaps=False),
                          row=1,
                          col=1)

            # 2. 时间序列对比
            # 标准化数据以便比较
            normalized_data = (merged_data -
                               merged_data.mean()) / merged_data.std()
            for col in normalized_data.columns:
                fig.add_trace(go.Scatter(x=normalized_data.index,
                                         y=normalized_data[col],
                                         name=col,
                                         mode='lines'),
                              row=2,
                              col=1)

            fig.update_layout(
                height=1000,
                title=
                f'场站{station_id}({station_type}) - {source}{data_source}数据相关性分析',
                showlegend=True)

            # 保存图表
            fig.write_html(
                self.output_dir /
                f'correlation_analysis_station_{station_id}_{source}_{data_source}.html'
            )

    def run_analysis(self, use_processed: bool = True):
        """
        运行所有分析
        
        Args:
            use_processed: 是否使用处理后的数据，默认为True
        """
        # 分析所有场站的功率数据
        for station_id in range(1, 11):
            self.analyze_power_data(station_id, use_processed)

        # 分析风电场和光伏电站的气象数据
        self.analyze_nwp_data(1, use_processed)  # 风电场
        self.analyze_nwp_data(6, use_processed)  # 光伏电站

        # 分析相关性
        self.analyze_correlation(1, use_processed)  # 风电场
        self.analyze_correlation(6, use_processed)  # 光伏电站


def main():
    # 设置数据目录和输出目录
    data_dir = os.path.join(project_root, 'dataset')
    output_dir = os.path.join(project_root, 'outputs/analysis')

    # 创建分析器并运行分析
    analyzer = DataAnalyzer(data_dir, output_dir)

    # 分析原始数据
    print("\n=== 分析原始数据 ===")
    analyzer.run_analysis(use_processed=True)

    # 分析处理后的数据
    print("\n=== 分析处理后数据 ===")
    analyzer.run_analysis(use_processed=True)


if __name__ == '__main__':
    main()
