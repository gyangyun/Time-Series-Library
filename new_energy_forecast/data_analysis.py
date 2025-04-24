import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# 添加项目根目录到系统路径
project_root = Path(__file__).parent
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

        # 定义风电站和光伏电站的特征组
        self.wind_features = {
            'wind': ['u100', 'v100'],  # 风速相关
            'weather': ['t2m', 'tp', 'tcc'],  # 天气相关
            'pressure': ['sp', 'msl']  # 气压相关
        }

        self.solar_features = {
            'radiation': ['poai', 'ghi'],  # 辐照度相关
            'weather': ['t2m', 'tp', 'tcc'],  # 天气相关
            'pressure': ['sp', 'msl']  # 气压相关
        }

    def analyze_power_data(self, station_id: int):
        """
        分析单个场站的功率数据
        
        Args:
            station_id: 场站ID
        """
        power_df = self.loader.load_processed_power_data(station_id)
        if power_df is None:
            print(f"无法加载场站{station_id}的功率数据")
            return

        station_type = self.loader.get_station_type(station_id)
        print(f"\n=== 场站{station_id}({station_type})功率数据分析 ===")

        # 基本统计信息
        print("\n基本统计信息：")
        print(power_df['power'].describe())

        # 缺失值信息
        print("\n缺失值统计：")
        missing = power_df.isnull().sum()
        print(missing)

        # 创建子图布局
        fig = make_subplots(rows=3,
                            cols=1,
                            subplot_titles=[
                                f'场站{station_id}({station_type})功率时间序列',
                                f'场站{station_id}({station_type})功率分布',
                                f'场站{station_id}({station_type})各小时功率分布'
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
        fig.update_layout(height=1200,
                          showlegend=False,
                          title_text=f"场站{station_id}({station_type})功率数据分析")

        # 更新x轴和y轴标签
        fig.update_xaxes(title_text="时间", row=1, col=1)
        fig.update_xaxes(title_text="功率", row=2, col=1)
        fig.update_xaxes(title_text="小时", row=3, col=1)

        fig.update_yaxes(title_text="功率", row=1, col=1)
        fig.update_yaxes(title_text="频次", row=2, col=1)
        fig.update_yaxes(title_text="功率", row=3, col=1)

        # 保存图表
        fig.write_html(self.output_dir /
                       f'power_analysis_station_{station_id}.html')

    def analyze_nwp_data(self, station_id: int):
        """
        分析单个场站的所有气象数据
        
        Args:
            station_id: 场站ID
        """
        station_type = self.loader.get_station_type(station_id)
        print(f"\n=== 场站{station_id}({station_type})气象数据分析 ===")

        # for source in self.loader.nwp_sources:
        for source in ['NWP_1']:
            df = self.loader.load_processed_nwp_data(station_id, source)
            if df is None:
                print(f"无法加载{source}数据")
                continue

            print(f"\n{source}数据概览：")
            print(df.info())

            # 分析每个变量
            # 只分析5,5位置的气象数据
            df.query('lat == 5 and lon == 5', inplace=True)
            df.set_index('time', inplace=True)
            for var in df.columns:
                print(f"\n{var}变量统计信息：")
                print(df[var].describe())

                # 创建子图
                fig = make_subplots(rows=2,
                                    cols=2,
                                    subplot_titles=[
                                        f'{var}时间序列', f'{var}分布统计',
                                        f'{var}日变化特征', f'{var}time特征'
                                    ])

                # 1. 时间序列图
                time_series = df[var]
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

                # 3. 日变化特征
                df['hour'] = df.index.hour
                daily_pattern = df.groupby('hour')[var].mean()
                fig.add_trace(go.Scatter(x=daily_pattern.index,
                                         y=daily_pattern.values,
                                         mode='lines+markers',
                                         name=f'{var}日变化'),
                              row=2,
                              col=1)

                # 4. 月变化特征
                df['month'] = df.index.month
                monthly_pattern = df.groupby('month')[var].mean()
                fig.add_trace(go.Scatter(x=monthly_pattern.index,
                                         y=monthly_pattern.values,
                                         mode='lines+markers',
                                         name=f'{var}月变化'),
                              row=2,
                              col=2)

                # 更新布局
                fig.update_layout(
                    height=1000,
                    title_text=f"场站{station_id} - {source} - {var}数据分析",
                    showlegend=False)

                # 保存图表
                fig.write_html(
                    self.output_dir /
                    f'nwp_analysis_station_{station_id}_{source}_{var}.html')

    def analyze_feature_importance(self):
        """
        分析特征重要性
        """
        print("\n=== 特征重要性分析 ===")

        # 分别分析风电站和光伏电站
        self._analyze_station_type(list(range(1, 6)), "风电站")
        self._analyze_station_type(list(range(6, 11)), "光伏电站")

    def _analyze_station_type(self, station_ids: list, station_type: str):
        """
        分析特定类型场站的特征重要性
        
        Args:
            station_ids: 场站ID列表
            station_type: 场站类型（"风电站"或"光伏电站"）
        """
        print(f"\n{station_type}特征重要性分析:")

        all_feature_importance = {}
        all_correlations = {}

        for station_id in station_ids:
            # 加载合并后的数据
            merged_data = self.loader.merge_power_and_nwp_data(
                station_id, 'train')
            if merged_data is None:
                continue

            # 不要2024-01-01，因为这天没有气象预报数据
            merged_data = merged_data[merged_data['time'] >= '2024-01-02']

            # 准备特征和目标变量
            features = []
            # for source in ['nwp_1', 'nwp_2', 'nwp_3']:
            for source in ['nwp_1']:
                for var in self.wind_features if station_type == "风电站" else self.solar_features:
                    for feature in self.wind_features[
                            var] if station_type == "风电站" else self.solar_features[
                                var]:
                        col = f"{source}_{feature}"
                        if col in merged_data.columns:
                            features.append(col)

            X = merged_data[features]
            y = merged_data['power']

            # 处理缺失值
            # 1. 删除所有特征都是NaN的行
            mask = ~X.isna().all(axis=1)
            X = X[mask]
            y = y[mask]

            # 2. 对剩余的NaN值进行插值
            X = X.interpolate(
                method='linear',
                axis=0).fillna(method='bfill').fillna(method='ffill')

            # 计算相关性
            correlations = pd.concat([X, y],
                                     axis=1).corr()['power'].drop('power')
            all_correlations[station_id] = correlations

            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 使用随机森林计算特征重要性
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y)

            # 保存特征重要性
            importance = pd.Series(rf.feature_importances_, index=features)
            all_feature_importance[station_id] = importance

            print(f"\n场站{station_id}的Top 10特征:")
            print("\n相关性最强的特征:")
            print(correlations.abs().sort_values(ascending=False).head(10))
            print("\n随机森林特征重要性:")
            print(importance.sort_values(ascending=False).head(10))

        # 绘制特征重要性热力图
        self._plot_feature_importance(all_feature_importance, all_correlations,
                                      station_type)

    def _plot_feature_importance(self, feature_importance: dict,
                                 correlations: dict, station_type: str):
        """
        绘制特征重要性热力图
        
        Args:
            feature_importance: 特征重要性字典
            correlations: 相关性字典
            station_type: 场站类型
        """
        # 合并所有场站的特征重要性
        importance_df = pd.DataFrame(feature_importance).fillna(0)
        correlation_df = pd.DataFrame(correlations).fillna(0)

        # 计算平均特征重要性和相关性
        mean_importance = importance_df.mean(axis=1)
        mean_correlation = correlation_df.mean(axis=1)

        # 创建综合评分
        combined_score = pd.DataFrame({
            '特征重要性':
            mean_importance,
            '相关性系数':
            mean_correlation.abs(),
            '综合得分':
            mean_importance * mean_correlation.abs()
        })

        # 按综合得分排序
        combined_score = combined_score.sort_values('综合得分', ascending=False)

        # 创建图表
        fig = make_subplots(rows=1,
                            cols=2,
                            subplot_titles=['特征重要性分布', '特征相关性分布'],
                            specs=[[{
                                'type': 'bar'
                            }, {
                                'type': 'bar'
                            }]])

        # 添加特征重要性柱状图
        fig.add_trace(go.Bar(x=combined_score.index,
                             y=combined_score['特征重要性'],
                             name='特征重要性'),
                      row=1,
                      col=1)

        # 添加相关性柱状图
        fig.add_trace(go.Bar(x=combined_score.index,
                             y=combined_score['相关性系数'],
                             name='相关性系数'),
                      row=1,
                      col=2)

        # 更新布局
        fig.update_layout(title_text=f"{station_type}特征分析",
                          height=800,
                          showlegend=True,
                          barmode='group')

        # 保存图表
        fig.write_html(self.output_dir /
                       f'feature_importance_{station_type}.html')

        # 打印综合分析结果
        print(f"\n{station_type}特征综合分析结果:")
        print("\nTop 10最重要特征（综合评分）:")
        print(combined_score.head(10))

        # 按特征组分类分析
        feature_groups = self.wind_features if station_type == "风电站" else self.solar_features
        print("\n特征组分析:")
        for group_name, features in feature_groups.items():
            group_scores = []
            for feature in features:
                # for source in ['nwp_1', 'nwp_2', 'nwp_3']:
                for source in ['nwp_1']:
                    col = f"{source}_{feature}"
                    if col in combined_score.index:
                        group_scores.append(combined_score.loc[col, '综合得分'])
            if group_scores:
                print(f"\n{group_name}组平均得分: {np.mean(group_scores):.4f}")

    def analyze_nwp_sources_comparison(self, station_id: int):
        """
        分析同一站点不同气象数据源之间的关系
        
        Args:
            station_id: 场站ID
        """
        print(f"\n=== 场站{station_id}不同气象数据源对比分析 ===")

        # 加载三个数据源的数据
        nwp_data = {}
        for source in ['NWP_1', 'NWP_2', 'NWP_3']:
            df = self.loader.load_processed_nwp_data(station_id, source)
            if df is not None:
                # 只分析中心点(5,5)的数据
                df = df.query('lat == 5 and lon == 5').copy()
                df.set_index('time', inplace=True)
                nwp_data[source] = df

        if len(nwp_data) < 2:
            print("数据源不足，无法进行对比分析")
            return

        # 加载功率数据用于对比预测能力
        power_df = self.loader.load_processed_power_data(station_id)
        if power_df is not None:
            power_df.set_index('time', inplace=True)

        # 1. 分析不同数据源同一特征的相关性
        station_type = self.loader.get_station_type(station_id)
        features = self.wind_features if station_type == "风电站" else self.solar_features

        print("\n1. 不同数据源同一特征的相关性分析:")
        for feature_group, feature_list in features.items():
            print(f"\n{feature_group}组特征对比:")
            for feature in feature_list:
                # 收集所有数据源的这个特征
                feature_data = {}
                for source, df in nwp_data.items():
                    if feature in df.columns:
                        feature_data[source] = df[feature]

                if len(feature_data) > 1:
                    # 计算相关性矩阵
                    corr_matrix = pd.DataFrame(feature_data).corr()
                    print(f"\n{feature}特征在不同数据源间的相关性:")
                    print(corr_matrix)

                    # 创建相关性热力图
                    fig = go.Figure(
                        data=go.Heatmap(z=corr_matrix.values,
                                        x=corr_matrix.columns,
                                        y=corr_matrix.index,
                                        text=np.round(corr_matrix.values, 3),
                                        texttemplate='%{text}',
                                        textfont={"size": 10},
                                        hoverongaps=False))

                    fig.update_layout(
                        title=f"场站{station_id} - {feature}特征不同数据源相关性",
                        height=500,
                        width=500)

                    fig.write_html(
                        self.output_dir /
                        f'nwp_sources_correlation_station_{station_id}_{feature}.html'
                    )

        # 2. 分析不同数据源的预测能力
        if power_df is not None:
            print("\n2. 不同数据源的预测能力分析:")
            prediction_corr = {}
            for source, df in nwp_data.items():
                # 对齐时间索引
                common_index = df.index.intersection(power_df.index)
                if len(common_index) > 0:
                    source_corr = {}
                    for col in df.columns:
                        corr = df.loc[common_index,
                                      col].corr(power_df.loc[common_index,
                                                             'power'])
                        source_corr[col] = corr
                    prediction_corr[source] = source_corr

            # 转换为DataFrame便于对比
            pred_corr_df = pd.DataFrame(prediction_corr)
            print("\n各数据源特征与功率的相关性:")
            print(pred_corr_df)

            # 创建柱状图对比
            fig = go.Figure()
            for source in pred_corr_df.columns:
                fig.add_trace(
                    go.Bar(
                        name=source,
                        x=pred_corr_df.index,
                        y=pred_corr_df[source],
                        text=np.round(pred_corr_df[source], 3),
                        textposition='auto',
                    ))

            fig.update_layout(title=f"场站{station_id} - 不同数据源特征与功率的相关性对比",
                              barmode='group',
                              height=600,
                              width=1000)

            fig.write_html(
                self.output_dir /
                f'nwp_sources_power_correlation_station_{station_id}.html')

        # 3. 分析不同数据源特征的时序差异
        print("\n3. 不同数据源特征的时序差异分析:")
        for feature_group, feature_list in features.items():
            for feature in feature_list:
                feature_data = {}
                for source, df in nwp_data.items():
                    if feature in df.columns:
                        feature_data[source] = df[feature]

                if len(feature_data) > 1:
                    # 创建时间序列对比图
                    fig = go.Figure()
                    for source, series in feature_data.items():
                        fig.add_trace(
                            go.Scatter(x=series.index,
                                       y=series.values,
                                       mode='lines',
                                       name=f'{source}-{feature}'))

                    fig.update_layout(
                        title=f"场站{station_id} - {feature}特征不同数据源时序对比",
                        height=500,
                        width=1000)

                    fig.write_html(
                        self.output_dir /
                        f'nwp_sources_timeseries_station_{station_id}_{feature}.html'
                    )

                    # 计算统计指标
                    stats = pd.DataFrame({
                        source: {
                            '均值': series.mean(),
                            '标准差': series.std(),
                            '最小值': series.min(),
                            '最大值': series.max(),
                            '中位数': series.median()
                        }
                        for source, series in feature_data.items()
                    })
                    print(f"\n{feature}特征在不同数据源的统计指标:")
                    print(stats)

    def run_analysis(self):
        """运行所有分析"""
        # 运行特征重要性分析
        # self.analyze_feature_importance()

        # 运行数据源对比分析
        for station_id in range(1, 11):
            self.analyze_nwp_sources_comparison(station_id)

        # 运行其他分析...
        # for station_id in range(1, 11):
        #     self.analyze_power_data(station_id)
        #     self.analyze_nwp_data(station_id)


def main():
    # 设置数据目录和输出目录
    data_dir = os.path.join(project_root, 'dataset')
    output_dir = os.path.join(project_root, 'outputs/analysis')

    # 创建分析器并运行分析
    analyzer = DataAnalyzer(data_dir, output_dir)
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
