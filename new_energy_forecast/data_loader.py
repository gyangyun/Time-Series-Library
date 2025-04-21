import os
import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from glob import glob


class DataLoader:
    """数据加载器类，用于加载和预处理气象数据和功率数据"""

    def __init__(self, data_dir: str):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据根目录路径
        """
        self.data_dir = data_dir
        # 创建raw和processed目录
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        # 定义数据源和变量
        self.nwp_sources = ['NWP_1', 'NWP_2', 'NWP_3']
        self.variables = {
            'NWP_1': ['u100', 'v100', 't2m', 'tp', 'tcc', 'sp', 'poai', 'ghi'],
            'NWP_2':
            ['u100', 'v100', 't2m', 'tp', 'tcc', 'msl', 'poai', 'ghi'],
            'NWP_3': ['u100', 'v100', 't2m', 'tp', 'tcc', 'sp', 'poai', 'ghi']
        }

        # 定义训练集数据文件路径
        self.train_data_dir = os.path.join(self.raw_dir, '初赛训练集')
        self.train_nwp_dir = os.path.join(self.train_data_dir,
                                          'nwp_data_train')
        self.train_fact_dir = os.path.join(self.train_data_dir, 'fact_data')

        # 定义测试集数据文件路径
        self.test_data_dir = os.path.join(self.raw_dir, '初赛测试集')
        self.test_nwp_dir = os.path.join(self.test_data_dir, 'nwp_data_test')

    def load_raw_nwp_data(self,
                          station_id: int,
                          source: str,
                          dataset: str = 'train') -> Optional[pd.DataFrame]:
        """
        从原始目录加载指定场站的所有气象数据
        
        Args:
            station_id: 场站ID
            source: 气象数据源（NWP_1, NWP_2, NWP_3）
            dataset: 数据集类型，'train' 或 'test'
            
        Returns:
            pandas.DataFrame对象或None（如果文件不存在）
        """
        # 根据数据集类型选择目录
        if dataset == 'train':
            raw_dir = os.path.join(self.train_nwp_dir, str(station_id), source)
        elif dataset == 'test':
            raw_dir = os.path.join(self.test_nwp_dir, str(station_id), source)
        else:
            raise ValueError("dataset参数必须是'train'或'test'")

        if not os.path.exists(raw_dir):
            print(f"原始目录不存在: {raw_dir}")
            return None

        try:
            # 读取所有文件并合并
            datasets = []
            for file in sorted(glob(os.path.join(raw_dir, '*.nc'))):
                ds = xr.open_dataset(file)
                # 预处理数据
                ds = self.preprocess_nwp_data(ds)
                # 转换为DataFrame
                df = self._process_nwp_to_df(ds)
                datasets.append(df)

            # 合并所有DataFrame
            return pd.concat(datasets, axis=0) if datasets else None
        except Exception as e:
            print(
                f"Error loading raw NWP data for station {station_id}, source {source}: {e}"
            )
            return None

    def load_processed_nwp_data(
            self,
            station_id: int,
            source: str,
            dataset: str = 'train') -> Optional[pd.DataFrame]:
        """
        从处理后目录加载指定场站的所有气象数据
        
        Args:
            station_id: 场站ID
            source: 气象数据源（NWP_1, NWP_2, NWP_3）
            dataset: 数据集类型，'train' 或 'test'
            
        Returns:
            pandas.DataFrame对象或None（如果文件不存在）
        """
        processed_file = os.path.join(
            self.processed_dir,
            f'station_{station_id}_{source}_{dataset}_processed.parquet')

        if not os.path.exists(processed_file):
            print(f"处理后的文件不存在: {processed_file}")
            return None

        try:
            return pd.read_parquet(processed_file)
        except Exception as e:
            print(f"Error loading processed file {processed_file}: {e}")
            return None

    def load_raw_power_data(self, station_id: int) -> Optional[pd.DataFrame]:
        """
        从原始目录加载指定场站的功率数据
        
        Args:
            station_id: 场站ID
            
        Returns:
            pandas.DataFrame对象或None（如果文件不存在）
        """
        raw_file = os.path.join(self.train_fact_dir,
                                f"{station_id}_normalization_train.csv")
        if not os.path.exists(raw_file):
            print(f"原始文件不存在: {raw_file}")
            return None

        try:
            power_df = pd.read_csv(raw_file, parse_dates=['时间'])
            power_df.rename(columns={
                '时间': 'time',
                '功率(MW)': 'power'
            },
                            inplace=True)
            return power_df
        except Exception as e:
            print(f"Error loading raw file {raw_file}: {e}")
            return None

    def load_processed_power_data(self,
                                  station_id: int) -> Optional[pd.DataFrame]:
        """
        加载处理后的功率数据，如果处理后的数据不存在，则加载原始数据并进行处理
        
        Args:
            station_id: 场站ID
            
        Returns:
            处理后的功率数据DataFrame，如果数据不存在则返回None
        """
        # 构建处理后数据文件路径
        processed_file = os.path.join(
            self.processed_dir,
            f'station_{station_id}_power_processed.parquet')

        try:
            # 尝试加载处理后的数据
            if os.path.exists(processed_file):
                return pd.read_parquet(processed_file)

            # 如果处理后的数据不存在，加载原始数据并处理
            raw_df = self.load_raw_power_data(station_id)
            if raw_df is None:
                return None

            # 预处理数据
            processed_df = self.preprocess_power_data(raw_df)

            # 保存处理后的数据
            os.makedirs(self.processed_dir, exist_ok=True)
            processed_df.to_parquet(processed_file)
            print(f"已将处理后的功率数据保存到: {processed_file}")

            return processed_df

        except Exception as e:
            print(f"加载处理后的功率数据时出错: {e}")
            return None

    def preprocess_nwp_data(self, ds: xr.Dataset) -> xr.Dataset:
        """
        预处理气象数据
        
        Args:
            ds: 原始气象数据Dataset
            
        Returns:
            处理后的Dataset
        """
        # 基础数据清洗
        ds = ds.copy()

        # 处理异常值
        # 处理每个channel的异常值
        for channel in ds.channel.values:
            # 获取当前channel的数据
            channel_data = ds.sel(channel=channel)['data']

            # 替换无效值
            channel_data = channel_data.where(channel_data != -999, np.nan)

            # 计算分位数
            q1 = channel_data.quantile(0.01)
            q99 = channel_data.quantile(0.99)

            # 根据变量类型设置不同的异常值处理策略
            if channel in ['t2m', 'sp', 'msl']:  # 温度、气压类变量
                # 使用更严格的阈值
                q1 = channel_data.quantile(0.005)
                q99 = channel_data.quantile(0.995)
            elif channel in ['u100', 'v100']:  # 风速类变量
                # 风速通常有更大的波动范围
                q1 = channel_data.quantile(0.02)
                q99 = channel_data.quantile(0.98)
            elif channel in ['ghi', 'poai']:  # 辐照度类变量
                # 辐照度不能为负
                q1 = 0
                q99 = channel_data.quantile(0.99)
            elif channel == 'tp':  # 降雨量变量
                # 降雨量不能为负，且通常有较大的偏态分布
                q1 = 0
                q99 = channel_data.quantile(0.999)  # 使用更严格的阈值，因为降雨量通常有长尾分布
            elif channel == 'tcc':  # 云量变量
                # 云量范围在0-1之间
                q1 = 0
                q99 = 1

            # 应用异常值处理
            channel_data = channel_data.clip(q1, q99)

            # 更新数据集
            ds['data'].loc[dict(channel=channel)] = channel_data

        return ds

    def preprocess_power_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理功率数据
        
        Args:
            df: 原始功率数据DataFrame
            
        Returns:
            处理后的DataFrame
        """
        df = df.copy()

        # 处理缺失值
        df['power'] = df['power'].interpolate(method='linear', limit=4)

        # 异常值处理
        q1 = df['power'].quantile(0.01)
        q99 = df['power'].quantile(0.99)
        df.loc[df['power'] < q1, 'power'] = q1
        df.loc[df['power'] > q99, 'power'] = q99

        return df

    def get_station_type(self, station_id: int) -> str:
        """
        获取场站类型
        
        Args:
            station_id: 场站ID
            
        Returns:
            str: 'wind' 或 'solar'
        """
        return 'wind' if station_id <= 5 else 'solar'

    def _process_nwp_to_df(self, ds: xr.Dataset) -> pd.DataFrame:
        """
        将xarray Dataset转换为pandas DataFrame,保留所有维度信息
        
        Args:
            ds: 气象数据Dataset
            
        Returns:
            处理后的DataFrame,包含time、lead_time、lat、lon和所有气象变量
        """
        # 使用xarray的to_dataframe方法直接转换
        df = ds.to_dataframe().reset_index()

        # 重命名列
        df = df.rename(columns={'data': 'value'})

        # 将value列拆分成多个channel列
        df = df.pivot_table(index=['time', 'lead_time', 'lat', 'lon'],
                            columns='channel',
                            values='value').reset_index()

        # 处理时间列
        df.rename(columns={'time': 'date', 'lead_time': 'hour'}, inplace=True)
        # 气象数据每个文件是第二天北京时间0点开始的未来24小时气象预报，1230的数据是对1231的预报。本次比赛数据是从对2024年1月2日的预报数据开始的。可以理解为少了20240101的天气数据，但功率里面有20240101。
        df['date'] = df['date'].dt.normalize() + pd.Timedelta(days=1)
        df['time'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')
        df.drop(columns=['date', 'hour'], inplace=True)
        return df

    def process_and_save_all_nwp_data(self, dataset: str = 'train') -> None:
        """
        处理并保存所有场站的所有气象数据为DataFrame格式
        
        Args:
            dataset: 数据集类型，'train' 或 'test'
        """
        if dataset not in ['train', 'test']:
            raise ValueError("dataset参数必须是'train'或'test'")

        print(f"\n处理{dataset}集数据...")
        for station_id in range(1, 11):
            print(f"处理场站 {station_id} 的数据...")

            for source in self.nwp_sources:
                print(f"处理 {source} 数据...")

                # 加载并处理数据
                df = self.load_raw_nwp_data(station_id, source, dataset)
                if df is None:
                    print(f"无法加载场站 {station_id} 的 {source} {dataset}数据")
                    continue

                # 保存为parquet格式
                output_file = os.path.join(
                    self.processed_dir,
                    f'station_{station_id}_{source}_{dataset}_processed.parquet'
                )
                df.to_parquet(output_file)
                print(f"已保存到: {output_file}")

    def process_and_save_all_power_data(self) -> None:
        """
        处理并保存所有指定场站的功率数据
        
        Args:
            station_ids: 场站ID列表
        """
        for station_id in range(1, 11):
            try:
                # 加载原始数据
                raw_df = self.load_raw_power_data(station_id)
                if raw_df is None:
                    print(f"无法加载场站 {station_id} 的原始功率数据，跳过处理")
                    continue

                # 预处理数据
                processed_df = self.preprocess_power_data(raw_df)

                # 保存处理后的数据
                output_file = os.path.join(
                    self.processed_dir,
                    f'station_{station_id}_power_processed.parquet')
                processed_df.to_parquet(output_file)
                print(f"已将场站 {station_id} 的处理后功率数据保存到: {output_file}")

            except Exception as e:
                print(f"处理场站 {station_id} 的功率数据时出错: {e}")
                continue

    def merge_power_and_nwp_data(
            self,
            station_id: int,
            dataset: str = 'train',
            force_remerge: bool = False) -> Optional[pd.DataFrame]:
        """
        合并功率数据和气象数据，并缓存结果
        
        Args:
            station_id: 场站ID
            dataset: 数据集类型，'train' 或 'test'
            force_remerge: 是否强制重新合并数据，即使缓存文件存在
            
        Returns:
            合并后的DataFrame，包含功率数据和三个气象数据源的数据，如果数据不存在则返回None
        """
        # 构建缓存文件路径
        cache_file = os.path.join(
            self.processed_dir,
            f'station_{station_id}_{dataset}_merged_data.parquet')

        # 如果缓存文件存在且不强制重新合并，直接返回缓存数据
        if not force_remerge and os.path.exists(cache_file):
            try:
                return pd.read_parquet(cache_file)
            except Exception as e:
                print(f"读取缓存文件失败: {e}")

        try:
            # 初始化合并后的数据框
            if dataset == 'train':
                # 训练集包含功率数据
                power_df = self.load_processed_power_data(station_id)
                if power_df is None:
                    print(f"无法加载场站 {station_id} 的功率数据")
                    return None
                merged_df = power_df.copy()
            else:
                # 创建15分钟间隔的时间序列
                time_index = pd.date_range(start='2025-01-01 00:00:00',
                                           end='2025-02-28 23:45:00',
                                           freq='15min')
                merged_df = pd.DataFrame({'time': time_index})

            # 加载并合并每个气象数据源的数据
            for source in self.nwp_sources:
                # 加载气象数据
                nwp_df = self.load_processed_nwp_data(station_id, source,
                                                      dataset)
                if nwp_df is None:
                    print(f"无法加载场站 {station_id} 的 {source} 数据")
                    continue

                # 只选择中心点数据
                nwp_df = nwp_df[nwp_df['lat'] == 5]
                nwp_df = nwp_df[nwp_df['lon'] == 5]

                # 删除lat和lon列
                nwp_df = nwp_df.drop(['lat', 'lon'], axis=1)

                # 重命名列，添加数据源前缀
                rename_dict = {
                    var: f"{source.lower()}_{var}"
                    for var in self.variables[source]
                }
                nwp_df = nwp_df.rename(columns=rename_dict)

                # 将时间向下取整到15分钟
                nwp_df['time'] = pd.to_datetime(
                    nwp_df['time']).dt.floor('15min')

                # 如果有重复的时间戳，取平均值
                nwp_df = nwp_df.groupby('time').mean().reset_index()

                # 合并到主数据框
                merged_df = pd.merge(merged_df, nwp_df, on='time', how='left')

            # 按时间排序
            merged_df = merged_df.sort_values('time', ascending=True)

            # 对气象数据进行插值，填充15分钟间隔的缺失值
            numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
            merged_df[numeric_cols] = merged_df[numeric_cols].interpolate(
                method='linear')

            # 保存合并后的数据到缓存文件
            try:
                merged_df.to_parquet(cache_file)
                print(f"已将合并数据保存到: {cache_file}")
            except Exception as e:
                print(f"保存合并数据时出错: {e}")

            return merged_df

        except Exception as e:
            print(f"合并场站 {station_id} 的数据时出错: {e}")
            return None

    def get_merged_data_info(self,
                             station_id: int,
                             dataset: str = 'train') -> None:
        """
        打印合并后数据的信息
        
        Args:
            station_id: 场站ID
            dataset: 数据集类型，'train' 或 'test'
        """
        merged_df = self.merge_power_and_nwp_data(station_id, dataset)
        if merged_df is not None:
            print(f"\n场站 {station_id} {dataset}集合并后的数据信息:")
            print(f"数据形状: {merged_df.shape}")
            print(
                f"时间范围: {merged_df['time'].min()} 到 {merged_df['time'].max()}")
            print("\n列名:")
            for col in merged_df.columns:
                print(f"- {col}")
            print("\n数据预览:")
            print(merged_df.head())
            print("\n数据统计信息:")
            print(merged_df.describe())
        else:
            print(f"无法获取场站 {station_id} 的合并数据信息")


if __name__ == "__main__":
    project_root = Path(__file__).parent
    data_dir = os.path.join(project_root, 'dataset')  # 修改数据目录路径
    loader = DataLoader(data_dir=data_dir)

    # # 处理并保存所有气象数据和功率数据
    # print("开始处理所有气象数据...")
    # # 处理训练集
    # loader.process_and_save_all_nwp_data(dataset='train')
    # # 处理测试集
    # loader.process_and_save_all_nwp_data(dataset='test')

    # print("\n开始处理所有功率数据...")
    # loader.process_and_save_all_power_data()

    # # 测试数据加载功能
    # print("\n测试数据加载功能...")

    # # 测试气象数据加载
    # print("\n测试气象数据加载:")
    # # for station_id in range(1, 11):
    # for station_id in range(1, 2):
    #     for source in ['NWP_1', 'NWP_2', 'NWP_3']:
    #         print(f"\n加载场站 {station_id} 的 {source} 数据...")

    #         # 测试加载原始数据
    #         print("从原始目录加载:")
    #         df = loader.load_raw_nwp_data(station_id, source)
    #         if df is not None:
    #             print(f"成功加载数据，时间范围: {df['time'].min()} 到 {df['time'].max()}")
    #             print(f"数据列: {df.columns.tolist()}")
    #             print(f"数据形状: {df.shape}")

    #         # 测试加载处理后的数据
    #         print("\n从处理后目录加载:")
    #         df = loader.load_processed_nwp_data(station_id, source)
    #         if df is not None:
    #             print(f"成功加载数据，时间范围: {df['time'].min()} 到 {df['time'].max()}")
    #             print(f"数据列: {df.columns.tolist()}")
    #             print(f"数据形状: {df.shape}")

    # # 测试功率数据加载
    # print("\n测试功率数据加载:")
    # # for station_id in range(1, 11):
    # for station_id in range(1, 2):
    #     print(f"\n加载场站 {station_id} 的功率数据...")

    #     # 测试加载原始数据
    #     print("从原始目录加载:")
    #     power_df = loader.load_raw_power_data(station_id)
    #     if power_df is not None:
    #         print(
    #             f"成功加载数据，时间范围: {power_df['time'].min()} 到 {power_df['time'].max()}"
    #         )
    #         print(f"数据列: {power_df.columns.tolist()}")
    #         print(f"数据行数: {len(power_df)}")

    #     # 测试加载处理后的数据
    #     print("\n从处理后目录加载:")
    #     power_df = loader.load_processed_power_data(station_id)
    #     if power_df is not None:
    #         print(
    #             f"成功加载数据，时间范围: {power_df['time'].min()} 到 {power_df['time'].max()}"
    #         )
    #         print(f"数据列: {power_df.columns.tolist()}")
    #         print(f"数据行数: {len(power_df)}")
    # print("\n数据加载测试完成")

    # # 测试数据合并和缓存功能
    print("\n测试数据合并和缓存功能:")
    for station_id in range(1, 11):
        for dataset in ['train', 'test']:
            print(f"\n处理场站 {station_id} 的{dataset}数据:")

            # 首次合并数据（会创建缓存）
            print("首次合并数据...")
            merged_data = loader.merge_power_and_nwp_data(station_id, dataset)
            if merged_data is not None:
                print(f"合并数据形状: {merged_data.shape}")

            # # 从缓存加载数据
            # print("\n从缓存加载数据...")
            # cached_data = loader.merge_power_and_nwp_data(station_id, dataset)
            # if cached_data is not None:
            #     print(f"缓存数据形状: {cached_data.shape}")

            # # 强制重新合并数据
            # print("\n强制重新合并数据...")
            # remerged_data = loader.merge_power_and_nwp_data(station_id,
            #                                                 dataset,
            #                                                 force_remerge=True)
            # if remerged_data is not None:
            #     print(f"重新合并数据形状: {remerged_data.shape}")

    train1 = loader.merge_power_and_nwp_data(1, 'train')
    test1 = loader.merge_power_and_nwp_data(1, 'test')
