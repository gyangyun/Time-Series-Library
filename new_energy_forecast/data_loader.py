import os
import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from data_preprocess import Preprocesser

class PowerDataset(Dataset):
    """电力预测数据集类，用于生成CNN训练数据"""
    
    def __init__(self, data_dir: str, station_id: int, dataset: str = 'train'):
        """
        初始化数据集
        
        Args:
            data_dir: 数据根目录路径
            station_id: 场站ID
            dataset: 数据集类型，'train' 或 'test'
        """
        self.preprocesser = Preprocesser(data_dir)
        self.station_id = station_id
        self.dataset = dataset
        
        # 定义数据源和变量
        self.nwp_sources = ['NWP_1', 'NWP_2', 'NWP_3']
        self.variables = {
            'NWP_1': ['u100', 'v100', 't2m', 'tp', 'tcc', 'sp', 'poai', 'ghi'],
            'NWP_2': ['u100', 'v100', 't2m', 'tp', 'tcc', 'msl', 'poai', 'ghi'],
            'NWP_3': ['u100', 'v100', 't2m', 'tp', 'tcc', 'sp', 'poai', 'ghi']
        }
        
        # 加载数据
        self._load_data()
        
        # 如果是训练集，计算归一化参数
        if dataset == 'train':
            self._compute_normalization_stats()
        else:
            # 如果是测试集，加载训练集的归一化参数
            self._load_normalization_stats()
            
    def _compute_normalization_stats(self):
        """计算归一化参数并保存"""
        # 将数据重塑为[N, C, H, W]格式，方便计算统计量
        data = self.nwp_data.values.reshape(-1, 24, 11, 11)  # 24 = 8变量 * 3源
        
        # 计算每个通道的均值和标准差
        means = torch.tensor([np.mean(data[:, i, :, :]) for i in range(24)])
        stds = torch.tensor([np.std(data[:, i, :, :]) for i in range(24)])
        
        # 防止除零
        stds[stds < 1e-6] = 1.0
        
        # 创建归一化transform
        self.normalize = transforms.Normalize(mean=means, std=stds)
        
        # 保存归一化参数
        stats_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(stats_dir, exist_ok=True)
        torch.save({
            'means': means,
            'stds': stds
        }, os.path.join(stats_dir, f'norm_stats_station_{self.station_id}.pt'))
        
    def _load_normalization_stats(self):
        """加载归一化参数"""
        stats_path = os.path.join(
            os.path.dirname(__file__),
            'models',
            f'norm_stats_station_{self.station_id}.pt'
        )
        if os.path.exists(stats_path):
            stats = torch.load(stats_path)
            self.normalize = transforms.Normalize(
                mean=stats['means'],
                std=stats['stds']
            )
        else:
            raise ValueError(f"找不到场站 {self.station_id} 的归一化参数文件")
        
    def _load_data(self):
        """加载并预处理数据"""
        # 加载功率数据
        if self.dataset == 'train':
            power_df = self.preprocesser.load_processed_power_data(self.station_id)
            if power_df is None:
                raise ValueError(f"无法加载场站 {self.station_id} 的功率数据")
            # 把2024-01-01剔除掉
            power_df.query('time >= "2024-01-02"', inplace=True)
            self.power_data = power_df.set_index('time')['power']
        else:
            # 测试集时间范围
            test_dates = pd.date_range(start='2025-01-01 00:00:00',
                                     end='2025-02-28 23:45:00',
                                     freq='15min')
            self.power_data = pd.Series(index=test_dates, data=np.nan)

        # 获取所有时间点
        self.time_points = sorted(self.power_data.index)
        
        # 将三个气象数据源的数据按time、lat、lon进行对齐
        df_list = {}
        for source in self.nwp_sources:
            df = self.preprocesser.load_processed_nwp_data(
                self.station_id, source, self.dataset)
            if df is None:
                raise ValueError(f"无法加载场站 {self.station_id} 的 {source} 数据")
            df.set_index(['time', 'lat', 'lon'], inplace=True)
            df.columns = [f"{source.lower()}_{col}" for col in df.columns]
            df_list[source] = df

        # 初始化特征图字典，用于存储每个时间点的特征
        self.nwp_data = pd.concat(df_list.values(), axis=1)
            
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.time_points)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (特征图, 标签)
            特征图形状: [24, 11, 11]  # 24 = 8变量 * 3源
            标签形状: [1]
        """
        # 获取当前时间点
        time = self.time_points[idx]
        
        time_data = self.nwp_data.query('time == @time').sort_index(level=['lat', 'lon'])
        # 直接将数据重塑为[24, 11, 11]的形状
        time_data = time_data.values.reshape(24, 11, 11)  # shape: [24, 11, 11]
        
        # 转换为torch.Tensor并归一化
        X = torch.from_numpy(time_data).float()
        X = self.normalize(X)
        
        y = torch.tensor([self.power_data[time]], dtype=torch.float32)
        
        return X, y
    
    def get_feature_info(self) -> Dict:
        """
        获取特征信息
        
        Returns:
            Dict: 包含特征维度信息的字典
        """
        # 计算特征通道数
        n_channels = sum(len(vars) for vars in self.variables.values())
        
        return {
            'n_channels': n_channels,
            'height': 11,
            'width': 11,
            'variables': self.variables
        }
    
    def get_station_type(self) -> str:
        """
        获取场站类型
        
        Returns:
            str: 'wind' 或 'solar'
        """
        return 'wind' if self.station_id <= 5 else 'solar'

if __name__ == "__main__":
    # 测试代码
    project_root = Path(__file__).parent
    data_dir = os.path.join(project_root, 'dataset')
    
    # 创建数据集
    station_id = 1
    dataset = PowerDataset(data_dir=data_dir, station_id=station_id, dataset='train')
    
    # 测试数据加载
    X, y = dataset[0]  # 获取第一个样本
    
    # 打印数据形状
    print(f"特征图形状: {X.shape}")  # [24, 11, 11]
    print(f"标签形状: {y.shape}")    # [1]
    
    # 打印特征信息
    feature_info = dataset.get_feature_info()
    print("\n特征信息:")
    print(f"通道数: {feature_info['n_channels']}")
    print(f"特征图大小: {feature_info['height']}x{feature_info['width']}")
    
    # 测试DataLoader
    from torch.utils.data import DataLoader
    batch_size = 32
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 获取一个batch的数据
    batch_X, batch_y = next(iter(train_loader))
    print(f"\n批次数据形状:")
    print(f"X shape: {batch_X.shape}")  # [batch_size, 24, 11, 11]
    print(f"y shape: {batch_y.shape}")  # [batch_size, 1] 