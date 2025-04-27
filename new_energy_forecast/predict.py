import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from models import WeatherCNN
from data_loader import PowerDataset
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import logging
from datetime import datetime

def setup_logging(log_dir: str):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'predict_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def predict(
    data_dir: str,
    station_id: int,
    model_dir: str,
    output_dir: str,
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    模型推理
    
    Args:
        data_dir: 数据目录
        station_id: 场站ID
        model_dir: 模型目录
        output_dir: 输出目录
        batch_size: 批次大小
        device: 推理设备
    """
    # 设置日志
    log_dir = os.path.join(output_dir, 'logs')
    logger = setup_logging(log_dir)
    logger.info(f'开始对场站 {station_id} 进行预测')
    
    # 加载模型配置
    config_path = os.path.join(model_dir, f'config_station_{station_id}.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 创建测试数据集
    test_dataset = PowerDataset(data_dir=data_dir, station_id=station_id, dataset='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device=='cuda' else False
    )
    
    # 创建模型
    model = WeatherCNN(in_channels=24)
    model = model.to(device)
    
    # 加载模型权重
    checkpoint = torch.load(
        os.path.join(model_dir, f'best_model_station_{station_id}.pth'),
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 预测
    predictions = []
    time_points = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(test_loader)):
            data = data.to(device)
            output = model(data)
            
            predictions.extend(output.cpu().numpy().flatten().tolist())
            # 获取时间点
            batch_times = test_dataset.time_points[
                batch_idx * batch_size:(batch_idx + 1) * batch_size
            ]
            time_points.extend(batch_times)
    
    # 创建预测结果DataFrame
    df_pred = pd.DataFrame({
        'time': time_points,
        'power': predictions
    })
    df_pred.set_index('time', inplace=True)
    
    # 保存预测结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'output{station_id}.csv')
    df_pred.to_csv(output_path)
    logger.info(f'预测结果已保存到: {output_path}')
    
    return df_pred

def predict_all_stations(
    data_dir: str,
    model_dir: str,
    output_dir: str,
    batch_size: int = 32
):
    """预测所有场站"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 预测每个场站
    all_predictions = {}
    for station_id in range(1, 11):
        df_pred = predict(
            data_dir=data_dir,
            station_id=station_id,
            model_dir=model_dir,
            output_dir=output_dir,
            batch_size=batch_size
        )
        all_predictions[station_id] = df_pred
    
    # 压缩所有输出文件
    import zipfile
    with zipfile.ZipFile(os.path.join(output_dir, 'output.zip'), 'w') as zipf:
        for station_id in range(1, 11):
            output_file = os.path.join(output_dir, f'output{station_id}.csv')
            zipf.write(output_file, os.path.basename(output_file))

if __name__ == '__main__':
    # 设置基本参数
    project_root = Path(__file__).parent
    data_dir = os.path.join(project_root, 'dataset')
    model_dir = os.path.join(project_root, 'models')
    output_dir = os.path.join(project_root, 'predictions')
    
    # 预测所有场站
    predict_all_stations(
        data_dir=data_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        batch_size=32
    ) 