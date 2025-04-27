import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import numpy as np
from models import WeatherCNN
from data_loader import PowerDataset
from datetime import datetime
import json

def setup_logging(log_dir: str):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train_model(
    data_dir: str,
    station_id: int,
    model_dir: str,
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    训练模型
    
    Args:
        data_dir: 数据目录
        station_id: 场站ID
        model_dir: 模型保存目录
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        device: 训练设备
    """
    # 设置日志
    log_dir = os.path.join(model_dir, 'logs')
    logger = setup_logging(log_dir)
    logger.info(f'开始训练场站 {station_id} 的模型')
    
    # 创建数据集
    train_dataset = PowerDataset(data_dir=data_dir, station_id=station_id, dataset='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device=='cuda' else False
    )
    
    # 创建模型
    model = WeatherCNN(in_channels=24)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 保存模型配置
    model_config = {
        'station_id': station_id,
        'in_channels': 24,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'device': device
    }
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, f'config_station_{station_id}.json'), 'w') as f:
        json.dump(model_config, f, indent=4)
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                          f'Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch} Average Loss: {avg_loss:.6f}')
        
        # 更新学习率
        scheduler.step(avg_loss)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(model_dir, f'best_model_station_{station_id}.pth'))
            logger.info(f'保存最佳模型, loss: {best_loss:.6f}')
    
    logger.info(f'训练完成, 最佳loss: {best_loss:.6f}')

if __name__ == '__main__':
    # 设置基本参数
    project_root = Path(__file__).parent
    data_dir = os.path.join(project_root, 'dataset')
    model_dir = os.path.join(project_root, 'models')
    
    # 训练所有场站的模型
    for station_id in range(1, 11):
        train_model(
            data_dir=data_dir,
            station_id=station_id,
            model_dir=model_dir,
            batch_size=32,
            epochs=100,
            learning_rate=0.001
        ) 