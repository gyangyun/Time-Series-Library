import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop

def train_model(
    data_dir: str,
    station_id: int,
    model_dir: str,
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 0.001,
    patience: int = 7,
    val_ratio: float = 0.2,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    训练模型
    
    Args:
        data_dir: 数据目录
        station_id: 场站ID
        model_dir: 模型保存目录
        batch_size: 批次大小
        epochs: 最大训练轮数
        learning_rate: 学习率
        patience: 早停耐心值
        val_ratio: 验证集比例
        device: 训练设备
    """
    # 设置日志
    log_dir = os.path.join(model_dir, 'logs')
    logger = setup_logging(log_dir)
    logger.info(f'开始训练场站 {station_id} 的模型')
    
    # 创建数据集
    dataset = PowerDataset(data_dir=data_dir, station_id=station_id, dataset='train')
    
    # 划分训练集和验证集
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device=='cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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
    
    # 初始化早停
    early_stopping = EarlyStopping(patience=patience)
    
    # 保存模型配置
    model_config = {
        'station_id': station_id,
        'in_channels': 24,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'device': device,
        'val_ratio': val_ratio,
        'patience': patience
    }
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, f'config_station_{station_id}.json'), 'w') as f:
        json.dump(model_config, f, indent=4)
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                          f'Loss: {loss.item():.6f}')
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f'Epoch {epoch} Train Loss: {avg_train_loss:.6f} '
                   f'Val Loss: {avg_val_loss:.6f}')
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(model_dir, f'best_model_station_{station_id}.pth'))
            logger.info(f'保存最佳模型, val_loss: {best_val_loss:.6f}')
        
        # 早停检查
        if early_stopping(avg_val_loss):
            logger.info(f'触发早停机制，在epoch {epoch}')
            break
    
    logger.info(f'训练完成, 最佳val_loss: {best_val_loss:.6f}')

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
            learning_rate=0.001,
            patience=7,
            val_ratio=0.2
        ) 