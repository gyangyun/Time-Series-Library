import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime


def update_dataset(data_path, pred_date, target_cols):
    """
    更新数据集，将预测结果添加到数据集中
    
    Args:
        data_path: 数据集路径
        pred_date: 预测日期 (YYYYMMDD格式)
        target_cols: 目标列名列表
    """
    # 读取当前数据集
    df = pd.read_parquet(data_path) if data_path.endswith(
        '.parquet') else pd.read_pickle(data_path)

    # 读取预测结果
    pred_dir = os.path.dirname(data_path)
    pred_file = os.path.join(pred_dir,
                             "predict_results/data/real_prediction.npy")
    predictions = np.load(pred_file)

    # 将预测日期转换为datetime对象
    pred_date = datetime.strptime(str(pred_date), "%Y%m%d")

    # 生成该天的时间索引（根据频率）
    freq = 'd'  # 默认为天
    periods = predictions.shape[1]  # 预测长度

    time_index = pd.date_range(start=pred_date, periods=periods, freq=freq)

    # 将预测结果转换为DataFrame
    pred_df = pd.DataFrame({'date': time_index})

    # 添加每个目标列的预测值
    for i, col in enumerate(target_cols):
        if col != 'date':  # 跳过日期列
            pred_df[col] = predictions[0, :, i]

    # 将预测结果合并到原始数据集
    # 首先删除可能存在的同一天的数据
    df = df[~df['date'].dt.date.isin([pred_date.date()])]

    # 然后添加新的预测结果
    df = pd.concat([df, pred_df], ignore_index=True)

    # 按时间排序
    df = df.sort_values('date')

    # 保存更新后的数据集
    if data_path.endswith('.parquet'):
        df.to_parquet(data_path, index=False)
    else:
        df.to_pickle(data_path)

    print(f"已将{pred_date.strftime('%Y%m%d')}的预测结果更新到数据集中")


def main():
    parser = argparse.ArgumentParser(description='更新数据集脚本')
    parser.add_argument('--data_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--pred_date',
                        type=str,
                        required=True,
                        help='预测日期 (YYYYMMDD格式)')
    parser.add_argument('--target_cols',
                        type=str,
                        nargs='+',
                        required=True,
                        help='目标列名列表')

    args = parser.parse_args()
    update_dataset(args.data_path, args.pred_date, args.target_cols)


if __name__ == '__main__':
    main()
