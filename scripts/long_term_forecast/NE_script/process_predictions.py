import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


def process_station_predictions(station_id):
    # 构建预测结果文件路径
    pred_path = f'new_energy_forecast/long_term_forecast_{station_id}_TimesNet_NEv2_ftMS_sl96_ll0_pl96_dm256_nh8_el4_dl2_df512_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/predict_results/data/real_prediction.npy'

    # 检查文件是否存在
    if not os.path.exists(pred_path):
        print(f"警告: 站点 {station_id} 的预测文件不存在: {pred_path}")
        return

    # 读取预测结果
    predictions = np.load(pred_path)

    # 生成时间序列
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    end_time = datetime(2025, 2, 28, 23, 45, 0)
    freq = '15min'

    # 创建时间索引
    time_index = pd.date_range(start=start_time, end=end_time, freq=freq)

    # 确保预测结果的长度与时间索引匹配
    if len(predictions.flatten()) != len(time_index):
        print(
            f"警告: 站点 {station_id} 的预测长度 ({len(predictions.flatten())}) 与时间序列长度 ({len(time_index)}) 不匹配"
        )
        # 如果预测结果比时间序列长，截断；如果短，用NaN填充
        if len(predictions.flatten()) > len(time_index):
            predictions = predictions.flatten()[:len(time_index)]
        else:
            temp = np.full(len(time_index), np.nan)
            temp[:len(predictions.flatten())] = predictions.flatten()
            predictions = temp
    else:
        predictions = predictions.flatten()

    # 创建DataFrame
    df = pd.DataFrame({'': time_index, 'Power': predictions})

    # 格式化时间列
    df[''] = df[''].dt.strftime('%Y/%-m/%-d %H:%M')

    # 创建输出目录
    output_path = 'new_energy_forecast/outputs/data'
    os.makedirs(output_path, exist_ok=True)

    # 保存为CSV文件
    output_path = os.path.join(output_path, f'output{station_id}.csv')

    # 保存为CSV，不包含列名，使用逗号分隔，保留一位小数
    df.to_csv(output_path, index=False)
    print(f"站点 {station_id} 的预测结果已保存到: {output_path}")


def main():
    print("开始处理预测结果...")

    # 处理站点1-10的预测结果
    for station_id in range(1, 11):
        print(f"\n处理站点 {station_id} 的预测结果...")
        process_station_predictions(station_id)

    print("\n所有预测结果处理完成！")


if __name__ == "__main__":
    main()
