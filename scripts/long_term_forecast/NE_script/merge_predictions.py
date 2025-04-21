import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def merge_predictions():
    """合并所有场站的预测结果并保存为比赛要求的格式"""

    # 创建时间索引
    start_time = datetime(2025, 1, 1)
    end_time = datetime(2025, 2, 28, 23, 45)  # 最后一个时间点是23:45
    time_index = pd.date_range(start=start_time, end=end_time, freq='15min')

    # 循环处理每个场站的预测结果
    for station_id in range(1, 11):
        print(f"Processing station {station_id}")

        # 加载预测结果
        pred_path = f"./outputs/TimesNet_custom_ftMS_sl672_ll192_pl96_dm512_nh8_el3_dl1_df2048_fc3_ebtimeF_dtTrue_test_1/predict_results/data/real_prediction.npy"
        predictions = np.load(pred_path)

        # 创建DataFrame
        df = pd.DataFrame({
            'time': time_index,
            'power': predictions.reshape(-1)
        })

        # 确保预测值在合理范围内
        df['power'] = df['power'].clip(0, 1)  # 归一化后的功率应该在0-1之间

        # 保存为CSV文件
        output_file = f"./outputs/output{station_id}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved predictions to {output_file}")

    # 将所有输出文件打包
    os.system('cd ./outputs && zip output.zip output*.csv')
    print(
        "All predictions have been merged and zipped to ./outputs/output.zip")


if __name__ == "__main__":
    merge_predictions()
