from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from run import create_setting, create_parser

def load_and_concatenate_results(args):
    dataset_path = args.dataset_path
    province_names = args.dataset_path
    order_nos = args.dataset_path
    pred_start = args.pred_start
    pred_end = args.pred_end
    freq = args.freq

    # 将空格分隔的字符串转换为列表
    province_names = args.province_names.split()
    order_nos = [int(no) for no in args.order_nos.split()]

    all_results = []

    for province_name in province_names:
        for order_no in order_nos:
            root_path = Path(dataset_path) / province_name / str(order_no)
            preprocessor_path = root_path / "preprocessor_d.bin"

            # 检查预处理器是否存在
            if not preprocessor_path.exists():
                print(f"预处理器路径不存在: {preprocessor_path}")
                continue

            preprocessor = joblib.load(preprocessor_path)
            scaler_y = preprocessor["scaler_y"]

            # 生成 setting 变量
            args['province_name'] = province_name
            args['order_no'] = order_no
            setting = create_setting(args, ii=0)  # 假设ii=0

            results_path = root_path / "results" / setting

            # 检查结果路径是否存在
            if not results_path.exists():
                print(f"结果路径不存在: {results_path}")
                continue

            # [1, 待预测时间步长, output_size]
            pred = np.load(results_path / "real_prediction.npy")
            # output_size只取第一维
            pred = pred[0, :, 0]
            # 反归一化
            pred = scaler_y.inverse_transform(
                np.array(pred).reshape(-1, 1)
            ).squeeze()

            time_range = pd.date_range(
                pd.to_datetime(pred_start),
                pd.to_datetime(pred_end),
                freq=freq,
            )

            df = pd.DataFrame(
                {"order": order_no, 'province_name': province_name, "datetime": time_range, "pred": pred}
            )

            all_results.append(df)

    # 将所有结果拼接成一个DataFrame
    if all_results:
        concatenated_df = pd.concat(all_results, axis=0, ignore_index=True)
        return concatenated_df
    else:
        print("没有找到任何结果文件")
        return None

if __name__ == '__main__':
    # parser = create_parser()
    import argparse
    parser = argparse.ArgumentParser(description="TimesNet")
    parser.add_argument("--province_names", type=str, help="Comma-separated list of province names.")
    parser.add_argument("--order_nos", type=str, help="Comma-separated list of order numbers.")

    # 添加其他参数
    args = parser.parse_args()
    from IPython import embed;embed()

    load_and_concatenate_results(args)

