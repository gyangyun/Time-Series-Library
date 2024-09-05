from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from run import create_parser, create_setting


def merge_df(
    df, by_cols, new_col="sfmc", new_col_value="全网", aggfunc=None, is_append=False
):
    """按分组聚合，聚合后拼接到原DataFrame"""

    if not aggfunc:
        aggfunc = {"dl": "sum"}

    grouped_df = df.groupby(by_cols, as_index=False).agg(aggfunc)
    grouped_df[new_col] = new_col_value

    if is_append:
        return_df = pd.concat([df, grouped_df], axis=0)
        return_df.reset_index(drop=True, inplace=True)
    else:
        return_df = grouped_df[[col for col in df.columns if col in grouped_df.columns]]

    return return_df


def load_and_concatenate_results(args, is_merge_qsh, is_merge_qw, output_path=None):
    dataset_path = args.dataset_path
    province_names = args.dataset_path
    industry_ids = args.dataset_path
    pred_start = args.pred_start
    pred_end = args.pred_end
    freq = args.freq

    # 将空格分隔的字符串转换为列表
    province_names = args.province_names.split()
    industry_ids = args.industry_ids.split()

    all_results = []

    for province_name in province_names:
        for industry_id in industry_ids:
            root_path = Path(dataset_path) / province_name / industry_id
            preprocessor_path = root_path / "preprocessor.bin"

            # 检查预处理器是否存在
            if not preprocessor_path.exists():
                print(f"预处理器路径不存在: {preprocessor_path}")
                continue

            preprocessor = joblib.load(preprocessor_path)
            scaler_y = preprocessor["scaler_y"]

            # 生成 setting 变量
            setting = create_setting(args, ii=0)  # 假设ii=0

            results_path = root_path / setting / "predict_results" / "data"

            # 检查结果路径是否存在
            if not results_path.exists():
                print(f"结果路径不存在: {results_path}")
                continue

            # [1, 待预测时间步长, output_size]
            pred = np.load(results_path / "real_prediction.npy")
            # output_size只取第一维
            pred = pred[0, :, 0]
            # 反归一化
            pred = scaler_y.inverse_transform(np.array(pred).reshape(-1, 1)).squeeze()

            time_range = pd.date_range(
                pd.to_datetime(pred_start),
                pd.to_datetime(pred_end),
                freq=freq,
            )

            df = pd.DataFrame(
                {
                    "province_name": province_name,
                    "industry_id": industry_id,
                    "datetime": time_range,
                    "pred_value": pred,
                }
            )

            all_results.append(df)

    # 将所有结果拼接成一个DataFrame
    if all_results:
        concatenated_df = pd.concat(all_results, axis=0, ignore_index=True)
    else:
        print("没有找到任何结果文件")
        return None

    if is_merge_qsh:
        industry_id_lite_l = [
            "[3]第一产业",
            "[4]第二产业",
            "[5]第三产业",
            "[6]B、城乡居民生活用电合计",
            "[9]C、趸售",
        ]
        industry_id_lite_df = concatenated_df.query("industry_id in @industry_id_lite_l")
        qsh_df = merge_df(
            industry_id_lite_df,
            by_cols=["province_name", "datetime"],
            new_col="industry_id",
            new_col_value="[1]全社会用电总计（汇总）",
            aggfunc={"pred_value": "sum"},
            is_append=False,
        )
        concatenated_df = pd.concat([concatenated_df, qsh_df], axis=0)

    if is_merge_qw:
        province_name_lite_l = ["广东", "广西", "云南", "贵州", "海南", "全网"]
        province_name_lite_df = concatenated_df.query(
            "province_name in @province_name_lite_l"
        )
        qsh_df = merge_df(
            province_name_lite_df,
            by_cols=["industry_id", "datetime"],
            new_col="province_name",
            new_col_value="全网（汇总）",
            aggfunc={"pred_value": "sum"},
            is_append=False,
        )
        concatenated_df = pd.concat([concatenated_df, qsh_df], axis=0)

    if not output_path:
        output_path = (
            Path(args.dataset_path)
            / f"predict_results_{args.pred_start}_{args.pred_start}"
            / "real_prediction.csv"
        )

        if not output_path.parent.is_dir():
            output_path.parent.mkdir(parents=True)

    concatenated_df.to_csv(output_path, index=False)
    return concatenated_df


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--dataset_path", type=str, help="Dataset path")
    parser.add_argument(
        "--province_names", type=str, help="Comma-separated list of province names."
    )
    parser.add_argument(
        "--industry_ids", type=str, help="Comma-separated list of order numbers."
    )

    # 添加其他参数
    args = parser.parse_args()
    df = load_and_concatenate_results(args, is_merge_qsh=True, is_merge_qw=True)
