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


def load_and_concatenate_test_results(args, is_merge_qsh, is_merge_qw):
    dataset_path = args.dataset_path
    test_start = args.test_start
    test_end = args.test_end
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

            results_path = root_path / setting / "test_results" / "data"

            # 检查结果路径是否存在
            if not results_path.exists():
                print(f"结果路径不存在: {results_path}")
                continue

            # [batch_size, pred_len, output_size]
            pred = np.load(results_path / "pred.npy")
            true = np.load(results_path / "true.npy")

            # 只取每个时间步长预测结果的第一维
            pred = pred[:, :, 0]
            true = true[:, :, 0]

            # 反归一化
            pred = scaler_y.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
            true = scaler_y.inverse_transform(true.reshape(-1, 1)).reshape(true.shape)

            time_range = pd.date_range(
                pd.to_datetime(test_start),
                pd.to_datetime(test_end),
                freq=freq,
            )

            # 创建一个包含所有预测和真实值的DataFrame
            # 针对单步长输出
            df = pd.DataFrame(
                {
                    "datetime": time_range,
                    "pred_value": pred.flatten(),
                    "true_value": true.flatten(),
                }
            )
            df["province_name"] = province_name
            df["industry_id"] = industry_id
            df = df[
                ["province_name", "industry_id", "datetime", "pred_value", "true_value"]
            ]

            all_results.append(df)

            # 针对多步长输出
            # for i in range(len(pred)):
            #     tmp_pred = pred[i]
            #     tmp_true = true[i]
            #     tmp_time_range = time_range[i : i+tmp_pred.shape[0]]

            #     df = pd.DataFrame({
            #         "province_name": province_name,
            #         "industry_id": industry_id,
            #         "datetime": tmp_time_range,
            #         "pred_value": tmp_pred,
            #         "true_value": tmp_true
            #     })

            #     all_results.append(df)

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
        industry_id_lite_df = concatenated_df.query(
            "industry_id in @industry_id_lite_l"
        )
        qsh_df = merge_df(
            industry_id_lite_df,
            by_cols=["province_name", "datetime"],
            new_col="industry_id",
            new_col_value="[1]全社会用电总计（汇总）",
            aggfunc={"pred_value": "sum", "true_value": "sum"},
            is_append=False,
        )
        concatenated_df = pd.concat([concatenated_df, qsh_df], axis=0)

    if is_merge_qw:
        province_name_lite_l = ["广东", "广西", "云南", "贵州", "海南"]
        province_name_lite_df = concatenated_df.query(
            "province_name in @province_name_lite_l"
        )
        qsh_df = merge_df(
            province_name_lite_df,
            by_cols=["industry_id", "datetime"],
            new_col="province_name",
            new_col_value="全网（汇总）",
            aggfunc={"pred_value": "sum", "true_value": "sum"},
            is_append=False,
        )
        concatenated_df = pd.concat([concatenated_df, qsh_df], axis=0)

    return concatenated_df


def load_and_concatenate_pred_results(args, is_merge_qsh, is_merge_qw):
    dataset_path = args.dataset_path
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
        industry_id_lite_df = concatenated_df.query(
            "industry_id in @industry_id_lite_l"
        )
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
        province_name_lite_l = ["广东", "广西", "云南", "贵州", "海南"]
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
    return concatenated_df


def results_post_processing(date_df, values):
    date_df["order_no"] = date_df["industry_id"].str.extract(r"\[(\d+)\]").astype("int")
    date_df["year"] = date_df["date"].dt.year
    date_df["month"] = date_df["date"].dt.month

    # 日电量明细
    date_df = date_df[
        [
            "province_name",
            "order_no",
            "industry_id",
            "year",
            "month",
            "date",
        ]
        + values
    ]
    # 月电量明细
    month_df = date_df.groupby(
        by=["province_name", "order_no", "industry_id", "year", "month"], as_index=False
    )[values].sum()

    province_names = [
        "广东",
        "广西",
        "云南",
        "贵州",
        "海南",
        "全网",
        "广东（不含广深）",
        "广东（含广州）",
        "广州",
        "深圳",
    ]
    province_lite_names = [
        province_name
        for province_name in province_names
        if province_name in month_df["province_name"].unique()
    ]

    # 日电量透视图
    date_pvt = pd.pivot_table(
        date_df,
        index=["order_no", "industry_id", "year", "month", "date"],
        columns="province_name",
        values=values,
        aggfunc="sum",
    )
    date_pvt = date_pvt[province_lite_names]
    date_pvt.reset_index(drop=False, inplace=True)

    # 月电量透视图
    month_pvt = pd.pivot_table(
        date_df,
        index=["order_no", "industry_id", "year", "month"],
        columns="province_name",
        values=values,
        aggfunc="sum",
    )
    month_pvt = month_pvt[province_lite_names]
    month_pvt.reset_index(drop=False, inplace=True)
    return date_df, month_df, date_pvt, month_pvt


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

    # =========================处理预测数据=========================
    # 合并预测结果
    pred_date_df = load_and_concatenate_pred_results(
        args, is_merge_qsh=True, is_merge_qw=True
    )
    pred_date_df, pred_month_df, pred_date_pvt, pred_month_pvt = (
        results_post_processing(
            pred_date_df,
            values=["pred_value"],
        )
    )
    # 保存结果
    pred_output_dp = (
        Path(args.dataset_path) / f"predict_results_{args.pred_start}_{args.pred_end}"
    )

    if not pred_output_dp.is_dir():
        pred_output_dp.mkdir(parents=True)

    pred_date_df.to_excel(
        pred_output_dp.joinpath("日电量明细表【预测】.xlsx"), index=False
    )
    pred_month_df.to_excel(
        pred_output_dp.joinpath("月电量明细表【预测】.xlsx"), index=False
    )
    pred_date_pvt.to_excel(
        pred_output_dp.joinpath("日电量透视表【预测】.xlsx"), index=False
    )
    pred_month_pvt.to_excel(
        pred_output_dp.joinpath("月电量透视表【预测】.xlsx"), index=False
    )

    # =========================处理真实数据=========================
    test_date_df = load_and_concatenate_test_results(
        args, is_merge_qsh=True, is_merge_qw=True
    )
    test_date_df, test_month_df, test_date_pvt, test_month_pvt = (
        results_post_processing(
            test_date_df,
            values=["pred_value", "true_value"],
        )
    )

    # 保存结果
    test_output_dp = (
        Path(args.dataset_path) / f"testict_results_{args.test_start}_{args.test_end}"
    )

    if not test_output_dp.is_dir():
        test_output_dp.mkdir(parents=True)

    test_date_df.to_excel(
        test_output_dp.joinpath("日电量明细表【测试】.xlsx"), index=False
    )
    test_month_df.to_excel(
        test_output_dp.joinpath("月电量明细表【测试】.xlsx"), index=False
    )
    test_date_pvt.to_excel(
        test_output_dp.joinpath("日电量透视表【测试】.xlsx"), index=False
    )
    test_month_pvt.to_excel(
        test_output_dp.joinpath("月电量透视表【测试】.xlsx"), index=False
    )
    # =========================处理真实数据=========================
    # hydl_r_raw_df = pd.read_pickle(Path(args.dataset_path).parent.parent.joinpath("行业电量/hydl_r_raw_df.pkl"))
    # # 保存行业电量日数据
    # hydl_r_df = hydl_r_raw_df[["sfmc", "order_no", "hyflmc", "sjsj", "dl"]].copy()
    # hydl_r_df.query('sjsj >= "2020-01-01" and order_no <= 10', inplace=True)
    # hydl_r_df.insert(
    #     hydl_r_df.columns.get_loc("order_no"),
    #     "hyflid",
    #     "[" + hydl_r_df["order_no"].astype("int").astype("str") + "]" + hydl_r_df["hyflmc"],
    # )

    # hydl_r_qw_df = hydl_r_df.query('sfmc == "全网"')
    # hydl_r_qw_df['sfmc'] = '全网（汇总）'
    # hydl_r_qsh_df = hydl_r_df.query('hyflid == "[1]全社会用电总计"')
    # hydl_r_qsh_df['hyflid'] = '[1]全社会用电总计（汇总）'

    # hydl_r_df = pd.concat([hydl_r_df, hydl_r_qw_df, hydl_r_qsh_df], axis=0)
    # hydl_r_df.rename(columns={"sfmc": "province_name", "sjsj": "date", "hyflid": "industry_id", "dl": "electricity_consumption"}, inplace=True)
    # hydl_date_df, hydl_month_df, hydl_date_pvt, hydl_month_pvt = results_post_processing(
    #     hydl_r_df,
    #     values="electricity_consumption",
    # )

    # # =========================生成对比结果=========================
    # contrast_date_df = pd.merge(date_df, hydl_date_df, on=['province_name', 'order_no', 'industry_id', 'year', 'month', 'date'], how='inner', suffixes=('_pred', '_true'))
    # contrast_month_df = pd.merge(month_df, hydl_month_df, on=['province_name', 'order_no', 'industry_id', 'year', 'month'], how='inner', suffixes=('_pred', '_true'))
    # contrast_date_pvt = pd.merge(date_pvt, hydl_date_pvt, on=['order_no', 'industry_id', 'year', 'month', 'date'], how='inner', suffixes=('_pred', '_true'))
    # contrast_month_pvt = pd.merge(month_pvt, hydl_month_pvt, on=['order_no', 'industry_id', 'year', 'month'], how='inner', suffixes=('_pred', '_true'))

    # # =========================再加上一个气温值，方便核查原因=========================
    # # 加载天气数据
    # tq_r_df = pd.read_pickle(Path(args.dataset_path).parent.parent.joinpath("天气/sf_tq_r_df.pkl"))
    # tq_r_df.query('sjsj >= "2020-01-01"', inplace=True)

    # tqyb_r_df = pd.read_pickle(Path(args.dataset_path).parent.parent.joinpath("天气预报/sf_tqyb_r_df.pkl"))
    # max_date = tq_r_df.sjsj.max()
    # tq_cols = tq_r_df.columns.to_list()
    # tqyb_r_df = tqyb_r_df[tqyb_r_df.sjsj > max_date][tq_cols]
    # tq_r_df = pd.concat([tq_r_df, tqyb_r_df])

    # tq_r_df.rename(columns={"sfmc": "province_name", "sjsj": "date"}, inplace=True)
    # tq_r_df.sort_values(by=["province_name", "date"], ascending=True, inplace=True)
    # contrast_date_df = pd.merge(contrast_date_df, tq_r_df, on=['province_name', 'date'], how='inner')

    # # =========================保存结果=========================
    # output_dp = (
    #     Path(args.dataset_path) / f"predict_results_{args.pred_start}_{args.pred_end}"
    # )

    # if not output_dp.is_dir():
    #     output_dp.mkdir(parents=True)

    # # date_df.to_excel(output_dp.joinpath("日电量明细表【预测】.xlsx"), index=False)
    # # month_df.to_excel(output_dp.joinpath("月电量明细表【预测】.xlsx"), index=False)
    # # date_pvt.to_excel(output_dp.joinpath("日电量透视表【预测】.xlsx"), index=False)
    # # month_pvt.to_excel(output_dp.joinpath("月电量透视表【预测】.xlsx"), index=False)
    # #
    # # hydl_date_df.to_excel(output_dp.joinpath("日电量明细表【真实】.xlsx"), index=False)
    # # hydl_month_df.to_excel(output_dp.joinpath("月电量明细表【真实】.xlsx"), index=False)
    # # hydl_date_pvt.to_excel(output_dp.joinpath("日电量透视表【真实】.xlsx"), index=False)
    # # hydl_month_pvt.to_excel(output_dp.joinpath("月电量透视表【真实】.xlsx"), index=False)

    # contrast_date_df.to_excel(output_dp.joinpath("日电量明细表【对比】.xlsx"), index=False)
    # contrast_month_df.to_excel(output_dp.joinpath("月电量明细表【对比】.xlsx"), index=False)
    # # contrast_date_pvt.to_excel(output_dp.joinpath("日电量透视表【对比】.xlsx"), index=False)
    # # contrast_month_pvt.to_excel(output_dp.joinpath("月电量透视表【对比】.xlsx"), index=False)
