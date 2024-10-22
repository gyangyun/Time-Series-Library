import argparse
import json
import os
import random

import joblib
import numpy as np
import torch
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import grid_search

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_imputation import Exp_Imputation
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from utils.plot_result import plot_predict_result, plot_test_result
from utils.print_args import print_args


def create_parser():
    parser = argparse.ArgumentParser(description="TimesNet")

    # basic config
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        default="long_term_forecast",
        help=
        "task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]",
    )
    # 新增一个2模式，表示预测。即{0: 'test', 1: 'train', 2, 'predict'}
    parser.add_argument("--is_training",
                        type=int,
                        required=True,
                        default=1,
                        help="status")
    parser.add_argument("--model_id",
                        type=str,
                        required=True,
                        default="test",
                        help="model id")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="Autoformer",
        help="model name, options: [Autoformer, Transformer, TimesNet]",
    )

    # data loader
    parser.add_argument("--data",
                        type=str,
                        required=True,
                        default="ETTm1",
                        help="dataset type")
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="root path of the data file",
    )
    parser.add_argument("--data_path",
                        type=str,
                        default="ETTh1.csv",
                        help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help=
        "forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument("--target",
                        type=str,
                        default="OT",
                        help="target feature in S or MS task")
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help=
        "freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )

    # forecasting task
    parser.add_argument("--seq_len",
                        type=int,
                        default=96,
                        help="input sequence length")
    parser.add_argument("--label_len",
                        type=int,
                        default=48,
                        help="start token length")
    parser.add_argument("--pred_len",
                        type=int,
                        default=96,
                        help="prediction sequence length")
    parser.add_argument("--seasonal_patterns",
                        type=str,
                        default="Monthly",
                        help="subset for M4")
    parser.add_argument("--inverse",
                        action="store_true",
                        help="inverse output data",
                        default=False)

    # inputation task
    parser.add_argument("--mask_rate",
                        type=float,
                        default=0.25,
                        help="mask ratio")

    # anomaly detection task
    parser.add_argument("--anomaly_ratio",
                        type=float,
                        default=0.25,
                        help="prior anomaly ratio (%)")

    # model define
    parser.add_argument("--expand",
                        type=int,
                        default=2,
                        help="expansion factor for Mamba")
    parser.add_argument("--d_conv",
                        type=int,
                        default=4,
                        help="conv kernel size for Mamba")
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--num_kernels",
                        type=int,
                        default=6,
                        help="for Inception")
    parser.add_argument("--enc_in",
                        type=int,
                        default=7,
                        help="encoder input size")
    parser.add_argument("--dec_in",
                        type=int,
                        default=7,
                        help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model",
                        type=int,
                        default=512,
                        help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers",
                        type=int,
                        default=2,
                        help="num of encoder layers")
    parser.add_argument("--d_layers",
                        type=int,
                        default=1,
                        help="num of decoder layers")
    parser.add_argument("--d_ff",
                        type=int,
                        default=2048,
                        help="dimension of fcn")
    parser.add_argument("--moving_avg",
                        type=int,
                        default=25,
                        help="window size of moving average")
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help=
        "whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation",
                        type=str,
                        default="gelu",
                        help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in ecoder",
    )
    parser.add_argument(
        "--channel_independence",
        type=int,
        default=1,
        help="0: channel dependence 1: channel independence for FreTS model",
    )
    parser.add_argument(
        "--decomp_method",
        type=str,
        default="moving_avg",
        help=
        "method of series decompsition, only support moving_avg or dft_decomp",
    )
    parser.add_argument(
        "--use_norm",
        type=int,
        default=1,
        help="whether to use normalize; True 1 False 0",
    )
    parser.add_argument(
        "--down_sampling_layers",
        type=int,
        default=0,
        help="num of down sampling layers",
    )
    parser.add_argument("--down_sampling_window",
                        type=int,
                        default=1,
                        help="down sampling window size")
    parser.add_argument(
        "--down_sampling_method",
        type=str,
        default=None,
        help="down sampling method, only support avg, max, conv",
    )
    parser.add_argument(
        "--seg_len",
        type=int,
        default=48,
        help="the length of segmen-wise iteration of SegRNN",
    )

    # optimization
    parser.add_argument("--num_workers",
                        type=int,
                        default=10,
                        help="data loader num workers")
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs",
                        type=int,
                        default=10,
                        help="train epochs")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="batch size of train input data")
    parser.add_argument("--patience",
                        type=int,
                        default=3,
                        help="early stopping patience")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.0001,
                        help="optimizer learning rate")
    parser.add_argument("--des",
                        type=str,
                        default="test",
                        help="exp description")
    parser.add_argument("--loss",
                        type=str,
                        default="MSE",
                        help="loss function")
    parser.add_argument("--lradj",
                        type=str,
                        default="type1",
                        help="adjust learning rate")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--use_multi_gpu",
                        action="store_true",
                        help="use multiple gpus",
                        default=False)
    parser.add_argument("--devices",
                        type=str,
                        default="0,1,2,3",
                        help="device ids of multile gpus")

    # de-stationary projector params
    parser.add_argument(
        "--p_hidden_dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="hidden layer dimensions of projector (List)",
    )
    parser.add_argument(
        "--p_hidden_layers",
        type=int,
        default=2,
        help="number of hidden layers in projector",
    )

    # metrics (dtw)
    parser.add_argument(
        "--use_dtw",
        type=bool,
        default=False,
        help=
        "the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)",
    )

    # Augmentation
    parser.add_argument("--augmentation_ratio",
                        type=int,
                        default=0,
                        help="How many times to augment")
    parser.add_argument("--seed",
                        type=int,
                        default=2,
                        help="Randomization seed")
    parser.add_argument(
        "--jitter",
        default=False,
        action="store_true",
        help="Jitter preset augmentation",
    )
    parser.add_argument(
        "--scaling",
        default=False,
        action="store_true",
        help="Scaling preset augmentation",
    )
    parser.add_argument(
        "--permutation",
        default=False,
        action="store_true",
        help="Equal Length Permutation preset augmentation",
    )
    parser.add_argument(
        "--randompermutation",
        default=False,
        action="store_true",
        help="Random Length Permutation preset augmentation",
    )
    parser.add_argument(
        "--magwarp",
        default=False,
        action="store_true",
        help="Magnitude warp preset augmentation",
    )
    parser.add_argument(
        "--timewarp",
        default=False,
        action="store_true",
        help="Time warp preset augmentation",
    )
    parser.add_argument(
        "--windowslice",
        default=False,
        action="store_true",
        help="Window slice preset augmentation",
    )
    parser.add_argument(
        "--windowwarp",
        default=False,
        action="store_true",
        help="Window warp preset augmentation",
    )
    parser.add_argument(
        "--rotation",
        default=False,
        action="store_true",
        help="Rotation preset augmentation",
    )
    parser.add_argument(
        "--spawner",
        default=False,
        action="store_true",
        help="SPAWNER preset augmentation",
    )
    parser.add_argument(
        "--dtwwarp",
        default=False,
        action="store_true",
        help="DTW warp preset augmentation",
    )
    parser.add_argument(
        "--shapedtwwarp",
        default=False,
        action="store_true",
        help="Shape DTW warp preset augmentation",
    )
    parser.add_argument(
        "--wdba",
        default=False,
        action="store_true",
        help="Weighted DBA preset augmentation",
    )
    parser.add_argument(
        "--discdtw",
        default=False,
        action="store_true",
        help="Discrimitive DTW warp preset augmentation",
    )
    parser.add_argument(
        "--discsdtw",
        default=False,
        action="store_true",
        help="Discrimitive shapeDTW warp preset augmentation",
    )
    parser.add_argument("--extra_tag",
                        type=str,
                        default="",
                        help="Anything extra")

    # 新增的自定义参数
    parser.add_argument("--train_start",
                        type=str,
                        default="",
                        help="train start")
    parser.add_argument("--train_end", type=str, default="", help="train end")
    parser.add_argument("--test_start",
                        type=str,
                        default="",
                        help="test start")
    # 注意：test_end表示测试集中y能取到的边界
    parser.add_argument("--test_end", type=str, default="", help="test end")
    parser.add_argument("--pred_start",
                        type=str,
                        default="",
                        help="pred start")
    # 注意：pred_end表示预测数据集中能预测的边界
    parser.add_argument("--pred_end", type=str, default="", help="pred end")
    parser.add_argument("--cols",
                        type=str,
                        default="",
                        help="Comma-separated list of features")
    parser.add_argument("--use_autoregression",
                        type=int,
                        default=0,
                        help="is autoregression flag")
    parser.add_argument("--use_best_params",
                        type=int,
                        default=0,
                        help="是否使用最优参数,0表示不使用,1表示使用")
    return parser


def create_setting(args, ii):
    """根据提供的 args 参数创建 setting 字符串"""
    setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        ii,
    )
    return setting


def parse_setting(setting):
    """根据提供的 setting 字符串逆向解析成一个包含参数的 dict"""
    parts = setting.split("_")

    # 依次对应 create_setting 中各个字段的名称
    args_dict = {
        "task_name": parts[0],
        "model_id": parts[1],
        "model": parts[2],
        "data": parts[3],
        "features": parts[4],
        "seq_len": int(parts[5]),
        "label_len": int(parts[6]),
        "pred_len": int(parts[7]),
        "d_model": int(parts[8]),
        "n_heads": int(parts[9]),
        "e_layers": int(parts[10]),
        "d_layers": int(parts[11]),
        "d_ff": int(parts[12]),
        "expand": int(parts[13]),
        "d_conv": int(parts[14]),
        "factor": int(parts[15]),
        "embed": parts[16],
        "distil": parts[17] == "True",  # 将字符串 'True'/'False' 转换为布尔类型
        "des": parts[18],
        "ii": int(parts[19]),
    }

    return args_dict


def load_best_args(args):
    """
    加载最佳参数文件，并更新args中的部分参数
    """
    best_args_path = os.path.join(args.root_path, "best_args.json")
    if os.path.exists(best_args_path):
        print("发现最佳参数文件，将使用最佳参数")
        with open(best_args_path, "r") as f:
            best_args = json.load(f)
        # 更新args中的部分参数，其实就是定位到参数文件的目录的那些参数
        update_keys = [
            'task_name',
            'model_id',
            'model',
            'data',
            'features',
            'seq_len',
            'label_len',
            'pred_len',
            'd_model',
            'n_heads',
            'e_layers',
            'd_layers',
            'd_ff',
            'expand',
            'd_conv',
            'factor',
            'embed',
            'distil',
            'des',
            'ii',
        ]

        for key, value in best_args.items():
            # if key not in not_update_keys:
            if key in update_keys:
                setattr(args, key, value)
        print("已更新参数:", best_args)
    else:
        print("未找到最佳参数文件，将使用默认参数")
    return args


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = create_parser()
    args = parser.parse_args()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print_args(args)

    if args.task_name == "long_term_forecast":
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == "short_term_forecast":
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == "imputation":
        Exp = Exp_Imputation
    elif args.task_name == "anomaly_detection":
        Exp = Exp_Anomaly_Detection
    elif args.task_name == "classification":
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    # 训练
    if args.is_training == 1:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = create_setting(args, ii)

            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(
                    setting))
            exp.train(setting)

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(
                    setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    # 测试
    elif args.is_training == 0:
        if args.use_best_params == 1:
            args = load_best_args(args)

        ii = 0
        setting = create_setting(args, ii)

        exp = Exp(args)  # set experiments
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(
            setting))
        exp.test(setting, load=True)

        # 新增代码，用于绘制test结果
        scaler_y = joblib.load(os.path.join(args.root_path,
                                            "preprocessor.bin"))["scaler_y"]
        result_path = os.path.join(args.root_path, setting, "test_results",
                                   "data")
        fig_path = os.path.join(args.root_path, setting, "test_results",
                                "figure")
        plot_test_result(
            result_path,
            fig_path,
            args.test_start,
            args.test_end,
            args.freq,
            scaler_y,
        )

        torch.cuda.empty_cache()
    # 预测
    elif args.is_training == 2:
        if args.use_best_params == 1:
            args = load_best_args(args)

        ii = 0
        setting = create_setting(args, ii)

        exp = Exp(args)  # set experiments
        print(">>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(
            setting))
        exp.predict(setting, load=True)
        torch.cuda.empty_cache()

        # 新增代码，用于绘制test结果
        scaler_y = joblib.load(os.path.join(args.root_path,
                                            "preprocessor.bin"))["scaler_y"]
        result_path = os.path.join(args.root_path, setting, "predict_results",
                                   "data")
        fig_path = os.path.join(args.root_path, setting, "predict_results",
                                "figure")
        plot_predict_result(
            result_path,
            fig_path,
            args.pred_start,
            args.pred_end,
            args.freq,
            scaler_y,
        )
    # 超参搜索
    elif args.is_training == 3:
        print(">>>>>>>开始超参数搜索>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # 定义超参数搜索空间
        search_space = {
            # "model": tune.choice(["TimesNet", "Autoformer", "Transformer", "Informer"]),
            # "model": tune.choice(["TimesNet", "iTransformer"]),
            # 可以添加其他超参数
            # "learning_rate": tune.loguniform(1e-4, 1e-1),
            # "batch_size": tune.choice([16, 32, 64, 128]),
            # "n_heads": tune.choice([8, 16]),
            # "e_layers": tune.randint(4, 8),
            # "d_layers": tune.randint(2, 6),
            # "d_model": tune.choice([64, 128, 256, 512, 1024]),
            # "d_ff": tune.choice([128, 256, 512, 1024]),
            # "dropout": tune.uniform(0.1, 0.5),
            # "seq_len": tune.choice([14, 28, 56, 84]),
            "seq_len": grid_search([14, 28, 56, 84]),
            # "label_len": tune.choice([7, 14, 28]),
            # "factor": tune.choice([1, 3, 5]),
        }

        # 定义目标函数
        def objective(config):
            # 更新参数
            for key, value in config.items():
                setattr(args, key, value)

            exp = Exp(args)
            setting = create_setting(args, 0)

            # 训练模型并获取验证集loss
            best_model_path, val_loss = exp.train(setting)

            # 报告验证集loss给Ray Tune
            train.report({"val_loss": val_loss})

        # 设置搜索算法
        # OptunaSearch是随机搜索算法，所以无法使用grid_search
        # 要使用grid_search请使用tune的默认搜索算法，即不指定search_alg
        # 而使用grid_search时，num_samples表示重复次数，请勿设置
        # search_alg = OptunaSearch()

        # 设置Tuner
        tuner = tune.Tuner(
            tune.with_resources(
                # objective, resources={"cpu": 2}
                objective,
                resources={
                    "cpu": 0,
                    "gpu": 0.25
                }),  # 如果使用GPU，可以改为 {"cpu": 1, "gpu": 1}
            tune_config=tune.TuneConfig(
                metric="val_loss",
                mode="min",
                # search_alg=search_alg,
                # num_samples=4,  # 可以增加样本数量以获得更好的结果
            ),
            run_config=train.RunConfig(
                name="optimized_timesnet_tuning",
                stop={"training_iteration": args.train_epochs},
            ),
            param_space=search_space,
        )

        # 运行超参数搜索
        results = tuner.fit()

        # 获取最佳配置
        best_result = results.get_best_result("val_loss", "min")
        print("最佳试验的配置: ", best_result.config)
        print("最佳试验的验证集损失: ", best_result.metrics["val_loss"])

        # 更新args以包含最佳超参数
        for key, value in best_result.config.items():
            setattr(args, key, value)

        # 保存最佳参数（args）
        best_args_path = os.path.join(args.root_path, "best_args.json")
        with open(best_args_path, "w") as f:
            json.dump(vars(args), f, indent=4)
        print(f"最佳参数已保存至: {best_args_path}")

        # 使用最佳参数重新训练模型
        print("使用最佳参数重新训练模型")
        exp = Exp(args)
        setting = create_setting(args, 0)
        exp.train(setting)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
