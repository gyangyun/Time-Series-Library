export CUDA_VISIBLE_DEVICES=0
# =========================公用参数=========================
task_name="long_term_forecast"

# -------------------------模型参数-------------------------
# model_name="Informer"
# model_name="Autoformer"
model_name="TimesNet"
# model_name="TimeMixer"
# model_name="iTransformer"

# 序列长度相关参数
seq_len=14
label_len=7
pred_len=1

# 模型结构参数
n_heads=8
e_layers=6
d_layers=4
factor=3
# enc_in=5
# dec_in=5
enc_in=8
dec_in=8
c_out=1
d_model=72
d_ff=128

# TimeMixer特定参数
# 注意，TimeMixer的c_out要和enc_in/dec_in保持一致
# c_out=5
# c_out=8
down_sampling_layers=3
down_sampling_window=1
down_sampling_method="avg"

# -------------------------训练参数-------------------------
itr=1
train_epochs=30
# train_epochs=1
batch_size=16
top_k=5
description="Exp"
# model_id="IEd1_${seq_len}_${label_len}_${pred_len}"
model_id="IEd1_${seq_len}_${label_len}_${pred_len}"

# -------------------------数据集相关参数-------------------------
dataset_path="/home/guoyy/Workspace/ts/lib/ElecForcastPrep/cache/dataset/deep_learning"
# dataset_path="/Users/guoyangyun/计量中心/16.分析报告/分析/智能报表/ElecForcastPrep/cache/dataset/deep_learning"

data_name="IEd1"
freq="d"
features="MS"
target="electricity_consumption"

# -------------------------时间范围参数-------------------------
# train_start="2024-01-01"
# train_end="2024-02-29"
# test_start="2024-03-07"
# test_end="2024-03-09"
# pred_start="2024-03-10"
# pred_end="2024-03-15"

train_start="2022-01-01"
train_end="2024-06-30"
test_start="2024-07-01"
test_end="2024-09-30"
pred_start="2024-10-01"
pred_end="2024-10-31"
# -------------------------特征列表-------------------------
cols=(
    "date"
    "wd_-1_shift"
    "wd_max_-1_shift"
    "wd_min_-1_shift"
    "holiday_flag_te"
    "electricity_consumption"
)

# cols=("date" "wd_-1_shift" "wd_max_-1_shift" "wd_min_-1_shift" "holiday_flag_te" "electricity_consumption")
# cols=("date" "wd_-1_shift" "wd_max_-1_shift" "wd_min_-1_shift" "wd_-1_diff" "wd_max_-1_diff" "wd_min_-1_diff" "holiday_flag_te" "electricity_consumption")
# cols=("date" "wd_-1_shift" "wd_max_-1_shift" "wd_min_-1_shift" "wd_max_rolling_7_mean_-1_shift" "wd_min_rolling_7_mean_-1_shift" "wd_rolling_7_mean_-1_shift" "holiday_flag_te" "electricity_consumption")
# cols=("date" "wd_-1_shift" "wd_max_-1_shift" "wd_min_-1_shift" "wd_-1_shift_group_te" "wd_max_-1_shift_group_te" "wd_min_-1_shift_group_te" "holiday_flag_te" "electricity_consumption")
# cols=("date" "wd_-1_shift" "wd_max_-1_shift" "wd_min_-1_shift" "wd_rolling_7_mean_-1_shift" "wd_max_rolling_7_mean_-1_shift" "wd_min_rolling_7_mean_-1_shift" "wd_-1_shift_group_te" "wd_max_-1_shift_group_te" "wd_min_-1_shift_group_te" "wd_rolling_7_mean_-1_shift_group_te" "wd_max_rolling_7_mean_-1_shift_group_te" "wd_min_rolling_7_mean_-1_shift_group_te" "holiday_flag_te" "electricity_consumption")

# "date"
# "wd"
# "wd_max"
# "wd_min"
# "is_holiday"
# "wd_rolling_7_mean"
# "wd_max_rolling_7_mean"
# "wd_min_rolling_7_mean"
# "wd_-1_shift"
# "wd_max_-1_shift"
# "wd_min_-1_shift"
# "wd_rolling_7_mean_-1_shift"
# "wd_max_rolling_7_mean_-1_shift"
# "wd_min_rolling_7_mean_-1_shift"
# "holiday_flag_te"
# "wd_-1_shift_group_te"
# "wd_max_-1_shift_group_te"
# "wd_min_-1_shift_group_te"
# "wd_rolling_7_mean_-1_shift_group_te"
# "wd_max_rolling_7_mean_-1_shift_group_te"
# "wd_min_rolling_7_mean_-1_shift_group_te"
# "wd_max_-1_shift_is_extreme"
# "wd_min_-1_shift_is_extreme"
# "electricity_consumption"

# =========================实验设置=========================
# province_names=("广东")
# industry_ids=("[6]B、城乡居民生活用电合计")

province_names=("广东" "广西" "云南" "贵州" "海南" "广州" "深圳" "广东（含广州）")
# province_names=("广东（含广州）")
# province_names=("广东")
# province_names=("贵州")
# industry_ids=("[1]全社会用电总计" "[2]A、全行业用电合计" "[3]第一产业" "[4]第二产业" "[5]第三产业" "[6]B、城乡居民生活用电合计" "[7]城镇居民" "[8]乡村居民" "[9]C、趸售" "[10]D、其他、无行业分类")
industry_ids=("[1]全社会用电总计" "[3]第一产业" "[4]第二产业" "[5]第三产业" "[6]B、城乡居民生活用电合计" "[9]C、趸售" "[10]D、其他、无行业分类")
# industry_ids=("[1]全社会用电总计")
# industry_ids=("[3]第一产业")
# industry_ids=("[6]B、城乡居民生活用电合计")

# 遍历所有 province_name 和 industry_id 的组合
for province_name in "${province_names[@]}"; do
    for industry_id in "${industry_ids[@]}"; do
        # 相关路径设置
        root_path="${dataset_path}/${province_name}/${industry_id}"
        data_path="train_dataset.pkl"
        checkpoints="${root_path}"

        # 在这里进行你需要的操作，例如打印路径
        echo "处理路径: ${root_path}"
        # =========================train=========================
        # is_training=1
        # use_autoregression=1
        # use_best_params=0
        # =========================test=========================
        is_training=0
        use_autoregression=0
        use_best_params=1
        # =========================tune=========================
        # is_training=3
        # use_autoregression=1
        # use_best_params=0
        # =========================predict=========================
        is_training=2
        data_path="predict_dataset.pkl"
        use_autoregression=1
        use_best_params=1
        # =========================运行脚本=========================
        python -u run.py \
        --task_name $task_name \
        --is_training $is_training \
        --root_path $root_path \
        --data_path $data_path \
        --model_id $model_id \
        --model $model_name \
        --data $data_name \
        --features $features \
        --checkpoints $checkpoints \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --n_heads $n_heads \
        --e_layers $e_layers \
        --d_layers $d_layers \
        --factor $factor \
        --enc_in $enc_in \
        --dec_in $dec_in \
        --c_out $c_out \
        --d_model $d_model \
        --d_ff $d_ff \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_window $down_sampling_window \
        --down_sampling_method $down_sampling_method \
        --des $description \
        --itr $itr \
        --train_epochs $train_epochs \
        --batch_size $batch_size \
        --top_k $top_k \
        --freq $freq \
        --target $target \
        --train_start $train_start \
        --train_end $train_end \
        --test_start $test_start \
        --test_end $test_end \
        --pred_start $pred_start \
        --pred_end $pred_end \
        --cols "${cols[*]}" \
        --use_autoregression $use_autoregression \
        --use_best_params $use_best_params
    done
done
# =========================合并结果=========================
is_training=0
use_autoregression=1
use_best_params=1
root_path="${dataset_path}"
data_path="${dataset_path}"
checkpoints="${dataset_path}"

python -u combine_result.py \
--task_name $task_name \
--is_training $is_training \
--root_path $root_path \
--data_path $data_path \
--dataset_path $dataset_path \
--model_id $model_id \
--model $model_name \
--data $data_name \
--features $features \
--checkpoints $checkpoints \
--seq_len $seq_len \
--label_len $label_len \
--pred_len $pred_len \
--n_heads $n_heads \
--e_layers $e_layers \
--d_layers $d_layers \
--factor $factor \
--enc_in $enc_in \
--dec_in $dec_in \
--c_out $c_out \
--d_model $d_model \
--d_ff $d_ff \
--down_sampling_layers $down_sampling_layers \
--down_sampling_window $down_sampling_window \
--down_sampling_method $down_sampling_method \
--des $description \
--itr $itr \
--train_epochs $train_epochs \
--batch_size $batch_size \
--top_k $top_k \
--freq $freq \
--target $target \
--province_names "${province_names[*]}" \
--industry_ids "${industry_ids[*]}"  \
--train_start $train_start \
--train_end $train_end \
--test_start $test_start \
--test_end $test_end \
--pred_start $pred_start \
--pred_end $pred_end \
--cols "${cols[*]}" \
--use_autoregression $use_autoregression \
--use_best_params $use_best_params