export CUDA_VISIBLE_DEVICES=0

# =========================公用参数=========================
task_name="long_term_forecast"
# -------------------------模型参数-------------------------
model_name="TimesNet"
# model_name="TimeMixer"
# model_name="iTransformer"
# model_name="Nonstationary_Transformer"

data_name="IEd1"
features="MS"
seq_len=60
label_len=31
pred_len=31
e_layers=4
d_layers=2
factor=3
enc_in=95
dec_in=95
c_out=1
d_model=72
d_ff=128

# TimeMixer
down_sampling_layers=2
down_sampling_window=1
down_sampling_method="avg"
batch_size=16

model_id="IEd1_${seq_len}_${label_len}_${pred_len}"
# -------------------------训练参数-------------------------
description="Exp"
itr=1
train_epochs=30
# train_epochs=1
top_k=5
freq="d"
target="electricity_consumption"
# -------------------------自定义参数-------------------------
train_start="2022-01-01"
train_end="2024-02-29"
test_start="2024-06-01"
test_end="2024-07-31"

# train_start="2022-01-01"
# train_end="2022-02-28"
# test_start="2022-03-07"
# test_end="2022-03-14"
# cols=("date" "wd_-1_shift" "wd_max_-1_shift" "wd_min_-1_shift" "holiday_flag_te" "electricity_consumption")
# cols=("date" "wd_-1_shift" "wd_max_-1_shift" "wd_min_-1_shift" "wd_max_rolling_7_mean_-1_shift" "wd_min_rolling_7_mean_-1_shift" "wd_rolling_7_mean_-1_shift" "holiday_flag_te" "electricity_consumption")
# cols=("date" "wd_-1_shift" "wd_max_-1_shift" "wd_min_-1_shift" "wd_-1_shift_group_te" "wd_max_-1_shift_group_te" "wd_min_-1_shift_group_te" "holiday_flag_te" "electricity_consumption")


cols=("date" "wd_-1_shift" "wd_-2_shift" "wd_-3_shift" "wd_-4_shift" "wd_-5_shift" "wd_-6_shift" "wd_-7_shift" "wd_-8_shift" "wd_-9_shift" "wd_-10_shift" "wd_-11_shift" "wd_-12_shift" "wd_-13_shift" "wd_-14_shift" "wd_-15_shift" "wd_-16_shift" "wd_-17_shift" "wd_-18_shift" "wd_-19_shift" "wd_-20_shift" "wd_-21_shift" "wd_-22_shift" "wd_-23_shift" "wd_-24_shift" "wd_-25_shift" "wd_-26_shift" "wd_-27_shift" "wd_-28_shift" "wd_-29_shift" "wd_-30_shift" "wd_-31_shift" "wd_max_-1_shift" "wd_max_-2_shift" "wd_max_-3_shift" "wd_max_-4_shift" "wd_max_-5_shift" "wd_max_-6_shift" "wd_max_-7_shift" "wd_max_-8_shift" "wd_max_-9_shift" "wd_max_-10_shift" "wd_max_-11_shift" "wd_max_-12_shift" "wd_max_-13_shift" "wd_max_-14_shift" "wd_max_-15_shift" "wd_max_-16_shift" "wd_max_-17_shift" "wd_max_-18_shift" "wd_max_-19_shift" "wd_max_-20_shift" "wd_max_-21_shift" "wd_max_-22_shift" "wd_max_-23_shift" "wd_max_-24_shift" "wd_max_-25_shift" "wd_max_-26_shift" "wd_max_-27_shift" "wd_max_-28_shift" "wd_max_-29_shift" "wd_max_-30_shift" "wd_max_-31_shift" "wd_min_-1_shift" "wd_min_-2_shift" "wd_min_-3_shift" "wd_min_-4_shift" "wd_min_-5_shift" "wd_min_-6_shift" "wd_min_-7_shift" "wd_min_-8_shift" "wd_min_-9_shift" "wd_min_-10_shift" "wd_min_-11_shift" "wd_min_-12_shift" "wd_min_-13_shift" "wd_min_-14_shift" "wd_min_-15_shift" "wd_min_-16_shift" "wd_min_-17_shift" "wd_min_-18_shift" "wd_min_-19_shift" "wd_min_-20_shift" "wd_min_-21_shift" "wd_min_-22_shift" "wd_min_-23_shift" "wd_min_-24_shift" "wd_min_-25_shift" "wd_min_-26_shift" "wd_min_-27_shift" "wd_min_-28_shift" "wd_min_-29_shift" "wd_min_-30_shift" "wd_min_-31_shift" "holiday_flag_te" "electricity_consumption")



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

# =========================根目录调整=========================
data_path="predict_dataset.pkl"
# province_names=("广东" "广西" "云南" "贵州" "海南")
# province_names=("广东")
province_names=("贵州")
# industry_ids=("[1]全社会用电总计" "[2]A、全行业用电合计" "[3]第一产业" "[4]第二产业" "[5]第三产业" "[6]B、城乡居民生活用电合计" "[7]城镇居民" "[8]乡村居民" "[9]C、趸售" "[10]D、其他、无行业分类")
# industry_ids=("[1]全社会用电总计")
# industry_ids=("[4]第二产业")
industry_ids=("[6]B、城乡居民生活用电合计")

dataset_path="/home/guoyy/Workspace/ts/lib/ElecForcastPrep/cache/dataset/deep_learning"
# dataset_path="/Users/guoyangyun/计量中心/16.分析报告/分析/智能报表/ElecForcastPrep/cache/dataset/deep_learning"

# 遍历所有 province_name 和 industry_id 的组合
for province_name in "${province_names[@]}"; do
    for industry_id in "${industry_ids[@]}"; do
        root_path="${dataset_path}/${province_name}/${industry_id}"
        checkpoints="${root_path}"

        # 在这里进行你需要的操作，例如打印路径
        echo "处理路径: ${root_path}"

        # =========================train=========================
        is_training=1

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
        --province_name $province_name \
        --industry_id $industry_id \
        --train_start $train_start \
        --train_end $train_end \
        --test_start $test_start \
        --test_end $test_end \
        --cols "${cols[*]}"

        # =========================test=========================
        is_training=0

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
        --province_name $province_name \
        --industry_id $industry_id \
        --train_start $train_start \
        --train_end $train_end \
        --test_start $test_start \
        --test_end $test_end \
        --cols "${cols[*]}"

        # =========================predict=========================
        # is_training=2
        # data_path="predict_dataset.pkl"
        # pred_start="2024-07-01"
        # pred_end="2024-07-31"
        # # pred_start="2024-06-01"
        # # pred_end="2024-06-30"
        # use_autoregression=0
        #
        # python -u run.py \
        # --task_name $task_name \
        # --is_training $is_training \
        # --root_path $root_path \
        # --data_path $data_path \
        # --model_id $model_id \
        # --model $model_name \
        # --data $data_name \
        # --features $features \
        # --checkpoints $checkpoints \
        # --seq_len $seq_len \
        # --label_len $label_len \
        # --pred_len $pred_len \
        # --e_layers $e_layers \
        # --d_layers $d_layers \
        # --factor $factor \
        # --enc_in $enc_in \
        # --dec_in $dec_in \
        # --c_out $c_out \
        # --d_model $d_model \
        # --d_ff $d_ff \
        # --down_sampling_layers $down_sampling_layers \
        # --down_sampling_window $down_sampling_window \
        # --down_sampling_method $down_sampling_method \
        # --des $description \
        # --itr $itr \
        # --train_epochs $train_epochs \
        # --batch_size $batch_size \
        # --top_k $top_k \
        # --freq $freq \
        # --target $target \
        # --province_name $province_name \
        # --industry_id $industry_id \
        # --train_start $train_start \
        # --train_end $train_end \
        # --test_start $test_start \
        # --test_end $test_end \
        # --pred_start $pred_start \
        # --pred_end $pred_end \
        # --use_autoregression $use_autoregression \
        # --cols "${cols[*]}"

    done
done

# =========================合并结果=========================

# python -u combine_result.py \
# --task_name $task_name \
# --is_training $is_training \
# --model_id $model_id \
# --model $model_name \
# --data $data_name \
# --features $features \
# --seq_len $seq_len \
# --label_len $label_len \
# --pred_len $pred_len \
# --e_layers $e_layers \
# --d_layers $d_layers \
# --factor $factor \
# --enc_in $enc_in \
# --dec_in $dec_in \
# --c_out $c_out \
# --d_model $d_model \
# --d_ff $d_ff \
# --down_sampling_layers $down_sampling_layers \
# --down_sampling_window $down_sampling_window \
# --down_sampling_method $down_sampling_method \
# --des $description \
# --itr $itr \
# --train_epochs $train_epochs \
# --batch_size $batch_size \
# --top_k $top_k \
# --freq $freq \
# --target $target \
# --train_start $train_start \
# --train_end $train_end \
# --test_start $test_start \
# --test_end $test_end \
# --pred_start $pred_start \
# --pred_end $pred_end \
# --cols "${cols[*]}" \
# --dataset_path $dataset_path \
# --province_names "${province_names[*]}" \
# --industry_ids "${industry_ids[*]}"
