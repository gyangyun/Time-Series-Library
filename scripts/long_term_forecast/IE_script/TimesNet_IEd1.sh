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
seq_len=14
label_len=7
pred_len=1
e_layers=2
d_layers=1
factor=3
enc_in=8
dec_in=8
c_out=1
d_model=16
d_ff=32

model_id="IEd1_${seq_len}_${label_len}_${pred_len}"
# -------------------------训练参数-------------------------
description="Exp"
itr=1
# train_epochs=30
train_epochs=1
top_k=5
freq="d"
target="electricity_consumption"
# -------------------------自定义参数-------------------------
# train_start="2022-01-01"
# train_end="2024-02-29"
# test_start="2024-06-01"
# test_end="2024-06-30"

train_start="2022-01-01"
train_end="2022-02-28"
test_start="2022-03-07"
test_end="2022-03-14"
# cols="date,wd,wd_max,wd_min,holiday_code,electricity_consumption"
cols="date,wd,wd_max,wd_min,wd_1,wd_max_1,wd_min_1,holiday_code,electricity_consumption"
# =========================根目录调整=========================
data_path="predict_dataset.pkl"
# province_names=("广东" "广西" "云南" "贵州" "海南")
province_names=("广东" "广西")
# industry_ids=("[1]全社会用电总计" "[2]A、全行业用电合计" "[3]第一产业" "[4]第二产业" "[5]第三产业" "[6]B、城乡居民生活用电合计" "[7]城镇居民" "[8]乡村居民" "[9]C、趸售" "[10]D、其他、无行业分类")
industry_ids=("[1]全社会用电总计")

# dataset_path="/home/guoyy/Workspace/ts/lib/全社会用电量预测（省级）/cache/dataset/deep_learning"
dataset_path="/Users/guoyangyun/计量中心/16.分析报告/分析/智能报表/ElecForcastPrep/cache/dataset/deep_learning"

# 遍历所有 province_name 和 industry_id 的组合
# for province_name in "${province_names[@]}"; do
#     for industry_id in "${industry_ids[@]}"; do
#         root_path="${dataset_path}/${province_name}/${industry_id}"
#
#         # 在这里进行你需要的操作，例如打印路径
#         echo "处理路径: ${root_path}"
#
#         # =========================train=========================
#         is_training=1
#
#         python -u run.py \
#         --task_name $task_name \
#         --is_training $is_training \
#         --root_path $root_path \
#         --data_path $data_path \
#         --model_id $model_id \
#         --model $model_name \
#         --data $data_name \
#         --features $features \
#         --seq_len $seq_len \
#         --label_len $label_len \
#         --pred_len $pred_len \
#         --e_layers $e_layers \
#         --d_layers $d_layers \
#         --factor $factor \
#         --enc_in $enc_in \
#         --dec_in $dec_in \
#         --c_out $c_out \
#         --d_model $d_model \
#         --d_ff $d_ff \
#         --des $description \
#         --itr $itr \
#         --train_epochs $train_epochs \
#         --top_k $top_k \
#         --freq $freq \
#         --target $target \
#         --province_name $province_name \
#         --industry_id $industry_id \
#         --train_start $train_start \
#         --train_end $train_end \
#         --test_start $test_start \
#         --test_end $test_end \
#         --cols $cols
#
#         # =========================test=========================
#         is_training=0
#
#         python -u run.py \
#         --task_name $task_name \
#         --is_training $is_training \
#         --root_path $root_path \
#         --data_path $data_path \
#         --model_id $model_id \
#         --model $model_name \
#         --data $data_name \
#         --features $features \
#         --seq_len $seq_len \
#         --label_len $label_len \
#         --pred_len $pred_len \
#         --e_layers $e_layers \
#         --d_layers $d_layers \
#         --factor $factor \
#         --enc_in $enc_in \
#         --dec_in $dec_in \
#         --c_out $c_out \
#         --d_model $d_model \
#         --d_ff $d_ff \
#         --des $description \
#         --itr $itr \
#         --train_epochs $train_epochs \
#         --top_k $top_k \
#         --freq $freq \
#         --target $target \
#         --province_name $province_name \
#         --industry_id $industry_id \
#         --train_start $train_start \
#         --train_end $train_end \
#         --test_start $test_start \
#         --test_end $test_end \
#         --cols $cols
#
#         # =========================predict=========================
#         is_training=2
#
#         data_path="predict_dataset.pkl"
#         pred_start="2022-03-14"
#         pred_end="2022-03-21"
#         is_autoregression=1
#
#         python -u run.py \
#         --task_name $task_name \
#         --is_training $is_training \
#         --root_path $root_path \
#         --data_path $data_path \
#         --model_id $model_id \
#         --model $model_name \
#         --data $data_name \
#         --features $features \
#         --seq_len $seq_len \
#         --label_len $label_len \
#         --pred_len $pred_len \
#         --e_layers $e_layers \
#         --d_layers $d_layers \
#         --factor $factor \
#         --enc_in $enc_in \
#         --dec_in $dec_in \
#         --c_out $c_out \
#         --d_model $d_model \
#         --d_ff $d_ff \
#         --des $description \
#         --itr $itr \
#         --train_epochs $train_epochs \
#         --top_k $top_k \
#         --freq $freq \
#         --target $target \
#         --province_name $province_name \
#         --industry_id $industry_id \
#         --train_start $train_start \
#         --train_end $train_end \
#         --test_start $test_start \
#         --test_end $test_end \
#         --pred_start $pred_start \
#         --pred_end $pred_end \
#         --is_autoregression $is_autoregression \
#         --cols $cols
#
#     done
# done

# =========================合并结果=========================

pred_start="2022-03-14"
pred_end="2022-03-21"
is_training=2

python -u combine_result.py \
--task_name $task_name \
--is_training $is_training \
--model_id $model_id \
--model $model_name \
--data $data_name \
--features $features \
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
--des $description \
--itr $itr \
--train_epochs $train_epochs \
--top_k $top_k \
--freq $freq \
--target $target \
--train_start $train_start \
--train_end $train_end \
--test_start $test_start \
--test_end $test_end \
--pred_start $pred_start \
--pred_end $pred_end \
--cols $cols \
--dataset_path "$dataset_path" \
--province_names "${province_names[*]}" \
--industry_ids "${industry_ids[*]}"
