export CUDA_VISIBLE_DEVICES=0

# =========================公用参数=========================
task_name="long_term_forecast"
root_path="/home/guoyy/Workspace/ts/lib/全社会用电量预测（省级）/cache"
# root_path = "/Users/guoyangyun/计量中心/16.分析报告/分析/智能报表/全社会用电量预测（省级）/cache"
data_path="train_ppd_hydl_tq_r_df.pkl"
model_id="IEd1_28_1"
model_name="TimesNet"
# model_name="TimeMixer"
# model_name="iTransformer"
# model_name="Nonstationary_Transformer"
data_name="IEd1"
features="MS"
seq_len=56
label_len=7
pred_len=1
e_layers=2
d_layers=1
factor=3
# enc_in=5
# dec_in=5
enc_in=8
dec_in=8
c_out=1
d_model=16
d_ff=32
description="Exp"
itr=1
train_epochs=30
top_k=5
freq="d"
target="electricity_consumption"
province_name="广东"
order_no="1"
train_start="2022-01-01"
train_end="2024-02-29"
test_start="2024-06-01"
test_end="2024-06-30"
# cols="date,wd,wd_max,wd_min,holiday_code,electricity_consumption"
cols="date,wd,wd_max,wd_min,wd_1,wd_max_1,wd_min_1,holiday_code,electricity_consumption"

# # =========================train=========================
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
  --province_name $province_name \
  --order_no $order_no \
  --train_start $train_start \
  --train_end $train_end \
  --test_start $test_start \
  --test_end $test_end \
  --cols $cols

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
  --province_name $province_name \
  --order_no $order_no \
  --train_start $train_start \
  --train_end $train_end \
  --test_start $test_start \
  --test_end $test_end \
  --cols $cols

# =========================predict=========================
is_training=2
data_path="pred_ppd_hydl_tq_r_df.pkl"
pred_start="2024-07-01"
pred_end="2024-07-31"
is_autoregression=1

python -u run.py \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
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
  --province_name $province_name \
  --order_no $order_no \
  --train_start $train_start \
  --train_end $train_end \
  --test_start $test_start \
  --test_end $test_end \
  --pred_start $pred_start \
  --pred_end $pred_end \
  --is_autoregression $is_autoregression \
  --cols $cols

