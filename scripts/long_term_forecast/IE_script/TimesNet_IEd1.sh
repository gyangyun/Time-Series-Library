export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet
# model_name=Autoformer

# --root_path /home/guoyy/Workspace/ts/lib/全社会用电量预测（省级）/cache \
# --cols 'date,wd,wd_max,wd_min,holiday_code,electricity_consumption'

# =========================train=========================
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --train_epochs 1 \
#   --root_path /Users/guoyangyun/计量中心/16.分析报告/分析/智能报表/全社会用电量预测（省级）/cache \
#   --data_path train_ppd_hydl_tq_r_df.pkl \
#   --model_id IEd1_28_1 \
#   --model $model_name \
#   --data IEd1 \
#   --features MS \
#   --seq_len 28 \
#   --label_len 7 \
#   --pred_len 7 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 5 \
#   --dec_in 5 \
#   --c_out 1 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 \
#   --freq 'd' \
#   --target 'electricity_consumption' \
#   --province_name '广东' \
#   --order_no '1' \
#   --train_start '2022-01-01' \
#   --train_end '2022-02-28' \
#   --test_start '2022-03-08' \
#   --test_end '2022-03-15' \
#   --cols 'date,wd,wd_max,wd_min,holiday_code,electricity_consumption'

# =========================test=========================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --train_epochs 1 \
  --root_path /Users/guoyangyun/计量中心/16.分析报告/分析/智能报表/全社会用电量预测（省级）/cache \
  --data_path train_ppd_hydl_tq_r_df.pkl \
  --model_id IEd1_28_1 \
  --model $model_name \
  --data IEd1 \
  --features MS \
  --seq_len 28 \
  --label_len 7 \
  --pred_len 7 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 1 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --freq 'd' \
  --target 'electricity_consumption' \
  --province_name '广东' \
  --order_no 1 \
  --train_start '2022-01-01' \
  --train_end '2022-02-28' \
  --test_start '2022-03-08' \
  --test_end '2022-03-15' \
  --cols 'date,wd,wd_max,wd_min,holiday_code,electricity_consumption'

# =========================predict=========================
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 2 \
#   --root_path /Users/guoyangyun/计量中心/16.分析报告/分析/智能报表/全社会用电量预测（省级）/cache \
#   --data_path train_ppd_hydl_tq_r_df.pkl \
#   --model_id IEd1_28_1 \
#   --model $model_name \
#   --data IEd1 \
#   --features MS \
#   --seq_len 28 \
#   --label_len 7 \
#   --pred_len 7 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 5 \
#   --dec_in 5 \
#   --c_out 1 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 \
#   --freq 'd' \
#   --target 'electricity_consumption' \
#   --province_name '广东' \
#   --order_no 1 \
#   --train_start '2022-01-01' \
#   --train_end '2022-02-28' \
#   --test_start '2022-03-08' \
#   --test_end '2022-03-15' \
#   --pred_start '2022-03-16' \
#   --pred_end '2022-03-18' \
#   --is_autoregression 1 \
#   --cols 'date,wd,wd_max,wd_min,holiday_code,electricity_consumption'
