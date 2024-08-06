export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet
# model_name=Autoformer

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /home/guoyy/Workspace/ts/lib/全社会用电量预测（省级）/cache \
#   --data_path train_ppd_hydl_tq_r_df.pkl \
#   --model_id IEd1_28_1 \
#   --model $model_name \
#   --data IEd1 \
#   --features MS \
#   --seq_len 28 \
#   --label_len 7 \
#   --pred_len 1 \
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
#   --industry_name '全社会用电总计' \
#   --train_start '2022-01-01' \
#   --train_end '2024-02-29' \
#   --test_start '2024-06-01' \
#   --test_end '2024-06-30' \
#   --cols 'date,wd,wd_max,wd_min,holiday_code,electricity_consumption'

  # --cols 'date,wd,wd_max,wd_min,holiday_code,electricity_consumption'

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path /home/guoyy/Workspace/ts/lib/全社会用电量预测（省级）/cache \
  --data_path train_ppd_hydl_tq_r_df.pkl \
  --model_id IEd1_28_1 \
  --model $model_name \
  --data IEd1 \
  --features MS \
  --seq_len 28 \
  --label_len 7 \
  --pred_len 1 \
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
  --industry_name '全社会用电总计' \
  --train_start '2022-01-01' \
  --train_end '2024-02-29' \
  --test_start '2024-06-01' \
  --test_end '2024-06-30' \
  --pred_start '2024-07-01' \
  --pred_end '2024-07-07' \
  --is_autoregression 1 \
  --cols 'date,wd,wd_max,wd_min,holiday_code,electricity_consumption'

  # --cols 'date,wd,wd_max,wd_min,holiday_code,electricity_consumption'

