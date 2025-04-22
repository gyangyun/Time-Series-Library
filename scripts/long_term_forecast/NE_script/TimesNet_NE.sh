# 设置实验名称
export model_name=TimesNet

# 设置数据参数
export data_path=dataset/processed/station_1_train_merged_data.parquet
export root_path=new_energy_forecast
export data=NEm
export target=power

# 设置训练参数
export train_epochs=10
export patience=5
export learning_rate=0.0001
export batch_size=16
export pred_len=96  # 24小时 * 4(15分钟)
export seq_len=960  # 调整为2的幂次方
export label_len=192  # 2天 * 24小时 * 4(15分钟)

# 设置模型参数
export e_layers=3
export d_layers=1
export factor=3
export enc_in=25  # 输入特征维度(24个气象特征 + 1个功率)
export dec_in=25
export c_out=1   # 输出维度(功率)
export d_model=512
export d_ff=1024
export top_k=5
export des='Exp'

# 设置训练时间范围
export train_start=20240101
export train_end=20241030
export test_start=20241201
export test_end=20241231

# 设置GPU参数
export gpu=0
export devices='0'
export use_gpu=1
export gpu_type='cuda'

# 设置实验ID
export id=1

# 设置输出路径
export checkpoints=$root_path/checkpoints/

# 创建必要的目录
mkdir -p $checkpoints

# 运行训练和测试


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $id \
  --model $model_name \
  --data $data \
  --features MS \
  --target $target \
  --freq t \
  --checkpoints $checkpoints \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --d_model $d_model \
  --d_ff $d_ff \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --factor $factor \
  --top_k $top_k \
  --des $des \
  --train_epochs $train_epochs \
  --patience $patience \
  --lradj type1 \
  --learning_rate $learning_rate \
  --batch_size $batch_size \
  --train_start $train_start \
  --train_end $train_end \
  --test_start $test_start \
  --test_end $test_end \
  --gpu $gpu \
  --gpu_type $gpu_type \
  --devices $devices \
  --use_gpu $use_gpu \
  --inverse

# 运行预测
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 0 \
#   --root_path $root_path \
#   --data_path $data_path \
#   --model_id $id \
#   --model $model_name \
#   --data $data \
#   --features MS \
#   --target $target \
#   --freq 15min \
#   --checkpoints $checkpoints \
#   --seq_len $seq_len \
#   --label_len $label_len \
#   --pred_len $pred_len \
#   --enc_in $enc_in \
#   --dec_in $dec_in \
#   --c_out $c_out \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --e_layers $e_layers \
#   --d_layers $d_layers \
#   --factor $factor \
#   --top_k $top_k \
#   --des $des \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --lradj type1 \
#   --learning_rate $learning_rate \
#   --batch_size $batch_size \
#   --train_start $train_start \
#   --train_end $train_end \
#   --test_start $test_start \
#   --test_end $test_end \
#   --use_gpu $use_gpu \
#   --gpu $gpu \
#   --use_multi_gpu $use_multi_gpu \
#   --devices $devices \
#   --output_attention false \
#   --use_amp true \
#   --use_dtw false \
#   --inverse true \
#   --use_autoregression false \
#   --output_dir $output_dir \
#   --pred_start 20250101 \
#   --pred_end 20250228 