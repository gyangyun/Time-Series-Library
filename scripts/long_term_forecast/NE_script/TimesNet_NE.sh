# 设置实验名称
export model_name=TimesNet

# 设置数据参数
export data_path=dataset/processed/station_1_train_merged_data.parquet
export root_path=new_energy_forecast
export data=NEm
export target=power

# 设置特征选择参数
station_id=$(echo $data_path | grep -o 'station_[0-9]*' | cut -d'_' -f2)
if [ $station_id -le 5 ]; then
    # 风电站特征 - 使用三个数据源的关键特征
    export cols="nwp_1_v100,nwp_1_t2m,nwp_1_sp,nwp_1_u100,\
nwp_2_v100,nwp_2_t2m,nwp_2_msl,nwp_2_u100,\
nwp_3_v100,nwp_3_t2m,nwp_3_sp,nwp_3_u100,\
power"
    export enc_in=13  # 12个特征 + 1个功率
    export dec_in=13
else
    # 光伏站特征 - 使用三个数据源的关键特征
    export cols="nwp_1_poai,nwp_1_ghi,nwp_1_t2m,nwp_1_tcc,\
nwp_2_poai,nwp_2_ghi,nwp_2_t2m,nwp_2_tcc,\
nwp_3_poai,nwp_3_ghi,nwp_3_t2m,nwp_3_tcc,\
power"
    export enc_in=13  # 12个特征 + 1个功率
    export dec_in=13
fi

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

# 首先进行训练
python -u run.py \
  --task_name new_energy_forecast \
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
  --inverse \
  --cols $cols

# 设置预测开始和结束日期
start_date=20250101
end_date=20250228

# 创建临时数据文件
cp $data_path ${data_path%.parquet}_temp.parquet

# 逐天预测循环
current_date=$start_date
while [ $current_date -le $end_date ]; do
    echo "正在预测日期: $current_date"
    
    # 设置预测参数
    next_date=$(date -d "$current_date" +%Y%m%d)
    
    # 运行预测
    python -u run.py \
      --task_name new_energy_forecast \
      --is_training 0 \
      --root_path $root_path \
      --data_path ${data_path%.parquet}_temp.parquet \
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
      --pred_start $current_date \
      --pred_end $current_date \
      --gpu $gpu \
      --gpu_type $gpu_type \
      --devices $devices \
      --use_gpu $use_gpu \
      --inverse \
      --use_autoregression 1 \
      --cols $cols
      
    # 更新数据集
    python -u scripts/long_term_forecast/NE_script/update_dataset.py \
      --data_path ${data_path%.parquet}_temp.parquet \
      --pred_date $current_date
      
    # 更新日期到下一天
    current_date=$(date -d "$current_date + 1 day" +%Y%m%d)
done

# 清理临时文件
rm ${data_path%.parquet}_temp.parquet

echo "预测完成！"

