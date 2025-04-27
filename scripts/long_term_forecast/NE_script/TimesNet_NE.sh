#!/bin/bash

# 设置基础参数
export model_name=TimesNet
export task_name=long_term_forecast
export root_path=new_energy_forecast

# 设置训练参数
export train_epochs=20
export patience=3
export learning_rate=0.0001
export batch_size=16
export pred_len=96
export seq_len=96
export label_len=0

# 设置模型参数
export e_layers=4
export d_layers=2
export factor=3
export c_out=1   # 输出维度(功率)
export d_model=256
export d_ff=512
export top_k=1
export des='Exp'

# 设置损失函数参数
export loss_alpha=1.5
export loss_beta=0.8
export loss_gamma=0.5
export loss_delta=0
export loss_omega=1

# 设置时间范围
export train_start=20240102000000
export train_end=20241030234500
export test_start=20241201000000
export test_end=20241231234500
export pred_start=20250101000000
export pred_end=20250228234500

# 设置GPU参数
export gpu=0
export devices='0'
export gpu_type='cuda'
export freq=15min
export target=power
export data=NEv2

# 创建必要的目录
export checkpoints=$root_path/checkpoints/
mkdir -p $checkpoints

# 定义通用的运行函数
run_model() {
    local station_id=$1
    local is_training=$2
    local data_path=$3
    
    # 根据站点类型设置特征
    if [ $station_id -le 5 ]; then
        # 风电站特征
        export cols="nwp_1_v100,nwp_1_t2m,nwp_1_sp,nwp_1_u100,\
nwp_2_v100,nwp_2_t2m,nwp_2_msl,nwp_2_u100,\
nwp_3_v100,nwp_3_t2m,nwp_3_sp,nwp_3_u100"
        export enc_in=12
        export dec_in=12
    else
        # 光伏站特征
        export cols="nwp_1_poai,nwp_1_ghi,nwp_1_t2m,nwp_1_tcc,\
nwp_2_poai,nwp_2_ghi,nwp_2_t2m,nwp_2_tcc,\
nwp_3_poai,nwp_3_ghi,nwp_3_t2m,nwp_3_tcc"
        export enc_in=12
        export dec_in=12
    fi

    # 设置数据路径
    export scaler_path=${root_path}/dataset/processed/station_${station_id}_train_merged_data_scaler.pkl

    echo "正在处理站点 ${station_id}, 模式: ${is_training}"
    
    python -u run.py \
        --task_name $task_name \
        --is_training $is_training \
        --root_path $root_path \
        --data_path $data_path \
        --model_id $station_id \
        --model $model_name \
        --data $data \
        --features MS \
        --target $target \
        --freq $freq \
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
        --use_gpu \
        --inverse \
        --stride 96 \
        --scale \
        --scaler_path $scaler_path \
        --cols $cols \
        --loss renewable \
        --loss_alpha $loss_alpha \
        --loss_beta $loss_beta \
        --loss_gamma $loss_gamma \
        --loss_delta $loss_delta \
        --loss_omega $loss_omega \
        ${is_training_extra_args}
}

# 遍历所有站点
for station_id in {1..10}
do
    echo "开始处理站点 ${station_id}"
    
    # # 1. 训练阶段
    # export is_training=1
    # export data_path=dataset/processed/station_${station_id}_train_merged_data.parquet
    # run_model $station_id $is_training $data_path
    
    # # 2. 测试阶段
    # export is_training=0
    # export data_path=dataset/processed/station_${station_id}_train_merged_data.parquet
    # run_model $station_id $is_training $data_path
    
    # 3. 预测阶段
    export is_training=2
    export data_path=dataset/processed/station_${station_id}_test_merged_data.parquet
    export is_training_extra_args="--pred_start $pred_start --pred_end $pred_end"
    run_model $station_id $is_training $data_path
    
    echo "站点 ${station_id} 处理完成"
    echo "----------------------------------------"
done

echo "所有站点处理完成！"