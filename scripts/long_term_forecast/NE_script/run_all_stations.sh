#!/bin/bash

# 循环处理所有场站
for station_id in {1..10}
do
    echo "Processing station ${station_id}"
    
    # 设置数据路径
    export data_path=dataset/processed/station_${station_id}_merged_data.parquet
    
    # 根据场站类型调整模型参数
    if [ $station_id -le 5 ]; then
        echo "Wind power station"
        export model_name=TimesNet
        export train_epochs=10
        export d_model=512
        export d_ff=2048
    else
        echo "Solar power station"
        export model_name=TimesNet
        export train_epochs=15
        export d_model=256
        export d_ff=1024
    fi
    
    # 设置实验ID
    export id=${station_id}
    
    # 运行训练脚本
    bash scripts/long_term_forecast/NE_script/TimesNet_NE.sh
    
    echo "Finished processing station ${station_id}"
    echo "----------------------------------------"
done

# 合并所有预测结果
python scripts/long_term_forecast/NE_script/merge_predictions.py 