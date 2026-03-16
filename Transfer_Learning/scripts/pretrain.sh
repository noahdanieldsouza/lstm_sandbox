PROJECT_NAME="test_pretrain" 
DEVICE=$1
LAKE_CONFIG=config.json
MODEL_TYPE=lstm
LEARNING_RATE=0.0001
RUN_NAME=$2

EPOCHS=$3
LOOKBACK_WINDOW=$4
HORIZON_WINDOW=$5
LABEL_LEN=12
DROPOUT=$6
WD=$7
BATCH_SIZE=$8
HIDDEN_FEATURE_SIZE=$9

python -u executor.py \
       --project_name $PROJECT_NAME \
       --config_name $LAKE_CONFIG \
       --run_name $RUN_NAME \
       --task_name pretrain \
       --lookback_window $LOOKBACK_WINDOW \
       --horizon_window $HORIZON_WINDOW \
       --device $DEVICE \
       --max_epochs $EPOCHS \
       --weight_decay $WD \
       --dropout $DROPOUT \
       --ntrials 1 \
       --label_len $LABEL_LEN \
       --learning_rate $LEARNING_RATE \
       --model_type $MODEL_TYPE \
       --horizon_csv_path ./horizon_csv_${MODEL_TYPE} \
       --source_filename "mendota_observed.csv" \
       --c_out 3 \
       --target_cols poc DO_filled secchi_m \
       --num_layers 4 \
       --hidden_feature_size $HIDDEN_FEATURE_SIZE \
       --batch_size $BATCH_SIZE \
       --max_lr 0.003 \
       --div_factor 10 \
       --final_div_factor 1.05 \
       --pct_start 0.3