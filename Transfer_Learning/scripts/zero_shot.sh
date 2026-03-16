PROJECT_NAME="zeroshot_tf" 
DEVICE=0
LAKE_CONFIG=config.json
MODEL_TYPE=lstm
LEARNING_RATE=0.001
RUN_NAME="test_lstm"
PRETRAIN_RUN_NAME="${RUN_NAME}_${MODEL_TYPE}_pretrain"

EPOCHS=1
LOOKBACK_WINDOW=24
HORIZON_WINDOW=5
LABEL_LEN=7
DROPOUT=0.01
WD=0.001

DEVICE=0

python -u executor.py \
       --project_name $PROJECT_NAME \
       --config_name $LAKE_CONFIG \
       --run_name $RUN_NAME \
       --task_name zeroshot \
       --freeze_enc true \
       --freeze_dec true \
       --pretrain_run_name $PRETRAIN_RUN_NAME \
       --lookback_window $LOOKBACK_WINDOW \
       --device $DEVICE \
       --max_epochs $EPOCHS \
       --weight_decay $WD \
       --dropout $DROPOUT \
       --horizon_window $HORIZON_WINDOW \
       --ntrials 3 \
       --label_len $LABEL_LEN \
       --learning_rate $LEARNING_RATE \
       --model_type $MODEL_TYPE \
       --horizon_csv_path ./horizon_csv_${MODEL_TYPE} \
       --source_filename "mendota_observed.csv" \
       --target_cols poc DO_filled secchi_m