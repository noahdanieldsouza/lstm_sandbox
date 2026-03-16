PROJECT_NAME="lstm_project"
DEVICE=0
CONFIG_NAME=config.json
MODEL_TYPE=lstm
LEARNING_RATE=0.001
RUN_NAME="t0"
PRETRAIN_RUN_NAME="t0"

EPOCHS=20
LOOKBACK_WINDOW=24
HORIZON_WINDOW=1
LABEL_LEN=12
DROPOUT=0.01
WD=0.001
FREEZE_ENC=true
FREEZE_DEC=true

python -u executor.py \
       --project_name $PROJECT_NAME \
       --config_name $CONFIG_NAME \
       --run_name $RUN_NAME \
       --task_name finetune \
       --pretrain_run_name $PRETRAIN_RUN_NAME \
       --lookback_window $LOOKBACK_WINDOW \
       --horizon_window $HORIZON_WINDOW \
       --device $DEVICE \
       --max_epochs $EPOCHS \
       --weight_decay $WD \
       --dropout $DROPOUT \
       --ntrials 1 \
       --label_len $LABEL_LEN \
       --learning_rate $LEARNING_RATE \
       --freeze_enc $FREEZE_ENC \
       --freeze_dec $FREEZE_DEC \
       --model_type $MODEL_TYPE \
       --horizon_csv_path ./horizon_csv_${MODEL_TYPE} \
       --source_filename "mendota_observed.csv" \
       --c_out 3 \
       --target_cols poc DO_filled secchi_m \
       --masked_columns poc DO_filled secchi_m \
       --num_layers 4 \
       --hidden_feature_size 128 \
       --batch_size 256 \
       --max_lr 0.003 \
       --div_factor 10 \
       --final_div_factor 1.05 \
       --pct_start 0.3 \
