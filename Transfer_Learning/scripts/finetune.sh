PROJECT_NAME="finetune_tf" 
DEVICE=$1
LAKE_CONFIG=$2
MODEL_TYPE=$3
LEARNING_RATE=$4
RUN_NAME=$5
PRETRAIN_RUN_NAME=$6

EPOCHS=100
LOOKBACK_WINDOW=21
HORIZON_WINDOW=14
LABEL_LEN=7
DROPOUT=0.01
WD=0.001
FREEZE_ENC=true
FREEZE_DEC=true

python -u executor.py \
       --project_name $PROJECT_NAME \
       --config_name $LAKE_CONFIG \
       --run_name $RUN_NAME \
       --task_name finetune \
       --pretrain_run_name $PRETRAIN_RUN_NAME \
       --lookback $LOOKBACK_WINDOW \
       --device $DEVICE \
       --max_epochs $EPOCHS \
       --weight_decay $WD \
       --dropout $DROPOUT \
       --horizon_window $HORIZON_WINDOW \
       --ntrials 5 \
       --label_len $LABEL_LEN \
       --learning_rate $LEARNING_RATE \
       --freeze_enc $FREEZE_ENC \
       --freeze_dec $FREEZE_DEC \
       --model_type $MODEL_TYPE \
       --horizon_csv_path ./horizon_csv_${MODEL} \