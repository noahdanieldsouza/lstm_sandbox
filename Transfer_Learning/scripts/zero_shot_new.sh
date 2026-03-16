PROJECT_NAME="lstm_project"
DEVICE=0
LAKE_CONFIG=config.json
MODEL_TYPE=lstm
RUN_NAME="t0"

LOOKBACK_WINDOW=24
HORIZON_WINDOW=1
LABEL_LEN=12
DROPOUT=0.01

# Specify the checkpoint path from which we load the model
CKPT_PATH="./pretrain_checkpoints/t0_lstm_pretrain/ckpt_latest.pth"
SOURCE_FILENAME="mendota_observed.csv" # specify the dataset to evaluate on
CONFIG_NAME="config_zeroshot.json" # if you want to change the splits

python -u executor.py \
       --project_name $PROJECT_NAME \
       --config_name $CONFIG_NAME \
       --run_name $RUN_NAME \
       --task_name zeroshot \
       --ckpt_path $CKPT_PATH \
       --lookback_window $LOOKBACK_WINDOW \
       --horizon_window $HORIZON_WINDOW \
       --device $DEVICE \
       --max_epochs 1 \
       --dropout $DROPOUT \
       --ntrials 1 \
       --label_len $LABEL_LEN \
       --learning_rate 0.001 \
       --model_type $MODEL_TYPE \
       --horizon_csv_path ./horizon_csv_${MODEL_TYPE} \
       --source_filename "$SOURCE_FILENAME" \
       --c_out 3 \
       --target_cols poc DO_filled secchi_m \
       --masked_columns poc DO_filled secchi_m \
       --num_layers 4 \
       --hidden_feature_size 128 \
       --batch_size 256 \
