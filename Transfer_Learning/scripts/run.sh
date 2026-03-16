#!/bin/bash

mkdir -p logs

bash scripts/pretrain.sh 0 "run_01_standard" 200 24 5 0.01 0.001 128 256 > logs/run_01_standard.log 2>&1 &
bash scripts/pretrain.sh 1 "run_02_b64" 200 24 5 0.01 0.001 64 256 > logs/run_02_b64.log 2>&1 &
bash scripts/pretrain.sh 2 "run_03_b256" 200 24 5 0.01 0.001 256 256 > logs/run_03_b256.log 2>&1 &
bash scripts/pretrain.sh 3 "run_04_hf128" 200 24 5 0.01 0.001 128 128 > logs/run_04_hf128.log 2>&1 &
bash scripts/pretrain.sh 4 "run_05_b64" 200 24 5 0.01 0.001 128 512 > logs/run_05_b64.log 2>&1 &
bash scripts/pretrain.sh 5 "run_07_ep400" 400 24 5 0.01 0.001 128 256 > logs/run_07_ep400.log 2>&1 &
bash scripts/pretrain.sh 0 "run_08_wd0.0001" 200 24 5 0.01 0.0001 128 256 > logs/run_08_wd0.0001.log 2>&1 &
bash scripts/pretrain.sh 1 "run_09_do0" 200 24 5 0 0.001 128 256 > logs/run_09_do0.log 2>&1 &
bash scripts/pretrain.sh 2 "run_10_do0.02" 200 24 5 0.02 0.001 128 256 > logs/run_10_do0.02.log 2>&1 &
bash scripts/pretrain.sh 3 "run_11_do0.05" 200 24 5 0.05 0.001 128 256 > logs/run_11_do0.05.log 2>&1 &
bash scripts/pretrain.sh 4 "run_12_do0.1" 200 24 5 0.1 0.001 128 256 > logs/run_12_do0.1.log 2>&1 &
bash scripts/pretrain.sh 5 "run_13_h1" 200 24 1 0.01 0.001 128 256 > logs/run_13_h1.log 2>&1 &
bash scripts/pretrain.sh 6 "run_14_lb48" 200 48 5 0.01 0.001 128 256 > logs/run_14_lb48.log 2>&1 &