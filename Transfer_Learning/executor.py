import argparse
import torch
import random
import numpy as np
import os
import json
import pandas as pd
import math
import datetime
import wandb

from utility import Utils
from functools import partial
from model import TFModel, seq2seq
from trainer import init_wandb, Trainer

import sys


parser = argparse.ArgumentParser(description='TL')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='train', choices=['pretrain', 'finetune', 'zeroshot'], help='task name, options:[train, evaluate]')
# parser.add_argument('--zero_shot', type=str, default='True', help='zero-shot evaluation perform')

# data loader
parser.add_argument('--root_path', type=str, default='./', help='root path of the data, code and model files')
parser.add_argument('--data_path', type=str, default='../data', help='path to the data dir')
parser.add_argument('--source_filename', type=str, default='TransferLearningData.csv', help='name of the data file')
parser.add_argument('--non_null_ratio', type=float, default=0.8, help='non null ratio required for considering one window')
parser.add_argument('--config_base', type=str, default='../config', help='path to the config file')
parser.add_argument('--config_name', type=str, default='config.json', help='config file')
parser.add_argument('--horizon_csv_path', type=str, default='./horizon_csv_lstm', help='path to horizon window vs rmse csvs')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly')

# model loader
parser.add_argument('--pretrain_checkpoints_dir', type=str, default='./pretrain_checkpoints/', help='location of model training checkpoints')
parser.add_argument('--finetune_checkpoints_dir', type=str, default='./finetune_checkpoints/', help='location of model finetuning checkpoints')
parser.add_argument('--pretrain_run_name', type=str, default='pretrain_all_lakes_lstm_pretrain', help='run name in wandb')
parser.add_argument('--load_pretrain', type=str, default='True', help='If False will not load pretrained model')
parser.add_argument('--pretrain_ckpt_name', type=str, default='ckpt_latest.pth', help='checkpoints we will use to finetune, options:[ckpt_best.pth, ckpt10.pth, ckpt20.pth...]')
parser.add_argument('--ckpt_name', type=str, default='ckpt_latest', help='name of the checkpoint to be saved for the current task')
parser.add_argument('--ckpt_path', type=str, default='', help='direct path to a checkpoint file (overrides pretrain_checkpoints_dir/pretrain_run_name/pretrain_ckpt_name)')

# pretraining task
parser.add_argument('--lookback_window', type=int, default=21, help='Input window')
parser.add_argument('--horizon_window', type=int, default=14, help='Output window')
parser.add_argument('--label_len', type=int, default=7, help='Input length to the decoder')
parser.add_argument('--horizon_range', nargs='+', type=int, default=[1], help='List of integers')

# model define
parser.add_argument('--model_type', type=str, default='lstm', help='type of model - lstm or transformer')
# lstm 
parser.add_argument('--num_layers', type=int, default=1, help='number of lstm layers')
parser.add_argument('--hidden_feature_size', type=int, default=8, help='size of the hidden layer')
parser.add_argument('--rnn_type', type=str, default="LSTM", help='type of model: LSTM, GRU or RNN')

# transformer
parser.add_argument('--enc_in', type=int, default=8, help='encoder input size') ## this needs to be updated based on the input feature configuration
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--output_attention', default=True, action='store_true', help='whether to output attention in encoder')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--c_out', type=int, default=1, help='output size')

# training 
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--batch_shuffle', type=bool, default=False)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--max_lr', type=float, default=5e-2)
parser.add_argument('--div_factor', type=float, default=100)
parser.add_argument('--pct_start', type=float, default=0.05)
parser.add_argument('--anneal_strategy', type=str, default='cos')
parser.add_argument('--final_div_factor', type=float, default=10000.0)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--dropout', type=float, default=0.001)
parser.add_argument('--eval_freq', type=int, default=1, help='frequency at which we are evaluating the model during training')
parser.add_argument('--early_stop', type=bool, default=False, help='Set to True if we want Early stopping')
parser.add_argument('--early_stop_thres', type=int, default=5, help='If there is no improvement for N epochs we stop the training process')
parser.add_argument('--early_stop_delta', type=float, default=0.5, help='Amount of improvement needed for early stopping criteria')
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--training_prediction', type=str, default='recursive', help='teacher_forcing or recursive or mixed_teacher_forcing')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.0)
parser.add_argument('--dynamic_tf', type=bool, default=False, help='Dynamic Teacher forcing to reduce teacher forcing ratio uniformly every epoch')
def list_of_strings(arg):
    return arg.split(',')
parser.add_argument('--flag_feature', type=list_of_strings, default=[], help='features that are going to be excluded')
parser.add_argument('--flag_noise', type=str, default='', help='options: add_all, replace_all, add_drivers, replace_drivers')
parser.add_argument('--frac_std_noise', type=float, default=0.001, help='')
parser.add_argument('--ntrials', type=int, default=1, help='number of trials we want to run our model for')

# fine-tuning
parser.add_argument('--freeze_enc', type=str, default='false', help='whether to freeze the encoder')
parser.add_argument('--freeze_dec', type=str, default='false', help='whether to freeze the decoder')
parser.add_argument('--freeze_linear', type=str, default='false', help='whether to freeze the linear layer')

# GPU
parser.add_argument('--device', type=str, default='3', help='cuda device')

# weights and biases
parser.add_argument('--project_name', type=str, default='lstm_eval', help='project name in wandb')
parser.add_argument('--run_name', type=str, required=True, default='lstm_train', help='run name in wandb')
parser.add_argument('--save_code', type=str, default='True', help='whether to log code in wandb or not')
parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'], help='wandb logging mode')

parser.add_argument("--target_cols", type=str, nargs='+', default=["poc"], help="Space-separated list of target variable names")
parser.add_argument('--masked_columns', type=str, nargs='+', default=[], help='Space-separated list of features to mask')


args = parser.parse_args()

args.run_name = args.run_name + '_' + args.model_type + '_' + args.task_name
args.pretrain_run_name = args.pretrain_run_name + '_' + args.model_type + '_' + "pretrain"

'''
read config file
'''
config_path = os.path.join(args.config_base, args.config_name)
with open(config_path, 'r') as json_file:
    config = json.load(json_file)

'''
set the cuda device
'''
args.device = 'cuda:' + args.device if torch.cuda.is_available() else 'cpu'

'''
read the file
'''
filepath = os.path.join(args.root_path, args.data_path, args.source_filename)
df = pd.read_csv(filepath)

'''
Define Input Features
'''
all_features_col = [feat for feat in df.columns if feat!='datetime']
input_features_col = [feat for feat in all_features_col if feat not in args.masked_columns]

print("included features: ", input_features_col)
 
df.datetime = df.datetime.astype('datetime64[ns]')
train_df = df.copy(deep=True)

flag_cols = [col for col in train_df.columns if col.startswith('Flag')]

'''
initialize utils object
'''
print("target cols: ", args.target_cols)
utils = Utils(inp_cols=input_features_col,
              target_cols=args.target_cols,
              all_cols=all_features_col,
              date_col=['datetime'],
              args=args,
              stride=1)

'''
Read data path    
'''
data_path = os.path.join(args.root_path, args.data_path)

'''
Train-test-val split and then window each split
'''
train_X, train_lake_names = utils.split_and_window(df, config, data_path, split='train')
val_X, val_lake_names = utils.split_and_window(df, config, data_path, split='val')
test_X, test_lake_names = utils.split_and_window(df, config, data_path, split='test')

'''
normalize the data
'''
train_X = torch.from_numpy(train_X).type(torch.Tensor)
test_X = torch.from_numpy(test_X).type(torch.Tensor)
val_X = torch.from_numpy(val_X).type(torch.Tensor)

train_X = utils.normalize_tensor(train_X, use_stat=False)
test_X = utils.normalize_tensor(test_X, use_stat=True)
val_X = utils.normalize_tensor(val_X, use_stat=True)

'''
extract the input-output
'''
X_train, Y_train = utils.extract_io(train_X, model=args.model_type)
X_test, Y_test = utils.extract_io(test_X, model=args.model_type)
X_val, Y_val = utils.extract_io(val_X, model=args.model_type)

start_seed = 2000
seeds = [start_seed + 20 * i for i in range(args.ntrials)]

train_dicts = []
test_dicts = []
val_dicts = []

base_run_name = args.run_name

if args.task_name=='pretrain':
    
    print(f"|| STARTING PRE-TRAINING ||")

    fix_seed = seeds[0]
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    '''
    update run name
    '''
    run_name = args.run_name + "_{}_{}"
    
    args.run_name = run_name.format(str(datetime.datetime.now().date()), str(datetime.datetime.now().time()))

    '''
    model
    '''
    train_checkpoints_dir = os.path.join(args.pretrain_checkpoints_dir, base_run_name) 
    print(f"ckpt dir = {train_checkpoints_dir}")

    if not os.path.exists(train_checkpoints_dir):
        os.makedirs(train_checkpoints_dir)

    if args.model_type=='transformer':
        model = TFModel(args)
    else:
        model = seq2seq(input_size=len(input_features_col),
                        utils=utils,
                        args=args) 
    '''
    Starting to train
    '''
    trainer = Trainer(vars(args), model, utils)
    train_eval_dict, test_eval_dict, val_eval_dict = trainer.train_model(
                                                          X_train=X_train,
                                                          Y_train=Y_train,
                                                          X_test=X_test,
                                                          Y_test=Y_test,
                                                          args=vars(args),
                                                          X_val=X_val,
                                                          Y_val=Y_val,
                                                          train_lake_names=train_lake_names, 
                                                          )
    # each train_eval_dict correspond to one target
    # train_eval_dict[0].keys()
    # dict_keys(['y_pred', 'y_true', 'rmse', 'mask', 'plot_table', 'plot_gt_values', 'horizon_pred_table', 'horizon_gt_values'])
    train_dicts.append(train_eval_dict) 
    test_dicts.append(test_eval_dict)
    val_dicts.append(val_eval_dict)

    ckpt_name = args.ckpt_name +'.pth'
    model_path = os.path.join(train_checkpoints_dir, ckpt_name)
    print(f"model_path = {model_path}")

    torch.save(model, model_path)

    '''
    plot the mean predictions with the corresponding error bars
    '''
    args.run_name = run_name.format("final", str(datetime.datetime.now().date()), str(datetime.datetime.now().time()))

    trainer.evaluate_uncertainty_multivariate(args=vars(args), eval_dicts=train_dicts, train_or_val='train')
    trainer.evaluate_uncertainty_multivariate(args=vars(args), eval_dicts=val_dicts, train_or_val='val')
    trainer.evaluate_uncertainty_multivariate(args=vars(args), eval_dicts=test_dicts, train_or_val='test')
    wandb.finish()
    print(f"Done with model {args.task_name} ")

    
elif args.task_name=='finetune':
    
    print(f"|| STARTING FINE-TUNING ||")
    
    args.finetune_checkpoints_dir = os.path.join(args.finetune_checkpoints_dir, base_run_name) 
    
    if not os.path.exists(args.finetune_checkpoints_dir):
        os.makedirs(args.finetune_checkpoints_dir)
    
    load_model_path = os.path.join(args.pretrain_checkpoints_dir, args.pretrain_run_name, args.pretrain_ckpt_name)
    print(f"load model from path = {load_model_path}")
    '''
    Load already pre-trained model
    '''
    model = torch.load(load_model_path, weights_only=False)
    
    model.freeze_or_no_freeze(args)
    
    trainer = Trainer(vars(args), model, utils)
    
    run_name = args.run_name + "_trial_{}_{}_{}"
    
    for trial in range(args.ntrials):
    
        '''
        update run name
        '''
        args.run_name = run_name.format(str(trial), str(datetime.datetime.now().date()), str(datetime.datetime.now().time()))

        fix_seed = seeds[trial]

        random.seed(fix_seed)
        torch.manual_seed(fix_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(fix_seed)

        np.random.seed(fix_seed)

        '''
        Starting to train
        '''
        
        train_eval_dict, test_eval_dict, val_eval_dict = trainer.train_model(
                                                              X_train=X_train,
                                                              Y_train=Y_train,
                                                              X_test=X_test,
                                                              Y_test=Y_test,
                                                              args=vars(args),
                                                              X_val=X_val,
                                                              Y_val=Y_val,
                                                              train_lake_names=train_lake_names,
                                                              )

        train_dicts.append(train_eval_dict)
        test_dicts.append(test_eval_dict)
        val_dicts.append(val_eval_dict)

        ckpt_name = args.ckpt_name + '_trial_' + str(trial) + '.pth'

        save_model_path = os.path.join(args.finetune_checkpoints_dir, ckpt_name)

        print(f"model_path = {save_model_path}")

        torch.save(model, save_model_path)

    '''
    plot the mean predictions with the corresponding error bars
    '''
    args.run_name = run_name.format("final", str(datetime.datetime.now().date()), str(datetime.datetime.now().time()))

    config = init_wandb(vars(args), args.task_name)

    trainer.evaluate_uncertainty_multivariate(args=vars(args), eval_dicts=train_dicts, train_or_val='train')
    trainer.evaluate_uncertainty_multivariate(args=vars(args), eval_dicts=val_dicts, train_or_val='val')
    trainer.evaluate_uncertainty_multivariate(args=vars(args), eval_dicts=test_dicts, train_or_val='test')
    wandb.finish()

    print(f"Done with model {args.task_name} ")

elif args.task_name=='zeroshot':
    
    print(f"|| STARTING ZERO-SHOT ||")
    
    if args.ckpt_path:
        load_model_path = args.ckpt_path
    else:
        load_model_path = os.path.join(args.pretrain_checkpoints_dir, args.pretrain_run_name, args.pretrain_ckpt_name)
    
    print(f"load model from path = {load_model_path}")

    '''
    Load already pre-trained or fine-tuned model
    '''
    model = torch.load(load_model_path, weights_only=False)
    
    run_name = args.run_name + "_trial_{}_{}_{}"
    
    trainer = Trainer(vars(args), model, utils)
    
    for trial in range(args.ntrials):
    
        '''
        update run name
        '''
        args.run_name = run_name.format(str(trial), str(datetime.datetime.now().date()), str(datetime.datetime.now().time()))

        fix_seed = seeds[trial]

        random.seed(fix_seed)
        torch.manual_seed(fix_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(fix_seed)

        np.random.seed(fix_seed)

        '''
        Perform zero-shot
        '''
        train_eval_dict, test_eval_dict, val_eval_dict = trainer.perform_zero_shot(
                                                                    X_train=X_train,
                                                                    Y_train=Y_train,
                                                                    X_test=X_test,
                                                                    Y_test=Y_test,
                                                                    X_val=X_val,
                                                                    Y_val=Y_val,
                                                                    )

        train_dicts.append(train_eval_dict)
        test_dicts.append(test_eval_dict)
        val_dicts.append(val_eval_dict)

    '''
    plot the mean predictions with the corresponding error bars
    '''
    args.run_name = run_name.format("final", str(datetime.datetime.now().date()), str(datetime.datetime.now().time()))

    config = init_wandb(vars(args), args.task_name)

    trainer.evaluate_uncertainty_multivariate(args=vars(args), eval_dicts=train_dicts, train_or_val='train')
    trainer.evaluate_uncertainty_multivariate(args=vars(args), eval_dicts=val_dicts, train_or_val='val')
    trainer.evaluate_uncertainty_multivariate(args=vars(args), eval_dicts=test_dicts, train_or_val='test')
    wandb.finish()

    print(f"Done with model {args.task_name} ")
else:
    print("!! Wrong task name - Aborting !!")