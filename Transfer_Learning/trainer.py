import random
import numpy as np
import pandas as pd
import copy
import wandb
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch import optim
from datetime import timedelta, datetime
from tqdm import trange, tqdm


def _to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, features, target, Mx, My):
        'Initialization'
        self.features = features
        self.target = target
        self.mask_x = Mx
        self.mask_y = My

    def __len__(self):
        'Denotes the total number of samples'
        return self.features.__len__()

    def __getitem__(self, index):
        'Generates one sample of data'
        X = self.features[index]
        y = self.target[index]
        mask_y = self.mask_y[index]
        mask_x = self.mask_x[index]

        return X, y, mask_x, mask_y

class EarlyStopping:
    
    def __init__(self, thres=2, min_delta=0):
        
        self.thres = thres
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            
            if self.counter >= self.thres:
                return True
        else:
            self.counter -= 1
            if self.counter < 0:
                self.counter = 0

        return False
    
def init_wandb(args, task_name):
    wandb_mode = args.get('wandb_mode', 'online')
    save_code = _to_bool(args.get('save_code', True))
    wandb.init(project=args['project_name'], 
               name="_".join([task_name, args['run_name']]), 
               config=args, 
               save_code=save_code,
               mode=wandb_mode)
    config = wandb.config
    return config

class Trainer():
    
    def __init__(self, args, model, utils):
        
        self.model = model
        self.task_name = args['task_name']
        self.training_prediction = args['training_prediction']
        self.teacher_forcing_ratio = args['teacher_forcing_ratio']
        self.learning_rate = args['learning_rate']
        self.target_len = args['horizon_window']
        self.label_len = args['label_len']
        self.dynamic_tf = args['dynamic_tf']
        self.batch_size = args['batch_size']
        self.batch_shuffle = args['batch_shuffle']
        self.output_size = args['c_out']
        self.device = args['device']
        self.args = args
        self.utils = utils
        
        self.model = self.model.to(self.device)
        
        # use for normalization of target variables
        self.std_targets = self.utils.feat_std[:, :, self.utils.targets_index].to(self.device) 
        self.mean_targets = self.utils.feat_mean[:, :, self.utils.targets_index].to(self.device)
        # self.std_targets = self.utils.feat_std[:, :, self.utils.targets_sub_index].to(self.device) 
        # self.mean_targets = self.utils.feat_mean[:, :, self.utils.targets_sub_index].to(self.device)  
    
        self.alpha = (-self.mean_targets/self.std_targets)
        
    def compute_loss(self, target, pred, mask):
        loss = (pred - target) ** 2
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def convert_2d_to_1d_mask(self, mask, threshold=0.8):
        
        # Calculate the percentage of True values (masked features) along each timestep
        percentage_masked = mask.float().mean(dim=1)

        # Compare the percentage to the threshold to determine if the entire timestep should be masked
        timestep_mask = percentage_masked >= threshold
        
        return timestep_mask

    def train_model(self,
                    X_train, 
                    Y_train, 
                    X_test, 
                    Y_test,
                    train_lake_names,
                    args,
                    X_val=None,
                    Y_val=None,
                    val_lake_names=None):
        if args['model_type']=='transformer':
            return self.train_tf_model(X_train, 
                                       Y_train, 
                                       X_test, 
                                       Y_test,
                                       args,
                                       X_val,
                                       Y_val,
                                       train_lake_names,
                                       val_lake_names)
        else:
            return self.train_lstm_model(X_train, 
                                         Y_train, 
                                         X_test, 
                                         Y_test,
                                         args,
                                         X_val,
                                         Y_val,
                                         )

    def train_lstm_model(self,
                        X_train, 
                        Y_train, 
                        X_test, 
                        Y_test,
                        args,
                        X_val=None,
                        Y_val=None,
                        train_lake_names=None,
                        val_lake_names=None):

        '''
        train lstm encoder-decoder

        : param X_train:              input data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param Y_train:             target data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param n_epochs:                  number of epochs
        : param target_len:                number of values to predict. Time horizon
        : param batch_size:                number of samples per gradient update
        : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
        :                                  'mixed_teacher_forcing'); default is 'recursive'
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
        :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
        :                                  teacher forcing.
        : param learning_rate:             float >= 0; learning rate
        : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
        :                                  reduces the amount of teacher forcing for each epoch
        : return losses:                   array of loss function for each epoch
        '''
        
        project_name = args['project_name']
        run_name = args['run_name']
        self.task_name = args['task_name']
        self.learning_rate = args['learning_rate']
        self.dynamic_tf = args['dynamic_tf']
        self.batch_size = args['batch_size']
        self.batch_shuffle = args['batch_shuffle']
        self.output_size = args['c_out']
        
        config = init_wandb(args, self.task_name)
        
        n_epochs = config.max_epochs
        
        # initialize array of losses
        losses = np.full(n_epochs, np.nan)
        val_rmse = []
        train_rmse = []
        test_rmse = []
        
        # n_batches = int(math.ceil(X_train.shape[0] / config.batch_size))
        early_stop = config.early_stop
        early_stopper = EarlyStopping(thres=config.early_stop_thres, min_delta=config.early_stop_delta)
        
        params = {
                  'batch_size': config.batch_size,
                  'shuffle': config.batch_shuffle
                }
        
        X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)
        if X_val is not None:
            X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)
        X_test, Y_test = X_test.to(self.device), Y_test.to(self.device)
        
        '''
        Training generator
        '''
        M_x = 1 - (1 * (torch.isnan(X_train)))
        M_x = M_x.float().to(self.device)
        
        M_y = 1 - (1 * (torch.isnan(Y_train)))
        M_y = M_y.float().to(self.device)
        
        X_train = torch.nan_to_num(X_train)
        Y_train = torch.nan_to_num(Y_train)
        
        
        training_set = Dataset(X_train, Y_train, M_x, M_y)
        training_generator = torch.utils.data.DataLoader(training_set, **params)
        
        '''
        Validation generator
        '''
        if X_val is not None:
            M_x_val = 1 - (1 * (torch.isnan(X_val)))
            M_x_val = M_x_val.float().to(self.device)

            M_y_val = 1 - (1 * (torch.isnan(Y_val)))
            M_y_val = M_y_val.float().to(self.device)

            X_val = torch.nan_to_num(X_val)
            Y_val = torch.nan_to_num(Y_val)

            validation_set = Dataset(X_val, Y_val, M_x_val, M_y_val)
            validation_generator = torch.utils.data.DataLoader(validation_set, **params)

        '''
        Test generator
        '''
        M_x_test = 1 - (1 * (torch.isnan(X_test)))
        M_x_test = M_x_test.float().to(self.device)
        
        M_y_test = 1 - (1 * (torch.isnan(Y_test)))
        M_y_test = M_y_test.float().to(self.device)
        
        X_test = torch.nan_to_num(X_test)
        Y_test = torch.nan_to_num(Y_test)
        
        testing_set = Dataset(X_test, Y_test, M_x_test, M_y_test)
        testing_generator = torch.utils.data.DataLoader(testing_set, **params)
        
        # ----------------------------------------
        n_batches = len(training_generator)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=config.weight_decay)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr=config.max_lr, 
                                                        epochs=n_epochs, 
                                                        div_factor=config.div_factor, 
                                                        pct_start=config.pct_start, 
                                                        anneal_strategy=config.anneal_strategy, 
                                                        final_div_factor=config.final_div_factor,
                                                        steps_per_epoch=n_batches, 
                                                        )
        
        
        wandb.watch(self.model.encoder)
        
        self.model.to(self.device)
        
        torch.autograd.set_detect_anomaly(True)
        
        # ----------------------------------------
        with trange(n_epochs) as tr:
            for it in tr:

                batch_loss = 0.
                
                self.model.train()
                
                for input_batch, target_batch, maskX, maskY in tqdm(training_generator):
                    # if all values are missing, skip it
                    if maskY.eq(0).all() or self.utils.less_data(maskX):
                        continue
                    
                    # zero the gradient
                    optimizer.zero_grad()
                    
                    outputs = self.model(input_batch=input_batch, alpha=self.alpha, target_batch=target_batch)

                    # compute the loss
                    loss = self.compute_loss(target_batch, outputs, maskY)
                    # loss = criterion(outputs, target_batch)
                    # print(f"loss item = {loss.item()}")
                    batch_loss += loss.item()

                    # backpropagation
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # loss for epoch
                batch_loss /= n_batches
                losses[it] = batch_loss

                # dynamic teacher forcing
                if self.dynamic_tf and self.teacher_forcing_ratio > 0:
                    self.teacher_forcing_ratio = self.teacher_forcing_ratio - 0.002

                if it % config.eval_freq == 0:
                    with torch.no_grad():
                        self.model.eval()
                       
                        if X_val is not None:
                            val_eval_dict = self.evaluate_batch(test_generator=validation_generator)
                            batch_val_loss = val_eval_dict["rmse"].item()
                            val_rmse.append(batch_val_loss)
                        
                        train_eval_dict = self.evaluate_batch(test_generator=training_generator)
                        test_eval_dict = self.evaluate_batch(test_generator=testing_generator)
                        
                        batch_train_loss = train_eval_dict["rmse"].item()
                        batch_test_loss = test_eval_dict["rmse"].item()
                        
                        train_rmse.append(batch_train_loss)
                        test_rmse.append(batch_test_loss)
                    # if early_stop and early_stopper.early_stop(batch_val_loss):
                    #     print("Early stopping")
                    #     break
                # progress bar
                if X_val is not None:
                    metrics = {
                        "loss":batch_loss,
                        "val_rmse":batch_val_loss,
                        "train_rmse":batch_train_loss,
                        "test_rmse":batch_test_loss
                        }
                else:
                    metrics = {
                        "loss":batch_loss,
                        "train_rmse":batch_train_loss,
                        "test_rmse":batch_test_loss
                        }

                tr.set_postfix(metrics)
                wandb.log(metrics)
        
        val_eval_dicts = None
        with torch.no_grad():
            if X_val is not None:
                val_eval_dicts = self.evaluate_and_plot_multivariate(test_generator=validation_generator, split='val')

            train_eval_dicts = self.evaluate_and_plot_multivariate(test_generator=training_generator, split='train')

            test_eval_dicts = self.evaluate_and_plot_multivariate(test_generator=testing_generator, split='test')

        return train_eval_dicts, test_eval_dicts, val_eval_dicts

    def train_tf_model(self,
                    X_train, 
                    Y_train, 
                    X_test, 
                    Y_test,
                    args,
                    X_val=None,
                    Y_val=None,
                    train_lake_names=None,
                    val_lake_names=None):

        '''
        train lstm encoder-decoder

        : param X_train:              input data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param Y_train:             target data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param n_epochs:                  number of epochs
        : param target_len:                number of values to predict. Time horizon
        : param batch_size:                number of samples per gradient update
        : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
        :                                  'mixed_teacher_forcing'); default is 'recursive'
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
        :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
        :                                  teacher forcing.
        : param learning_rate:             float >= 0; learning rate
        : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
        :                                  reduces the amount of teacher forcing for each epoch
        : return losses:                   array of loss function for each epoch
        '''

        project_name = args['project_name']
        run_name = args['run_name']
        
        config = init_wandb(args, self.task_name)

        n_epochs = config.max_epochs

        # initialize array of losses
        losses = np.full(n_epochs, np.nan)
        val_rmse = []
        train_rmse = []
        test_rmse = []

        # n_batches = int(math.ceil(X_train.shape[0] / config.batch_size))
        early_stop = config.early_stop
        early_stopper = EarlyStopping(thres=config.early_stop_thres, min_delta=config.early_stop_delta)

        params = {
                  'batch_size': config.batch_size,
                  'shuffle': config.batch_shuffle
                }

        X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)
        if X_val is not None:
            X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)
        X_test, Y_test = X_test.to(self.device), Y_test.to(self.device)

        '''
        Training generator
        '''
        M_x = 1 - (1 * (torch.isnan(X_train)))
        M_x = M_x.float().to(self.device)

        M_y = 1 - (1 * (torch.isnan(Y_train)))
        M_y = M_y.float().to(self.device)

        X_train = torch.nan_to_num(X_train)
        Y_train = torch.nan_to_num(Y_train)

        training_set = Dataset(X_train, Y_train, M_x, M_y)
        training_generator = torch.utils.data.DataLoader(training_set, **params)

        '''
        Validation generator
        '''
        if X_val is not None:
            M_x_val = 1 - (1 * (torch.isnan(X_val)))
            M_x_val = M_x_val.float().to(self.device)

            M_y_val = 1 - (1 * (torch.isnan(Y_val)))
            M_y_val = M_y_val.float().to(self.device)

            X_val = torch.nan_to_num(X_val)
            Y_val = torch.nan_to_num(Y_val)

            validation_set = Dataset(X_val, Y_val, M_x_val, M_y_val)
            validation_generator = torch.utils.data.DataLoader(validation_set, **params)

        '''
        Test generator
        '''
        M_x_test = 1 - (1 * (torch.isnan(X_test)))
        M_x_test = M_x_test.float().to(self.device)

        M_y_test = 1 - (1 * (torch.isnan(Y_test)))
        M_y_test = M_y_test.float().to(self.device)

        X_test = torch.nan_to_num(X_test)
        Y_test = torch.nan_to_num(Y_test)

        testing_set = Dataset(X_test, Y_test, M_x_test, M_y_test)
        testing_generator = torch.utils.data.DataLoader(testing_set, **params)

        n_batches = len(training_generator)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=config.weight_decay)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.max_epochs)

        self.model.to(self.device)

        torch.autograd.set_detect_anomaly(True)


        with trange(n_epochs) as tr:
            for it in tr:

                batch_loss = 0.
                batch_loss_tf = 0.
                batch_loss_no_tf = 0.
                num_tf = 0
                num_no_tf = 0

                self.model.train()

                for input_batch, target_batch, maskX, maskY in tqdm(training_generator):
                    
                    maskY = maskY[:, -self.target_len:, :]
                    
                    # if all values are missing, skip it
                    if maskY.eq(0).all() or self.utils.less_data(maskX):
                        continue

                    # zero the gradient
                    optimizer.zero_grad()
                    
                    dec_inp = torch.zeros_like(target_batch[:, -self.target_len:, :]).float()
                    dec_inp = torch.cat([target_batch[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    maskX_1d = self.convert_2d_to_1d_mask(maskX)
                    
                    outputs, attns = self.model(x_enc=input_batch, x_dec=dec_inp, alpha=self.alpha, enc_self_mask=maskX_1d)
                    # print(f"outputs shape = {outputs.shape}")

                    # compute the loss
                    loss = self.compute_loss(target_batch[:, -self.target_len:, :], outputs, maskY)
                    batch_loss += loss.item()

                    # backpropagation
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # loss for epoch
                batch_loss /= n_batches
                losses[it] = batch_loss

                # dynamic teacher forcing
                if self.dynamic_tf and self.teacher_forcing_ratio > 0:
                    self.teacher_forcing_ratio = self.teacher_forcing_ratio - 0.002

                if it % config.eval_freq == 0:
                    with torch.no_grad():
                        self.model.eval()
                        # print(f"ckpt 1 \n")
                        if X_val is not None:
                            val_eval_dict = self.evaluate_batch(test_generator=validation_generator)
                            batch_val_loss = val_eval_dict["rmse"].item()
                            val_rmse.append(batch_val_loss)
                        # print(f"ckpt 2 \n")
                        train_eval_dict = self.evaluate_batch(test_generator=training_generator)
                        test_eval_dict = self.evaluate_batch(test_generator=testing_generator)

                        batch_train_loss = train_eval_dict["rmse"].item()
                        batch_test_loss = test_eval_dict["rmse"].item()

                        train_rmse.append(batch_train_loss)
                        test_rmse.append(batch_test_loss)

                # progress bar
                if X_val is not None:
                    metrics = {
                        "loss":batch_loss,
                        "val_rmse":batch_val_loss,
                        "train_rmse":batch_train_loss
                        }
                else:
                    metrics = {
                        "loss":batch_loss,
                        "train_rmse":batch_train_loss,
                        "test_rmse":batch_test_loss
                        }
                tr.set_postfix(metrics)
                wandb.log(metrics)

        val_eval_dicts = None
        with torch.no_grad():
            if X_val is not None:
                val_eval_dicts = self.evaluate_and_plot_multivariate(test_generator=validation_generator, split='val')

            train_eval_dicts = self.evaluate_and_plot_multivariate(test_generator=training_generator, split='train')
            test_eval_dicts = self.evaluate_and_plot_multivariate(test_generator=testing_generator, split='test')

        return train_eval_dicts, test_eval_dicts, val_eval_dicts

    def predict_batch_lstm(self, testing_generator, target_len):
        '''
        : param input_tensor:      input data (batch, seq_len, input_size); PyTorch tensor
        : param target_len:        number of target values to predict (30)
        : return np_outputs:       np.array containing predicted values; prediction done recursively
        '''
        
        eval_outputs = []
        eval_masks = []
        target_samples = []
        
        for input_batch, target_batch, maskX, maskY in tqdm(testing_generator):
            if input_batch.device!=self.device or target_batch.device!=self.device:
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

            with torch.cuda.amp.autocast():
                
                if not maskY.eq(0).all():
                    outputs = self.model(input_batch=input_batch, alpha=self.alpha)
                else:
                    outputs = torch.zeros_like(maskY)
                
                eval_outputs.append(outputs)
                eval_masks.append(maskY)
                target_samples.append(target_batch)
                
        # np_outputs = outputs.detach()
        eval_outputs = torch.cat(eval_outputs, dim=0)
        eval_masks = torch.cat(eval_masks, dim=0)
        target_samples = torch.cat(target_samples, dim=0)
        
        return eval_outputs, eval_masks, target_samples
    
    def predict_batch_tf(self, testing_generator, target_len):
        '''
        : param input_tensor:      input data (batch, seq_len, input_size); PyTorch tensor
        : param target_len:        number of target values to predict (30)
        : return np_outputs:       np.array containing predicted values; prediction done recursively
        '''

        eval_outputs = []
        eval_masks = []
        target_samples = []

        for input_batch, target_batch, maskX, maskY in tqdm(testing_generator):

            if input_batch.device!=self.device or target_batch.device!=self.device:
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

            dec_inp = torch.zeros_like(target_batch[:, -self.target_len:, :]).float()
            dec_inp = torch.cat([target_batch[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
            
            maskY = maskY[:, -self.target_len:, :]
            
            with torch.cuda.amp.autocast():

                if not maskY.eq(0).all():
                    outputs, attns = self.model(x_enc=input_batch, x_dec=dec_inp, alpha=self.alpha)
                else:
                    outputs = torch.zeros_like(maskY)

                eval_outputs.append(outputs)
                eval_masks.append(maskY)
                target_samples.append(target_batch[:, -self.target_len:, :])

        # np_outputs = outputs.detach()
        eval_outputs = torch.cat(eval_outputs, dim=0)
        eval_masks = torch.cat(eval_masks, dim=0)
        target_samples = torch.cat(target_samples, dim=0)

        return eval_outputs, eval_masks, target_samples


    def evaluate_batch(self, test_generator, unnorm=True):
        
        if self.args['model_type']=='transformer':
            y_pred, y_masks, Y_test = self.predict_batch_tf(test_generator, self.utils.output_window)
        else:
            y_pred, y_masks, Y_test = self.predict_batch_lstm(test_generator, self.utils.output_window)
        
        if unnorm:
            # unnormalize the data
            y_pred = y_pred*self.std_targets + self.mean_targets
            Y_test = Y_test*self.std_targets + self.mean_targets

        sqred_err = (y_pred-Y_test)**2 #.mean())**0.5
        rmse = ((sqred_err * y_masks).sum() / y_masks.sum())**0.5
        evaluate_dict = {
            "y_pred":y_pred,
            "y_true":Y_test,
            "rmse":rmse,
            "mask": y_masks
        }
        return evaluate_dict

    def evaluate_and_plot_multivariate(self, test_generator, split):
        """
        Wrapper to run evaluate_and_plot() on each target independently.
        Assumes model predicts multiple target variables.

        Returns:
            eval_dicts: List of eval_dicts (one per target)
        """

        full_eval = self.evaluate_batch(test_generator=test_generator)
        eval_dicts = []
        
        num_targets = len(self.utils.targets_index)
        for i in range(num_targets):
            sliced_eval_dict = {
            k: (v[:, :, i:i+1] if isinstance(v, torch.Tensor) and v.ndim == 3 else v)
            for k, v in full_eval.items()
            }

            eval_dict_i = self.evaluate_and_plot(sliced_eval_dict, split)
            # breakpoint()
            eval_dicts.append(eval_dict_i)
        
        return eval_dicts

    def evaluate_and_plot(self, eval_dict, split):

        masks = eval_dict['mask']
        gt_ = eval_dict['y_true']
        pred = eval_dict['y_pred'] # model predicted values

        pred_plot = copy.deepcopy(pred)
        gt_plot = copy.deepcopy(gt_)

        for i in range(masks.shape[0]):
            pred_plot[i][masks[i, :, 0]==0] = torch.nan
            gt_plot[i][masks[i, :, 0]==0] = torch.nan

        mask_sum = masks.sum(dim=(1, 2))

        gt_df = pd.DataFrame(gt_plot.cpu().numpy()[:,:,0])
        gt_values = np.append(gt_df[0].values, gt_df.iloc[-1,1:]) # ground-truth values

        T_pred_table, plot_df, plot_gt_values = self.utils.predictionTable(pred_df=pred_plot, gt_values=gt_values, split=split)
        eval_dict['plot_table'] = plot_df
        eval_dict['plot_gt_values'] = plot_gt_values

        '''
        compute rmse
        '''
        gt_df = pd.DataFrame(gt_.cpu().numpy()[:,:,0])
        gt_values = np.append(gt_df[0].values, gt_df.iloc[-1,1:]) # ground-truth values

        predtable, _, _ = self.utils.predictionTable(pred_df=pred, gt_values=gt_values, split=split)
        # breakpoint()
        eval_dict['horizon_pred_table'] = predtable
        eval_dict['horizon_gt_values'] = gt_values
        
        # rmse_values = self.utils.compute_horizon_rmse(T_pred_table=predtable, gt_values=gt_values, train_or_val=split)
        return eval_dict
    
    def perform_zero_shot(self, X_train, Y_train, X_test, Y_test, X_val=None, Y_val=None):
        
        self.model.to(self.device)
        config = init_wandb(self.args, self.task_name)
        
        params = {
                  'batch_size': self.batch_size,
                  'shuffle': self.batch_shuffle
                }
        
        M_x = 1 - (1 * (torch.isnan(X_train)))
        M_x = M_x.float().to(self.device)

        M_y = 1 - (1 * (torch.isnan(Y_train)))
        M_y = M_y.float().to(self.device)

        X_train = torch.nan_to_num(X_train)
        Y_train = torch.nan_to_num(Y_train)

        training_set = Dataset(X_train, Y_train, M_x, M_y)
        training_generator = torch.utils.data.DataLoader(training_set, **params)

        '''
        Validation generator
        '''
        if X_val is not None:
            M_x_val = 1 - (1 * (torch.isnan(X_val)))
            M_x_val = M_x_val.float().to(self.device)

            M_y_val = 1 - (1 * (torch.isnan(Y_val)))
            M_y_val = M_y_val.float().to(self.device)

            X_val = torch.nan_to_num(X_val)
            Y_val = torch.nan_to_num(Y_val)

            validation_set = Dataset(X_val, Y_val, M_x_val, M_y_val)
            validation_generator = torch.utils.data.DataLoader(validation_set, **params)

        '''
        Test generator
        '''
        M_x_test = 1 - (1 * (torch.isnan(X_test)))
        M_x_test = M_x_test.float().to(self.device)

        M_y_test = 1 - (1 * (torch.isnan(Y_test)))
        M_y_test = M_y_test.float().to(self.device)

        X_test = torch.nan_to_num(X_test)
        Y_test = torch.nan_to_num(Y_test)

        testing_set = Dataset(X_test, Y_test, M_x_test, M_y_test)
        testing_generator = torch.utils.data.DataLoader(testing_set, **params)
        
        val_eval_dicts = None
        with torch.no_grad():

            if X_val is not None:
                val_eval_dicts = self.evaluate_and_plot_multivariate(test_generator=validation_generator, split='val')

            train_eval_dicts = self.evaluate_and_plot_multivariate(test_generator=training_generator, split='train')
            test_eval_dicts = self.evaluate_and_plot_multivariate(test_generator=testing_generator, split='test')

        return train_eval_dicts, test_eval_dicts, val_eval_dicts
        
    def evaluate_uncertainty_multivariate(self, args, eval_dicts, train_or_val):
        """
        eval_dicts: list of lists → [trial][feature]
        Restructure into [feature][trial], then evaluate each.
        """

        n_trials = len(eval_dicts)
        n_features = len(eval_dicts[0])  # assuming each trial has the same # features

        # Restructure to [feature][trial]
        eval_dicts_by_feature = [[] for _ in range(n_features)]
        for trial_idx in range(n_trials):
            for feat_idx in range(n_features):
                eval_dicts_by_feature[feat_idx].append(eval_dicts[trial_idx][feat_idx])

        all_rmses = [] # each element corresponds to a feature
        # Evaluate uncertainty for each feature
        for i, eval_dicts_for_feature in enumerate(eval_dicts_by_feature): #for loop over features
            feature = self.utils.target_cols[i]
            feature_rmses = self.evaluate_uncertainty(args, eval_dicts_for_feature, f"{train_or_val}", feature) # (mean,std) over trials for each H
            # breakpoint()
            wandb.summary[train_or_val + "_" + feature + '_rmse'] = feature_rmses.RMSE.values.mean() 
            wandb.summary[train_or_val + "_" + feature + '_rmse_std'] = feature_rmses.STD.values.mean() 
            all_rmses.append(feature_rmses)

        # breakpoint()
        all_rmses = np.concatenate(all_rmses)
        wandb.summary[f"{train_or_val}_final_rmse"] = all_rmses.mean()
        wandb.summary[f"{train_or_val}_final_rmse_std"] = all_rmses.std()

    def evaluate_uncertainty(self, args, eval_dict, train_or_val, feature_id=0):
        rmse_values=self.utils.compute_horizon_rmse(eval_dict, train_or_val, feature_id)
        err_std = rmse_values.STD.values
        # breakpoint()
        self.utils.plotTable(eval_dict, train_or_val, err_std, feature_id)

        # rmses = []
        # for trial in range(len(eval_dict)):
        #     rmses.append(eval_dict[trial]['rmse'].item())
        # rmses = np.array(rmses)
        return rmse_values