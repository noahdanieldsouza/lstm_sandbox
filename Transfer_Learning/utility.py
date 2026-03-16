import numpy as np
import torch, os
import pandas as pd
import math
import argparse
import wandb
import matplotlib.pyplot as plt
import pickle

from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import datetime

class Utils:
    
    def __init__(self, 
                 inp_cols, 
                 target_cols,
                 all_cols, 
                 date_col,
                 args,
                 stride=1):
        
        self.inp_cols = inp_cols
        self.target_cols = target_cols
        self.date_col = date_col
        self.input_window = args.lookback_window
        self.output_window = args.horizon_window
        self.label_len = args.label_len
        self.num_out_features = args.c_out
        self.non_null_ratio = args.non_null_ratio
        self.device = args.device
        self.task_name = args.task_name
        self.stride = stride
        self.horizon_range = args.horizon_range
        self.horizon_csv_path = args.horizon_csv_path
        self.window = self.input_window + self.output_window
        self.lake = args.config_name[:-5]
        self.y_mean = None
        self.y_std = None
        self.args= args
        self.windowed_dataset_path = ""
        self.predictions_dir = args.run_name
        self.all_io_cols = all_cols
        # if all(tc in self.inp_cols for tc in self.target_cols):
        #     self.all_io_cols = self.inp_cols
        # else:
        #     self.all_io_cols = self.inp_cols + [tc for tc in self.target_cols if tc not in self.inp_cols]


        self.targets_sub_index = [self.target_cols.index(tc) for tc in self.target_cols]
        
        self.targets_index = [self.all_io_cols.index(tc) for tc in self.target_cols]
        
        self.feat_mean = None
        self.feat_std = None

        # self.chloro_sub_index = self.all_io_cols.index(self.target_cols) #index within the all_io_cols
        # self.chloro_index = 0 #index within the original set of columns
    
    def load_pickle(self, lake_path):
        
        lake_arr = None
        # lake_path = os.path.join(path, lake) + '_' + str(self.window) + '.pkl'
        
        with open(lake_path, 'rb') as pickle_file:
            lake_arr = pickle.load(pickle_file)
        
        return lake_arr
            
    def normalize_tensor(self, tensor, use_stat=False):
        eps = 1e-5 # epsilon for zero std
        if not use_stat: # for train split
            self.feat_mean = tensor.nanmean(dim=(0, 1))[None, None, :]
            mask = torch.isnan(tensor)
            filtered_data = tensor.clone()
            filtered_data[mask] = 0
            
            rev_mask = 1-(mask*1)
            
            sqred_values = rev_mask*((filtered_data - self.feat_mean)**2)
            sqred_sum = sqred_values.sum(dim=(0, 1))
            variance = sqred_sum/torch.sum(rev_mask, dim=(0, 1))
            
            self.feat_std = torch.sqrt(variance)[None, None, :]            
            # self.feat_mean = tensor.mean(dim=(0, 1))[None, None, :]
            # self.feat_std = tensor.std(dim=(0, 1))[None, None, :]

        tensor = (tensor-self.feat_mean)/(self.feat_std+eps)
        return tensor
    
    def extract_io(self, data, model='transformer'):
        """
        Extract input (X) and output (Y) from windowed data.
        
        data shape: [N, lookback+horizon, len(all_io_cols)]
        X will use only inp_cols (excluding masked columns)
        Y will use only target_cols
        """
        enc_end_index = self.input_window # context length
        inp_cols_index = [self.all_io_cols.index(col) for col in self.inp_cols]
        # Extract input features using inp_cols_index (maps inp_cols to positions in all_io_cols)
        X = data[:, :enc_end_index, inp_cols_index]

        # Extract target features using targets_index (maps target_cols to positions in all_io_cols)
        Y = data[:, enc_end_index:, self.targets_index]
        return X, Y
    
    def apply_noise(self, df, mean=0):
        for col in self.inp_cols:
            if col != 'Chla_ugL':
                if self.args.flag_noise == 'add':
                    df[col] += np.random.normal(mean, df[col].std(), df[col].shape)
            # if col == 'Chla_ugL':
            #     if self.args.flag_noise == 'add':
            #         df[col] += np.random.lognormal(mean, df[col].std()* self.args.frac_std_noise, df[col].shape)
            #         df[col] = df[col].clip(lower=0)
        return df
        
    def split_and_window(self, df, config, windowed_dataset_path, split, include_target=True):
        
        train_lake_names = []
        
        train_X = []
        
        train_fractions = config['train_fractions']
        val_fractions = config['val_fractions']
        test_fractions = config['test_fractions']
        
        for ind, lake in enumerate(config[split+'_lakes']):
            '''
            Assuming, we have the same set of val_lakes and test_lakes
            '''
            print(f"Splitting for {split} and for lake = {lake}")
            
            lakename, dtype = lake.split('_')

            df_temp = df.copy(deep=True)
            num_samples = df_temp.shape[0]
            
            train_frac = train_fractions[ind]
            val_frac = val_fractions[ind]
            test_frac = test_fractions[ind]
            
          
            lakefile = lakename + "_" + dtype + '_' + str(self.window) + '_' + split +'_.pkl'
            
            lake_path = os.path.join(windowed_dataset_path, lakefile)
            
            train_end_index = int(train_frac*num_samples)
            val_end_index = train_end_index + int(val_frac*num_samples)
            test_end_index = val_end_index + int(test_frac*num_samples)

            if split=='train':
                train_df = df_temp.iloc[:train_end_index, :].reset_index(drop='true')
                # print("original data: ", train_df.loc[100:200, self.inp_cols])
                if self.args.flag_noise != '':
                    train_df = self.apply_noise(train_df)
                # print("noisy data: ", train_df.loc[100:200, self.inp_cols])
                self.train_size = train_df.shape[0]
                self.train_dates = train_df.datetime.values
                
                
            elif split=='val':
                train_df = df_temp.iloc[train_end_index:val_end_index, :].reset_index(drop='true')
                self.val_size = train_df.shape[0]
                self.val_dates = train_df.datetime.values
            else:
                train_df = df_temp.iloc[val_end_index:test_end_index, :].reset_index(drop='true')
                self.test_size = train_df.shape[0]
                self.test_dates = train_df.datetime.values
                    
            if os.path.exists(lake_path):
                print(f"Windowed dataset already exists for lake {lakename}, skipping windowing ...")
                lake_arr = self.load_pickle(lake_path)
                train_X.append(lake_arr)
                train_lake_names += [lake]*lake_arr.shape[0]
                
            else:
                
                lake_x = self.windowed_dataset(train_df)
                print(f"Lake {lake} has got shape = {lake_x.shape}")
                
                if lake_x.shape[0]==0:
                    continue
                
                # with open(lake_path, 'wb') as pickle_file:
                #     pickle.dump(lake_x, pickle_file)
                #     print(f"Pickled lake {lake}")
                
                train_X.append(lake_x)
                train_lake_names += [lake]*lake_x.shape[0]
        
        return np.concatenate(train_X, axis=0), train_lake_names
        
        
    def windowed_dataset(self, df, include_target=True):
        '''
        create a windowed dataset
    
        : param y:                time series feature (array)
        : param input_window:     number of y samples to give model
        : param output_window:    number of future y samples to predict
        : param stide:            spacing between windows
        : param num_features:     number of features (i.e., 1 for us, but we could have multiple features)
        : return X, Y:            arrays with correct dimensions for LSTM
        :                         (i.e., [input/output window size # examples, # features])
        '''

        L = df.shape[0]
        num_samples = (L - self.window) // self.stride + 1
        
        X = np.array([]) 
        
        for ii in tqdm(np.arange(num_samples)):
            start_x = self.stride * ii
            end_x = start_x + self.window
            
            subset_dfX = df.iloc[start_x:end_x, :].copy(deep=True).reset_index(drop='true')
            
            if X.shape[0]==0:
                X = np.expand_dims(subset_dfX.loc[:, self.all_io_cols], axis=0)
            else:
                toAdd = np.expand_dims(subset_dfX.loc[:, self.all_io_cols], axis=0)
                X = np.append(X, toAdd, axis=0)
                                 
        return X

    def less_data(self, mask):
        for i in range(mask.shape[0]):
            window = mask[i]
            total_ = window.shape[0]*window.shape[1]
            if window.sum()/total_ < self.non_null_ratio:
                return True
        return False
        
    def numpy_to_torch(self, Xtrain, Ytrain, Xtest, Ytest):
        '''
        convert numpy array to PyTorch tensor
    
        : param Xtrain:               windowed training input data (# examples, input window size, # features); np.array
        : param Ytrain:               windowed training target data (# examples, output window size, # features); np.array
        : param Xtest:                windowed test input data (# examples, input window size, # features); np.array
        : param Ytest:                windowed test target data (# examples, output window size, # features); np.array
        : return X_train_torch, Y_train_torch,
        :        X_test_torch, Y_test_torch:      all input np.arrays converted to PyTorch tensors

        '''

        X_train_torch = torch.from_numpy(Xtrain).type(torch.Tensor)
        Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)

        X_test_torch = torch.from_numpy(Xtest).type(torch.Tensor)
        Y_test_torch = torch.from_numpy(Ytest).type(torch.Tensor)

        return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch

    @staticmethod
    def plot_train_test_rmse(train_rmse, test_rmse, title="train_test_rmse"):
        plt.figure(figsize=(5,4), dpi=150)
        plt.plot(train_rmse, lw=2.0, label='train_rmse')
        plt.plot(test_rmse, lw=2.0, label='test_rmse')
        plt.yscale("log")
        plt.grid("on", alpha=0.2)
        plt.legend()
        wandb.log({title: wandb.Image(plt)})
        plt.close()
        
    def pred_per_step_helper(self, predictions, idx, pred_values, date):
        '''
        Compute all the predictions for a single date. i.e. as T+1, T+2, T+3, ... T+horizon timestep prediction
        '''
        c = 0
        rind = idx
        while c<self.output_window:
            rind = rind + c
            pred_values[date].append(predictions[rind].reshape(-1)[-(c+1)])
            c+=1
    
        return pred_values

    def prediction_per_step(self, df, predictions, gts, ids):
        pred_values = {}
        gts_ls = {}
        for idx in ids:
            date = df.loc[idx+self.input_window+self.output_window-1,self.date_col]
            pred_values[date] = []
            gts_ls['GT_'+date] = gts[idx].reshape(-1)[-1]
            pred_values = self.pred_per_step_helper(predictions, idx, pred_values, date)
    
        pred_values = {k:list(reversed(v)) for k,v in pred_values.items()}
        for k,v in pred_values.items():
            pred_values[k] = [i.cpu().numpy() for i in v]
            
        return pred_values, gts_ls
    
    def fillpredtable(self, r, table, pred):
        for i,k in enumerate(table.columns):
            if i-(r-1)>=0 and i-(r-1) < pred.shape[0]:
                table.loc[r, k] = pred[i-(r-1)][r-1].cpu().numpy()
        return table


    def predictionTable(self, pred_df, split, gt_values=None, plot=True):
            '''
            Create the prediction table
            '''
            if split=='train':
                size = self.train_size
                dates = self.train_dates[self.input_window:]
            elif split=='val':
                size = self.val_size
                dates = self.val_dates[self.input_window:]
            else:
                size = self.test_size
                dates = self.test_dates[self.input_window:]
            
            # print(f"dates = {dates[300:450]}")
            pred_table = np.zeros((self.output_window, size - self.input_window))
            pred_table = pd.DataFrame(pred_table)
            pred_table.columns = dates
            pred_table.index = range(1,self.output_window+1)
            pred_table.loc[:] = np.nan
            
            for r in range(1, self.output_window+1):
                pred_table = self.fillpredtable(r, pred_table, pred_df)
            
            if plot:
                start_idx = self.output_window - 1
                end_idx = -self.output_window + 1 if self.output_window > 1 else None

                 
                # plot_df = pred_table.iloc[:, self.output_window-1:-self.output_window+1]
                plot_df = pred_table.iloc[:, start_idx:end_idx]
                # plot_gt_values = gt_values[self.output_window-1:-self.output_window+1]
                plot_gt_values = gt_values[start_idx:end_idx]
                return pred_table, plot_df, plot_gt_values
            
            return pred_table

    
    def plotTable(self, eval_ls, train_or_val, err_std, feature_id):
        '''
        Plot the prediction table
        '''
        x_plot = eval_ls[0]['plot_table'].columns.values
        x_plot = [pd.Timestamp(d).strftime('%Y-%m-%d %H:%M') for d in x_plot]
        
        fig,ax = plt.subplots()
        
        fig.set_figheight(5)
        fig.set_figwidth(20)
        
        ax.grid(visible=True, alpha=0.2)
        
        '''
        get the mean predictions
        '''
        ntrials = len(eval_ls)
        
        stck = []
        for trial in range(ntrials):
            # print(f"plot table for {train_or_val} = {eval_ls[trial]['plot_table']}")
            stck.append(eval_ls[trial]['plot_table'])    
        
        stck = np.array([df.values for df in stck])
        mean_array = np.mean(stck, axis=0)
        std_array = np.std(stck, axis=0)
        
        plot_mean_df = pd.DataFrame(mean_array, columns=x_plot, index=eval_ls[0]['plot_table'].index)
        plot_std_df = pd.DataFrame(std_array, columns=x_plot, index=eval_ls[0]['plot_table'].index)
        
        # print(f"plot_mean df = {plot_mean_df}")
        
        for t in self.horizon_range:
            mean_predictions = plot_mean_df.loc[t,:].values
            std_predictions = plot_std_df.loc[t, :].values
            
            # print(f"x_plot being plot is = {x_plot[300:450]}")
            ax.plot(x_plot, mean_predictions, linestyle='-', label='T+'+str(t))
            
            
            lower_bounds = mean_predictions - 2.1701*err_std[t-1]
            upper_bounds = mean_predictions + 2.1701*err_std[t-1]
            ax.fill_between(x_plot, lower_bounds, upper_bounds, alpha=0.2, label='T+'+str(t)+' Confidence shading')
        
        # print(f"x_plot = {len(x_plot)}")
        # print(f"plot_gt = {plot_gt.shape}")
        plot_gt = eval_ls[0]['plot_gt_values']
        ax.plot(x_plot, plot_gt, linestyle='-', label='Ground-truth')
        

        ax.set_xlabel('Timeline')
        ax.set_ylabel('Variable')
        
        if len(x_plot)>6*365:
            every_nth = 200
        elif len(x_plot)>=3*365:
            every_nth = 100
        else:
            every_nth = 10
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
        x = range(len(x_plot))
        ax.set_xticks(x)
        ax.set_xticklabels(x_plot, rotation=45)
        plt.legend()
        
        title = f'T+n Prediction Performance on {train_or_val} data (Target {feature_id})'

        plt.tight_layout()
        plt.title(title, y=1.02)
        # plt.savefig(f"./plot_{train_or_val}.pdf")
        wandb.log({f'{train_or_val}_target_{feature_id}_plot': wandb.Image(plt)})

        # Save predictions and ground truth to CSV
        self.save_predictions_to_csv(plot_mean_df, plot_std_df, plot_gt, x_plot, train_or_val, feature_id)
        
        plt.close()
    
    def save_predictions_to_csv(self, plot_mean_df, plot_std_df, plot_gt, x_plot, train_or_val, feature_id):
        '''
        Save predictions and ground truth to CSV file
        '''
        # Create a DataFrame with all the data
        csv_data = {}
        
        # Add timestamps
        csv_data['timestamp'] = x_plot
        
        # Add ground truth
        csv_data['ground_truth'] = plot_gt
        
        # Add predictions for each horizon
        for t in range(1, self.output_window + 1):
            if t in plot_mean_df.index:
                csv_data[f'prediction_T{t}'] = plot_mean_df.loc[t, :].values
                # csv_data[f'std_T{t}'] = plot_std_df.loc[t, :].values
                
                # # Add confidence intervals
                # csv_data[f'lower_bound_T{t}'] = plot_mean_df.loc[t, :].values - 2.1701 * plot_std_df.loc[t, :].values
                # csv_data[f'upper_bound_T{t}'] = plot_mean_df.loc[t, :].values + 2.1701 * plot_std_df.loc[t, :].values
        
        # Create DataFrame and save
        results_df = pd.DataFrame(csv_data)
        
        # Create predictions folder if it doesn't exist
        predictions_path = os.path.join("prediction_results", self.predictions_dir)
        if not os.path.exists(predictions_path):
            os.makedirs(predictions_path)
        
        # Create filename with timestamp
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{train_or_val}_target_{feature_id}.csv"
        filepath = os.path.join(predictions_path, filename)
        
        # Save to CSV
        results_df.to_csv(filepath, index=False)
        print(f"Predictions saved to: {filepath}")
        
        # Also log to wandb
        wandb.log({f'{train_or_val}_target_{feature_id}_predictions_csv': wandb.Table(dataframe=results_df)})

    def compute_horizon_rmse(self, eval_ls, train_or_val, feature_id=0):
        
        rmse_values = []
        std_values = []
        
        ntrials = len(eval_ls)
        
        gt_values = eval_ls[0]['horizon_gt_values']
        
        for i in range(self.output_window):
            rmse_trials = []
            for trial in range(ntrials):
                T_pred_table = eval_ls[trial]['horizon_pred_table']
                # plot_table = eval_ls[trial]['plot_table']
                rmse_trials.append(self.compute_rmse(i, T_pred_table, gt_values))
                # rmse_trials.append(self.compute_rmse(i, plot_table, gt_values))
                
            rmse_trials = np.array(rmse_trials)
            rmse_values.append((rmse_trials.mean(), rmse_trials.std())) # (mean, std) over trials for each H 
        
        rmse_values = pd.DataFrame(rmse_values, columns=['RMSE', 'STD'], index=range(1,self.output_window+1))
        filename = '{}_target{}_H{}_{}_{}.csv'.format(
        train_or_val, feature_id, self.output_window, self.lake, self.args.run_name.partition('_2')[0]
        )
        path = os.path.join(self.horizon_csv_path, filename)
        os.makedirs(self.horizon_csv_path, exist_ok=True)
        rmse_values.to_csv(path, index=False)

        y = rmse_values.RMSE.values.tolist()
        y_std = rmse_values.STD.values.tolist()

        x = rmse_values.index.tolist()
        xlabel = ['T+'+str(i) for i in rmse_values.index]
        fig,ax = plt.subplots()
        
        fig.set_figheight(4)
        fig.set_figwidth(15)
        
        ax.grid(visible=True, alpha=0.2)
        ax.plot(x, y, linestyle='-', label='RMSE')
        ax.errorbar(x, y, yerr=y_std, label='Error Bars', fmt='o', color='green', alpha=0.5, capsize=5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(xlabel, rotation=90)
        plt.legend()
        
        title = '{} Target {}: Varying RMSE error w.r.t horizon window'.format(train_or_val, feature_id)

        plt.tight_layout()
        plt.title(title,y=1.02)
        plt.xlabel('Horizon Window')
        plt.ylabel('Root Mean Squared Error')
        
        wandb.log({f'{train_or_val}_target_{feature_id}_rmse_plot': wandb.Image(plt)})

        plt.close()
        
        return rmse_values
    
    def compute_rmse(self, i, ptable, gt_values):
    
        tk = ptable.iloc[i,:].values
        null_inds = np.where(np.isnan(tk))[0]
        mask = np.ones(gt_values.shape)
        mask[null_inds]= 0
        tk = np.nan_to_num(tk)
        unreduced_loss = (tk-gt_values)**2
        unreduced_loss = (unreduced_loss * mask).sum()
        
        non_zero_elements = mask.sum()
        loss = unreduced_loss / non_zero_elements
        
        rmse = loss**0.5
        # print(f"rmse = {rmse}")
        return rmse