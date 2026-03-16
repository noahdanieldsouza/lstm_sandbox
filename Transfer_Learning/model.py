import random
import numpy as np
import pandas as pd
import copy
import wandb
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from torch import optim
from datetime import timedelta, datetime
from tqdm import trange, tqdm

from utility import Utils

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, out_path, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.out_path = out_path
        
    def __call__(
        self, current_valid_loss, model, epoch, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(model.state_dict(), self.out_path)

class encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers=1, rnn_type='LSTM', dropout=0.0):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.dropout = dropout

        # define LSTM/GRU/RNN layer
        f = getattr(nn, self.rnn_type)
        self.model = f(input_size=input_size, hidden_size=hidden_size,
                       num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, x_input):

        '''
        : param x_input:               input of shape (# in batch, seq_len, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        '''
        
        lstm_out, self.hidden = self.model(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))

        return lstm_out, self.hidden

    def init_hidden(self, batch_size):

        '''
        initialize hidden state
        : param batch_size:    x_input.shape[0]
        : return:              zeroed hidden state and cell state
        '''
        if self.rnn_type == 'LSTM':
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size))
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size)


class decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, hidden_size, num_layers=1, rnn_type='LSTM', dropout=0.0):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.dropout = dropout

        # define LSTM/GRU/RNN layer
        f = getattr(nn, self.rnn_type)
        self.model = f(input_size=input_size, hidden_size=hidden_size,
                       num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # TODO: predict mean and max
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''
        lstm_out, hidden = self.model(x_input.unsqueeze(1), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(1))

        return output, hidden

class seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''

    def __init__(self, input_size,
                 utils,
                 args):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of lstm in both encoder and decoder
        '''

        super(seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = args.hidden_feature_size
        self.num_layers = args.num_layers
        self.rnn_type = args.rnn_type
        self.output_size = args.c_out
        self.device = args.device
        self.dropout = args.dropout
        self.target_len = args.horizon_window
        self.training_prediction = args.training_prediction
        self.teacher_forcing_ratio = args.teacher_forcing_ratio

        self.encoder = encoder(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, rnn_type=self.rnn_type, dropout=self.dropout).to(self.device)
        self.decoder = decoder(input_size=self.output_size, hidden_size=self.hidden_size, num_layers=self.num_layers, rnn_type=self.rnn_type, dropout=self.dropout).to(self.device)
        
        self.encoder_init = copy.deepcopy(self.encoder)
        self.decoder_init = copy.deepcopy(self.decoder)
        
        self.utils = utils
    
    def forward(self, input_batch, alpha, target_batch=None):
        
        # outputs tensor
        outputs = torch.zeros(input_batch.shape[0], self.target_len, self.output_size, device=self.device)
        
        # encoder outputs
        encoder_output, encoder_hidden = self.encoder(input_batch)

        decoder_input = torch.zeros([input_batch.shape[0], self.output_size], device=self.device)  # shape: (batch_size, input_size)
        decoder_hidden = encoder_hidden
        
        if self.training:
            if self.training_prediction == 'recursive':
                # predict recursively
                for t in range(self.target_len):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[:,t,:] = decoder_output
                    decoder_input = decoder_output

            if self.training_prediction == 'teacher_forcing':
                # use teacher forcing
                if random.random() < self.teacher_forcing_ratio:
                    for t in range(self.target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[:,t,:] = decoder_output
                        decoder_input = target_batch[:, t, :]

                # predict recursively
                else:
                    for t in range(self.target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[:,t,:] = decoder_output
                        decoder_input = decoder_output

            if self.training_prediction == 'mixed_teacher_forcing':
                # predict using mixed teacher forcing
                for t in range(self.target_len):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[:,t,:] = decoder_output

                    # predict with teacher forcing
                    if random.random() < self.teacher_forcing_ratio:
                        decoder_input = target_batch[:, t, :]

                    # predict recursively
                    else:
                        decoder_input = decoder_output
                        
        else:
            for t in range(self.target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:,t,:] = decoder_output
                decoder_input = decoder_output
            
        out_cloned = outputs.clone()

        # out_cloned[:, :, 0] = torch.clamp(outputs[:, :, 0], min=alpha)
        out_cloned = torch.clamp(outputs, min=alpha)
        
        return out_cloned
    
    def freeze_or_no_freeze(self, args):

        if args.freeze_enc=='true':
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"Encoder frozen")
            
        if args.freeze_dec=='true':
            for name, param in self.decoder.named_parameters():
                if 'linear' in name:
                    continue
                else:
                    param.requires_grad = False
            print(f"Decoder frozen")
            
        if args.freeze_linear=='true':
            for name, param in self.decoder.named_parameters():
                if 'linear' in name:
                    param.requires_grad = False

            print(f"Linear Layer frozen")

class TFModel(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    
    TODO:
    1. At-least use Reformer if not Autoformer/Informer/FedFormer
    """
    def __init__(self, configs):
        super(TFModel, self).__init__()
        self.horizon_window = configs.horizon_window
        self.output_attention = configs.output_attention
        self.lookback_window = configs.lookback_window
        
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        
        # Initialization
        self.initialize_weights()

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, 
                x_enc, 
                x_dec, 
                alpha,
                x_mark_enc=None, 
                x_mark_dec=None,
                enc_self_mask=None, 
                dec_self_mask=None, 
                dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # print(f"enc_out = {enc_out.shape}")
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        
        dec_out_cloned = dec_out.clone()
        
        dec_out_cloned[:, :, 0] = torch.clamp(dec_out[:, :, 0], min=alpha)
        
        if self.output_attention:
            return dec_out_cloned[:, -self.horizon_window:, :], attns
        else:
            return dec_out_cloned[:, -self.horizon_window:, :]  # [B, L, D]
    
    def initialize_weights(self):
        # Apply custom initialization to specific components if necessary
        # Example: Special tokens or specific embeddings not covered by default init
        # torch.nn.init.normal_(self.some_special_token, std=.02)

        # Apply a general initialization policy to Linear and LayerNorm layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.weight, 1.0)
            torch.nn.init.constant_(m.bias, 0)
        
    def freeze_or_no_freeze(self, args):
        if args.freeze_enc=='true':
            for param in self.enc_embedding.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False

            print(f"Encoder frozen")
            
        if args.freeze_dec=='true':
            for param in self.dec_embedding.parameters():
                param.requires_grad = False
            for name, param in self.decoder.named_parameters():
                if 'projection' in name:
                    continue
                else:
                    param.requires_grad = False
            print(f"Decoder frozen")
        
        if args.freeze_linear=='true':
            for name, param in self.decoder.named_parameters():
                if 'linear' in name:
                    param.requires_grad = False

            print(f"Linear Layer frozen")