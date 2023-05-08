import numpy as np
import torch
import torch.nn as nn
from action_conditional_lstm import ActCondLSTM
from encoder import ConvEncoder
from decoder import ConvDecoder

class RecEnvSimModel(nn.Module):
    # def __init__(self,
    #              input_shape, # =[3,210,160]
    #              hidden_size, # = 1024
    #              action_dim,
    #              fusion_size): # = 2048
    def __init__(self,
                 LSTM_config:tuple,
                 encoder_config:tuple,
                 decoder_config:tuple):
        '''
        Arguments:
            LSTM_config: a tuple of
                input_size: the dimensionality of input to the LSTM, int
                hidden_size: the number of hidden units, int
                action_dim: the dimensionality of the action space, int
                fusion_size: the fusioned vector size, int
            encoder_config: a tuple of
                in_channels: number of channels of an image
                num_filters_list: a list of ints for the number of filters of each conv2d layer
                filter_size_list: a list of ints or tuples for the filters size of each conv2d layer
                stride_list: a list of ints or tuples for the stride size of each conv2d layer
                padding_list: a list of ints or tuples for the stride size of each conv2d layer
            decoder_config: a tuple of
                input_length: the length of the input vector to the decoder, int
                deconv_input_shape: the shape of the input tensor to the decoder
                num_filters_list: a list of ints for the number of filters of each ConvTranspose2d layer
                filter_size_list: a list of ints or tuples for the filters size of each ConvTranspose2d layer
                stride_list: a list of ints or tuples for the stride size of each ConvTranspose2d layer
                padding_list: a list of ints or tuples for the stride size of each ConvTranspose2d layer
            
        '''
        super(RecEnvSimModel,self).__init__()
        self.encoder = ConvEncoder(*encoder_config)
        self.lstm = ActCondLSTM(*LSTM_config)
        self.decoder = ConvDecoder(*decoder_config)

    def forward(self,x,a,hc):
        '''
        Arguments:
            x: input state of the environment (the image of the game for example)
            a: action taken at state x
            hc: a tuple (h,c) where h is the hidden state vector of the LSTM
            and c is the cell state of the LSTM.
        '''
        h, c = hc
        encoded = self.encoder(x)
        new_h, new_c = self.lstm(encoded,a,h,c)
        decoded = self.decoder(new_h)
        return decoded, (new_h, new_c)
