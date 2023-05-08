import numpy as np
import torch
import torch.nn as nn
from action_conditional_lstm import ActCondLSTM
from encoder import ConvEncoder
from decoder import ConvDecoder

class RecEnvSimModel(nn.Module):
    def __init__(self,
                 input_shape, # =[3,210,160]
                 hidden_size, # = 1024
                 action_dim,
                 fusion_size): # = 2048
        super(RecEnvSimModel,self).__init__()
        self.encoder = ConvEncoder()
        self.lstm = ActCondLSTM(input_size= np.prod(input_shape),
                                hidden_size= hidden_size,
                                action_dim=action_dim,
                                fusion_size=fusion_size)
        self.decoder = ConvDecoder()

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
