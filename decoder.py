import torch
import torch.nn as nn
import numpy as np



class ConvDecoder(nn.Module):
    def __init__(self,
                 input_length = 1024,
                 deconv_input_shape = [32,11,8],
                 num_filters_list = [32,32,64,3],
                 filter_size_list = [(4,4),(6,6),(6,6),(8,8)],
                 stride_list=[2 for _ in range(4)],
                 padding_list=[(0,0),(1,1),(1,1),(0,1)]) -> None:
        '''
        Arguments
            input_length: the length of the input vector to the decoder, int
            deconv_input_shape: the shape of the input tensor to the transposedConv decoder
            num_filters_list: a list of ints for the number of filters of each ConvTranspose2d layer
            filter_size_list: a list of ints or tuples for the filters size of each ConvTranspose2d layer
            stride_list: a list of ints or tuples for the stride size of each ConvTranspose2d layer
            padding_list: a list of ints or tuples for the stride size of each ConvTranspose2d layer
        '''
        super(ConvDecoder,self).__init__()
        assert  len(num_filters_list) ==\
                len(filter_size_list) == \
                len(stride_list) == \
                len(padding_list)
        self.deconv_input_shape = list(deconv_input_shape)
        self.linear = (nn.Linear(input_length,np.prod(deconv_input_shape))) # a linear layer that keeps the dimensions the same
        self.model = nn.Sequential()
        in_channels = deconv_input_shape[0]
        for i_layer in range(len(num_filters_list)):
            self.model.append(nn.ConvTranspose2d(in_channels=in_channels,
                                                out_channels=num_filters_list[i_layer],
                                                kernel_size=filter_size_list[i_layer],
                                                stride=stride_list[i_layer],
                                                padding=padding_list[i_layer]))
            in_channels = num_filters_list[i_layer]
            if i_layer!=len(num_filters_list)-1: # if this is not the last layer
                self.model.append(nn.BatchNorm2d(num_filters_list[i_layer]))
                self.model.append(nn.RReLU())
        
    def forward(self,x):
        x = self.linear(x)
        x = torch.reshape(x,[-1]+self.deconv_input_shape)
        return self.model(x)