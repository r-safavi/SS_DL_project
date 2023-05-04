import torch
import torch.nn as nn



class ConvEncoder(nn.Module):
    def __init__(self,
                 state_size,
                 num_filters_list = [64,32,32,32],
                 filter_size_list = [(8,8),(6,6),(6,6),(4,4)],
                 stride_list=[2 for _ in range(4)],
                 padding_list=['same' for _ in range(4)]) -> None:
        '''
        Arguments:
            filter_size_list: a list of ints or tuples for the filters size of each conv2d layer
            stride_list: a 
        '''
        super(ConvEncoder,self).__init__()
        assert  len(num_filters_list) ==\
                len(filter_size_list) == \
                len(stride_list) == \
                len(padding_list)
        
        self.model = nn.Sequential()
        for i_layer in range(len(num_filters_list)):
            self.model.append(nn.Conv2d(in_channels=state_size,
                                        out_channels=num_filters_list[i_layer],
                                        kernel_size=filter_size_list[i_layer],
                                        stride=stride_list[i_layer],
                                        padding=padding_list[i_layer]))
            self.model.append(nn.BatchNorm2d(num_filters_list[i_layer]))
            self.model.append(nn.RReLU())


