import torch
import torch.nn as nn


def expand_dims(tensor:torch.Tensor):
    if tensor.ndim ==1: tensor = tensor[None, :, None]
    elif tensor.ndim==2: tensor = tensor[:, :, None]
    return tensor

class ActCondLSTM(nn.Module):
    def __init__(self, 
                 input_size,
                 hidden_size,
                 action_dim,
                 fusion_size,
                 bias=True):
        '''
        Arguments:
            input_size: the dimensionality of input to the LSTM
            hidden_size: the number of hidden units
            action_dim: the dimensionality of the action space
            fusion_size: the fusioned vector size
        '''
        super(ActCondLSTM,self).__init__()
        self.bias = bias

        # Action fusion parameters
        self.Wh = torch.FloatTensor(fusion_size,hidden_size).uniform_(-1/hidden_size,1/hidden_size).requires_grad_(True)
        self.Wa = torch.FloatTensor(fusion_size,action_dim).uniform_(-1/hidden_size,1/hidden_size).requires_grad_(True)
        
        # Gate Update parameters
        self.Wiv= torch.FloatTensor(hidden_size,fusion_size).uniform_(-1/hidden_size,1/hidden_size).requires_grad_(True)
        self.Wiz= torch.FloatTensor(hidden_size,input_size).uniform_(-1/hidden_size,1/hidden_size).requires_grad_(True)
        if bias:
            self.bi = torch.FloatTensor(hidden_size,1).uniform_(-1/hidden_size,1/hidden_size).requires_grad_(True)
        
        self.Wfv= torch.FloatTensor(hidden_size,fusion_size).uniform_(-1/hidden_size,1/hidden_size).requires_grad_(True)
        self.Wfz= torch.FloatTensor(hidden_size,input_size).uniform_(-1/hidden_size,1/hidden_size).requires_grad_(True)
        if bias:
            self.bf = torch.FloatTensor(hidden_size,1).uniform_(-1/hidden_size,1/hidden_size).requires_grad_(True)
        
        self.Wov= torch.FloatTensor(hidden_size,fusion_size).uniform_(-1/hidden_size,1/hidden_size).requires_grad_(True)
        self.Woz= torch.FloatTensor(hidden_size,input_size).uniform_(-1/hidden_size,1/hidden_size).requires_grad_(True)
        if bias:
            self.bo = torch.FloatTensor(hidden_size,1).uniform_(-1/hidden_size,1/hidden_size).requires_grad_(True)
        
        # Cell update parameters
        self.Wcv= torch.FloatTensor(hidden_size,fusion_size).uniform_(-1/hidden_size,1/hidden_size).requires_grad_(True)
        self.Wcz= torch.FloatTensor(hidden_size,input_size).uniform_(-1/hidden_size,1/hidden_size).requires_grad_(True)
        if bias:
            self.bc = torch.FloatTensor(hidden_size,1).uniform_(-1/hidden_size,1/hidden_size).requires_grad_(True)
        

    def forward(self, x , a , h , c):
        '''
        Arguments:
            x: input
            a: action
            h: hidden state input
            c: cell input
        Returns:
            new_hidden_state: output hidden state of the LSTM
            new_cell_output: output cell value of the LSTM
        '''
        x , a , h , c = map(expand_dims,[x , a , h , c]) # we need an extra dimension at the end of vectors
        # Action fusion
        fusioned = torch.mul(torch.matmul(self.Wh , h),torch.matmul(self.Wa, a))

        # Gate update
        new_input_gate = nn.functional.sigmoid(torch.matmul(self.Wiv,fusioned)+\
                                               torch.matmul(self.Wiz,x)+\
                                                (self.bi if self.bias else 0))
        new_forget_gate = nn.functional.sigmoid(torch.matmul(self.Wfv,fusioned)+\
                                               torch.matmul(self.Wfz,x)+\
                                                (self.bf if self.bias else 0))
        new_output_gate = nn.functional.sigmoid(torch.matmul(self.Wov,fusioned)+\
                                               torch.matmul(self.Woz,x)+\
                                                (self.bo if self.bias else 0))
        
        # Cell update
        new_cell_output = new_forget_gate * c +\
                        new_input_gate * nn.functional.tanh(torch.matmul(self.Wcv,fusioned) +\
                                                            torch.matmul(self.Wcz,x)+\
                                                            (self.bc if self.bias else 0))
        new_hidden_state = new_output_gate * nn.functional.tanh(new_cell_output)
        return new_hidden_state.squeeze(-1) , new_cell_output.squeeze(-1) # we have to drop the last dimension
        
