import torch
import torch.nn as nn
from action_conditional_lstm import ActCondLSTM

class RecEnvSimModel(nn.Module):
    def __init__(self):
        super(RecEnvSimModel,self).__init__()
