from torch.nn.utils.rnn import pack_padded_sequence
import torch
from typing import *
import random
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence

def pading_empty_tensor(context_LHL):  #context(QA) last hidden layer, this is padding for lstm because the context length may be difference
    seqlen = [s.size(0) for s in context_LHL]
    data = pack_padded_sequence(context_LHL, seqlen, batch_first=True, enforce_sorted=False)
    return data

def get_retrieve_context_matrix(SEPind:List, max_context_len, hidden_layer_size):
    #### we use this function to retrieve after the highway network
    #### because in highway network will add some constant tensor
    #### those tensor is zero array(padding) need to be eliminate to zero

    num = len(SEPind)  #通常是 batch size
    
    AOLN = [1] * hidden_layer_size  # a_output_hidden_layer_needed
    AOLD = [0] * hidden_layer_size  # a_output_hidden_DONT_needed

    ## -1 because of the SEPind is take [CLS] into consideration
    mtx = [[AOLN]*(SEPind[i]-1) + [AOLD]*(max_context_len-SEPind[i]+1) for i in range(num)]
    mtx = torch.tensor(mtx, requires_grad=False)
    
    return mtx

def get_context_from_LHL(LHL, SEPind:List):
    batch_size = len(SEPind)
    context = []
    for i in range(batch_size):
        context.append(LHL[i][1:SEPind[i]])
    
    ret = pad_sequence(context, batch_first=True)
    return ret

class Highway_layer(torch.nn.Module):
    def get_2d_init_tensor(self,m,n):
        w = torch.empty(m,n, requires_grad=True)
        w = torch.nn.init.xavier_normal_(w)
        return w

    def get_1d_init_tensor(self,m):
        w = [0] * m
        for i in range(m):
            w[i] = random.uniform(-0.8, 0.8)
        w = torch.tensor(w, requires_grad=True)
        return w

    def __init__(self) -> None:

        super(Highway_layer, self).__init__()

        self.Wp = Parameter(self.get_2d_init_tensor(768,768))
        self.bp = Parameter(self.get_1d_init_tensor(768))
        self.Wgate = Parameter(self.get_2d_init_tensor(768,768))
        self.bgate = Parameter(self.get_1d_init_tensor(768))

        # self.l = torch.nn.Linear(20,20)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, T) :
        Xp = torch.matmul(T, self.Wp) + self.bp
        Xp = torch.relu(Xp)
        Xgate = torch.matmul(T, self.Wgate) + self.bgate
        Xgate = torch.sigmoid(Xgate)

        Xhighway = Xgate * Xp + T - Xgate * T
        return Xhighway
    
class CLS_pooler_layer(torch.nn.Module):
    def get_2d_init_tensor(self,m,n):
        w = torch.empty(m,n, requires_grad=True)
        w = torch.nn.init.xavier_normal_(w)
        return w

    def get_1d_init_tensor(self,m):
        w = [0] * m
        for i in range(m):
            w[i] = random.uniform(-0.8, 0.8)
        w = torch.tensor(w, requires_grad=True)
        return w

    def __init__(self, num_labels) -> None:
        super(CLS_pooler_layer, self).__init__()

        self.Wcls = Parameter(self.get_2d_init_tensor(768,num_labels))
        self.bcls = Parameter(self.get_1d_init_tensor(num_labels))

        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, CLS) :
        ret = torch.matmul(CLS, self.Wcls) + self.bcls
        ret = torch.sigmoid(ret)
        return ret
