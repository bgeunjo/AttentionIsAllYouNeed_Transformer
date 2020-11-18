import torch
import torch.nn as nn
import numpy as np
from torch import matmul, sqrt
# [ ✔ ] positionalEncoding 
# [ _ ] scaledDotProduction 
# [ _ ] multiHeadAttention 
# [ _ ] maskedMultiHeadAttention 
# [ _ ] FeedForwardNetwork 
# [ _ ] layerNomalization  
# [ _ ] Encoder 
# [ _ ] Decoder 

def positionalEncoding(pos_num,d_model):
    def getPEArgu(pos,i):
        c = np.power(10000 , 2*i / d_model)
        return pos / c
    def getPE(pos):
        return [ getPEArgu(pos,i) for i in range(d_model) ]

    embeded = np.array([ getPE(pos) for pos in range(pos_num) ])
    embeded[:, 0::2], embeded[:,1::2] = np.sin(embeded[:,0::2]), np.cos(embeded[:,1::2])

    return torch.FloatTensor(embeded)

Q = positionalEncoding(3,64)
print (Q.shape)
K = positionalEncoding(4,64)
d_k = torch.FloatTensor(K.size(-1))
print (K.transpose(-1,-2).shape)
score =matmul(Q,K.transpose(-1,-2)) / sqrt(d_k)
# score = nn.Softmax(dim=1)(score)

class scaledDotProduction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,Q,K,V, opt_mask=False):
        d_k = K.size(-1)
        score = matmul(Q, K.transpose(-1, -2)) / sqrt(d_k)
        score.masked_fill_(opt_mask,float("-inf"))
        score = nn.Softmax(dim=1)(score)
