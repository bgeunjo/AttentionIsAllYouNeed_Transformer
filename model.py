import torch
import torch.nn as nn
import numpy as np
from torch import matmul, sqrt

# [ ✔ ] positionalEncoding 
# [ ✔ ] scaledDotProduction - 질문 !
# [ _ ] multiHeadAttention 
# [ _ ] maskedMultiHeadAttention 
# [ ✔ ] posWiseFeedForwardNetwork 
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

class scaledDotProduction(nn.Module):
    # Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
    def __init__(self):
        super(scaledDotProduction,self).__init__()

    def forward(self, Q, K, V, opt_mask=False):
        d_k = K.size(-1)
        score = matmul(Q, K.transpose(-1, -2)) / sqrt(d_k)
        score.masked_fill_(opt_mask,float("-inf"))
        score = nn.Softmax(dim=-1)(score)
        attention = matmul(score, V)
        return attention



class multiHeadAttention(nn.Module):
    # MultiHead(Q, K, V) = Concat(head_1,...,head_h)W^O
    # WHERE head_i=Attention(QW_i^Q, KW_i^K, VW_i^V)

    def __init__(self):
        super(multiHeadAttention,self).__init__()
    
    def forward(self, Q, K, V):
        return 

class posWiseFFN(nn.Module):
    def __init__(self,d_model,d_ff):
        super(posWiseFFN,self).__init__()
        self.layer1=nn.Linear(d_model,d_ff)
        self.relu=nn.ReLU(inplace=True)
        self.layer2=nn.Linear(d_ff,d_model)

        self.layerNom=nn.LayerNorm(d_model)
    def forward(self,x):
        out = self.layer2(self.relu(self.layer1(x)))
        out = self.layerNom(out + x) # residual connection
        return out

# ====TEST AREA====

#Q = positionalEncoding(3,64)
#print (Q.shape)
#K = positionalEncoding(4,64)
#d_k = torch.FloatTensor(K.size(-1))
#print (K.transpose(-1,-2).shape)
#score =matmul(Q,K) / sqrt(d_k)
#score = nn.Softmax(dim=1)(Q)
#print (score)