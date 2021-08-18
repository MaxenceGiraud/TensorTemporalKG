import numpy as np
import torch
from torch.nn.init import xavier_normal_


class TuckERTTR(torch.nn.Module):
    def __init__(self, d, de, dr,dt,ranks,cuda=False, **kwargs):
        super(TuckERTTR, self).__init__()

        if cuda == True :
            self.device = 'cuda'
        else :
            self.device  = 'cpu'

        # Embeddings dimensionality
        self.de = de
        self.dr = dr
        self.dt = dt

        # Data dimensionality
        self.ne = len(d.entities)
        self.nr = len(d.relations)
        self.nt = len(d.time)

        # Embedding matrices
        self.E = torch.nn.Embedding(self.ne, de).to(self.device)
        self.R = torch.nn.Embedding(self.nr, dr).to(self.device)
        self.T = torch.nn.Embedding(self.nt,dt).to(self.device)

        ni = [self.dr, self.de, self.de, self.dt]
        if isinstance(ranks,int) or isinstance(ranks,np.int64):
            ranks = [ranks for _ in range(5)]
        elif isinstance(ranks,list) and len(ranks)==5:
            pass
        else : 
            raise TypeError('ranks must be int or list of len 5')
        
        self.Zlist = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(np.random.uniform(-1e-1, 1e-1, (ranks[i], ni[i], ranks[i+1])), dtype=torch.float, requires_grad=True).to(self.device)) for i in range(4)])


        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bne = torch.nn.BatchNorm1d(de)
        self.bnr = torch.nn.BatchNorm1d(dr)
        self.bnt = torch.nn.BatchNorm1d(dt)
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)
        xavier_normal_(self.T.weight.data)

    def forward(self, e1_idx, r_idx,t_idx):
        W = torch.einsum('aib,bjc,ckd,dla->ijkl', list(self.Zlist))
        
        W = W.view(self.dr, self.de, self.de, self.dt)


        e1 = self.E(e1_idx)
        # x = self.bne(e1)
        # x = self.input_dropout(x)
        x = e1
        x = x.view(-1, 1, self.de)

        r = self.R(r_idx)
        # r = self.bnr(r)
        W_mat = torch.mm(r, W.view(self.dr, -1))
        W_mat = W_mat.view(-1, self.de, self.de*self.dt)
        # W_mat = self.hidden_dropout1(W_mat)
        x = torch.bmm(x, W_mat) 

        t = self.T(t_idx)
        # t = self.bnt(t)
        x = x.view(-1, self.de,self.dt)
        x = torch.bmm(x,t.view(*t.shape,-1))

        # x = self.hidden_dropout2(x)

        x= x.view(-1,self.de)
        x = torch.mm(x, self.E.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred