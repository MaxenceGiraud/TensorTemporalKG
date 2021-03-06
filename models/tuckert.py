import numpy as np
import torch
from torch.nn.init import xavier_normal_


class TuckERT(torch.nn.Module):
    def __init__(self, d, de, dr,dt,device="cpu",input_dropout=0.,hidden_dropout1=0.,hidden_dropout2=0.,**kwargs):
        super(TuckERT, self).__init__()

        self.device = device

        # Embeddings dimensionality
        self.de = de
        self.dr = dr
        self.dt = dt

        # Data dimensionality
        self.ne = len(d.entities)
        self.nr = len(d.relations)
        self.nt = len(d.time)
        

        # Embedding matrices
        self.E = torch.nn.Embedding(self.ne, de)
        self.R = torch.nn.Embedding(self.nr, dr)
        self.T = torch.nn.Embedding(self.nt,dt)

        # Core tensor
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-0.1, 0.1, (dr, de, de,dt)), dtype=torch.float, device=self.device, requires_grad=True))


        # "Specia"l layers
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = torch.nn.Dropout(hidden_dropout2)
        self.loss = torch.nn.BCELoss()

        self.bne = torch.nn.BatchNorm1d(de)
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)
        xavier_normal_(self.T.weight.data)

    def forward(self, e1_idx, r_idx,t_idx):
        # Mode 1 product with entity vector
        e1 = self.E(e1_idx)
        x = self.bne(e1)
        x = self.input_dropout(x)
        x = e1
        x = x.view(-1, 1, self.de)

        # Mode 2 product with relation vector
        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, self.de, self.de*self.dt)
        x = torch.bmm(x, W_mat) 

        # Mode 3 product with time vector
        t = self.T(t_idx)
        x = x.view(-1, self.de,self.dt)
        x = torch.bmm(x,t.view(*t.shape,-1))
        
        # Mode 4 product with entity matrix
        x= x.view(-1,self.de)
        x = torch.mm(x, self.E.weight.transpose(1,0))

        pred = torch.sigmoid(x)
        return pred