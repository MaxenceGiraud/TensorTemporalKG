import numpy as np
import torch
from torch.nn.init import xavier_normal_


class TuckERTNT(torch.nn.Module):
    def __init__(self, d, de, dr,dt,cuda=False, **kwargs):
        super(TuckERTNT, self).__init__()

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
        self.E = torch.nn.Embedding(self.ne, de)
        self.R = torch.nn.Embedding(self.nr, dr)
        self.T = torch.nn.Embedding(self.nt,dt)

        # Core tensor
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-0.1, 0.1, (dr, de, de,dt)), dtype=torch.float, device=self.device, requires_grad=True))

        # "Special" Layers
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bne = torch.nn.BatchNorm1d(de)
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)
        xavier_normal_(self.T.weight.data)

    def forward(self, e1_idx, r_idx,t_idx):
        ### Temporal part
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
        xt = x.view(-1, self.de,self.dt)
        xt = torch.bmm(xt,t.view(*t.shape,-1))
        
        # Mode 4 product with entity matrix # TODO not repeat this operation (using maybe bmm or reshaping core tensor to do operations)
        xt= xt.view(-1,self.de)
        xt = torch.mm(xt, self.E.weight.transpose(1,0))


        ### Non temporal part
        # mode 3 product with identity matrix
        x = x.view(-1,self.dt)
        x = torch.mm(x,torch.ones(self.dt).view(self.dt,1)) 

        # Mode 4 product with entity matrix
        x= x.view(-1,self.de)
        x = torch.mm(x, self.E.weight.transpose(1,0))


        # Sum of the 2 models
        x = x + xt
        
        # Turn results into "probabilities"
        pred = torch.sigmoid(x)
        return pred