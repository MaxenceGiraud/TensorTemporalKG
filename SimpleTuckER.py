from torch.nn.init import xavier_normal_
import torch 

class SimpleTuckER:
    def __init__(self,n_entities,n_relations,entities_embedding_dim,relation_embedding_dim):
        # super(SimpleTuckER,self).__init__()

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.entities_embedding_dim = entities_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim

        self.E = torch.nn.Embedding(n_entities,entities_embedding_dim) # Entities embeddings
        self.R = torch.nn.Embedding(n_relations,relation_embedding_dim) # Relation embeddings
        self.W  = torch.nn.Parameter(torch.FloatTensor(entities_embedding_dim,relation_embedding_dim,entities_embedding_dim).uniform_(-1,1),requires_grad=True) # Core Tensor

        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self,e1_idx, r_idx):
        e1 = self.E(e1_idx)
        r = self.R(r_idx)

        x = torch.tensordot(e1,self.W,dims=[[1],[0]])
        x = torch.bmm(r.view(-1,1,self.relation_embedding_dim),x)
        x = torch.tensordot(self.E.weight,x,dims=[[1],[2]])

        pred = torch.sigmoid(x.view(self.n_entities,-1))

        return pred
    
    def score(self,e1_idx,r_idx,e2_idx):
        e1 = self.E(e1_idx)
        r = self.R(r_idx)
        e2 = self.E(e2_idx)

        x = torch.tensordot(e1,self.W,dims=[[1],[0]])
        x = torch.tensordot(r,x,dims=[[1],[1]])
        x = torch.tensordot(e2,x,dims=[[1],[2]])

        pred = torch.sigmoid(x)

        return pred
        
