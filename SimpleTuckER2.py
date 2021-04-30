from torch.nn.init import xavier_normal_
import torch 

class SimpleTuckER2:
    def __init__(self,n_entities,n_relations,entities_embedding_dim,relation_embedding_dim):
        # super(SimpleTuckER2,self).__init__()

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.entities_embedding_dim = entities_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim

        self.E = torch.nn.Embedding(n_entities,entities_embedding_dim) # Entities embeddings
        self.R = torch.nn.Embedding(n_relations,relation_embedding_dim) # Relation embeddings
        self.W  = torch.nn.Parameter(torch.FloatTensor(entities_embedding_dim,relation_embedding_dim,entities_embedding_dim,relation_embedding_dim,entities_embedding_dim).uniform_(-1,1),requires_grad=True) # Core Tensor

        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self,e1_idx, r1_idx):
        e1 = self.E(e1_idx)
        r1 = self.R(r1_idx)

        x = torch.tensordot(e1,self.W,dims=[[1],[0]])
        x = torch.bmm(r1.view(-1,1,self.relation_embedding_dim),x.view(x.shape[0],x.shape[1],-1))
        x = x.view(x.shape[0],self.entities_embedding_dim,self.relation_embedding_dim,self.entities_embedding_dim)
        x = torch.tensordot(self.E.weight,x,dims=[[1],[1]])
        x = torch.tensordot(self.R.weight,x,dims=[[1],[2]])
        x = torch.tensordot(self.E.weight,x,dims=[[1],[3]])

        pred = torch.sigmoid(x.view(e1_idx.shape[0],self.n_entities,self.n_relations,self.n_entities))

        return pred
    
    def score(self,e1_idx,r1_idx,e2_idx,r2_idx,e3_idx):
        e1 = self.E(e1_idx)
        e2 = self.E(e2_idx)
        e3 = self.E(e3_idx)
        r1 = self.R(r1_idx)
        r2 = self.R(r2_idx)

        x = torch.tensordot(e1,self.W,dims=[[1],[0]])
        x = torch.tensordot(r1,x,dims=[[1],[1]])
        x = torch.tensordot(e2,x,dims=[[1],[2]])
        x = torch.tensordot(r2,x,dims=[[1],[3]])
        x = torch.tensordot(e3,x,dims=[[1],[4]])

        pred = torch.sigmoid(x)

        return pred
