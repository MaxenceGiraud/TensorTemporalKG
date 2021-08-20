import torch
import numpy as np
from load_data import Data
import time
from collections import defaultdict


def get_er_vocab(data):
    ''' Construct a dict of the data containing [E,R,T] as keys and target entities as values
    '''

    er_vocab = defaultdict(list)
    for quad in data:
        er_vocab[(quad[0], quad[1],quad[2])].append(quad[3])
    return er_vocab

def get_batch(batch_size,er_vocab, er_vocab_pairs, idx,n_entities,device='cpu'):
    ''' Return a batch of data for training
    batch_size : int,
        .
    er_vocab : dict,
        Dict containing [E,R,T] as keys and target entities as values
    er_vocab_pairs : list,
        list of er_vocab keys
    idx: int,
        Batch number 
    n_entities : int, 
        Total number of entities considered in the model (n_e)
    device :  {'cpu','cuda'}
        On which device to do the computation
    '''

    batch = er_vocab_pairs[idx:idx+batch_size]
    # targets = np.zeros((len(batch), len(data.entities)))
    targets = np.zeros((len(batch), n_entities))
    for idx, pair in enumerate(batch):
        targets[idx, er_vocab[pair]] = 1.
    targets = torch.FloatTensor(targets).to(device)
    return np.array(batch), targets


def train_temporal(model,data_idxs,data_idxs_valid,n_iter=200,learning_rate=0.0005,batch_size=128,print_loss_every=1,early_stopping=10,device='cpu'):
    ''' Train a temporal KG model

    Parameters
    -----------
    model : TuckER instance,
        TuckER model
    data_idxs : list of triples,    
        Train data idxs
    n_iter : int,
        Number of iterations
    learning_rate : float,
        Learning rate
    batch size : int,
        Batch size
    print_loss_every : int,
        Frequency for when to print the losses
    early_stopping : {False,int}:
        If False does nothing, if a number will perform early stopping using this int
    device : {'cpu','cuda'}
        On which device to do the computation
    '''

    if early_stopping == False : 
        early_stopping = n_iter + 1
        
    model.init()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    er_vocab = get_er_vocab(data_idxs)
    er_vocab = {k: v for (k, v) in er_vocab.items() if (type(k) is list or type(k) is tuple) and len(k) == 3}

    n_entities = int(max(data_idxs[:,0])+1)

    er_vocab_pairs = list(er_vocab.keys())

    # Validation set for early stopping
    targets_valid = np.zeros((data_idxs_valid.shape[0],n_entities))
    for idx,ent_id in enumerate(data_idxs_valid[:,-1]):
        targets_valid[idx,ent_id] = 1
    targets_valid = torch.FloatTensor(targets_valid).to(device)

    # Init params
    model.train()
    losses = []
    loss_valid_all =[]

    for i in range(n_iter):
        loss_batch =  []
        loss_valid = []

        # Compute validation loss
        for j in range(0,len(targets_valid)//32):
            data_j = torch.tensor(data_idxs_valid[j*32:(j+1)*32]).to(device)
            pred_valid = model.forward(data_j[:,0],data_j[:,1],data_j[:,2]).detach()
            loss_valid_j = model.loss(pred_valid,targets_valid[j*32:(j+1)*32]).item()
            loss_valid.append(loss_valid_j)
        
        data_j = torch.tensor(data_idxs_valid[(j+1)*32:]).to(device)
        pred_valid = model.forward(data_j[:,0],data_j[:,1],data_j[:,2]).detach()
        loss_valid_j = model.loss(pred_valid,targets_valid[(j+1)*32:]).item()
        loss_valid.append(loss_valid_j)
        


        loss_valid_all.append(np.mean(loss_valid))

        for j in range(0, len(er_vocab_pairs), batch_size):
            data_batch, targets = get_batch(batch_size,er_vocab, er_vocab_pairs, j,n_entities=n_entities,device=device)

            opt.zero_grad()
            e1_idx = torch.tensor(data_batch[:,0]).to(device)
            r_idx = torch.tensor(data_batch[:,1]).to(device)
            t_idx = torch.tensor(data_batch[:,2]).to(device)

            predictions = model.forward(e1_idx, r_idx,t_idx)
            loss = model.loss(predictions, targets)

            
            loss.backward()
            opt.step()
            loss_batch.append(loss.item())
        
        losses.append(np.mean(loss_batch))

        if i % print_loss_every == 0 :
            print(f"{i+1}/{n_iter} loss = {losses[-1]}")#, valid loss = {loss_valid}")

        # Early Stopping 
        # if i > early_stopping : 
        #     if min(loss_valid_all[-early_stopping:]) < min(loss_valid_all):
        #         print(f"{i}/{n_iter} loss = {losses[-1]}, valid loss = {loss_valid}")
        #         break

    model.eval()

    return model