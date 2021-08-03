import torch
import numpy as np
from load_data import Data
import time
from collections import defaultdict


def get_er_vocab(data):
        er_vocab = defaultdict(list)
        for quad in data:
            er_vocab[(quad[0], quad[1],quad[2])].append(quad[3])
        return er_vocab
    
def get_batch(batch_size,er_vocab, er_vocab_pairs, idx,n_entities,cuda=False):
        batch = er_vocab_pairs[idx:idx+batch_size]
        # targets = np.zeros((len(batch), len(data.entities)))
        targets = np.zeros((len(batch), n_entities))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if cuda:
            targets = targets.cuda()
        return np.array(batch), targets

def train_temporal(model,data_idxs,n_iter=200,learning_rate=0.0005,batch_size=128,print_loss_every=1,early_stopping=20):
    ''' Train a TuckERT model

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
    '''
    if early_stopping == False : 
        early_stopping = n_iter + 1
        
    model.init()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    er_vocab = get_er_vocab(data_idxs)
    er_vocab = {k: v for (k, v) in er_vocab.items() if (type(k) is list or type(k) is tuple) and len(k) == 3}

    n_entities = int(max(data_idxs[:,0])+1)

    er_vocab_pairs = list(er_vocab.keys())

    # Init params
    model.train()
    losses = []

    for i in range(n_iter):

        for j in range(0, len(er_vocab_pairs), batch_size):
            data_batch, targets = get_batch(batch_size,er_vocab, er_vocab_pairs, j,n_entities=n_entities)

            opt.zero_grad()
            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])  
            t_idx = torch.tensor(data_batch[:,2])

            predictions = model.forward(e1_idx, r_idx,t_idx)
            loss = model.loss(predictions, targets)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        if i % print_loss_every == 0 :
            print(f"{i}/{n_iter} loss = {np.mean(losses)}")

        # Early Spotting (#TODO on validation set ????)
        if i > early_stopping : 
            if losses[-early_stopping] >= max(losses[-early_stopping:]):
                print(f"{i}/{n_iter} loss = {np.mean(losses)}")
                break



    model.eval()

    return model