import torch
import numpy np

def get_ranks(model,torch_data_idxs,true,batch_size=128):
    ranks=[]

    for i in range(torch_data_idxs.shape[0]//batch_size):
        data_batch = torch_data_idxs[i*batch_size:(i+1)*batch_size]
        predictions = model.forward(data_batch[:,0],data_batch[:,1],data_batch[:,2])

        _, sort_idxs = torch.sort(predictions, dim=1, descending=True)
        for j in range(predictions.shape[0]):
            rank = np.where(sort_idxs[j]==true[i*batch_size + j])[0][0]
            ranks.append(rank+1)

    data_batch = torch_data_idxs[(i+1)*batch_size:]
    predictions = model.forward(data_batch[:,0],data_batch[:,1],data_batch[:,2])

    _, sort_idxs = torch.sort(predictions, dim=1, descending=True)
    for j in range(predictions.shape[0]):
        rank = np.where(sort_idxs[j]==true[(i+1)*batch_size + j])[0][0]
        ranks.append(rank+1)
    
    return ranks


def compute_MRR(ranks):
    return np.mean(1/np.array(ranks))

def compute_hits(ranks,n):
    return len(np.where(np.array(ranks)<=n)[0])/len(ranks)
