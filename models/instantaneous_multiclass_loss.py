import torch

def instantaneous_multiclass_loss(input,target):
    target = torch.tensor(target,dtype=bool)
    return torch.sigmoid(-input[target].sum() + torch.log(torch.exp(input[~target]).sum()))