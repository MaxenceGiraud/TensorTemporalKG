from models.tuckert import TuckERT
from models.tuckerttr import TuckERTTR
from models.tuckercpd import TuckERCPD
from models.tuckertnt import TuckERTNT

import torch

from train import train_temporal
from load_data import Data

from metrics import get_ranks,compute_MRR,compute_hits

import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TuckERCPD", nargs="?",
                    help="Which model to use: TuckERT, TuckERTTR,TuckERCPD.")
    parser.add_argument("--dataset", type=str, default="icews14", nargs="?",
                    help="Which dataset to use: icews14, icews05-15.")
    parser.add_argument("--n_iter", type=int, default=200, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.001, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--de", type=int, default=50, nargs="?",
                    help="Entity embedding dimensionality.")
    parser.add_argument("--dr", type=int, default=50, nargs="?",
                    help="Relation embedding dimensionality.")
    parser.add_argument("--dt", type=int, default=50, nargs="?",
                    help="Temporal embedding dimensionality.")
    parser.add_argument("--ranks", type=int, default=10, nargs="?",
                    help="Ranks of tensor for TR model.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--early_stopping", type=int, default=False, nargs="?",
                    help="Early stopping value")
    parser.add_argument("--input_dropout", type=float, default=0., nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0., nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0., nargs="?",
                    help="Dropout after the second hidden layer.")
    # parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                    # help="Amount of label smoothing.")

    args = parser.parse_args()

    return args

def train_model_from_args(args,print_scores=True):

    data_dir = "data/" + args.dataset + "/"
    data = Data(data_dir=data_dir)

    model =  globals()[args.model](d=data,**vars(args))

    # TODO rewrite train with cuda and not device
    if args.cuda :
        device = 'cuda'
    else :
        device = 'cpu'

    print("\n----------------------------------- TRAINING -------------------------")
    model = train_temporal(model,data,device=device,n_iter=args.n_iter,learning_rate=args.learning_rate,batch_size=args.batch_size,early_stopping=args.early_stopping)


    print("\n----------------------------------- Metrics --------------------------\n")
    # TODO rewrite get_ranks with cuda and not device
    # Compute metrics
    train_ranks = get_ranks(model,torch.tensor(data.train_data_idxs),torch.tensor(data.train_data_idxs[:,-1]),device=device,batch_size=args.batch_size)
    train_mrr = compute_MRR(train_ranks)
    train_hits1 = compute_hits(train_ranks,1)
    train_hits3 = compute_hits(train_ranks,3)
    train_hits10 = compute_hits(train_ranks,10)

    test_ranks = get_ranks(model,torch.tensor(data.test_data_idxs),torch.tensor(data.test_data_idxs[:,-1]),device=device,batch_size=args.batch_size)
    test_mrr = compute_MRR(test_ranks)
    test_hits1 = compute_hits(test_ranks,1)
    test_hits3 = compute_hits(test_ranks,3)
    test_hits10 = compute_hits(test_ranks,10)

    if print_scores : 
        print(f"Train\n MRR : {train_mrr}, Hits@1 : {train_hits1}, Hits@3 : {train_hits3}, Hits@10 : {train_hits10}\n") 
        print(f"Test\n MRR : {test_mrr}, Hits@1 : {test_hits1}, Hits@3 : {test_hits3}, Hits@10 : {test_hits10}\n") 

    return [train_mrr,train_hits1,train_hits3,train_hits10,test_mrr,test_hits1,test_hits3,test_hits10]

def main():
    args = parse()
    train_model_from_args(args,True)


if __name__ == '__main__':
    main()