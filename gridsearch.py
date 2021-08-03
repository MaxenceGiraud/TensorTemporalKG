import numpy as np
import torch
from tuckert import TuckERT
from tuckerttr import TuckERTTR
from tuckercpd import TuckERCPD
from train import train_temporal

from load_data import Data
from sklearn.model_selection import ParameterGrid

from metrics import get_ranks,compute_MRR,compute_hits

import pandas as pd


def grid_search(model,data,param_model_grid,learning_grid):
    # Get param list to try
    param_model_list = list(ParameterGrid(param_model_grid))
    learning_list = list(ParameterGrid(learning_grid))
    
    score_table = []

    for i_param_model in param_model_list : 
        for i_learning_param in learning_list : 

            # Create model
            modeli = model(d=data,**i_param_model)

            # Train 
            modeli = train_temporal(modeli,data.train_data_idxs,**i_learning_param)

            # Compute MRR
            print("train ranks")
            train_ranks = get_ranks(modeli,torch.tensor(data.train_data_idxs),torch.tensor(data.train_data_idxs[:,-1]))
            print('mrr + hits')
            train_mrr = compute_MRR(train_ranks)
            train_hits1 = compute_hits(train_ranks,1)
            train_hits3 = compute_hits(train_ranks,1)
            train_hits10 = compute_hits(train_ranks,1)

            test_ranks = get_ranks(modeli,torch.tensor(data.test_data_idxs),torch.tensor(data.test_data_idxs[:,-1]))
            test_mrr = compute_MRR(test_ranks)
            test_hits1 = compute_hits(test_ranks,1)
            test_hits3 = compute_hits(test_ranks,1)
            test_hits10 = compute_hits(test_ranks,1)

            score_table.append([i_param_model,i_learning_param,train_mrr,train_hits1,train_hits3,train_hits10,test_mrr,test_hits1,test_hits3,test_hits10])
    
    
    return score_table


def main():
    # load data and preprocess
    data = Data() 

    models = [TuckERTTR,TuckERCPD]


    # TODO WHAT VALUES FOR de, dr, dt ??????????
    # parameter_grid_tr = {'de':[10],'dr':[10],'dt':[10],'ranks':np.arange(5,10,dtype=int),'cuda':[False],"input_dropout":[0.],"hidden_dropout1":[0.],"hidden_dropout2":[0.]}
    # parameter_grid_cpd = {'de':[10],'dr':[10],'dt':[10],'cuda':[False],"input_dropout":[0.],"hidden_dropout1":[0.],"hidden_dropout2":[0.]}

    # learning_param_grid = {'learning_rate':[0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001],'n_iter':[1000],'batch_size':[128]}


    ### TESTS ##############################
    parameter_grid_tr = {'de':[10],'dr':[10],'dt':[10],'ranks':[5],'cuda':[False],"input_dropout":[0.],"hidden_dropout1":[0.],"hidden_dropout2":[0.]}
    parameter_grid_cpd = {'de':[10],'dr':[10],'dt':[10],'cuda':[False],"input_dropout":[0.],"hidden_dropout1":[0.],"hidden_dropout2":[0.]}
    learning_param_grid = {'learning_rate':[0.0001],'n_iter':[1],'batch_size':[128]}
    ##############################


    print("------------------------------------------------------\n Traing TuckER with Tensor Ring core decomposition")
    scores_tr = grid_search(TuckERTTR,data,parameter_grid_tr,learning_param_grid)

    cols = columns=["Model Parameters","Learning Parameters","Train MRR", "Train Hits@1", "Train Hits@3", "Train Hits@10","Test MRR", "Test Hits@1", "Test Hits@3", "Test Hits@10"]

    print("Saving to csv")
    pd.DataFrame(scores_tr,columns=cols).to_csv("TR.csv")

    print("------------------------------------------------------\n Traing TuckER with CPD on core")
    scores_cpd = grid_search(TuckERCPD,data,parameter_grid_cpd,learning_param_grid)

    pd.DataFrame(scores_tr,columns=cols).to_csv("CPD.csv")


if __name__ == "__main__":
    main()
