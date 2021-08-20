import numpy as np
import torch
from models.tuckert import TuckERT
from models.tuckerttr import TuckERTTR
from models.tuckercpd import TuckERCPD
from train import train_temporal

from load_data import Data
from sklearn.model_selection import ParameterGrid

from metrics import get_ranks,compute_MRR,compute_hits

import pandas as pd

import time


def grid_search(model,data,param_model_grid,learning_grid,file,cols):
    # Get param list to try
    param_model_list = list(ParameterGrid(param_model_grid))
    learning_list = list(ParameterGrid(learning_grid))

    n_training = len(param_model_list)*len(learning_list)
    
    score_table = []

    for i_param_model in param_model_list : 
        
        for i_learning_param in learning_list : 

            t_start = time.time()

            # Create model
            modeli = model(d=data,**i_param_model)

            # Train 
            modeli = train_temporal(modeli,data.train_data_idxs,torch.tensor(data.valid_data_idxs).to(i_learning_param['device']),**i_learning_param)

            # Compute MRR
            print("train ranks")
            train_ranks = get_ranks(modeli,torch.tensor(data.train_data_idxs),torch.tensor(data.train_data_idxs[:,-1]),device=i_learning_param['device'])
            print('mrr + hits')
            train_mrr = compute_MRR(train_ranks)
            train_hits1 = compute_hits(train_ranks,1)
            train_hits3 = compute_hits(train_ranks,3)
            train_hits10 = compute_hits(train_ranks,10)

            test_ranks = get_ranks(modeli,torch.tensor(data.test_data_idxs),torch.tensor(data.test_data_idxs[:,-1]),device=i_learning_param['device'])
            test_mrr = compute_MRR(test_ranks)
            test_hits1 = compute_hits(test_ranks,1)
            test_hits3 = compute_hits(test_ranks,3)
            test_hits10 = compute_hits(test_ranks,10)

            total_time = time.time()-t_start

            score_table.append([i_param_model,i_learning_param,train_mrr,train_hits1,train_hits3,train_hits10,test_mrr,test_hits1,test_hits3,test_hits10,total_time])

            pd.DataFrame(score_table,columns=cols).to_csv(file)


    
    
    return score_table


def main():
    # load data and preprocess
    data = Data() 

    DEVICE = 'cuda' #'cpu' or 'cuda'

        cud = True
    else :
        cud = False 


    # Parameter Grid to test
    parameter_grid_tr = {'de':np.linspace(40,400,4,dtype=int),'dr':[20,50],'dt':[20,50],'ranks':[10],'cuda':[cud],"input_dropout":[0.],"hidden_dropout1":[0.],"hidden_dropout2":[0.]}
    parameter_grid_cpd = {'de':np.linspace(40,400,4,dtype=int),'dr':[20,50],'dt':[20,50],'cuda':[cud],"input_dropout":[0.],"hidden_dropout1":[0.],"hidden_dropout2":[0.]}

    learning_param_grid = {'learning_rate':[0.2,0.05,0.01,0.005,0.001],'n_iter':[100],'batch_size':[128],'device':[DEVICE]} 


    cols = ["Model Parameters","Learning Parameters","Train MRR", "Train Hits@1", "Train Hits@3", "Train Hits@10","Test MRR", "Test Hits@1", "Test Hits@3", "Test Hits@10","Time"]



    ### TESTS ##############################
    # parameter_grid_tr = {'de':[2,2],'dr':[2],'dt':[2],'ranks':[2],'cuda':[cud],"input_dropout":[0.],"hidden_dropout1":[0.],"hidden_dropout2":[0.]}
    # parameter_grid_cpd = {'de':[2],'dr':[2],'dt':[2],'cuda':[cud],"input_dropout":[0.],"hidden_dropout1":[0.],"hidden_dropout2":[0.]}
    # learning_param_grid = {'learning_rate':[0.05],'n_iter':[5],'batch_size':[128],'device':[DEVICE]}
    ##############################


    print("------------------------------------------------------\n Traning TuckER with Tensor Ring core decomposition")
    scores_tr = grid_search(TuckERTTR,data,parameter_grid_tr,learning_param_grid,file='TR.csv',cols=cols)  

    print("------------------------------------------------------\n Training TuckER with CPD on core")
    scores_cpd = grid_search(TuckERCPD,data,parameter_grid_cpd,learning_param_grid,file="CPD.csv",cols=cols)


if __name__ == "__main__":
    main()
