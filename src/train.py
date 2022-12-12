import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from colorama import Back, Fore, Style
from tqdm import tqdm

import src.load_data as D
from src.AdaPU import AdaBoost_PU
from src.utils import *

DATASET = {"breastcancer": {"num_pos": 10,   "num_unlabeled": 455},
           "epsilon":      {"num_pos": 1000, "num_unlabeled": 40000},
           "cifar10_emb":  {"num_pos": 1000, "num_unlabeled": 50000},
           "unsw":         {"num_pos": 1000, "num_unlabeled": 175340}}
    
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='breastcancer')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_estimator', type=int, default=100)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--random', type=int, default=1)
    args = parser.parse_args()

    x_train_pos, x_train_unlabeled, prior, x_test_pos, x_test_neg = \
    D.load_dataset(args.dataset, 
                   DATASET[args.dataset]['num_pos'], 
                   DATASET[args.dataset]['num_unlabeled'],
                   args.seed)
    
    NUM_ESTIMATOR = int(args.num_estimator)
    
    model = AdaBoost_PU(NUM_ESTIMATOR, 
                        prior, 
                        args.random, 
                        args.beta)

    print(Fore.BLUE + "Training model...")
    print(Fore.BLUE + "-"*30)
    print("\n")
    model.fit(x_train_pos, x_train_unlabeled)

    print("\n")
    print(Fore.BLUE + "Evaluating...")
    print(Fore.BLUE + "-"*30)
    print("\n")
    print(Fore.GREEN + f"{'#classifiers':^7} | {'Acc':^10}")
    print(Fore.GREEN + "-"*52)
    
    for i in range(1, NUM_ESTIMATOR + 1):
        y_pred_p = model.staged_predict(x_train_pos, i)
        y_pred_n = model.staged_predict(x_train_unlabeled, i)
        
        pu_accuracy = accuracy(y_pred_p_test, y_pred_n_test)
        
        print(Fore.GREEN + f"{i:^12} | {pu_accuracy:^10.6f}")
        print(Fore.GREEN + "-"*52)

    print(Fore.BLUE + "Save results...")
    print(Fore.BLUE + "-"*52)
    print(Fore.RED + "Finished!!!")
    
    # savename = f'./results/{args.dataset}_{args.beta}_{args.random}_{args.seed}.mat'
    # scio.savemat(savename,
    #             {'accuracy':test_accuracy,
    #             'prior':prior,
    #             'f1': test_f1,
    #             'precision': test_precision,
    #             'recall': test_recall,
    #             'training_loss': training_losses,
    #             'nn_training_loss': nn_training_losses,
    #             'exp_training_loss': exp_training_losses,
    #             'nn_exp_training_loss': nn_exp_training_losses,
    #             'threshold': threshold_list,
    #             'feature': features_list})
