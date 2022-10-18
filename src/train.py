import argparse
import copy
import json
import platform
from types import SimpleNamespace

import cupy as np
import scipy.io as scio
from colorama import Back, Fore, Style
from tqdm import tqdm

import load_data as D
from adapu import Adaboost
from utils import *


DATASET = {"mnist":             {"num_pos": 1000, "num_unlabeled": 60000},
           "breastcancer":      {"num_pos": 10,   "num_unlabeled": 455},
           "epsilon":           {"num_pos": 1000, "num_unlabeled": 40000},
           "cifar10":           {"num_pos": 1000, "num_unlabeled": 50000},
           "cifar10_embedding": {"num_pos": 1000, "num_unlabeled": 50000},
           "unsw":              {"num_pos": 1000, "num_unlabeled": 175340}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='breastcancer')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_clf', type=int, default=100)
    parser.add_argument('--nnpu', type=int, default=1)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--random', type=int, default=1)
    args = parser.parse_args()

    training_set = {
        'num_pos': 0,
        'num_unlabeled': 0,
        'seed': 1,
        'random_selection': 0,
        'dataset_name': '',
        'num_clf': 100,
        'nnpu': 1,
        'beta': 1
    }

    training_set['random_selection'] = int(args.random)
    training_set['seed'] = int(args.seed)
    training_set['dataset_name'] = args.dataset
    training_set['num_pos'] = DATASET[args.dataset]["num_pos"]
    training_set['num_unlabeled'] = DATASET[args.dataset]["num_unlabeled"]
    training_set['num_clf'] = int(args.num_clf)
    training_set['nnpu'] = int(args.nnpu)
    training_set['beta'] = args.beta

    x_train_pos, x_train_unlabeled, prior, x_test_pos, x_test_neg = \
    D.load_dataset(training_set['dataset_name'], 
                   training_set['num_pos'], 
                   training_set['num_unlabeled'],
                   training_set['seed'])

    NUM_CLF = training_set['num_clf']
    
    model = Adaboost(NUM_CLF, 
                     prior, 
                     training_set['random_selection'], 
                     training_set['nnpu'],
                     training_set['beta'])
    
    print(Fore.BLUE + "Training model...")
    print(Fore.BLUE + "-"*30)
    print("\n")
    model.fit(x_train_pos, x_train_unlabeled)

    training_losses = []
    
    test_accuracy = []
    test_f1 = []
    test_recall = []
    test_precision = []
    
    threshold_list = []
    features_list = []

    print("\n")
    print(Fore.BLUE + "Evaluating...")
    print(Fore.BLUE + "-"*30)
    print("\n")
    print(Fore.RED + f"{'#classifiers':^7} | {'Train Loss':^12} | {'Test Acc':^10} | {'Test F1':^9}")
    print(Fore.RED + "-"*52)
    for i in range(1, NUM_CLF + 1):
        y_pred_p = model.predict(x_train_pos, i)
        y_pred_n = model.predict(x_train_unlabeled, i)
        
        y_pred_p_test = model.predict(x_test_pos, i)
        y_pred_n_test = model.predict(x_test_neg, i)
        
        training_loss = zero_one_training_loss(prior, y_pred_p, y_pred_n)
        training_losses.append(training_loss)
        
        pu_accuracy = accuracy(y_pred_p_test, y_pred_n_test)
        pu_f1, pu_precision, pu_recall = F1(y_pred_p_test, y_pred_n_test)
        
        test_accuracy.append(pu_accuracy)
        test_f1.append(pu_f1)
        test_precision.append(pu_precision)
        test_recall.append(pu_recall)

        threshold_list.append(model.clfs[i - 1].threshold)
        features_list.append(model.clfs[i - 1].feature_idx)
        
        if i % 10 == 0:
            print(Fore.RED + f"{i:^12} | {training_loss:^12.6f} | {pu_accuracy:^10.6f} | {pu_f1:^9.2f}")
            print(Fore.RED + "-"*52)
            
    print("\n")
    print(Fore.BLUE + "Save results...")
    print(Fore.BLUE + "-"*52)
    print(Fore.RED + "Finished!!!")
    savename = f'./results/{args.dataset}_{args.beta}_{round}.mat'
    scio.savemat(savename,
                {'accuracy':test_accuracy,
                'prior':prior,
                'f1': test_f1,
                'precision': test_precision,
                'recall': test_recall,
                'training_loss': training_losses,
                'threshold': threshold_list,
                'feature': features_list})
