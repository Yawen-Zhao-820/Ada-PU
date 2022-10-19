import random
from random import shuffle

import numpy as np
import pandas as pd
import tensorflow as tf
from catboost.datasets import epsilon
from colorama import Back, Fore, Style
from keras.datasets import cifar10
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow import keras


"""
MNIST Handwritten Digit Classification Dataset
"""
def get_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_train = x_train.reshape(60000, 784)
    x_test = x_test / 255.0
    x_test = x_test.reshape(10000, 784)

    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)

    return (x_train, y_train), (x_test, y_test)

def binarize_mnist_class(_trainY, _testY):
    trainY = np.ones(len(_trainY), dtype=np.int32)
    trainY[_trainY % 2 == 1] = -1
    testY = np.ones(len(_testY), dtype=np.int32)
    testY[_testY % 2 == 1] = - 1
    return trainY, testY


"""
UNSW-NB15 Dataset
"""
def get_unsw():
    x_train_data = pd.read_csv('./data/UNSW/UNSW_X_train.csv')
    x_test_data = pd.read_csv('./data/UNSW/UNSW_X_test.csv')
    y_train_data = pd.read_csv('./data/UNSW/UNSW_y_train.csv')
    y_test_data = pd.read_csv('./data/UNSW/UNSW_y_test.csv')

    x_train = np.asarray(x_train_data, dtype=np.float32)
    x_test = np.asarray(x_test_data, dtype=np.float32)
    y_train = np.asarray(y_train_data, dtype=np.int32)
    y_test = np.asarray(y_test_data, dtype=np.int32)

    return (x_train, y_train), (x_test, y_test)


def get_unsw_pn():
    x_train_data = pd.read_csv('./data/UNSW/UNSW_X_train.csv')
    x_test_data = pd.read_csv('./data/UNSW/UNSW_X_test.csv')
    y_train_data = pd.read_csv('./data/UNSW/UNSW_y_train.csv')
    y_test_data = pd.read_csv('./data/UNSW/UNSW_y_test.csv')

    x_train = np.asarray(x_train_data, dtype=np.float32)
    x_test = np.asarray(x_test_data, dtype=np.float32)
    y_train = np.asarray(y_train_data, dtype=np.int32)
    y_test = np.asarray(y_test_data, dtype=np.int32)

    return (x_train, y_train), (x_test, y_test)

def binarize_unsw_class(_trainY, _testY):
    _trainY = _trainY[:, 0]
    _testY = _testY[:, 0]

    trainY = np.ones(len(_trainY), dtype=np.int32)
    trainY[_trainY == 0] = -1

    testY = np.ones(len(_testY), dtype=np.int32)
    testY[_testY == 0] = - 1

    return trainY, testY


"""
CIFAR-10 Dataset
"""
def get_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    x_train /= 255
    x_test /= 255
    x_train = np.reshape(x_train, (50000, 3072))
    x_test = np.reshape(x_test, (10000, 3072))
    y_train = np.reshape(y_train, (50000,))
    y_test = np.reshape(y_test, (10000,))

    n_feature = x_train.shape[1]
    idx_feature = np.arange(n_feature)

    r = x_train[:, idx_feature % 3 == 0]
    g = x_train[:, idx_feature % 3 == 1]
    b = x_train[:, idx_feature % 3 == 2]
    x_train = np.concatenate((r, g, b), axis=1)

    r_t = x_test[:, idx_feature % 3 == 0]
    g_t = x_test[:, idx_feature % 3 == 1]
    b_t = x_test[:, idx_feature % 3 == 2]
    x_test = np.concatenate((r_t, g_t, b_t), axis=1)

    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)

    return (x_train, y_train), (x_test, y_test)

def binarize_cifar10_class(y_train, y_test):
    y_train_bin = np.ones(y_train.shape[0], dtype=np.int32)
    y_train_bin[
        (y_train == 2) | (y_train == 3) | (y_train == 4) | (y_train == 5) | (y_train == 6) | (y_train == 7)] = -1
    y_test_bin = np.ones(y_test.shape[0], dtype=np.int32)
    y_test_bin[(y_test == 2) | (y_test == 3) | (y_test == 4) | (y_test == 5) | (y_test == 6) | (y_test == 7)] = -1
    return y_train_bin, y_test_bin


"""
Breast Cancer Dataset
"""
def get_breastcancer():
    data = datasets.load_breast_cancer()
    x, y = data.data, data.target

    y[y == 0] = -1

    x_train = x[0:455]
    y_train = y[0:455]
    x_test = x[455:]
    y_test = y[455:]

    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)

    print(len(y_train))
    print(len(y_test))
    
    return (x_train, y_train), (x_test, y_test)

def get_breastcancer_pn():
    data = datasets.load_breast_cancer()
    x, y = data.data, data.target

    x_train = x[0:455]
    y_train = y[0:455]
    x_test = x[455:]
    y_test = y[455:]

    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)

    return (x_train, y_train), (x_test, y_test)


"""
Epsilon Dataset
"""
def get_epsilon():
    epsilon_train, epsilon_test = epsilon()
    data_train = np.asarray(epsilon_train, dtype=np.float32)
    data_test = np.asarray(epsilon_test, dtype=np.float32)
    num_data, num_feature = data_train.shape

    x_train = np.asarray(data_train[:, 1: num_feature], dtype=np.float32)
    x_test = np.asarray(data_test[:, 1: num_feature], dtype=np.float32)
    y_train = np.asarray(data_train[:, 0], dtype=np.int32)
    y_test = np.asarray(data_test[:, 0], dtype=np.int32)

    return (x_train, y_train), (x_test, y_test)


def get_pn(x_train, y_train, prior, num_pos, seed):
    np.random.seed(seed)

    x_train_pos = x_train[y_train == 1]
    x_train_neg = x_train[y_train == -1]

    num_neg = (1 - prior) / (prior * 2)
    num_neg = num_neg ** 2
    num_neg = int(num_neg * num_pos)
    
    pos_idx = np.random.choice(x_train_pos.shape[0], num_pos, replace=False)
    neg_idx = np.random.choice(x_train_neg.shape[0], num_neg, replace=False)

    if len(x_train.shape) == 1:
        x_train_p = x_train_pos[pos_idx]
        x_train_n = x_train_neg[neg_idx]
        y_train_p = np.ones(x_train_p.shape[0])
        y_train_n = np.ones(x_train_n.shape[0])
        y_train_n[y_train_n > 0] = -1
    else:
        x_train_p = x_train_pos[pos_idx, :]
        x_train_n = x_train_neg[neg_idx, :]
        y_train_p = np.ones(x_train_p.shape[0])
        y_train_n = np.ones(x_train_n.shape[0])
        y_train_n[y_train_n > 0] = -1

    x_train_ = np.concatenate((x_train_p, x_train_n), axis=0)
    y_train_ = np.concatenate((y_train_p, y_train_n), axis=0)

    return x_train_, y_train_


def make_dataset(dataset, n_labeled, n_unlabeled, seed):
    def make_pu_dataset_from_binary_dataset(x, y, seed, labeled=n_labeled, unlabeled=n_unlabeled):
        np.random.seed(seed)

        n_labeled_pos = labeled
        n_unlabeled = unlabeled

        idx_pos = np.where(y == 1)[0]
        num_pos = len(idx_pos)

        perm_pos = np.random.permutation(num_pos)
        selected_pos = idx_pos[perm_pos][0: n_labeled_pos]
        x_pos = x[selected_pos]
        
        perm = np.random.permutation(len(y))

        x_unlabeled = x[perm][0: n_unlabeled]
        y_unlabeled = y[perm][0: n_unlabeled]
        prior = float((y_unlabeled == 1).sum()) / float(n_unlabeled)

        return x_pos, x_unlabeled, prior

    def make_pu_testset_from_binary_dataset(x, y):
        idx_pos = np.where(y == 1)[0]
        x_pos = x[idx_pos]

        idx_neg = np.where(y == -1)[0]
        x_neg = x[idx_neg]

        return x_pos, x_neg

    (x_train, y_train), (x_test, y_test) = dataset

    x_train_pos, x_train_unlabeled, prior = make_pu_dataset_from_binary_dataset(x_train, y_train, seed)
    x_test_pos, x_test_negative = make_pu_testset_from_binary_dataset(x_test, y_test)
    return x_train_pos, x_train_unlabeled, prior, x_test_pos, x_test_negative


def load_pu_dataset(num_labeled, num_unlabeled):
    (x_train, y_train), (x_test, y_test) = get_mnist()
    y_train, y_test = binarize_mnist_class(y_train, y_test)
    x_train_pos, x_train_unlabeled, prior, x_test_pos, x_test_neg = make_dataset(
        ((x_train, y_train), (x_test, y_test)), num_labeled, num_unlabeled)
    return x_train_pos, x_train_unlabeled, prior, x_test_pos, x_test_neg


def load_dataset(dataset_name, n_labeled, n_unlabeled, seed):

    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = get_mnist()
        y_train, y_test = binarize_mnist_class(y_train, y_test)

    elif dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = get_cifar10()
        y_train, y_test = binarize_cifar10_class(y_train, y_test)

    elif dataset_name == "epsilon":
        (x_train, y_train), (x_test, y_test) = get_epsilon()

    elif dataset_name == "breastcancer":
        (x_train, y_train), (x_test, y_test) = get_breastcancer()

    elif dataset_name == "unsw":
        (x_train, y_train), (x_test, y_test) = get_unsw()
        y_train, y_test = binarize_unsw_class(y_train, y_test)

    elif dataset_name == "cifar10_embedding":
        x_train_p = np.load('./data/CifarEmb/trainloader_positive.npy')[:,:-1]
        x_train_u = np.load('./data/CifarEmb/trainloader_unlabeled.npy')[:,:-1]

        x_test_p = np.load('./data/CifarEmb/testloader_positive.npy')[:,:-1]
        x_test_n = np.load('./data/CifarEmb/testloader_negative.npy')[:,:-1]

        prior = 0.4
        return x_train_p, x_train_u, prior, x_test_p, x_test_n

    else:
        raise ValueError("dataset name {} is unknown.".format(dataset_name))
    
    print(Fore.BLUE + "Loading data...") 
    
    x_train_pos, x_train_unlabeled, prior, x_test_pos, x_test_neg = make_dataset(
        ((x_train, y_train), (x_test, y_test)), n_labeled, n_unlabeled, seed)
    
    print(Fore.RED + "-"*30) 
    print(Fore.RED + f"{'Positive training data shape: ':^10} {x_train_pos.shape}")
    print(Fore.RED + f"{'Unlabeled training data shape: ':^10} {x_train_unlabeled.shape}")
    print(Fore.RED + "-"*30)
    print(Fore.BLUE + f"prior: {prior}")
    print(Fore.RED + "-"*30) 
    
    return x_train_pos, x_train_unlabeled, prior, x_test_pos, x_test_neg


def get_dataset_pn(num_pos, dataset_name, prior, seed):
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = get_mnist()
        y_train, y_test = binarize_mnist_class(y_train, y_test)
        x_train, y_train = get_pn(x_train, y_train, prior, num_pos, seed)

    elif dataset_name == "epsilon":
        (x_train, y_train), (x_test, y_test) = get_epsilon()
        x_train, y_train = get_pn(x_train, y_train, prior, num_pos, seed)

    elif dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = get_cifar10()
        y_train, y_test = binarize_cifar10_class(y_train, y_test)
        x_train, y_train = get_pn(x_train, y_train, prior, num_pos, seed)

    elif dataset_name == "breastcancer":
        (x_train, y_train), (x_test, y_test) = get_breastcancer_pn()
        x_train, y_train = get_pn(x_train, y_train, prior, num_pos, seed)

    elif dataset_name == "unsw":
        (x_train, y_train), (x_test, y_test) = get_unsw_pn()
        y_train, y_test = binarize_unsw_class(y_train, y_test)
        x_train, y_train = get_pn(x_train, y_train, prior, num_pos, seed)
  
    else:
        raise ValueError("dataset name {} is unknown.".format(dataset_name))

    return x_train, y_train, x_test, y_test
