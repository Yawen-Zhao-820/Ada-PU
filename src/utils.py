import numpy as np


def accuracy(output_p, output_n):
    """
    Calculate the accuracy.

    :param y_true: Original data
    :param y_pred: Prediction data
    :return: accuracy rate
    """
    true_positive = np.sum((np.sign(output_p) + 1) / 2)
    true_negative = np.sum((np.sign(-output_n) + 1) / 2)
    all = len(output_p) + len(output_n)
    if true_positive == 0 and true_negative == 0:
        accuracy = 0
    else:
        accuracy = float(true_positive + true_negative) / float(all)
    return accuracy


def accuracy_pn(y_pred, y_true):
    """
    Calculate the accuracy.

    :param y_true: Original data
    :param y_pred: Prediction data
    :return: accuracy rate
    """
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def F1(output_p, output_n):
    true_positive = np.sum((np.sign(output_p) + 1) / 2)
    all_predicted_positive = np.sum((np.sign(output_p) + 1) / 2) + np.sum((np.sign(output_n) + 1) / 2)
    all_real_positive = len(output_p)
    if all_predicted_positive == 0:
        precision = 0
    else:
        precision = float(true_positive) / float(all_predicted_positive)
    recall = float(true_positive) / float(all_real_positive)
    if precision == 0 or recall == 0:
        F1 = 0
    else:
        F1 = 2 * precision * recall / (precision + recall)
    return F1, precision, recall


def zero_one_training_loss(prior, y_pred_p, y_pred_n):
    cost_p = prior * np.mean((1 - np.sign(y_pred_p)) / 2)
    cost_n = np.mean((1 + np.sign(y_pred_n)) / 2)
    cost_p_ = prior * np.mean((1 + np.sign(y_pred_p)) / 2)
    cost = cost_p + cost_n - cost_p_
    return cost


def nn_zero_one_training_loss(prior, y_pred_p, y_pred_n):
    cost = np.mean((1 + np.sign(y_pred_n)) / 2) - prior * np.mean((1 + np.sign(y_pred_p)) / 2)
    cost = max(cost, 0)
    cost = cost + prior * np.mean((1 - np.sign(y_pred_p)) / 2)
    return cost


def exp_training_loss(prior, y_pred_p, y_pred_n):
    cost_p = prior * np.mean(np.exp(-y_pred_p))
    cost_n = np.mean(np.exp(y_pred_n))
    cost_p_ = prior * np.mean(np.exp(y_pred_p))
    cost = cost_p + cost_n - cost_p_
    return cost


def nn_exp_training_loss(prior, y_pred_p, y_pred_n):
    cost = np.mean(np.exp(y_pred_n)) - prior * np.mean(np.exp(y_pred_p))
    cost = max(cost, 0)
    cost = cost + prior * np.mean(np.exp(-y_pred_p))
    return cost


def zero_one_test(output_p, output_n, prior):
    cost = prior * np.mean((1 - np.sign(output_p)) / 2)
    cost = cost + (1 - prior) * np.mean((1 + np.sign(output_n)) / 2)
    return cost
