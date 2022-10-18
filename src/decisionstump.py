import datetime
from time import time

import numpy as np


NUM_STEPS = 10
EPS = 1e-10


class DecisionStump:
    def __init__(self, 
                 prior, 
                 weight, 
                 HX_u, 
                 HX_p, 
                 unlabeled_losses, 
                 random_selection, 
                 nnpu, 
                 beta, 
                 id):

        self.beta = beta
        self.id = id
        
        self.inequal = 'lt'
        self.feature_idx = 0.0
        self.threshold = 0.0
        self.alpha = 1.0
        self.run_time = 0.0
        
        self.random_selection = random_selection
        self.nnpu = nnpu

        self.prior = prior

        self.min_error = float("inf")
        self.max_estimation = float("-inf")

        self.miss = {}
        self.weight = weight

        self.HX_u = HX_u
        self.HX_p = HX_p
        self.unlabeled_losses = unlabeled_losses

        self.best_p = None
        self.best_u = None

    def get_target(self, num_p, num_u):
        y_train_p = np.ones(num_p)
        y_train_n = np.ones(num_p)
        y_train_u = np.ones(num_u)
        y_train_n[y_train_n > 0] = -1
        y_train_u[y_train_u > 0] = -1

        return y_train_p, y_train_n, y_train_u
    
    
    def create_seed(self):
        current_time = datetime.datetime.now()
        current_time = current_time.microsecond
        
        np.random.seed(current_time)
        random_seed = np.random.uniform(low=0, high=1, size=None)
        
        return random_seed


    def predict(self, x_train, feature_idx, inequal, threshold):
        """
        The predict function of decision stump.

        :param x_train: training data
        :return: The result of prediction (Numpy array)
        """
        x_column = x_train[:, feature_idx]
        predictions = np.ones(x_train.shape[0])

        if inequal == 'lt':
            predictions[x_column <= threshold] = -1
        else:
            predictions[x_column > threshold] = -1

        return predictions


    def build(self, x_train_p, x_train_u):
        """
        Processing of building stump

        :param x_train_p: training positive data
        :param x_train_u: training unlabeled data
        :param y_train: label of unlabeled data
        :return: None, update the parameter of decision stump
        """
        begin_time = time()

        if len(x_train_p.shape) >= 3:
            n_samples_p = x_train_p.shape[0]
            n_samples_u = x_train_u.shape[0]
            n_features = x_train_p.shape[1] * x_train_p.shape[2] * x_train_p.shape[3]
        else:
            n_samples_p = x_train_p.shape[0]
            n_samples_u = x_train_u.shape[0]
            n_features = x_train_p.shape[1]

        y_train_p, y_train_n, y_train_u = self.get_target(n_samples_p, n_samples_u)

        self.best_p = y_train_p
        self.best_u = y_train_u

        for feature_i in range(n_features):

            range_min = x_train_u[:, feature_i].min()
            range_max = np.max(x_train_u[:, feature_i])
            step_size = (range_max - range_min) / NUM_STEPS

            for i in range(NUM_STEPS + 1):
                for inequal in ['lt', 'gt']:

                    if self.random_selection == 1:
                        threshold = self.create_seed() * (range_max - range_min) + range_min
                    else:
                        threshold = min(range_min + float(i) * step_size, range_max)

                    curr_pred_p = self.predict(x_train_p, feature_i, inequal, threshold)
                    curr_pred_u = self.predict(x_train_u, feature_i, inequal, threshold)

                    HX_u = self.HX_u
                    HX_p = self.HX_p

                    # Calculate the global error
                    error, error_n, error_p = self.sum_error_calc(curr_pred_p, curr_pred_u)

                    if self.nnpu == 1:  # nnpu
                        if error_n < 0:
                            continue
                    else:               # upu
                        if error < 0:
                            continue

                    if error >= 0.5:
                        continue

                    if len(self.unlabeled_losses) == 0:
                        estimation = self.max_estimation
                        if error < self.min_error:
                            self.max_estimation = estimation
                            self.min_error = error

                            self.inequal = inequal
                            self.threshold = threshold
                            self.feature_idx = feature_i

                            self.best_p = curr_pred_p
                            self.best_u = curr_pred_u
                    else:
                        estimation = self.target_function(self.prior, n_samples_p, n_samples_u, HX_p, HX_u, curr_pred_p, curr_pred_u)

                        if estimation > self.max_estimation:
                            self.max_estimation = estimation
                            self.min_error = error

                            self.inequal = inequal
                            self.threshold = threshold
                            self.feature_idx = feature_i

                            self.best_p = curr_pred_p
                            self.best_u = curr_pred_u

        self.alpha = 0.5 * np.log((1.0 - self.min_error + EPS) / (self.min_error + EPS))

        HX_u = HX_u * self.beta + np.multiply(self.alpha, self.best_u)
        HX_p = HX_p * self.beta + np.multiply(self.alpha, self.best_p)
        
        self.HX_u = HX_u
        self.HX_p = HX_p

        end_time = time()
        self.run_time = end_time - begin_time


    def target_function(self, 
                        prior: float, 
                        num_pos: int, 
                        num_unlabeled: int, 
                        y_preds_pos: np.array, 
                        y_preds_unlabeled: np.array, 
                        curr_pred_pos: np.array,
                        curr_pred_unlabeled: np.array):
        
        term_pos_pos = (prior / num_pos) * np.sum(np.exp(-1 * y_preds_pos) * curr_pred_pos)
        term_pos_neg = (prior / num_pos) * np.sum(np.exp(y_preds_pos) * curr_pred_pos)
        term_unlabeled = (1 / num_unlabeled) * (np.sum(np.exp(y_preds_unlabeled) * curr_pred_unlabeled))
     
        return term_pos_pos + term_pos_neg - term_unlabeled


    def sum_error_calc(self, curr_pred_p: np.array, curr_pred_u: np.array):
        error_pn = self.prior * np.sum(self.weight['n'][curr_pred_p != -1])
        error_un = np.sum(self.weight['u'][curr_pred_u != -1])
        error_pp = self.prior * np.sum(self.weight['p'][curr_pred_p != 1])

        error = error_pp + error_un - error_pn
        error_n = error_un - error_pn
        error_p = error_pp - error_pn

        return error, error_n, error_p
