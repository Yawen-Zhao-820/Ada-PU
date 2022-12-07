from time import time
import datetime
import numpy as np
import warnings

NUM_STEPS = 10
EPS = np.finfo(float).eps


class DecisionStump:
    """An Decision Stump as the weak learner of Ada-PU.
    
    This class implements the Decision Stump follow [1].
    
    .. versionadded:: 1.0.1
    
    Parameters
    ----------
    prior : float
        prior probability of positive data
    weight : dict
        samples' weight
    random_selection : int
        whether to use the random selection strategy
    ada_prediction_p : np.array
        positive data predictions (HX_positive)
    ada_prediction_u : np.array
        unlabeled data predictions (HX_unlabeled)
    beta:
        control parameter
        
    References
    ----------
    .. [1] Harrington P. Machine learning in action. Simon and Schuster, 2012.
    """
    def __init__(self, prior, weight, ada_predictions_u, ada_predictions_p, random_selection, beta):

        self.beta = beta
        self.random_selection = random_selection
        self.prior = prior
        self.weight = weight
        
        self.ada_predictions_u = ada_predictions_u
        self.ada_predictions_p = ada_predictions_p
        
        self.inequal = 'lt'
        self.feature_idx = 0.0
        self.threshold = 0.0
        self.estimator_weight = 1.0
        self.min_error = float("inf")
        self.max_estimation = float("-inf")

        self.best_prediction_p = None
        self.best_prediction_u = None
        

    def get_target(self, num_p: int, num_u: int) -> np.array:
        """Get the label of each sample.

        Args:
            num_p (_type_): number of positive samples
            num_u (_type_): number of unlabeled samples

        Returns:
            np.array: labels of data
        """
        y_train_p = np.ones(num_p)
        y_train_n = np.ones(num_p)
        y_train_u = np.ones(num_u)
        y_train_n[y_train_n > 0] = -1
        y_train_u[y_train_u > 0] = -1

        return y_train_p, y_train_n, y_train_u


    def fitted_predict(self, X):
        """Prediction function used after fitting progress.

        Args:
            X (np.array): training data  ``X``

        Returns:
            np.array: prediction results
        """
        x_column = X[:, int(self.feature_idx)]
        predictions = np.ones(X.shape[0])

        if self.inequal == 'lt':
            predictions[x_column <= self.threshold] = -1
        else:
            predictions[x_column > self.threshold] = -1

        return predictions


    def predict(self, x_train, feature_idx, inequal, threshold):
        """The predict function of decision stump.

        Args:
            x_train (np.array): training data 
            feature_idx (int): feature index
            inequal (str): ``<`` or ``>``, the side of decision tree, right/left
            threshold (float): the threshold of each feature index
            
        Returns:
            np.array: prediction results
        """
        x_column = x_train[:, int(feature_idx)]
        predictions = np.ones(x_train.shape[0])

        if inequal == 'lt':
            predictions[x_column <= threshold] = -1
        else:
            predictions[x_column > threshold] = -1

        return predictions
    

    def _get_threshold_range(self, range_min, range_max, lower_bound, higher_bound, step):
        """Get the feature's threshold

        Args:
            x_train_u (np.array): unlabeled training data
            feature_index (int): feature index
            step (int): step

        Returns:
            threshold (float): features' threshold
        """
            
        if self.random_selection == 1:
            time = datetime.datetime.now()
            time = time.microsecond
            np.random.seed(time)
            rd = np.random.uniform(low=lower_bound, high=higher_bound, size=None)
            threshold = rd * (range_max - range_min) + range_min
        else:
            lower_bound = lower_bound * (range_max - range_min) + range_min
            higher_bound = higher_bound * (range_max - range_min) + range_min
            step_size = (higher_bound - lower_bound) / NUM_STEPS
            threshold = min(lower_bound + float(step) * step_size, higher_bound)

        return threshold


    def build(self, x_train_p, x_train_u):
        """The function of building a decision stump.

        Args:
            x_train_p (np.array): positive training data 
            x_train_u (np.array): unlabeled training data 
        """

        n_samples_p = x_train_p.shape[0]
        n_samples_u = x_train_u.shape[0]
        n_features = x_train_p.shape[1]

        y_train_p, y_train_n, y_train_u = self.get_target(n_samples_p, n_samples_u)

        self.best_prediction_p = y_train_p
        self.best_prediction_u = y_train_u

        for feature_i in range(n_features):
            range_min = x_train_u[:, feature_i].min()
            range_max = np.max(x_train_u[:, feature_i])

            num_interval = len(np.unique(x_train_u[:, feature_i]))
            
            lower_bound = 0 - (1 / (num_interval-1))
            higher_bound = 1 + (1 / (num_interval-1))
            
            for step in range(NUM_STEPS + 1):

                for inequal in ['lt', 'gt']:
                    
                    threshold = self._get_threshold_range(range_min, range_max, lower_bound, higher_bound, step)
                    
                    curr_pred_p = self.predict(x_train_p, feature_i, inequal, threshold)
                    curr_pred_u = self.predict(x_train_u, feature_i, inequal, threshold)

                    ada_predictions_u = self.ada_predictions_u
                    ada_predictions_p = self.ada_predictions_p

                    # Calculate the global error
                    error, error_n, error_p = self.sum_error_calc(curr_pred_p, curr_pred_u)

                    if error_n < 0 or error < 0 or error >= 0.5:
                        continue
                    
                    estimation = self.hx_target_function(n_samples_p, 
                                                         n_samples_u, 
                                                         ada_predictions_p, 
                                                         ada_predictions_u, 
                                                         curr_pred_p, 
                                                         curr_pred_u)

                    if estimation > self.max_estimation:
                        self.max_estimation = estimation
                        self.min_error = error

                        self.inequal = inequal
                        self.threshold = threshold
                        self.feature_idx = feature_i

                        self.best_prediction_p = curr_pred_p
                        self.best_prediction_u = curr_pred_u
        
        self.estimator_weight = 0.5 * np.log((1.0 - self.min_error + EPS) / (self.min_error + EPS))
        self.estimator_weight *= self.beta

        ada_predictions_u = ada_predictions_u + np.multiply(self.estimator_weight, self.best_prediction_u)
        ada_predictions_p = ada_predictions_p + np.multiply(self.estimator_weight, self.best_prediction_p)

        self.ada_predictions_u = ada_predictions_u
        self.ada_predictions_p = ada_predictions_p


    def hx_target_function(self, num_pos, num_unlabeled, y_preds_pos, y_preds_unlabeled, curr_pred_pos,
                        curr_pred_unlabeled):
        """Calculating the target E_t. 

        Args:
            num_pos (int): positive samples' number
            num_unlabeled (int): unlabeled samples' number
            y_preds_pos (np.array): ada predict results of positive data (Hx)
            y_preds_unlabeled (np.array): ada predict results of unlabeled data (Hx)
            curr_pred_pos (np.array): current predict results of positive data
            curr_pred_unlabeled (np.array): current predict results of unlabeled data

        Returns:
            E_t: the target value for selecting stump in each boost iteration
        """

        term_pos_pos = (self.prior / num_pos) * np.sum(np.exp(-1 * y_preds_pos) * curr_pred_pos)
        term_pos_neg = (self.prior / num_pos) * np.sum(np.exp(y_preds_pos) * curr_pred_pos)
        term_unlabeled = (1 / num_unlabeled) * (np.sum(np.exp(y_preds_unlabeled) * curr_pred_unlabeled))
        
        return term_pos_pos + term_pos_neg - term_unlabeled


    def sum_error_calc(self, curr_pred_p, curr_pred_u):
        """Calculating the error of each stump. 

        Args:
            curr_pred_p (np.array): current predict results of positive data
            curr_pred_u (np.array): current predict results of unlabeled data

        Returns:
            current estimator's error
        """
        error_pn = np.sum(self.weight['n'][curr_pred_p != -1])
        error_pn = self.prior * error_pn / np.sum(self.weight['n'])

        error_un = np.sum(self.weight['u'][curr_pred_u != -1])
        error_un = error_un / np.sum(self.weight['u'])

        error_pp = np.sum(self.weight['p'][curr_pred_p != 1])
        error_pp = self.prior * error_pp / np.sum(self.weight['p'])

        error = error_pp + error_un - error_pn
        error_n = error_un - error_pn
        error_p = error_pp - error_pn

        return error, error_n, error_p
