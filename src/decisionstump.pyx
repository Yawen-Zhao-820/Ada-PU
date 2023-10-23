# cython: language_level=3

from time import time
import datetime
import warnings
import pandas as pd
from joblib import Memory

cimport numpy as cnp
import numpy as np

cdef int NUM_STEPS = 1
cdef float EPS = np.finfo(float).eps


cdef class DecisionStump:
    cdef public float beta
    cdef public int random_selection
    cdef public float prior
    cdef public dict weight  
    cdef public str inequal
    cdef public float feature_idx
    cdef public float threshold
    cdef public float estimator_weight
    cdef public float min_error
    cdef public cnp.ndarray best_prediction_p
    cdef public cnp.ndarray best_prediction_u
    cdef public float min_E_t
    cdef public dict sorted_indices
    cdef public dict sorted_weights
    cdef public float Z


    def __init__(self, prior, weight, random_selection, beta):
        self.beta = beta
        self.random_selection = random_selection
        self.prior = prior
        self.weight = weight   
        self.inequal = 'lt'
        self.feature_idx = 0.0
        self.threshold = 0.0
        self.estimator_weight = 1.0
        self.min_error = 0.5
        self.Z = 1.0
        self.best_prediction_p = None
        self.best_prediction_u = None
        self.min_E_t = float("inf")
        self.sorted_indices = {}
        self.sorted_weights = {}
        

    cdef tuple get_target(self, int num_p, int num_u):
        """Generates label arrays for positive, negative, and unlabeled samples.

        Args:
            num_p (int): The number of positive samples.
            num_u (int): The number of unlabeled samples.

        Returns:
            tuple: Three label arrays for positive, negative, and unlabeled samples respectively.
        """

        cdef cnp.ndarray[cnp.float32_t, ndim=1] y_train_p = np.ones(num_p, dtype = np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] y_train_n = -np.ones(num_p, dtype = np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] y_train_u = -np.ones(num_u, dtype = np.float32)

        return y_train_p, y_train_n, y_train_u
    

    cpdef fitted_predict(self, cnp.ndarray[cnp.float32_t, ndim=2] X):
        """Prediction function used after fitting progress.

        Args:
            X (cnp.ndarray[cnp.float32_t, ndim=2]): 2D array representing the input data.

        Returns:
            cnp.ndarray[cnp.float32_t, ndim=1]: An array of prediction results.
        """

        cdef cnp.ndarray[cnp.float32_t, ndim=1] x_column = X[:, int(self.feature_idx)]
        cdef cnp.ndarray[cnp.float32_t, ndim=1] predictions = np.ones(X.shape[0], dtype=np.float32)
        cdef int n_samples = X.shape[0]

        for i in range(n_samples):
            if self.inequal == 'lt':
                if x_column[i] <= self.threshold:
                    predictions[i] = -1
            else:
                if x_column[i] > self.threshold:
                    predictions[i] = -1

        return predictions


    cdef cnp.ndarray predict(self, cnp.ndarray x_train, int feature_idx, str inequal, float threshold):
        """The predict function of decision stump.

        Args:
        x_train (cnp.ndarray[cnp.float32_t, ndim=2]): The training data.
        feature_idx (int): The index of the feature to use for making predictions.
        inequal (str): A string representing the inequality used for decision-making. 
                       Expected values are 'lt' for less than or any other value for greater than.
        threshold (float): The value against which feature values are compared.
        Returns:
            np.array: prediction results
        """

        cdef cnp.ndarray[cnp.float32_t, ndim=1] x_column = x_train[:, int(feature_idx)]
        cdef cnp.ndarray[cnp.float32_t, ndim=1] predictions = np.ones(x_train.shape[0], dtype=np.float32)
        cdef int n_samples = x_train.shape[0]
        for i in range(n_samples):
            if inequal == 'lt':
                if x_column[i] <= threshold:
                    predictions[i] = -1
            else:
                if x_column[i] > threshold:
                    predictions[i] = -1
        return predictions

    cdef float _get_threshold_range(self, float range_min, float range_max, float lower_bound, float higher_bound, int step):
        """Determines the feature's threshold either randomly or based on a step size.
           Modifications can be applied based on what selection strategy is chosen.

        Args:
            range_min (float): Minimum value of the range.
            range_max (float): Maximum value of the range.
            lower_bound (float): Lower boundary for threshold determination.
            higher_bound (float): Upper boundary for threshold determination.
            step (int): Step size for threshold determination when not selecting randomly.

        Returns:
            float: Determined feature threshold.
        """

        cdef float threshold
            
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

    cpdef build(self, cnp.ndarray[cnp.float32_t, ndim=2] x_train_p, cnp.ndarray[cnp.float32_t, ndim=2] x_train_u, cnp.ndarray[cnp.float32_t, ndim=1] unique_counts, cnp.ndarray[cnp.float32_t, ndim=1] feature_mins, cnp.ndarray[cnp.float32_t, ndim=1] feature_maxs):
        """
        Constructs the PU decision stump based on the provided training data.

        Args:
            x_train_p (cnp.ndarray[cnp.float32_t, ndim=2]): Labeled positive training data.
            x_train_u (cnp.ndarray[cnp.float32_t, ndim=2]): Unlabeled training data.
            unique_counts (cnp.ndarray[cnp.float32_t, ndim=1]): The unique counts for each feature.
            feature_mins (cnp.ndarray[cnp.float32_t, ndim=1]): Minimum values for each feature.
            feature_maxs (cnp.ndarray[cnp.float32_t, ndim=1]): Maximum values for each feature.
        """

        cdef int n_samples_p = x_train_p.shape[0]
        cdef int n_samples_u = x_train_u.shape[0]
        cdef int n_features = x_train_p.shape[1]

        cdef cnp.ndarray[cnp.float32_t, ndim=1] y_train_p, y_train_n, y_train_u
        y_train_p, y_train_n, y_train_u = self.get_target(n_samples_p, n_samples_u)

        self.best_prediction_p = y_train_p
        self.best_prediction_u = y_train_u

        cdef int feature_i
        cdef float feature_min, feature_max
        cdef int num_interval
        cdef float lower_bound, higher_bound
        cdef int step
        cdef float threshold

        cdef cnp.ndarray[cnp.float32_t, ndim=1] curr_pred_p_lt, curr_pred_u_lt
        cdef float error_lt, error_n_lt, Z_lt, E_t_lt

        cdef cnp.ndarray[cnp.float32_t, ndim=1] curr_pred_p_gt, curr_pred_u_gt
        cdef float error_gt, error_n_gt, Z_gt, E_t_gt

        for feature_i in range(n_features):
            num_interval = <int>unique_counts[feature_i]
            feature_min = feature_mins[feature_i]
            feature_max = feature_maxs[feature_i]
            
            lower_bound = 0 - (1 / (num_interval-1))
            higher_bound = 1 + (1 / (num_interval-1))

            for step in range(NUM_STEPS):
                threshold = self._get_threshold_range(feature_min, feature_max, lower_bound, higher_bound, NUM_STEPS)
                curr_pred_p_lt = self.predict(x_train_p, feature_i, 'lt', threshold)
                curr_pred_u_lt = self.predict(x_train_u, feature_i, 'lt', threshold)
                error_lt, error_n_lt, Z_lt, E_t_lt = self.sum_error_calc(curr_pred_p_lt, curr_pred_u_lt)

                curr_pred_p_gt = self.predict(x_train_p, feature_i, 'gt', threshold)
                curr_pred_u_gt = self.predict(x_train_u, feature_i, 'gt', threshold)
                error_gt, error_n_gt, Z_gt, E_t_gt = self.sum_error_calc(curr_pred_p_gt, curr_pred_u_gt)

                if E_t_lt < E_t_gt:
                    error, error_n, Z, E_t = error_lt, error_n_lt, Z_lt, E_t_lt
                    inequal = 'lt'
                    curr_pred_p, curr_pred_u = curr_pred_p_lt, curr_pred_u_lt
                else:
                    error, error_n, Z, E_t = error_gt, error_n_gt, Z_gt, E_t_gt
                    inequal = 'gt'
                    curr_pred_p, curr_pred_u = curr_pred_p_gt, curr_pred_u_gt
                
                if error >= 0.5 or error_n < 0 or error < 0:
                    continue
                
                if E_t < self.min_E_t:
                    self.min_error = error
                    self.inequal = inequal
                    self.threshold = threshold
                    self.feature_idx = feature_i
                    self.best_prediction_p = curr_pred_p
                    self.best_prediction_u = curr_pred_u
                    self.min_E_t = E_t
                    self.Z = Z

        self.estimator_weight = 0.5 * np.log((1.0 - self.min_error + EPS) / (self.min_error + EPS))
        self.estimator_weight *= self.beta

    cdef sum_error_calc(self, cnp.ndarray curr_pred_p, cnp.ndarray curr_pred_u):
        """Calculate the error of the current decision stump based on its predictions.

        Args:
            curr_pred_p (cnp.ndarray): Current prediction results for the positive data.
            curr_pred_u (cnp.ndarray): Current prediction results for the unlabeled data.

        Returns:
            tuple: the weighted classification error rate, 
            the weighted classification error rate for the negative part,
            the sum of the data weights,
            and the the weighted classification error.
        """
        cdef float error_pp = np.sum(self.weight['p'][curr_pred_p != 1])
        cdef float error_un = np.sum(self.weight['u'][curr_pred_u != -1])
        cdef float error_pn = np.sum(self.weight['n'][curr_pred_p != -1])
        cdef float E_t = error_pp + error_un - error_pn

        error_un = error_un / np.sum(self.weight['u'])
        error_pp = self.prior * error_pp / np.sum(self.weight['p'])
        error_pn = self.prior * error_pn / np.sum(self.weight['n'])
        error_n = error_un - error_pn
        error = error_pp + error_n

        cdef float Z = np.sum(self.weight['p']) + np.sum(self.weight['u']) - np.sum(self.weight['n']) 

        return error, error_n, Z, E_t
