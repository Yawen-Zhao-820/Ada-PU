import warnings

import numpy as np
from colorama import Fore
from tqdm import tqdm

from src.stump import DecisionStump


class AdaBoost_PU:
    """An AdaBoost classifier for PU Learning.
    
    This class implements the algorithm known as AdaBoost [1, 2] for PU Learning (Ada-PU), 
    Positive-unlabeled (PU) learning deals with binary classiﬁcation problems when 
    only positive (P) and unlabeled (U) data are available. Ada-PU follows the general 
    procedure of AdaBoost, while P data are regarded as positive and negative simultaneously. 
    Three distributions of PU data are maintained and updated in Ada-PU instead of one in the 
    ordinary supervised (PN) learning.
    
    .. versionadded:: 1.0.1
    
    Parameters
    ----------
    n_estimators : int
        determine the number of estimators
    prior : float
        prior probability of positive data
    random_selection : int
        whether to use the random selection strategy
    beta : float
        control parameter beta for smoothing the boosting stage
        
    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.
           
    .. [2] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.
    """
    def __init__(self, 
                 n_estimators, 
                 prior,
                 random_selection, 
                 beta):
        
        self.n_estimators = n_estimators
        self.random_selection = random_selection
        self.prior = prior
        self.beta = beta
        
        self.estimators = []
        self.estimator_weights = np.zeros(self.n_estimators, dtype=np.float64)
        
    
    @staticmethod
    def _validate_weight(estimator_weight: float) -> bool:
        """Check the estimator weight."""
       
        if not np.isfinite(estimator_weight) or np.isnan(estimator_weight):
            return True
        return False
        
    
    def _initial_weight(self,
                        sample_weights: dict,
                        n_samples_p: int, 
                        n_samples_u: int) -> None:
        """Initialize the samples' weight at the begining of boosting.

        Args:
            n_samples_p (int): number of positive samples
            n_samples_u (int): number of unlabeled samples
        """
        sample_weights['u'] = np.full(n_samples_u, (1 / n_samples_u)) 
        sample_weights['p'] = np.full(n_samples_p, (self.prior / n_samples_p))  
        sample_weights['n'] = np.full(n_samples_p, (self.prior / n_samples_p))
        
        return sample_weights


    def fit(self, x_train_p: np.array, x_train_u: np.array) -> None:
        """Implement the boosting progress.

        Args:
            x_train_p (np.array): positive training data
            x_train_u (np.array): unlabeled training data
        """
        sample_weights = {}

        n_samples_p = x_train_p.shape[0]
        n_samples_u = x_train_u.shape[0]

        sample_weights = self._initial_weight(sample_weights, n_samples_p, n_samples_u)

        y_train_p = np.ones(n_samples_p, dtype=np.float64)
        y_train_n = np.ones(n_samples_p, dtype=np.float64)
        y_train_n[y_train_n > 0] = -1

        y_train_u = np.ones(n_samples_u, dtype=np.float64)
        y_train_u[y_train_u > 0] = -1

        ada_predictions_u = np.zeros(n_samples_u)
        ada_predictions_p = np.zeros(n_samples_p)
    
        for i_boost in tqdm(range(self.n_estimators)):
            
            estimator = DecisionStump(self.prior, 
                                      sample_weights, 
                                      ada_predictions_u, 
                                      ada_predictions_p, 
                                      self.random_selection, 
                                      self.beta)
            
            estimator.build(x_train_p, x_train_u)
            
            if self._validate_weight(estimator.estimator_weight):
                warnings.warn(
                    "Estimator weight have reached infinite/nan values,"
                    f" at iteration {i_boost}, causing overflow. "
                    "Iterations stopped. Try another parameter.",
                    stacklevel=2,
                )
                break
            
            estimator_weight = estimator.estimator_weight

            ada_predictions_u = estimator.ada_predictions_u
            ada_predictions_p = estimator.ada_predictions_p
            
            if not i_boost == self.n_estimators - 1:
                sample_weights['u'] = self.weight_calc(estimator.weight['u'], 
                                                       -1, 
                                                       estimator.best_prediction_u, 
                                                       estimator_weight)
                
                sample_weights['p'] = self.weight_calc(estimator.weight['p'], 
                                                       +1, 
                                                       estimator.best_prediction_p, 
                                                       estimator_weight)
                
                sample_weights['n'] = self.weight_calc(estimator.weight['n'], 
                                                       -1, 
                                                       estimator.best_prediction_p, 
                                                       estimator_weight)
            
            self.estimators.append(estimator)
            self.estimator_weights[i_boost] = estimator_weight


    def predict(self, X: np.array) -> list:
        """Compute decision function of ``X`` for each boosting iteration.

        Args:
            X (np.array): X data for predicting with shape (n_samples, n_features)

        Returns:
            list: prediction results
        """ 
        pred = sum(
            estimator.fitted_predict(X) * estimator_weight
            for estimator, estimator_weight in zip(self.estimators, self.estimator_weights)
        )
        return np.sign(pred)
    
    
    def staged_predict(self, X: np.array, num_estimate: int) -> list:
        """Compute decision function of ``X`` for each boosting iteration.
        This method allows monitoring (i.e. determine error on testing/training set)
        after each boosting iteration.

        Args:
            X (np.array): X data for predicting with shape (n_samples, n_features)
            num_estimate (int): target stage round for predicting

        Returns:
            list: staged prediction
        """
        pred = sum(
            estimator.fitted_predict(X) * estimator_weight
            for estimator, estimator_weight in zip(self.estimators[:num_estimate], 
                                                   self.estimator_weights[:num_estimate])
        )
        return np.sign(pred)
    
    
    def weight_calc(self, weight: np.array, y_train: int, predictions: np.array, estimator_weight: float) -> np.array:
        """Calculate the samples' weight after each boost iteration.

        Args:
            weight (np.array): samples' weight of last iteration
            y_train (int): symbol for changing +/-. 
            predictions (np.array): predict results
            estimator_weight (float): weight of single estimator

        Returns:
            dict: samples' weight of current iteration
        """
        sample_weight = np.multiply(weight, np.exp(-1 * estimator_weight * y_train * predictions))
        return sample_weight
