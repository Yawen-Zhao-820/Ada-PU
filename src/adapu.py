import numpy as np
from decisionstump import DecisionStump
from tqdm import tqdm


class Adaboost:
    def __init__(self, n_clf, prior, is_random, nnpu, beta):
        self.n_clf = n_clf
        self.is_random = is_random
        self.nnpu = nnpu
        self.beta = beta
        self.prior = prior
        
        self.clfs = []
        self.unlabeled_losses = []

    def get_alpha_list(self):
        return np.array([clf.alpha for clf in self.clfs])

    def fit(self, x_train_p, x_train_u):
        """
        The part of training in the Adaboost.

        :param x_pos:       Positive data
        :param x_unlabeled: Unlabeled data
        :param y_train:     The label of data
        :return: None, only append the final classifier to the classifier list
        """
        weight_dict = {}

        n_samples_p, n_samples_u = x_train_p.shape[0], x_train_u.shape[0]

        weight_dict['u'] = np.full(n_samples_u, (1 / n_samples_u))
        weight_dict['p'] = np.full(n_samples_p, (1 / n_samples_p))
        weight_dict['n'] = np.full(n_samples_p, (1 / n_samples_p))

        y_train_p = np.ones(n_samples_p, dtype=np.float64)
        y_train_n = np.ones(n_samples_p, dtype=np.float64)
        y_train_n[y_train_n > 0] = -1

        y_train_u = np.ones(n_samples_u, dtype=np.float64)
        y_train_u[y_train_u > 0] = -1

        HX_u = np.zeros(n_samples_u)
        HX_p = np.zeros(n_samples_p)
    
        for i in tqdm(range(self.n_clf)):
            
            weak_classifier = DecisionStump(self.prior, 
                                            weight_dict, 
                                            HX_u, 
                                            HX_p, 
                                            self.unlabeled_losses, 
                                            self.is_random, 
                                            self.nnpu, 
                                            self.beta, 
                                            i)
            weak_classifier.build(x_train_p, x_train_u)
            
            alpha = weak_classifier.alpha

            HX_u = weak_classifier.HX_u
            HX_p = weak_classifier.HX_p

            weight_dict['u'] = self.weight_calc(weak_classifier.weight['u'], 
                                                y_train_u, 
                                                weak_classifier.best_u, 
                                                alpha)
            
            weight_dict['p'] = self.weight_calc(weak_classifier.weight['p'], 
                                                y_train_p, 
                                                weak_classifier.best_p, 
                                                alpha)
            
            weight_dict['n'] = self.weight_calc(weak_classifier.weight['n'], 
                                                y_train_n, 
                                                weak_classifier.best_p, 
                                                alpha)
            
            self.clfs.append(weak_classifier)
            self.unlabeled_losses.append(weak_classifier.min_error)

    def predict(self, x_train, num_clf, test=True):
        """
        The predict function of Adaboost.

        :param x_train: Training data
        :return: The result of prediction
        """
        prediction = np.zeros(x_train.shape[0])

        for i in range(num_clf):
            pred_val = self.clfs[i].predict(x_train, self.clfs[i].feature_idx, self.clfs[i].inequal, self.clfs[i].threshold)
            prediction += np.multiply(self.clfs[i].alpha, pred_val)
 
        if test:
            return np.sign(prediction)
        else:
            return prediction
    
    def weight_calc(self, weight, y_train, predictions, alpha):
        """
        Calculate the weight 
        ·
        :param x_train: data of training
        :param y_train: label
        :param predictions: Predicted value
        :return: weight
        """
        data_weight = np.multiply(weight, np.exp(-1 * alpha * y_train * predictions))
        return data_weight / np.sum(weight)
