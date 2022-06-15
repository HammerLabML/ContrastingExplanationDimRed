import numpy as np

from counterfactual import CounterfactualExplanation
from utils import non_zero_threshold


class MemoryCounterfactual(CounterfactualExplanation):
    def __init__(self, X, y, sample_norm=1, pred_norm=2, C_reg=.1, C_diverse=1., **kwds):
        self.X = X
        self.Y = y
        self.sample_dist = self.__build_norm(sample_norm)
        self.pred_dist = self.__build_norm(pred_norm)
        self.C_reg = C_reg
        self.C_diverse = C_diverse

        super().__init__(**kwds)

    def __build_norm(self, norm_desc):
        return lambda x: np.linalg.norm(x, ord=norm_desc)

    def _compute_counterfactual(self, x_orig, y_cf, previous_cf_indices=[]):
        X_diff = self.X - x_orig
        Y_diff = self.Y - y_cf
        
        divesity_cost = []    # Diversity loss
        features_blacklist = []
        for idx in previous_cf_indices:
            features_blacklist += [int(i) for i in np.argwhere(np.abs(x_orig - self.X[idx,:]) > non_zero_threshold)]
        for i in range(self.X.shape[0]):
            divesity_cost.append(np.sum([np.abs(self.X[i,idx] - x_orig[idx]) > non_zero_threshold for idx in features_blacklist]))
        divesity_cost = np.array(divesity_cost)

        cost = [self.C_diverse * divesity_cost[i] + self.C_reg * self.sample_dist(X_diff[i,:].flatten()) + self.pred_dist(Y_diff[i,:].flatten()) for i in range(X_diff.shape[0])]
        
        for idx in previous_cf_indices: # Previously used samples must not be used again!
            cost[idx] = float("inf")
        
        idx = np.argmin(cost)

        return self.X[idx,:]

    def _compute_diverse_counterfactual(self, x_orig, y_cf, X_cf):
        indices = []
        for x_cf in X_cf:
            for i in range(self.X.shape[0]):
                if all(self.X[i,:] == x_cf):
                    indices.append(i)
                    break

        return self._compute_counterfactual(x_orig, y_cf, indices)

    def compute_explanation(self, x_orig, y_cf):
        return self._compute_counterfactual(x_orig, y_cf)
